# autoencoder.py — LSTM Autoencoder definition and training (PyTorch)

import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import (
    EMBEDDING_DIM, LSTM_HIDDEN, LSTM_LAYERS,
    AUTOENCODER_EPOCHS, AUTOENCODER_LR, AUTOENCODER_BATCH, SEED,
    CONTRASTIVE_LOSS, CONTRASTIVE_TEMPERATURE, CONTRASTIVE_MIN_WINDOWS,
    FALLBACK_LOG,
)

logging.getLogger("torch").setLevel(logging.ERROR)
torch.manual_seed(SEED)
np.random.seed(SEED)

FEATURE_DIM = 5   # body, upper_wick, lower_wick, gap, rel_range (legacy; actual dim is dynamic)

# PERF FIX 4A — auto-detect best available device
def get_device() -> str:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        print(f"[PERF] GPU detected: {name} — using CUDA")
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("[PERF] Apple Silicon GPU detected — using MPS")
        return "mps"
    print("[PERF] No GPU detected — using CPU. Install CUDA PyTorch for 5-10x speedup.")
    return "cpu"

# PERF FIX 4D — scale hidden size with seq_len to avoid overparameterisation
def _effective_hidden(seq_len: int, n_features: int) -> int:
    return min(LSTM_HIDDEN, max(16, seq_len * n_features * 2))

# Cap training windows to bound wall-clock time at large corpus sizes
MAX_TRAIN_WINDOWS = 2000   # stride-sample when corpus exceeds this


class SupervisedContrastiveLoss(nn.Module):
    """
    NT-Xent style contrastive loss with outcome-based labels.
    Positives: same direction, similar magnitude. Negatives: opposite direction.
    Ambiguous outcomes (label=0) are excluded from the loss.
    """
    def __init__(self, temperature: float = CONTRASTIVE_TEMPERATURE):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor,
                outcome_labels: torch.Tensor) -> torch.Tensor:
        mask = outcome_labels != 0
        if mask.sum() < 4:
            return torch.tensor(0.0, requires_grad=True)

        emb = embeddings[mask]
        labels = outcome_labels[mask]

        emb_norm = nn.functional.normalize(emb, dim=1)
        sim_matrix = torch.mm(emb_norm, emb_norm.t()) / self.temperature

        label_matrix = labels.unsqueeze(0) == labels.unsqueeze(1)
        eye = torch.eye(len(labels), dtype=torch.bool, device=emb.device)
        pos_mask = label_matrix & ~eye
        neg_mask = ~label_matrix

        if pos_mask.sum() == 0:
            return torch.tensor(0.0, requires_grad=True)

        exp_sim = torch.exp(sim_matrix)
        denom = (exp_sim * (pos_mask | neg_mask).float()).sum(dim=1, keepdim=True) + 1e-8
        log_prob = sim_matrix - torch.log(denom)
        loss = -(log_prob * pos_mask.float()).sum() / (pos_mask.sum() + 1e-8)
        return loss


class LSTMEncoder(nn.Module):
    def __init__(self, input_dim=FEATURE_DIM, hidden=LSTM_HIDDEN,
                 n_layers=LSTM_LAYERS, emb_dim=EMBEDDING_DIM):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, n_layers,
                            batch_first=True, dropout=0.1 if n_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden, emb_dim)

    def forward(self, x):
        # x: (batch, seq_len, features)
        _, (h, _) = self.lstm(x)
        # h: (n_layers, batch, hidden) — use last layer
        return self.fc(h[-1])


class LSTMDecoder(nn.Module):
    def __init__(self, emb_dim=EMBEDDING_DIM, hidden=LSTM_HIDDEN,
                 n_layers=LSTM_LAYERS, output_dim=FEATURE_DIM):
        super().__init__()
        self.fc = nn.Linear(emb_dim, hidden)
        self.lstm = nn.LSTM(hidden, hidden, n_layers,
                            batch_first=True, dropout=0.1 if n_layers > 1 else 0.0)
        self.out = nn.Linear(hidden, output_dim)

    def forward(self, z, seq_len):
        # z: (batch, emb_dim)
        h0 = torch.tanh(self.fc(z))               # (batch, hidden)
        h0 = h0.unsqueeze(0).repeat(self.lstm.num_layers, 1, 1)
        c0 = torch.zeros_like(h0)
        inp = h0[-1].unsqueeze(1).repeat(1, seq_len, 1)  # repeat as input
        out, _ = self.lstm(inp, (h0, c0))
        return self.out(out)                       # (batch, seq_len, output_dim)


class LSTMAutoencoder(nn.Module):
    def __init__(self, seq_len, input_dim=FEATURE_DIM,
                 hidden=LSTM_HIDDEN, n_layers=LSTM_LAYERS, emb_dim=EMBEDDING_DIM):
        super().__init__()
        self.seq_len = seq_len
        self.encoder = LSTMEncoder(input_dim, hidden, n_layers, emb_dim)
        self.decoder = LSTMDecoder(emb_dim, hidden, n_layers, input_dim)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z, self.seq_len)

    def encode(self, x):
        with torch.no_grad():
            return self.encoder(x)


def _build_windows(feat_array: np.ndarray, seq_len: int) -> np.ndarray:
    """Sliding windows of shape (n_windows, seq_len, n_features)."""
    n = len(feat_array)
    if n < seq_len:
        return np.empty((0, seq_len, feat_array.shape[1]), dtype=np.float32)
    idx = np.arange(n - seq_len + 1)
    return np.stack([feat_array[i: i + seq_len] for i in idx], axis=0).astype(np.float32)


def fallback_embedding(window: np.ndarray,
                       embedding_dim: int = EMBEDDING_DIM) -> np.ndarray:
    """
    BUG4_FIX: Geometrically meaningful fallback when LSTM is unavailable.

    Normalizes each feature column to zero mean / unit std within the window,
    flattens, zero-pads to embedding_dim, and L2-normalizes. Cosine similarity
    on these embeddings reflects shape similarity regardless of feature scale.
    """
    col_means = window.mean(axis=0)
    col_stds  = window.std(axis=0) + 1e-8
    normed = (window - col_means) / col_stds
    flat = normed.flatten().astype(np.float32)
    emb = np.zeros(embedding_dim, dtype=np.float32)
    n = min(len(flat), embedding_dim)
    emb[:n] = flat[:n]
    norm = np.linalg.norm(emb) + 1e-8
    return emb / norm


def train_autoencoder(feat_array: np.ndarray, seq_len: int,
                      device: str = "cpu",
                      outcome_returns: np.ndarray | None = None,
                      is_retrain: bool = False,
                      n_features: int | None = None):
    if feat_array.ndim == 3:
        windows = feat_array.astype(np.float32)
    else:
        windows = _build_windows(feat_array, seq_len)
    min_windows = max(8, seq_len * 2 + 4)  # BUG4_FIX: was 10+seq_len*5 (too high for early warmup)
    if len(windows) < min_windows:
        raise ValueError(f"Too few windows ({len(windows)}) for seq_len={seq_len}")

    n_windows = len(windows)

    # PERF FIX 4C — adaptive epochs: fewer for retrains, scale with corpus
    if n_windows < 50:
        epochs = 15
    elif n_windows < 100:
        epochs = 25
    elif is_retrain:
        epochs = 20   # retrains need fewer epochs — model already has good weights
    else:
        epochs = AUTOENCODER_EPOCHS

    # PERF FIX 4 — cap training windows via stride sampling to bound wall-clock time
    if n_windows > MAX_TRAIN_WINDOWS:
        stride = n_windows // MAX_TRAIN_WINDOWS
        windows = windows[::stride][:MAX_TRAIN_WINDOWS]
        if outcome_returns is not None and len(outcome_returns) == n_windows:
            outcome_returns = outcome_returns[::stride][:MAX_TRAIN_WINDOWS]
        n_windows = len(windows)

    input_dim = windows.shape[2]
    # PERF FIX 4D — scale hidden size with seq_len
    eff_hidden = _effective_hidden(seq_len, input_dim)

    # Reproducible model init: reset seed keyed on seq_len so every run produces identical weights
    torch.manual_seed(SEED + seq_len)
    np.random.seed(SEED + seq_len)
    model = LSTMAutoencoder(seq_len, input_dim=input_dim, hidden=eff_hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=AUTOENCODER_LR)
    mse_criterion = nn.MSELoss()

    # Supervised contrastive loss setup
    use_contrastive = (
        CONTRASTIVE_LOSS
        and outcome_returns is not None
        and len(outcome_returns) == n_windows
        and n_windows >= CONTRASTIVE_MIN_WINDOWS
    )
    contrastive_fn = SupervisedContrastiveLoss(CONTRASTIVE_TEMPERATURE) if use_contrastive else None

    # Build outcome labels for contrastive loss
    outcome_label_arr = None
    if use_contrastive:
        _std = float(np.nanstd(outcome_returns))
        outcome_label_arr = np.zeros(n_windows, dtype=np.int32)
        for i, r in enumerate(outcome_returns):
            if np.isnan(r):
                continue
            if r > 0.3 * _std:
                outcome_label_arr[i] = 1
            elif r < -0.3 * _std:
                outcome_label_arr[i] = -1
        # log info
        n_pos = int((outcome_label_arr == 1).sum())
        n_neg = int((outcome_label_arr == -1).sum())
        print(f"[AE] Contrastive loss: {n_windows} windows, {n_pos} pos, {n_neg} neg labels")
    else:
        if CONTRASTIVE_LOSS and n_windows < CONTRASTIVE_MIN_WINDOWS:
            with open(FALLBACK_LOG, "a") as _f:
                _f.write(f"[AE] Contrastive fallback to MSE: only {n_windows} windows\n")

    labels_tensor = (
        torch.from_numpy(outcome_label_arr).long().to(device)
        if outcome_label_arr is not None else None
    )

    ds = TensorDataset(torch.from_numpy(windows),
                       torch.arange(n_windows) if use_contrastive
                       else torch.zeros(n_windows, dtype=torch.long))
    loader = DataLoader(ds, batch_size=min(AUTOENCODER_BATCH, n_windows), shuffle=True,
                        drop_last=False)

    model.train()
    for _ in range(epochs):
        for (batch, idx) in loader:
            batch = batch.to(device)
            recon = model(batch)
            loss = mse_criterion(recon, batch)

            if use_contrastive and labels_tensor is not None:
                emb = model.encoder(batch)
                emb_norm = nn.functional.normalize(emb, dim=1)
                batch_labels = labels_tensor[idx]
                c_loss = contrastive_fn(emb_norm, batch_labels)
                loss = loss + 0.5 * c_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    model.eval()
    if n_features is not None:
        with torch.no_grad():
            emb_tensor = model.encode(torch.from_numpy(windows).to(device))
        return model, emb_tensor.detach().cpu().numpy()
    return model


def compute_embeddings(model: LSTMAutoencoder, feat_array: np.ndarray,
                       seq_len: int, device: str = "cpu") -> np.ndarray:
    """
    Returns embeddings of shape (n_windows, EMBEDDING_DIM).
    Window i corresponds to bars [i .. i+seq_len-1].
    """
    windows = _build_windows(feat_array, seq_len)
    if len(windows) == 0:
        return np.empty((0, EMBEDDING_DIM), dtype=np.float32)

    model.eval()
    with torch.no_grad():
        t = torch.from_numpy(windows).to(device)
        embs = model.encode(t).cpu().numpy()

    # Validate — fallback caller checks for NaN
    if np.isnan(embs).any():
        raise ValueError("NaN in embeddings")
    return embs
