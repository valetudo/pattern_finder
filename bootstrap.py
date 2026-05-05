import subprocess
import sys


BASE_RUNTIME_PACKAGES = [
    "pandas",
    "numpy",
    "yfinance",
    "matplotlib",
    "scipy",
    "torch",
    "lightgbm",
    "hmmlearn",
    "dtaidistance",
    "scikit-learn",
    "tqdm",
]

IMPORT_NAME_MAP = {
    "scikit-learn": "sklearn",
}


def ensure_runtime_dependencies(extra_packages: list[str] | None = None):
    required = list(BASE_RUNTIME_PACKAGES)
    if extra_packages:
        required.extend(extra_packages)

    pkg_map = {
        "sklearn": "scikit-learn",
    }

    for pkg in required:
        import_name = IMPORT_NAME_MAP.get(pkg, pkg.replace("-", "_"))
        try:
            __import__(import_name)
        except ImportError:
            install_name = pkg_map.get(import_name, pkg)
            print(f"[bootstrap] Installing {install_name}...")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", install_name, "-q"]
            )