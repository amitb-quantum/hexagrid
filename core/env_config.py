"""
Energia - Environment Configuration
Handles WSL2 TF + PyTorch CUDA context collision by isolating
PyTorch checks in a subprocess. Import this at the top of every
Energia module.
"""

import os
import sys
import subprocess

# ── Suppress TF noise ──────────────────────────────────────────────────────────
os.environ['TF_CPP_MIN_LOG_LEVEL']    = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS']   = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# ── TensorFlow GPU setup ───────────────────────────────────────────────────────
import tensorflow as tf

_gpus = tf.config.list_physical_devices('GPU')
if _gpus:
    for _gpu in _gpus:
        try:
            tf.config.experimental.set_memory_growth(_gpu, True)
        except RuntimeError:
            pass  # Already initialized

# ── Subprocess helper (avoids CUDA double-init segfault) ──────────────────────
def _run_isolated(code: str) -> str:
    """Run Python code in a clean subprocess, return stdout."""
    result = subprocess.run(
        [sys.executable, '-c', code],
        capture_output=True, text=True,
        env={**os.environ, 'TF_CPP_MIN_LOG_LEVEL': '3'}
    )
    return result.stdout.strip()

# ── Public report function ─────────────────────────────────────────────────────
def report():
    """Print full Energia environment health report."""

    # --- PyTorch check in subprocess to avoid CUDA segfault ---
    torch_info = _run_isolated("""
import warnings, torch
warnings.filterwarnings('ignore')
cuda = torch.cuda.is_available()
name = torch.cuda.get_device_name(0) if cuda else 'N/A'
mem  = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2) if cuda else 0
print(f"{torch.__version__}|{cuda}|{name}|{mem}GB")
""")

    torch_ver, torch_cuda, torch_device, torch_mem = "?", "?", "?", "?"
    if '|' in torch_info:
        parts = torch_info.split('|')
        torch_ver, torch_cuda, torch_device, torch_mem = parts

    # --- TFQ + Cirq ---
    tfq_info = _run_isolated("""
import os; os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
try:
    import tensorflow_quantum as tfq, cirq
    print(f"{tfq.__version__}|{cirq.__version__}|OK")
except Exception as e:
    print(f"ERR|ERR|{e}")
""")
    tfq_ver, cirq_ver, tfq_status = ("?","?","?")
    if '|' in tfq_info:
        tfq_ver, cirq_ver, tfq_status = tfq_info.split('|', 2)

    # --- Other packages ---
    try:
        import simpy
        simpy_ver = simpy.__version__
    except Exception as e:
        simpy_ver = f"ERROR: {e}"

    try:
        import numpy as np
        numpy_ver = np.__version__
    except Exception as e:
        numpy_ver = f"ERROR: {e}"

    try:
        import matplotlib
        mpl_ver = matplotlib.__version__
    except Exception as e:
        mpl_ver = f"ERROR: {e}"

    try:
        import pennylane as qml
        pl_ver = qml.__version__
    except Exception as e:
        pl_ver = f"ERROR: {e}"

    try:
        import pandas as pd
        pd_ver = pd.__version__
    except Exception as e:
        pd_ver = f"ERROR: {e}"

    try:
        import fastapi
        fa_ver = fastapi.__version__
    except Exception as e:
        fa_ver = f"ERROR: {e}"

    # --- Print report ---
    W  = '\033[93m'   # yellow
    G  = '\033[92m'   # green
    R  = '\033[91m'   # red
    B  = '\033[94m'   # blue
    NC = '\033[0m'    # reset
    OK = f"{G}OK{NC}"
    ER = f"{R}ERR{NC}"

    def status(val): return OK if "ERROR" not in str(val) and "ERR" not in str(val) else ER

    print(f"\n{B}{'='*54}{NC}")
    print(f"{B}   ⚡  ENERGIA ENVIRONMENT REPORT{NC}")
    print(f"{B}{'='*54}{NC}")
    print(f"  {'TensorFlow':<20} {tf.__version__:<12} {status(tf.__version__)}")
    print(f"  {'TF GPU':<20} {str([g.name for g in _gpus]):<12} {OK if _gpus else ER}")
    print(f"  {'TFQ':<20} {tfq_ver:<12} {OK if tfq_status=='OK' else ER}")
    print(f"  {'Cirq':<20} {cirq_ver:<12} {status(cirq_ver)}")
    print(f"  {'PennyLane':<20} {pl_ver:<12} {status(pl_ver)}")
    print(f"  {'-'*38}")
    print(f"  {'PyTorch':<20} {torch_ver:<12} {status(torch_ver)}")
    print(f"  {'PyTorch CUDA':<20} {torch_cuda:<12} {OK if torch_cuda=='True' else W+'CPU'+NC}")
    print(f"  {'GPU Device':<20} {torch_device}")
    print(f"  {'GPU Memory':<20} {torch_mem}")
    print(f"  {'-'*38}")
    print(f"  {'NumPy':<20} {numpy_ver:<12} {status(numpy_ver)}")
    print(f"  {'SimPy':<20} {simpy_ver:<12} {status(simpy_ver)}")
    print(f"  {'Pandas':<20} {pd_ver:<12} {status(pd_ver)}")
    print(f"  {'Matplotlib':<20} {mpl_ver:<12} {status(mpl_ver)}")
    print(f"  {'FastAPI':<20} {fa_ver:<12} {status(fa_ver)}")
    print(f"{B}{'='*54}{NC}")

    # ── Dependency conflict warnings ──────────────────────────────────────────
    print(f"\n{W}  ⚠  KNOWN CONFLICTS (non-blocking):{NC}")
    print(f"  TF 2.13 expects numpy<=1.24.3  → you have {numpy_ver}")
    print(f"  Resolve later when we upgrade TF. Energia Phase 1 is unaffected.")
    print(f"  protobuf pinned to 3.20.3 for TFQ compatibility.")
    print(f"{B}{'='*54}{NC}\n")


if __name__ == "__main__":
    report()
