#!/usr/bin/env python
"""
Crop-JEPA Stage B - headless cluster version of SW_Crop_JEPA_StageB.ipynb.
Scientific code = the notebook, cell for cell. Only these things differ:
  1. the configuration cell reads CJ_* environment variables (set by config.env / the job scripts),
  2. matplotlib uses the non-interactive Agg backend and display() prints text,
  3. CJ_SKIP_REPORT=1 lets parallel shard workers / the pilot stop before the reporting cells.
"""
import os, json
import matplotlib
matplotlib.use("Agg")


def _env(name, default=None):
    v = os.environ.get(name)
    return default if v is None or v == "" else v


def _flag(name, default=False):
    v = _env(name)
    return default if v is None else v.strip().lower() in ("1", "true", "yes", "on")


def _list(name, default):
    v = _env(name)
    return list(default) if v is None else [x.strip() for x in v.split(",") if x.strip()]


RUN_MODE = _env("CJ_MODE", "smoke")
PROFILE = _env("CJ_PROFILE", "auto")
DATA_ROOT = _env("CJ_DATA_ROOT", "./lucas_data")
GROUP_REGEX = r'^LUCAS\d{4}_(\d+)_[A-Za-z0-9]+\.jpg$'
LIMIT_PER_CLASS = None
CACHE_RES = int(_env("CJ_CACHE_RES", 192))
OUTPUT_DIR = _env("CJ_OUTPUT_DIR", "./outputs_stageB")
RESUME_FROM_DIR = None
CACHE_DATASET_DIR = None

ARMS = _list("CJ_ARMS", ["ijepa", "veg_pos", "veg_neg", "pos_matched"])
ENABLE_ATTN_ARM = _flag("CJ_ENABLE_ATTN", False)
ENABLE_LEJEPA = _flag("CJ_ENABLE_LEJEPA", False)
ENABLE_TAU_SWEEP = _flag("CJ_ENABLE_TAU_SWEEP", False)
ENABLE_SIZE_AXIS = _flag("CJ_ENABLE_SIZE_AXIS", False)
BASELINES = _list("CJ_BASELINES", ["random_init", "supervised"])
SEEDS = [int(s) for s in _list("CJ_SEEDS", ["0", "1", "2", "3", "4", "5"])]
TIER2_SEEDS = [int(s) for s in _list("CJ_TIER2_SEEDS", ["0", "1", "2"])]
TARGET_STEPS = int(_env("CJ_TARGET_STEPS", 9000))
PILOT_STEPS = int(_env("CJ_PILOT_STEPS", 26000))
VEG_TAU = 0.5
VEG_EPS = 0.2
PRIOR_IMAGES = 4000
CTX_TOKENS_CAP = None
EQUIV_MARGIN_PP = 1.5
BOOTSTRAP_B = 2000
ALPHA = 0.05
PRIMARY_ALTERNATIVE = "greater"
SHARD_ID, NUM_SHARDS = int(_env("CJ_SHARD_ID", 0)), int(_env("CJ_NUM_SHARDS", 1))
OVERRIDES = json.loads(_env("CJ_OVERRIDES", "{}"))
if _env("CJ_MAX_SESSION_HOURS"):
    OVERRIDES["max_session_hours"] = float(_env("CJ_MAX_SESSION_HOURS"))
ALLOW_PROFILE_CHANGE = False
ZIP_CHECKPOINTS = _flag("CJ_ZIP_CHECKPOINTS", False)

DOWNLOAD_THREADS = 20
ALLOW_INCOMPLETE_DOWNLOAD = False
IGNORE_GATE0 = _flag("CJ_IGNORE_GATE0", False)
DELETE_LAST_PT_AFTER_EVAL = _flag("CJ_DELETE_LAST_PT", False)
SKIP_REPORT = _flag("CJ_SKIP_REPORT", False)

# ======================= notebook cell 4 =======================
import os, sys, re, json, time, math, random, hashlib, shutil, platform, copy, warnings, traceback, zipfile, contextlib
import datetime, itertools, urllib.request, urllib.parse
from dataclasses import dataclass, asdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps

SESSION_START = time.time()
_missing = []
try:
    import torchvision
    from torchvision.ops import roi_align
except Exception:
    _missing.append("torchvision")
try:
    import sklearn
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.exceptions import ConvergenceWarning
except Exception:
    _missing.append("scikit-learn")
try:
    import scipy
    from scipy.stats import pearsonr, spearmanr, rankdata
except Exception:
    _missing.append("scipy")
try:
    import matplotlib
    import matplotlib.pyplot as plt
except Exception:
    _missing.append("matplotlib")
if _missing:
    raise ImportError(f"Missing packages: {_missing}. Install with: pip install {' '.join(_missing)}")

def display(x):
    try:
        print(x.to_string() if hasattr(x, "to_string") else x)
    except Exception:
        print(x)

IS_KAGGLE = os.path.isdir("/kaggle/input") or bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE"))
IS_WINDOWS = platform.system() == "Windows"
HAS_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda:0" if HAS_CUDA else "cpu")


def describe_gpu():
    if not HAS_CUDA:
        return {"name": "CPU only", "vram_gb": 0.0, "capability": "-", "bf16": False, "n_gpus": 0}
    p = torch.cuda.get_device_properties(0)
    return {"name": p.name, "vram_gb": round(p.total_memory / 1024**3, 2), "capability": f"{p.major}.{p.minor}",
            "bf16": p.major >= 8, "n_gpus": torch.cuda.device_count()}


def system_ram_gib():
    """Total physical RAM in GiB. psutil if installed, the Win32 API on Windows, sysconf elsewhere."""
    try:
        import psutil
        return psutil.virtual_memory().total / 1024**3
    except Exception:
        pass
    if IS_WINDOWS:
        try:
            import ctypes

            class _MemStatus(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            st = _MemStatus()
            st.dwLength = ctypes.sizeof(_MemStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
                return st.ullTotalPhys / 1024**3
        except Exception:
            pass
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3
    except Exception:
        return 8.0


def free_disk_gib(path="."):
    p = Path(path).resolve()
    while not p.exists() and p != p.parent:
        p = p.parent
    return shutil.disk_usage(p).free / 1024**3


GPU = describe_gpu()
RAM_GIB = system_ram_gib()
if HAS_CUDA:
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:  # catches "no kernel image is available" early (old GPUs vs new PyTorch builds)
        _ = (torch.ones(8, 8, device=DEVICE) @ torch.ones(8, 8, device=DEVICE)).sum().item()
    except Exception as e:
        raise RuntimeError(f"CUDA is visible but kernels do not run on {GPU['name']}: {e}. "
                           "On Kaggle switch the accelerator to 'GPU T4 x2'; on a laptop reinstall PyTorch for your CUDA version.")
    if float(GPU["capability"]) < 7.0:
        print("WARNING: Pascal-class GPU (e.g. P100): no tensor cores, mixed precision gives little; prefer a T4 on Kaggle.")

print(f"Python {sys.version.split()[0]} | PyTorch {torch.__version__} | torchvision {torchvision.__version__} | "
      f"numpy {np.__version__} | pandas {pd.__version__} | scikit-learn {sklearn.__version__} | scipy {scipy.__version__}")
print(f"Platform: {platform.system()} | Kaggle: {IS_KAGGLE} | Device: {DEVICE}")
print(f"GPU: {GPU['name']} | VRAM {GPU['vram_gb']} GB | compute capability {GPU['capability']} | bf16: {GPU['bf16']} | GPUs visible: {GPU['n_gpus']}")
print(f"System RAM: {RAM_GIB:.1f} GiB | free disk here ({Path('.').resolve()}): {free_disk_gib('.'):.1f} GiB")
if IS_WINDOWS:
    print("Windows detected: no DataLoader worker processes are used, so the notebook runs unchanged on Windows.")
if "onedrive" in str(Path(".").resolve()).lower():
    print("WARNING: the working folder is inside OneDrive. The image cache (~2 GB) and checkpoints would be synced, and "
          "sync locks can make file renames fail. Pause OneDrive syncing while this runs, or move the project folder "
          "outside OneDrive (e.g. C:\\research\\crop_jepa). File writes are retried, but syncing still costs disk and time.")

plt.show = lambda *a, **k: None   # headless


# ======================= notebook cell 6 =======================
@dataclass
class Config:
    profile: str = "laptop_4gb"
    # model (ViT-Tiny/16 by default)
    img_size: int = 160
    patch_size: int = 16
    enc_dim: int = 192
    enc_depth: int = 12
    enc_heads: int = 3
    pred_dim: int = 128
    pred_depth: int = 4
    pred_heads: int = 4
    grad_ckpt: bool = False
    # optimisation (I-JEPA style schedules, all defined over optimizer steps)
    batch_size: int = 128
    target_steps: int = 9000        # core grid budget; epochs = ceil(target_steps / steps_per_epoch)
    pilot_steps: int = 26000        # Gate 0 budget
    warmup_frac: float = 0.1
    blr: float = 5e-4               # peak lr per 256 images; peak = blr * batch / 256
    min_lr: float = 1e-6
    wd_start: float = 0.04
    wd_end: float = 0.4
    ema_start: float = 0.996
    ema_end: float = 1.0
    grad_clip: float = 3.0
    precision: str = "bf16"         # "bf16" (Ampere+), "fp16" (T4, with loss scaling), "fp32"
    # target placement
    num_targets: int = 4
    target_scale: tuple = (0.15, 0.2)
    target_aspect: tuple = (0.75, 1.5)
    context_scale: tuple = (0.85, 1.0)
    min_context: int = 10
    veg_index: str = "chromatic"    # ExG = (2g - r - b) / (r + g + b) on un-normalised [0, 1] pixels
    veg_tau: float = 0.5
    veg_eps: float = 0.2
    ctx_tokens_cap: object = None   # None or int: cap context tokens per image (equalises arms)
    prior_images: int = 4000        # augmented training crops used to estimate the pos_matched prior
    prior_seed: int = 424242
    accept_batches: int = 200       # batches for the pos_matched acceptance check
    # augmentation
    crop_scale: tuple = (0.3, 1.0)
    hflip: bool = True
    # data
    cache_res: int = 192
    val_frac: float = 0.15
    test_frac: float = 0.15
    split_seed: int = 42
    dedup_hamming: int = 4
    draw_seed: int = 1234
    # evaluation
    shots: tuple = (1, 5, 10, 25, 50, 100)
    percents: tuple = (10, 100)
    n_draws: int = 5
    probe_C: tuple = (0.001, 0.01, 0.1, 1.0, 10.0)
    probe_max_iter: int = 2000
    knn_k: int = 20
    knn_t: float = 0.07
    eval_batch: int = 256
    # supervised baseline (upper reference, not matched)
    sup_epochs: int = 100
    sup_blr: float = 5e-4
    sup_wd: float = 0.05
    label_smoothing: float = 0.1
    sup_eval_every: int = 5
    sup_crop_scale: tuple = (0.35, 1.0)
    sup_regimes: tuple = ("1pct", "10pct", "100pct")   # "1pct" is used only if that regime exists in the label draws
    # monitoring, checkpoints, time budget
    monitor_n: int = 512
    knn_monitor_every: int = 10
    knn_monitor_train: int = 2000
    ckpt_every: int = 5
    max_session_hours: float = 1e9
    eval_reserve_min: float = 15.0
    # statistics (pre-registered; see section 4.1)
    bootstrap_B: int = 2000
    primary_regimes: tuple = ("10shot", "25shot", "10pct")
    equiv_margin_pp: float = 1.5
    equiv_ci: float = 0.90          # TOST at alpha = 0.05 <=> 90 % interval inside the margin
    alpha: float = 0.05
    primary_alternative: str = "greater"
    # gates
    rankme_floor_frac: float = 0.05  # Gate 0: RankMe must stay >= this fraction of enc_dim
    g0_knn_margin_pp: float = 3.0    # Gate 0: final online k-NN F1 must beat random_init by this many points (design choice)
    pos_var_warn: float = 0.10       # Gate -1: warn if patch position explains more than this share of ExG variance
    # Tier 2
    tau_sweep: tuple = (0.25, 1.0)   # 0.5 = core veg_pos, inf = core ijepa (both re-used)
    size_fracs: tuple = (0.1, 0.3)   # 1.0 = core runs (re-used)
    lejepa_lambda: float = 0.02      # [UNVERIFIED]
    lejepa_slices: int = 256         # [UNVERIFIED]
    lejepa_knots: int = 17           # [UNVERIFIED]
    lejepa_tmax: float = 3.0         # [UNVERIFIED] quadrature range [0, tmax] (symmetric integrand)
    lejepa_n_global: int = 2         # [UNVERIFIED]
    lejepa_n_local: int = 6          # [UNVERIFIED]
    lejepa_local_size: int = 96      # pixels; must be a multiple of patch_size
    lejepa_global_scale: tuple = (0.3, 1.0)
    lejepa_local_scale: tuple = (0.05, 0.3)
    lejepa_proj_hidden: int = 1024   # [UNVERIFIED] projector width
    lejepa_proj_dim: int = 128       # [UNVERIFIED]
    lejepa_batch_size: int = 64
    lejepa_blr: float = 5e-4         # [UNVERIFIED]
    lejepa_wd: float = 0.05          # [UNVERIFIED]
    lejepa_steps: int = 0            # 0 = same as target_steps


VIT_S = dict(enc_dim=384, enc_heads=6, pred_dim=192, pred_depth=6, pred_heads=6)
PROFILES = {
    "cpu_debug":        dict(img_size=64, patch_size=8, enc_dim=96, enc_depth=4, enc_heads=3, pred_dim=64, pred_depth=2,
                             pred_heads=2, batch_size=32, target_steps=60, pilot_steps=90, blr=2e-3, sup_blr=2e-3,
                             precision="fp32", cache_res=72, min_context=4, shots=(1, 5, 10), percents=(10, 100), n_draws=2,
                             sup_epochs=12, sup_eval_every=4, knn_monitor_every=2, knn_monitor_train=200, monitor_n=64,
                             ckpt_every=1, bootstrap_B=200, eval_batch=64, prior_images=128, accept_batches=12,
                             primary_regimes=("5shot", "10shot", "10pct"), lejepa_local_size=32, lejepa_batch_size=16,
                             lejepa_n_local=2, lejepa_slices=64, lejepa_proj_hidden=128, lejepa_proj_dim=32),
    "laptop_4gb":       dict(),                                                   # ViT-Tiny/16 @160, bf16 (main profile)
    "laptop_4gb_vits":  dict(img_size=224, batch_size=64, **VIT_S),               # ViT-S/16 @224, slow but fits
    "kaggle_t4_fast":   dict(precision="fp16", max_session_hours=11.0),           # ViT-Tiny/16 @160 on a T4
    "kaggle_t4":        dict(img_size=224, batch_size=128, precision="fp16", max_session_hours=11.0, **VIT_S),  # confirmation only
    "kaggle_p100_fast": dict(precision="fp32", max_session_hours=11.0),
    "kaggle_p100":      dict(img_size=224, batch_size=128, precision="fp32", max_session_hours=11.0, **VIT_S),
    "big_gpu":          dict(precision="bf16" if GPU["bf16"] else "fp16"),        # ViT-Tiny @160 (comparable with the laptop)
    "big_gpu_vits":     dict(img_size=224, batch_size=128, precision="bf16" if GPU["bf16"] else "fp16", **VIT_S),
}
KNOWN_ARMS = ("ijepa", "veg_pos", "veg_neg", "pos_matched", "attn")


def auto_profile():
    """Stage B's core grid is ViT-Tiny/16 @160 everywhere; ViT-S profiles are chosen explicitly (confirmation runs)."""
    if not HAS_CUDA:
        return "cpu_debug"
    name, vram = GPU["name"].lower(), GPU["vram_gb"]
    if vram < 6:
        return "laptop_4gb"
    if "t4" in name:
        return "kaggle_t4_fast"
    if "p100" in name:
        return "kaggle_p100_fast"
    return "big_gpu"


def hardware_precision():
    if not HAS_CUDA:
        return "fp32"
    return "bf16" if GPU["bf16"] else ("fp16" if float(GPU["capability"]) >= 7.0 else "fp32")


assert RUN_MODE in ("smoke", "benchmark", "pilot", "full"), RUN_MODE
assert PRIMARY_ALTERNATIVE in ("greater", "two-sided"), PRIMARY_ALTERNATIVE
_unknown = [a for a in ARMS if a not in KNOWN_ARMS]
if _unknown:
    raise ValueError(f"Unknown arms {_unknown}; Stage B arms are {KNOWN_ARMS} (ms / veg_ms were dropped).")
ARMS = list(dict.fromkeys(ARMS))   # the attn arm normally runs in Tier 2 (section 15) on TIER2_SEEDS
QUICK = dict(target_steps=int(TARGET_STEPS), pilot_steps=int(PILOT_STEPS), veg_tau=float(VEG_TAU), veg_eps=float(VEG_EPS),
             prior_images=int(PRIOR_IMAGES), ctx_tokens_cap=CTX_TOKENS_CAP, equiv_margin_pp=float(EQUIV_MARGIN_PP),
             bootstrap_B=int(BOOTSTRAP_B), cache_res=int(CACHE_RES), alpha=float(ALPHA), primary_alternative=PRIMARY_ALTERNATIVE)
STAT_KEYS = ("veg_tau", "veg_eps", "equiv_margin_pp", "alpha", "primary_alternative")

profile_name = auto_profile() if PROFILE == "auto" else PROFILE
if RUN_MODE == "smoke":
    cfg = Config(**{**PROFILES["cpu_debug"], **{k: QUICK[k] for k in STAT_KEYS}, "precision": hardware_precision()})
    cfg.profile = "smoke"
    SEEDS_RUN, TIER2_SEEDS_RUN = [0, 1, 2], [0, 1]
else:
    if profile_name not in PROFILES:
        raise ValueError(f"Unknown profile {profile_name}; choose from {list(PROFILES)}")
    cfg = Config(**{**PROFILES[profile_name], **QUICK, **OVERRIDES})
    cfg.profile = profile_name
    SEEDS_RUN, TIER2_SEEDS_RUN = list(SEEDS), list(TIER2_SEEDS)
if cfg.precision == "bf16" and not GPU["bf16"]:
    print("bf16 not supported here; switching to", hardware_precision())
    cfg.precision = hardware_precision()
if IS_KAGGLE and cfg.max_session_hours > 11.5:
    cfg.max_session_hours = 11.0
if cfg.lejepa_steps <= 0:
    cfg.lejepa_steps = cfg.target_steps
assert cfg.img_size % cfg.patch_size == 0 and cfg.enc_dim % cfg.enc_heads == 0 and cfg.pred_dim % cfg.pred_heads == 0
assert cfg.enc_dim % 4 == 0 and cfg.pred_dim % 4 == 0, "sin-cos position embeddings need dims divisible by 4"
assert cfg.lejepa_local_size % cfg.patch_size == 0, "lejepa_local_size must be a multiple of patch_size"
for _a in ("ijepa", "veg_pos"):
    if _a not in ARMS:
        print(f"WARNING: arm '{_a}' is not in ARMS — the pre-registered primary analysis cannot be computed.")

BASE_OUT = Path(OUTPUT_DIR) if OUTPUT_DIR else (Path("/kaggle/working/outputs_stageB") if IS_KAGGLE else Path("./outputs_stageB"))
OUT = BASE_OUT / "smoke" if RUN_MODE == "smoke" else BASE_OUT
for sub in ["runs", "cache", "figures", "stats", "priors", "paper_tables"]:
    (OUT / sub).mkdir(parents=True, exist_ok=True)

if RESUME_FROM_DIR and Path(RESUME_FROM_DIR).exists() and RUN_MODE != "smoke":
    copied = 0
    for src in Path(RESUME_FROM_DIR).rglob("*"):
        dst = OUT / src.relative_to(RESUME_FROM_DIR)
        if src.is_file() and not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied += 1
    print(f"Resumed from {RESUME_FROM_DIR}: copied {copied} files into {OUT}")

GRID = cfg.img_size // cfg.patch_size
print(f"Mode: {RUN_MODE} | profile: {cfg.profile} | precision: {cfg.precision} | output: {OUT.resolve()}")
print(f"Encoder: ViT dim {cfg.enc_dim} x depth {cfg.enc_depth}, patch {cfg.patch_size}, {cfg.img_size}px -> {GRID}x{GRID} tokens | "
      f"predictor dim {cfg.pred_dim} x depth {cfg.pred_depth} | batch {cfg.batch_size}")
print(f"Budget: {cfg.target_steps} optimizer steps per core run, {cfg.pilot_steps} for the Gate-0 pilot (epochs derived from the data size)")
print(f"Seeds: {SEEDS_RUN} | arms: {ARMS} | baselines: {BASELINES} | Tier-2 seeds: {TIER2_SEEDS_RUN} | shard {SHARD_ID}/{NUM_SHARDS}")
print(f"Tier 2: attn={ENABLE_ATTN_ARM or RUN_MODE == 'smoke'} tau_sweep={ENABLE_TAU_SWEEP or RUN_MODE == 'smoke'} "
      f"size_axis={ENABLE_SIZE_AXIS or RUN_MODE == 'smoke'} lejepa={ENABLE_LEJEPA or RUN_MODE == 'smoke'} (smoke mode exercises all of them)")


# ======================= notebook cell 8 =======================
class TimeBudgetExceeded(Exception):
    """Raised when the session is about to hit its time limit; progress is checkpointed first."""


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if HAS_CUDA:
        torch.cuda.manual_seed_all(seed)


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def _retry_io(fn, tries=8, wait=0.5):
    """Windows: antivirus, the search indexer or OneDrive can hold a file for a moment; retry instead of crashing."""
    for t in range(tries):
        try:
            return fn()
        except PermissionError:
            if t == tries - 1:
                raise
            time.sleep(wait * (t + 1))


def _tmp_path(path):
    return path.with_name(path.name + ".tmp")


def _atomic_replace(tmp, path):
    _retry_io(lambda: os.replace(tmp, path))


def save_json(obj, path):
    path = Path(path)
    tmp = _tmp_path(path)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)
    _atomic_replace(tmp, path)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_text(text, path):
    path = Path(path)
    tmp = _tmp_path(path)
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    _atomic_replace(tmp, path)


def save_csv(df, path, **kw):
    path = Path(path)
    tmp = _tmp_path(path)
    df.to_csv(tmp, index=kw.pop("index", False), **kw)
    _atomic_replace(tmp, path)


def save_npy(arr, path):
    path = Path(path)
    tmp = _tmp_path(path)
    with open(tmp, "wb") as f:
        np.save(f, arr)
    _atomic_replace(tmp, path)


def atomic_torch_save(obj, path):
    """Write to a temp file then rename, so a crash mid-save never corrupts the previous checkpoint."""
    path = Path(path)
    tmp = _tmp_path(path)
    torch.save(obj, tmp)
    _atomic_replace(tmp, path)


def safe_torch_load(path, map_location="cpu"):
    """weights_only=True first (safe). Checkpoints this notebook wrote itself inside OUT may fall back to
    weights_only=False (they contain plain RNG states and logs); anything else is refused."""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except Exception as e:
        own = str(Path(path).resolve()).startswith(str(OUT.resolve()))
        if not own:
            raise RuntimeError(f"Refusing to unpickle {path} with weights_only=False (not written by this notebook): {e}")
        return torch.load(path, map_location=map_location, weights_only=False)


TRAIN_KEYS = ["img_size", "patch_size", "enc_dim", "enc_depth", "enc_heads", "pred_dim", "pred_depth", "pred_heads",
              "batch_size", "warmup_frac", "blr", "min_lr", "wd_start", "wd_end", "ema_start", "ema_end", "grad_clip",
              "precision", "num_targets", "target_scale", "target_aspect", "context_scale", "min_context", "veg_index",
              "ctx_tokens_cap", "crop_scale", "hflip", "cache_res"]
LEJEPA_KEYS = ["lejepa_lambda", "lejepa_slices", "lejepa_knots", "lejepa_tmax", "lejepa_n_global", "lejepa_n_local",
               "lejepa_local_size", "lejepa_global_scale", "lejepa_local_scale", "lejepa_proj_hidden", "lejepa_proj_dim",
               "lejepa_batch_size", "lejepa_blr", "lejepa_wd"]


def config_hash(cfg, arm, seed, steps, tau=None, eps=None, prior=None, data=None, extra=None):
    """Everything that changes what a run learns. tau/eps enter only for arms that use them, the prior hash only for
    pos_matched, so changing VEG_TAU never invalidates finished ijepa runs."""
    d = {k: getattr(cfg, k) for k in TRAIN_KEYS}
    if str(arm).startswith("lejepa"):
        d.update({k: getattr(cfg, k) for k in LEJEPA_KEYS})
    d.update(arm=arm, seed=seed, steps=steps, tau=tau, eps=eps, prior=prior, data=data, extra=extra)
    return hashlib.sha1(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:12]


def hours_left():
    return cfg.max_session_hours - (time.time() - SESSION_START) / 3600


def budget_str():
    return "no session limit" if cfg.max_session_hours >= 1e6 else f"{hours_left():.1f} h left in session budget"


def get_rng_state():
    """RNG states as tensors/lists only, so checkpoints load with torch.load(weights_only=True)."""
    npst = np.random.get_state()
    py = random.getstate()
    st = {"python": [py[0], list(py[1]), py[2]],
          "numpy": [npst[0], torch.from_numpy(np.asarray(npst[1], dtype=np.int64)), int(npst[2]), int(npst[3]), float(npst[4])],
          "torch": torch.get_rng_state()}
    if HAS_CUDA:
        st["cuda"] = [s.cpu() for s in torch.cuda.get_rng_state_all()]
    return st


def set_rng_state(st):
    py = st["python"]
    random.setstate((py[0], tuple(py[1]), py[2]))
    n = st["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.int64).astype(np.uint32), int(n[2]), int(n[3]), float(n[4])))
    torch.set_rng_state(st["torch"].cpu())
    if HAS_CUDA and "cuda" in st and len(st["cuda"]) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all([s.cpu() for s in st["cuda"]])


def autocast_ctx():
    if DEVICE.type != "cuda" or cfg.precision == "fp32":
        return contextlib.nullcontext()
    return torch.autocast("cuda", dtype=torch.bfloat16 if cfg.precision == "bf16" else torch.float16)


def free_memory():
    import gc
    gc.collect()
    if HAS_CUDA:
        torch.cuda.empty_cache()


def state_hash(*modules):
    """Hash of the initial weights (CPU, before any training) — equal across arms with the same seed (paired design)."""
    h = hashlib.sha1()
    for m in modules:
        for k, v in sorted(m.state_dict().items()):
            h.update(k.encode())
            h.update(v.detach().float().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()[:16]


def strings_hash(items):
    return hashlib.sha1("\n".join(map(str, items)).encode()).hexdigest()[:12]


def veg_patch_scores(x01, patch, kind="chromatic"):
    """Excess-green per patch on un-normalised [0, 1] pixels: chromatic ExG = (2g - r - b) / (r + g + b)."""
    r, g, b = x01[:, 0].float(), x01[:, 1].float(), x01[:, 2].float()
    exg = (2 * g - r - b) / (r + g + b + 1e-6) if kind == "chromatic" else (2 * g - r - b)
    return F.avg_pool2d(exg.unsqueeze(1), patch).squeeze(1)  # (B, grid, grid)


def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    m = int(ok.sum())
    if m == 0:
        return out
    idx = np.argsort(p[ok])
    adj = np.minimum(1, np.maximum.accumulate((m - np.arange(m)) * p[ok][idx]))
    tmp = np.empty(m)
    tmp[idx] = adj
    out[ok] = tmp
    return out


print("Utilities ready.")


# ======================= notebook cell 10 =======================
PRIMARY_PAIRS = [("veg_pos", "ijepa"), ("veg_pos", "pos_matched"), ("veg_pos", "veg_neg")]
SECONDARY_PAIRS = [("veg_neg", "ijepa"), ("pos_matched", "ijepa"), ("attn", "ijepa")]


def min_attainable_p(n, alternative):
    """Smallest exact signed-rank p with n non-zero paired differences."""
    return (1.0 if alternative == "greater" else 2.0) / 2 ** n


def write_preregistration(prereg):
    path = OUT / "stats" / "preregistration.json"
    n_results = len(list((OUT / "runs").glob("*/eval.json")))
    core = {k: v for k, v in prereg.items() if k not in ("created", "results_existing_at_registration")}
    if path.exists():
        old = load_json(path)
        old_core = {k: v for k, v in old.items() if k not in ("created", "results_existing_at_registration")}
        if json.loads(json.dumps(core, default=_json_default)) == old_core:
            print(f"Pre-registration unchanged (registered {old.get('created')}, "
                  f"{old.get('results_existing_at_registration')} results existed then).")
            return old
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        amend = OUT / "stats" / f"preregistration_amendment_{stamp}.json"
        changed = sorted(k for k in set(core) | set(old_core) if json.loads(json.dumps(core.get(k), default=_json_default)) != old_core.get(k))
        save_json({**prereg, "created": datetime.datetime.now().isoformat(), "results_existing_at_registration": n_results,
                   "changed_keys": changed}, amend)
        print("!" * 100)
        print(f"WARNING: the analysis plan differs from the ORIGINAL pre-registration in: {changed}.")
        print(f"The original file is kept unchanged; this version is logged as {amend.name} "
              f"({n_results} finished runs existed at this point). Report the amendment in the paper.")
        print("!" * 100)
        return old
    reg = {**prereg, "created": datetime.datetime.now().isoformat(), "results_existing_at_registration": n_results}
    save_json(reg, path)
    print(f"Pre-registration written to {path} ({n_results} finished runs existed at registration).")
    return reg


n_seeds = len(SEEDS_RUN)
p_min_alt = min_attainable_p(n_seeds, cfg.primary_alternative)
holm_min = len(PRIMARY_PAIRS) * p_min_alt
PREREG = {
    "study": "Crop-JEPA Stage B: greenness-guided target placement in I-JEPA (LUCASVision)",
    "hypotheses": {
        "H1": "veg_pos > ijepa on the primary endpoint",
        "H2": "veg_pos > pos_matched (greenness beyond image position)",
        "H3": "veg_pos > veg_neg (polarity matters)",
        "H4 (Tier 2, not in the Holm family)": "the veg_pos effect is smaller under SIGReg/LeJEPA than under EMA I-JEPA",
    },
    "primary_endpoint": {
        "metric": "test macro-F1",
        "evaluation": "linear probe (logistic regression, C chosen on validation macro-F1) on frozen target-encoder features",
        "regimes": list(cfg.primary_regimes),
        "aggregation": "mean over the label draws of a regime, then mean over the regimes",
        "unit_of_analysis": "pretraining seed (label draws are nested within a seed)",
    },
    "arms": list(ARMS), "seeds": list(SEEDS_RUN), "baselines": list(BASELINES),
    "training_budget_steps": {"core": cfg.target_steps, "pilot": cfg.pilot_steps},
    "placement": {"tau": cfg.veg_tau, "eps": cfg.veg_eps, "num_targets": cfg.num_targets, "target_scale": cfg.target_scale},
    "primary_family": [f"{a} - {b}" for a, b in PRIMARY_PAIRS],
    "multiplicity": "Holm over the primary family (size 3)",
    "test": {"name": "exact Wilcoxon signed-rank on the seed-level differences (zero differences dropped, exact enumeration)",
             "alternative": cfg.primary_alternative, "alpha": cfg.alpha},
    "also_reported": ["paired sign-flip permutation p (exact)", "Wilcoxon p for the other alternatives",
                      "hierarchical bootstrap CI: resample seeds -> label draws within seed -> test images (shared by arms)"],
    "bootstrap": {"B": cfg.bootstrap_B, "interval": "percentile", "levels": [0.95, cfg.equiv_ci]},
    "decision_rules": {
        "A > B (confirmed)": f"Holm-adjusted p ({cfg.primary_alternative}) < {cfg.alpha} AND 95% CI lower bound > 0",
        "A < B": "95% CI upper bound < 0 (descriptive; not a pre-registered hypothesis)",
        "A ~ B (equivalent)": f"{int(cfg.equiv_ci * 100)}% CI inside +/- {cfg.equiv_margin_pp} pp (TOST, alpha 0.05)",
        "otherwise": "inconclusive",
    },
    "interpretation_table": [
        "veg_pos beats ijepa, pos_matched and veg_neg -> greenness-guided placement helps (positive result)",
        "veg_pos ~ pos_matched > ijepa -> the effect is a spatial prior, not greenness",
        "veg_pos ~ ijepa ~ others (all three equivalent) -> placement does not matter at this scale (negative result)",
        "veg_pos < ijepa -> greenness-biased placement hurts",
        "anything else -> mixed / inconclusive (reported as such)",
    ],
    "secondary_exploratory": ["each regime separately", "k-NN (same endpoint and per regime)", "veg_neg - ijepa",
                              "pos_matched - ijepa", "attn - ijepa (Tier 2)", "per-class F1 differences at 100 % labels",
                              "tau sweep, data-size axis, LeJEPA arms (Tier 2)"],
    "gates": {"G0": f"online k-NN val F1 of the pilot >= random_init + {cfg.g0_knn_margin_pp} pp AND min RankMe >= "
                    f"{cfg.rankme_floor_frac} x enc_dim",
              "G1": "interpretation table above, with the decision rules above"},
    "power_note": {"n_seeds": n_seeds, "min_exact_p": p_min_alt, "min_attainable_holm_p": holm_min},
}
if holm_min >= cfg.alpha and RUN_MODE == "smoke":
    print(f"(smoke mode, {n_seeds} seeds: the Holm-adjusted test cannot reach alpha here; that is expected for a pipeline check)")
elif holm_min >= cfg.alpha:
    print("!" * 100)
    print(f"NOTE: with {n_seeds} seeds the smallest exact Wilcoxon p ({cfg.primary_alternative}) is {p_min_alt:.4f}; "
          f"after Holm over 3 comparisons the smallest attainable adjusted p is {holm_min:.4f} >= alpha {cfg.alpha}.")
    print("The confirmatory test can then NEVER succeed. Use PRIMARY_ALTERNATIVE = 'greater' (the hypotheses are directional) "
          "or run more seeds (two-sided needs >= 7) BEFORE any result exists.")
    print("!" * 100)
else:
    print(f"Smallest attainable Holm-adjusted p with {n_seeds} seeds ({cfg.primary_alternative}): {holm_min:.4f} "
          f"(< alpha {cfg.alpha}; requires every seed to agree in sign for the strongest comparison).")
PREREG_SAVED = write_preregistration(PREREG)


# ======================= notebook cell 12 =======================
LUCAS_BASE_URL = "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/DRLL/LUCASvision/"
LUCAS_CROP_NAMES = {
    "B11": "Common wheat", "B12": "Durum wheat", "B13": "Barley", "B14": "Rye", "B15": "Oats", "B16": "Maize",
    "B21": "Potatoes", "B22": "Sugar beet", "B31": "Sunflower", "B32": "Rapeseed", "B33": "Soya", "B55": "Temporary grassland",
}
LUCAS_CLASSES = list(LUCAS_CROP_NAMES)
_UA = {"User-Agent": "Mozilla/5.0 (Crop-JEPA research notebook)"}


def http_get(url, timeout=60, tries=4):
    last = None
    for t in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=_UA), timeout=timeout) as r:
                return r.read()
        except Exception as e:
            last = e
            time.sleep(1.5 * (t + 1))
    raise last


def list_class_files(cls):
    html = http_get(f"{LUCAS_BASE_URL}{cls}/").decode("utf-8", "replace")
    links = re.findall(r'href="([^"]+\.jpe?g)"', html, flags=re.I)
    return sorted({urllib.parse.unquote(l.split("?")[0].split("/")[-1]) for l in links})


def is_writable(d):
    try:
        d = Path(d)
        d.mkdir(parents=True, exist_ok=True)
        t = d / ".write_test"
        t.write_text("x")
        t.unlink()
        return True
    except Exception:
        return False


def _download_one(task):
    cls, name, dest = task
    path = dest / cls / name
    if path.exists() and path.stat().st_size > 0:
        return cls, name, "exists", ""
    url = f"{LUCAS_BASE_URL}{cls}/{urllib.parse.quote(name)}"
    tmp = path.with_name(path.name + ".part")
    err = ""
    for t in range(3):
        try:
            data = http_get(url, timeout=90, tries=1)
            if len(data) < 1024:
                raise IOError(f"suspiciously small file ({len(data)} bytes)")
            with open(tmp, "wb") as f:
                f.write(data)
            _atomic_replace(tmp, path)   # a half-written file never carries the final name
            return cls, name, "downloaded", ""
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            time.sleep(2 * (t + 1))
    try:
        if tmp.exists():
            tmp.unlink()
    except Exception:
        pass
    return cls, name, "failed", err


def setup_lucas_data(dest_dir, limit_per_class=None, threads=20, seed=42, cache_res=192):
    """Lists every class folder on the JRC server, downloads what is missing (resumable: existing files are skipped,
    partial files never keep their final name, failures are retried and logged) and returns the selected file names."""
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    listing_path = dest / "listing_stageB.json"
    listing, errors = {}, {}
    for cls in LUCAS_CLASSES:
        (dest / cls).mkdir(exist_ok=True)
        try:
            listing[cls] = list_class_files(cls)
        except Exception as e:
            errors[cls] = f"{type(e).__name__}: {e}"
    if errors:
        old = load_json(listing_path) if listing_path.exists() else {}
        unresolved = []
        for cls in errors:
            if old.get(cls):
                listing[cls] = old[cls]
                print(f"  listing of {cls} failed ({errors[cls]}); using the listing saved on {time.ctime(listing_path.stat().st_mtime)}")
            else:
                unresolved.append(cls)
        if unresolved and not ALLOW_INCOMPLETE_DOWNLOAD:
            raise RuntimeError(
                f"Could not list {unresolved} on the JRC server (e.g. {errors[unresolved[0]]}). Check the internet connection "
                "and re-run this cell. The notebook stops here instead of silently continuing with only the images already on "
                "disk (e.g. Stage A's 100 per class), which would lock the splits to an incomplete dataset. To deliberately use "
                "only the local files, set ALLOW_INCOMPLETE_DOWNLOAD = True.")
        for cls in unresolved:
            listing[cls] = sorted(p.name for p in (dest / cls).glob("*.jpg"))
            print(f"  WARNING: listing of {cls} failed ({errors[cls]}); ALLOW_INCOMPLETE_DOWNLOAD = True, so only the "
                  f"{len(listing[cls])} local files of this class are used")
    else:
        save_json(listing, listing_path)
    rng = random.Random(seed)
    selected = {}
    for cls in LUCAS_CLASSES:
        names = sorted(listing.get(cls, []))
        selected[cls] = sorted(rng.sample(names, limit_per_class)) if limit_per_class and len(names) > limit_per_class else names
    tasks = [(cls, n, dest) for cls in LUCAS_CLASSES for n in selected[cls]]
    present = [dest / c / n for c, n, _ in tasks if (dest / c / n).exists()]
    missing = [t for t in tasks if not (t[2] / t[0] / t[1]).exists()]
    if present:
        sample = present[:: max(1, len(present) // 500)]
        mean_bytes = float(np.mean([p.stat().st_size for p in sample]))
    elif missing:
        probe = [_download_one(t) for t in missing[:5]]
        got = [dest / c / n for c, n, s, _ in probe if s == "downloaded"]
        mean_bytes = float(np.mean([p.stat().st_size for p in got])) if got else 1.5e6
        missing = [t for t in missing if not (t[2] / t[0] / t[1]).exists()]
    else:
        mean_bytes = 0.0
    need_gib = len(missing) * mean_bytes / 1024**3
    cache_gib = len(tasks) * cache_res * cache_res * 3 / 1024**3
    reserve_gib = cache_gib + 4.0   # image cache + checkpoints / logs of the grid
    free = free_disk_gib(dest)
    print(f"LUCASVision: {len(tasks)} images listed | {len(tasks) - len(missing)} already on disk | {len(missing)} to download "
          f"(≈{need_gib:.1f} GiB at {mean_bytes / 1e6:.2f} MB/image) | free disk {free:.1f} GiB "
          f"(cache ≈{cache_gib:.1f} GiB + ≈4 GiB checkpoints also needed)")
    if need_gib + reserve_gib > free:
        raise RuntimeError(f"Not enough disk space: need ≈{need_gib + reserve_gib:.1f} GiB, {free:.1f} GiB free. "
                           "Free some space (or move DATA_ROOT / OUTPUT_DIR to another drive) and re-run this cell.")
    results = []
    if missing:
        t0 = time.time()
        step = max(1, len(missing) // 20)
        with ThreadPoolExecutor(threads) as ex:
            futs = [ex.submit(_download_one, t) for t in missing]
            for i, fut in enumerate(as_completed(futs)):
                results.append(fut.result())
                if (i + 1) % step == 0 or i + 1 == len(missing):
                    done = i + 1
                    rate = done / max(1e-6, time.time() - t0)
                    n_fail = sum(1 for r in results if r[2] == "failed")
                    print(f"  {done}/{len(missing)} processed ({rate:.1f} files/s, {n_fail} failed, "
                          f"≈{(len(missing) - done) / max(rate, 1e-6) / 60:.0f} min left)")
    failures = [r for r in results if r[2] == "failed"]
    if failures:
        save_csv(pd.DataFrame(failures, columns=["class", "file", "status", "error"]), dest / "download_failures.csv")
    elif (dest / "download_failures.csv").exists():
        (dest / "download_failures.csv").unlink()
    per_class = {cls: {"listed": len(selected[cls]), "on_disk": sum((dest / cls / n).exists() for n in selected[cls])}
                 for cls in LUCAS_CLASSES}
    report = {"listed_total": len(tasks), "downloaded_now": sum(1 for r in results if r[2] == "downloaded"),
              "failed": len(failures), "listing_errors": errors, "per_class": per_class, "mean_file_mb": mean_bytes / 1e6,
              "time": datetime.datetime.now().isoformat()}
    save_json(report, dest / "download_report.json")
    print("Per-class counts (listed / on disk):", {c: f"{v['listed']}/{v['on_disk']}" for c, v in per_class.items()})
    if failures:
        print(f"{len(failures)} downloads failed (see {dest / 'download_failures.csv'}); first errors: {failures[:3]}")
        if not ALLOW_INCOMPLETE_DOWNLOAD:
            raise RuntimeError("Some downloads failed. Re-run this cell to retry them (finished files are skipped). If the same "
                               "files keep failing, set ALLOW_INCOMPLETE_DOWNLOAD = True — but do it BEFORE the splits are "
                               "written, because splits are never regenerated silently.")
    return selected, report


def write_lucas_metadata(dest, selected, out_dir):
    """metadata_stageB.csv: path (relative, forward slashes), label. Group ids are parsed from the filenames below."""
    dest = Path(dest)
    rows = [{"path": f"{cls}/{n}", "label": cls} for cls in LUCAS_CLASSES for n in selected.get(cls, [])
            if (dest / cls / n).exists()]
    target = dest / "metadata_stageB.csv" if is_writable(dest) else Path(out_dir) / "metadata_stageB.csv"
    save_csv(pd.DataFrame(rows), target)
    print(f"Metadata written: {target} ({len(rows)} images)")
    return target


# ======================= notebook cell 13 =======================
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SPLIT_ALIASES = {"train": "train", "training": "train", "val": "val", "valid": "val", "validation": "val",
                 "test": "test", "testing": "test"}


def make_synthetic_dataset(root, n_classes=6, per_class=60, res=96, seed=0):
    """Generated 'crop' photos named like LUCAS files (LUCAS<year>_<point>_<view>.jpg) so the group regex, its fallback,
    repeated survey points across years and planted duplicates are all exercised. Vegetation is denser towards the
    bottom of the image, like real field photos, so the ExG-position analysis has something to find. Smoke test only."""
    root = Path(root)
    if (root / "SYNTHETIC_OK").exists():
        return root
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:res, 0:res] / res
    point, made = 10_000_000, []
    for c in range(n_classes):
        d = root / f"S{c + 1:02d}"
        d.mkdir(parents=True, exist_ok=True)
        theta, freq = np.pi * c / n_classes, 3 + 2 * (c % 3)
        for i in range(per_class):
            img = np.ones((res, res, 3)) * (np.array([0.45, 0.35, 0.25]) + rng.normal(0, 0.03, 3))
            img += rng.normal(0, 0.04, (res, res, 3))
            stripes = np.sin(2 * np.pi * freq * (xx * np.cos(theta) + yy * np.sin(theta)) + rng.uniform(0, 6.28)) > (0.3 if c % 2 else -0.3)
            cx, cy, r = rng.uniform(0.3, 0.7), rng.uniform(0.45, 0.85), rng.uniform(0.25, 0.45)
            veg = stripes & ((((xx - cx) ** 2 + (yy - cy) ** 2) < r ** 2) | (yy > 0.75))
            img[veg] = np.array([0.15, 0.50 + 0.08 * (c % 2), 0.12]) + rng.normal(0, 0.05, (int(veg.sum()), 3))
            img[yy < 0.15] = np.array([0.55, 0.65, 0.85])  # sky band at the top
            repeat = i % 10 == 9                             # same survey point photographed in another year
            if not repeat:
                point += 1
            year = 2022 if repeat else int(rng.choice([2006, 2009, 2012, 2015, 2018]))
            name = f"LUCAS{year}_{point:08d}_Cover.jpg"
            if i == per_class - 1 and c < 2:
                name = f"LUCAS{year}_{point:08d}_Cover_copy.jpg"  # does not match the regex: exercises the fallback
            Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(d / name, quality=92)
            made.append(d / name)
    for k in range(3):  # planted exact duplicates under another point id: the hygiene check must merge their groups
        src = made[k * 7]
        point += 1
        shutil.copy(src, src.parent / f"LUCAS2018_{point:08d}_Dup.jpg")
    (root / "SYNTHETIC_OK").write_text("ok")
    return root


def _has_images(folder, limit=50):
    try:
        for i, e in enumerate(os.scandir(folder)):
            if i > limit:
                break
            if e.is_file() and Path(e.name).suffix.lower() in IMG_EXT:
                return True
    except (PermissionError, FileNotFoundError):
        pass
    return False


def autodetect_data_root(base="/kaggle/input", max_depth=5):
    base = Path(base)
    if not base.exists():
        return None
    for depth in range(max_depth):
        for d in sorted(p for p in base.glob("/".join(["*"] * (depth + 1))) if p.is_dir()):
            subs = [s for s in d.iterdir() if s.is_dir()]
            if sum(_has_images(s) for s in subs[:200]) >= 2:
                return d
    return None


def list_images(folder):
    return sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in IMG_EXT)


def discover_dataset(data_root, metadata_csv=None):
    """Returns path (relative to data_root, forward slashes: the key used everywhere, identical on Windows and Kaggle),
    abs_path (for reading), label, and any extra metadata columns."""
    data_root = Path(data_root) if data_root else None
    if metadata_csv:
        df = pd.read_csv(metadata_csv, keep_default_na=False, dtype=str)
        if not {"path", "label"} <= set(df.columns):
            raise ValueError("The metadata CSV needs at least the columns: path, label")
        base = data_root or Path(metadata_csv).parent
    else:
        if data_root is None or not data_root.exists():
            raise FileNotFoundError(f"DATA_ROOT not found: {data_root}. Set DATA_ROOT in the configuration cell.")
        rows = []
        for cd in sorted(x for x in data_root.iterdir() if x.is_dir()):
            rows += [dict(path=p.relative_to(data_root).as_posix(), label=cd.name) for p in list_images(cd)]
        df = pd.DataFrame(rows)
        base = data_root
    if df.empty:
        raise ValueError("No images found. Expected one sub-folder per class.")
    df["path"] = [Path(p).as_posix() for p in df["path"]]
    df["abs_path"] = [p if os.path.isabs(p) else str(Path(base) / p) for p in df["path"]]
    df["label"] = df["label"].astype(str)
    return df.reset_index(drop=True)


def filename_sample(df, n=30, path=None):
    lines = []
    for lab, sub in df.groupby("label"):
        names = [Path(p).name for p in sorted(sub["path"])[:n]]
        lines.append(f"[{lab}] {len(sub)} files; first {len(names)}:")
        lines += ["   " + "   ".join(names[i:i + 3]) for i in range(0, len(names), 3)]
    text = "\n".join(lines)
    print(text)
    if path:
        save_text(text + "\n", path)


def assign_groups(df, regex, min_match=0.95):
    """Group id = LUCAS survey-point id = capture group 1 of `regex` on the file name (confirmed by the user on the
    Stage A sample). Files that do not match fall back to the second '_' token (printed); a match rate below
    `min_match` stops the notebook instead of silently treating every image as its own group."""
    pat = re.compile(regex, re.I)
    year_pat = re.compile(r"LUCAS(\d{4})", re.I)
    groups, src, years, bad = [], [], [], []
    for p in df["path"]:
        name = Path(p).name
        m = pat.match(name)
        if m:
            groups.append(m.group(1))
            src.append("regex")
        else:
            parts = name.split("_")
            groups.append(parts[1] if len(parts) > 1 else Path(name).stem)
            src.append("fallback_2nd_token")
            bad.append(name)
        y = year_pat.search(name)
        years.append(int(y.group(1)) if y else -1)
    n = len(df)
    rate = 1 - len(bad) / max(1, n)
    print(f"Group id regex {regex!r} matched {n - len(bad)}/{n} file names ({100 * rate:.2f} %).")
    if bad:
        print(f"  {len(bad)} names did not match (second '_' token used instead); examples: {bad[:10]}")
    if rate < min_match:
        raise RuntimeError(f"Only {100 * rate:.1f} % of file names match GROUP_REGEX (< {100 * min_match:.0f} %). Look at the "
                           "filename sample above, fix GROUP_REGEX, and re-run. Per-image groups would leak.")
    out = df.copy()
    out["group"] = [str(g) for g in groups]
    out["point_id"] = out["group"]
    out["group_source"] = src
    out["year"] = years
    report = {"regex": regex, "files": n, "matched": n - len(bad), "match_rate": rate, "fallback_examples": bad[:10],
              "unique_points": int(out["group"].nunique())}
    return out, report


def _process_image(path, res):
    try:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            g = np.asarray(im.convert("L").resize((9, 8), Image.BILINEAR), dtype=np.int16)
            bits = (g[:, 1:] > g[:, :-1]).flatten()
            h = int.from_bytes(np.packbits(bits).tobytes(), "big")
            arr = np.asarray(ImageOps.fit(im, (res, res), Image.BICUBIC), dtype=np.uint8)
        return arr, h, True
    except Exception:
        return None, 0, False


def build_cache(df, out_dir, res):
    """Reads every image once (EXIF-transposed, RGB, ImageOps.fit to res x res) into one uint8 .npy built through a
    memmap, plus a dHash per image for the duplicate check. Re-used while the file list is unchanged."""
    cache_dir = Path(out_dir) / "cache"
    img_path, meta_path = cache_dir / f"images_{res}.npy", cache_dir / f"images_{res}.csv"
    if img_path.exists() and meta_path.exists():
        meta = pd.read_csv(meta_path, keep_default_na=False, dtype={"path": str, "label": str, "group": str, "dhash": str})
        if list(meta["path"]) == list(df["path"]):
            for col in df.columns:   # refresh metadata columns (group ids etc.); pixels and hashes are kept
                meta[col] = df[col].values
            save_csv(meta, meta_path)
            print(f"Image cache found: {img_path} ({len(meta)} images)")
            return img_path, meta
        print("Dataset file list changed: rebuilding the image cache.")
    n = len(df)
    need = n * res * res * 3 / 1024**3
    print(f"Building the image cache: {n} images at {res}px ≈ {need:.2f} GiB (free disk {free_disk_gib(cache_dir):.1f} GiB)")
    if need + 0.5 > free_disk_gib(cache_dir):
        raise RuntimeError("Not enough free disk space for the image cache.")
    tmp = cache_dir / f"images_{res}.tmp.npy"
    arr = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=(n, res, res, 3))
    hashes, ok = [0] * n, np.zeros(n, dtype=bool)
    t0 = time.time()
    workers = min(8, (os.cpu_count() or 2))
    with ThreadPoolExecutor(workers) as ex:
        for i, (a, h, good) in enumerate(ex.map(lambda p: _process_image(p, res), df["abs_path"])):
            if good:
                arr[i], hashes[i], ok[i] = a, h, True
            if (i + 1) % max(1, n // 10) == 0:
                print(f"  cached {i + 1}/{n} images ({time.time() - t0:.0f}s)")
    arr.flush()
    del arr
    _atomic_replace(tmp, img_path)
    meta = df.copy()
    meta["dhash"] = [f"{h:016x}" for h in hashes]
    meta["ok"] = ok
    save_csv(meta, meta_path)
    print(f"Image cache written: {img_path} | unreadable images: {(~ok).sum()}")
    return img_path, meta


def load_prebuilt_cache(cache_dir, res):
    """Kaggle: a cache built on the laptop and uploaded as a private dataset. Splits and label draws travel with it,
    so the Kaggle runs use exactly the same split."""
    cache_dir = Path(cache_dir)
    img_path, meta_path = cache_dir / f"images_{res}.npy", cache_dir / f"images_{res}.csv"
    if not (img_path.exists() and meta_path.exists()):
        raise FileNotFoundError(f"{img_path} / {meta_path} not found in CACHE_DATASET_DIR")
    for name in ["splits.csv", "label_draws.json"]:
        for cand in [cache_dir / name, cache_dir.parent / name]:
            if cand.exists() and not (OUT / name).exists():
                shutil.copy2(cand, OUT / name)
                print(f"Copied {name} from {cand}")
                break
    meta = pd.read_csv(meta_path, keep_default_na=False, dtype={"path": str, "label": str, "group": str, "dhash": str})
    meta["ok"] = meta["ok"].astype(str).str.lower().isin(["true", "1"])
    print(f"Pre-built image cache: {img_path} ({len(meta)} images)")
    return img_path, meta


class UnionFind:
    def __init__(self, items):
        self.p = {x: x for x in items}

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def hygiene_and_groups(meta, max_dist):
    """Exact and near-duplicate images (dHash Hamming distance <= max_dist) are merged into one group, so duplicates
    can never end up in different splits. Returns meta with 'group_final' and a report."""
    meta = meta[meta["ok"].astype(str).str.lower().isin(["true", "1"])].reset_index(drop=True).copy()
    n = len(meta)
    hashes = np.array([int(h, 16) for h in meta["dhash"]], dtype=np.uint64)
    uf_img = UnionFind(range(n))
    seen = {}
    for i, h in enumerate(hashes.tolist()):
        if h in seen and h != 0:
            uf_img.union(seen[h], i)
        elif h != 0:
            seen[h] = i
    near_checked = max_dist > 0 and n <= 60000
    if near_checked:
        b = hashes.view(np.uint8).reshape(n, 8)
        lut = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
        chunk = max(1, int(2e7 // max(n, 1)))
        for s in range(0, n, chunk):
            dist = lut[b[s:s + chunk, None, :] ^ b[None, :, :]].sum(-1, dtype=np.int32)
            ii, jj = np.nonzero(dist <= max_dist)
            for i, j in zip(ii + s, jj):
                if j > i:
                    uf_img.union(int(i), int(j))
    clusters = {}
    for i in range(n):
        clusters.setdefault(uf_img.find(i), []).append(i)
    dup_clusters = [c for c in clusters.values() if len(c) > 1]
    uf_grp = UnionFind(meta["group"].astype(str).unique().tolist())
    for c in dup_clusters:
        g = meta["group"].iloc[c].astype(str).tolist()
        for x in g[1:]:
            uf_grp.union(g[0], x)
    meta["group_final"] = [uf_grp.find(str(g)) for g in meta["group"]]
    conflicting = [c for c in dup_clusters if meta["label"].iloc[c].nunique() > 1]
    report = {"images_ok": n, "duplicate_clusters": len(dup_clusters),
              "images_in_duplicate_clusters": int(sum(len(c) for c in dup_clusters)),
              "duplicate_clusters_with_conflicting_labels": len(conflicting), "near_duplicate_check": near_checked,
              "groups_before": int(meta["group"].nunique()), "groups_after": int(meta["group_final"].nunique())}
    return meta, report


def group_stratified_split(meta, val_frac, test_frac, seed):
    """Whole groups (survey point + duplicate merges) go to one split, class by class (group label = majority label)."""
    rng = np.random.default_rng(seed)
    g = meta.groupby("group_final").agg(label=("label", lambda s: s.mode().iloc[0]), size=("label", "size")).reset_index()
    order = [("test", test_frac), ("val", val_frac)]
    assign = {}
    for lab, sub in g.groupby("label"):
        sub = sub.iloc[rng.permutation(len(sub))]
        if len(sub) < 3:
            assign.update({gid: "train" for gid in sub["group_final"]})
            continue
        total = sub["size"].sum()
        quota = {name: int(round(frac * total)) for name, frac in order}
        filled = {name: 0 for name, _ in order}
        labs = []
        for gid, size in zip(sub["group_final"], sub["size"]):
            dest = "train"
            for name, _ in order:
                if filled[name] < quota[name]:
                    dest = name
                    filled[name] += size
                    break
            assign[gid] = dest
            labs.append((gid, dest))
        if all(d != "train" for _, d in labs):
            assign[labs[-1][0]] = "train"
    out = meta.copy()
    out["split"] = out["group_final"].map(assign).fillna("train")
    return out


def _assert_no_split_overlap(splits_df):
    for grp_col in ["group_final", "group"]:
        if grp_col in splits_df.columns:
            sets = {s: set(splits_df.loc[splits_df["split"] == s, grp_col]) for s in ["train", "val", "test"]}
            for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
                inter = sets[a] & sets[b]
                assert not inter, f"Leakage: {a} and {b} share {len(inter)} values of {grp_col}, e.g. {sorted(inter)[:5]}"
    print("PASS: zero group / survey-point overlap across train, val and test.")


def make_or_load_splits(meta, out_dir):
    """Written once. If splits.csv exists but no longer matches the data, the notebook STOPS instead of re-splitting."""
    path = Path(out_dir) / "splits.csv"
    if path.exists():
        s = pd.read_csv(path, keep_default_na=False, dtype={"path": str, "label": str, "group": str, "group_final": str})
        paths_match = sorted(s["path"]) == sorted(meta["path"])
        groups_match = set(s["group_final"]) == set(meta["group_final"]) and set(s["group"]) == set(meta["group"].astype(str))
        if not (paths_match and groups_match):
            raise RuntimeError(f"{path} exists but does not match the current data (paths_match={paths_match}, "
                               f"groups_match={groups_match}). Splits are never regenerated silently. Either restore the "
                               "data the splits were made from, or deliberately delete splits.csv AND label_draws.json "
                               "(and all runs) in this OUTPUT_DIR — better: use a new OUTPUT_DIR.")
        print(f"Splits loaded from {path}")
        _assert_no_split_overlap(s)
        return s
    s = group_stratified_split(meta, cfg.val_frac, cfg.test_frac, cfg.split_seed)
    classes = sorted(s["label"].unique())
    s["label_idx"] = s["label"].map({c: i for i, c in enumerate(classes)})
    _assert_no_split_overlap(s)
    save_csv(s, path)
    print(f"Splits written to {path}")
    return s


def make_or_load_draws(y_train, n_classes, out_dir, seed=1234):
    """Fixed labelled subsets per regime (k per class, or p % per class), identical for every arm and seed."""
    path = Path(out_dir) / "label_draws.json"
    sig = {"n_train": int(len(y_train)), "y_hash": strings_hash(y_train.tolist()), "shots": list(cfg.shots),
           "percents": list(cfg.percents), "n_draws": cfg.n_draws, "seed": seed}
    counts = np.bincount(y_train, minlength=n_classes)
    print(f"Training images per class: {dict(zip(CLASSES, counts.tolist()))}")
    if path.exists():
        d = load_json(path)
        if d.get("_signature") != sig:
            raise RuntimeError(f"{path} was made for different training labels or regimes. It is never regenerated silently; "
                               "use a new OUTPUT_DIR (or delete it deliberately together with all runs).")
        draws = {k: [np.array(x, dtype=np.int64) for x in v] for k, v in d.items() if not k.startswith("_")}
        print(f"Label draws loaded from {path}; skipped regimes: {d.get('_skipped') or 'none'}")
        return draws
    rng = np.random.default_rng(seed)
    by_class = [np.where(y_train == c)[0] for c in range(n_classes)]
    draws, skipped = {}, []
    for k in cfg.shots:
        if (counts >= k).mean() < 0.8:
            skipped.append(f"{k}shot")
            print(f"  skipping {k}-shot: fewer than 80% of classes have {k} training images")
            continue
        draws[f"{k}shot"] = [np.concatenate([rng.choice(ix, min(k, len(ix)), replace=False) for ix in by_class if len(ix)])
                             for _ in range(cfg.n_draws)]
    for p in cfg.percents:
        if p >= 100:
            draws["100pct"] = [np.arange(len(y_train))]
        else:
            draws[f"{p}pct"] = [np.concatenate([rng.choice(ix, max(1, int(round(len(ix) * p / 100))), replace=False)
                                                for ix in by_class if len(ix)]) for _ in range(cfg.n_draws)]
    save_json({"_signature": sig, "_skipped": skipped, **{k: [x.tolist() for x in v] for k, v in draws.items()}}, path)
    print(f"Label draws written to {path}; skipped regimes: {skipped or 'none'}")
    return draws


def regime_order(names):
    shots = sorted([n for n in names if n.endswith("shot")], key=lambda s: int(s[:-4]))
    pcts = sorted([n for n in names if n.endswith("pct")], key=lambda s: int(s[:-3]))
    return shots + pcts


def leakage_audit(splits):
    gs = splits.groupby("group_final")
    sizes = gs.size()
    spanning = int((gs["split"].nunique() > 1).sum())
    pts = splits.groupby("group")
    pt_sizes = pts.size()
    years = pts["year"].nunique() if "year" in splits.columns else pd.Series(dtype=int)
    multi_cls = int((gs["label"].nunique() > 1).sum())
    audit = {"grouped_by": "group_final = LUCAS survey-point id (filename regex) merged with near-duplicate clusters",
             "n_groups": int(len(sizes)), "max_group_size": int(sizes.max()), "median_group_size": float(sizes.median()),
             "share_images_in_groups_gt1": float(sizes[sizes > 1].sum() / sizes.sum()),
             "groups_spanning_splits": spanning,
             "survey_points": int(len(pt_sizes)), "points_with_more_than_one_image": int((pt_sizes > 1).sum()),
             "points_spanning_several_years": int((years > 1).sum()) if len(years) else 0,
             "max_images_per_point": int(pt_sizes.max()), "groups_with_several_crop_labels": multi_cls}
    print(f"Split is grouped by `group_final` (survey point + duplicate merges): {audit['n_groups']} groups; "
          f"max group size {audit['max_group_size']}; median {audit['median_group_size']:.0f}; "
          f"{100 * audit['share_images_in_groups_gt1']:.1f} % of images are in groups of size > 1; "
          f"groups spanning splits: {spanning}")
    print(f"Survey points: {audit['survey_points']} | appearing more than once: {audit['points_with_more_than_one_image']} | "
          f"spanning several years: {audit['points_spanning_several_years']} | max images per point: {audit['max_images_per_point']} | "
          f"groups with more than one crop label (rotation): {multi_cls}")
    return audit


# ======================= notebook cell 15 =======================
DOWNLOAD_REPORT, GROUP_REPORT = {}, {}
if RUN_MODE == "smoke":
    DATA_ROOT_USED = make_synthetic_dataset(OUT / "synthetic_data")
    raw = discover_dataset(DATA_ROOT_USED)
elif CACHE_DATASET_DIR:
    DATA_ROOT_USED, raw = None, None
else:
    DATA_ROOT_USED = Path(DATA_ROOT) if DATA_ROOT else (autodetect_data_root() if IS_KAGGLE else None)
    if DATA_ROOT_USED is None:
        raise FileNotFoundError("Set DATA_ROOT (or CACHE_DATASET_DIR) in the configuration cell.")
    print("Data root:", Path(DATA_ROOT_USED).resolve())
    if is_writable(DATA_ROOT_USED):
        selected, DOWNLOAD_REPORT = setup_lucas_data(DATA_ROOT_USED, LIMIT_PER_CLASS, DOWNLOAD_THREADS, seed=42,
                                                     cache_res=cfg.cache_res)
        meta_csv = write_lucas_metadata(DATA_ROOT_USED, selected, OUT)
    else:
        print("DATA_ROOT is read-only (e.g. an attached Kaggle dataset): using the images found there, no download.")
        meta_csv = None
    raw = discover_dataset(DATA_ROOT_USED, meta_csv)

if raw is not None:
    print("\nFirst 30 file names per class (check the naming pattern the group regex relies on):")
    filename_sample(raw, 30, OUT / "stats" / "filename_sample.txt")
    if "group" in raw.columns and raw["group"].astype(str).str.len().gt(0).all() and RUN_MODE != "smoke" and \
            not raw["path"].map(lambda p: Path(p).name.upper().startswith("LUCAS")).any():
        GROUP_REPORT = {"source": "group column of the metadata CSV"}
        print("Using the group column of the metadata CSV (file names are not LUCAS names).")
    else:
        raw, GROUP_REPORT = assign_groups(raw, GROUP_REGEX)
    cache_path, cache_meta = build_cache(raw, OUT, cfg.cache_res)
else:
    cache_path, cache_meta = load_prebuilt_cache(Path(CACHE_DATASET_DIR), cfg.cache_res)
    GROUP_REPORT = {"source": f"pre-built cache {CACHE_DATASET_DIR}", "unique_points": int(cache_meta["group"].nunique())}
    filename_sample(cache_meta, 30, OUT / "stats" / "filename_sample.txt")

meta_ok, hygiene = hygiene_and_groups(cache_meta, cfg.dedup_hamming)
save_json(hygiene, OUT / "stats" / "hygiene_report.json")
print("Hygiene:", hygiene)
splits = make_or_load_splits(meta_ok, OUT)

row_of_path = {p: i for i, p in enumerate(cache_meta["path"])}
splits["cache_row"] = splits["path"].map(row_of_path)
if splits["cache_row"].isna().any():
    raise RuntimeError("Some split rows are missing from the image cache; the cache and splits.csv do not belong together.")
splits["cache_row"] = splits["cache_row"].astype(np.int64)
CLASSES = sorted(splits["label"].unique())
K = len(CLASSES)
ROWS = {s: splits.loc[splits["split"] == s, "cache_row"].to_numpy(np.int64) for s in ["train", "val", "test"]}
Y = {s: splits.loc[splits["split"] == s, "label_idx"].to_numpy(np.int64) for s in ["train", "val", "test"]}
Y_ALL = np.full(len(cache_meta), -1, dtype=np.int64)
for s in ROWS:
    Y_ALL[ROWS[s]] = Y[s]
DRAWS = make_or_load_draws(Y["train"], K, OUT, seed=cfg.draw_seed)
REGIMES = regime_order(list(DRAWS))
save_npy(Y["test"], OUT / "stats" / "y_test.npy")
LEAK = leakage_audit(splits)
missing_primary = [r for r in cfg.primary_regimes if r not in REGIMES]
if missing_primary:
    print(f"WARNING: primary-endpoint regimes {missing_primary} are not available (too few images); "
          f"the endpoint will use {[r for r in cfg.primary_regimes if r in REGIMES]} only.")
print(f"\n{K} classes | train {len(ROWS['train'])} | val {len(ROWS['val'])} | test {len(ROWS['test'])} | chance = {100 / K:.1f}%")
print("Label regimes (draws):", {r: len(DRAWS[r]) for r in REGIMES})


# ======================= notebook cell 16 =======================
names = LUCAS_CROP_NAMES if RUN_MODE != "smoke" else {}
split_table = splits.groupby(["label", "split"]).size().unstack(fill_value=0).reindex(columns=["train", "val", "test"], fill_value=0)
split_table["total"] = split_table.sum(axis=1)
split_table.insert(0, "crop", [names.get(c, c) for c in split_table.index])
print("Images per class and split:")
display(split_table)
print(f"Test set: {len(ROWS['test'])} images (Stage A had 180), per class: "
      f"{dict(zip(CLASSES, np.bincount(Y['test'], minlength=K).tolist()))}")

DATASET_AUDIT = {
    "dataset_source": "synthetic (smoke test)" if RUN_MODE == "smoke" else "LUCASVision (JRC)",
    "official_url": LUCAS_BASE_URL,
    "prepared": datetime.datetime.now().isoformat(),
    "image_count_reported_elsewhere": "≈15,876 [UNVERIFIED secondary source]; this audit is the measured count",
    "total_images_discovered": int(len(cache_meta)),
    "total_usable_images": int(len(meta_ok)),
    "unreadable_images": int((~cache_meta["ok"].astype(str).str.lower().isin(["true", "1"])).sum()),
    "number_of_classes": K, "class_names": CLASSES, "label_mapping": names,
    "images_per_class": meta_ok["label"].value_counts().sort_index().to_dict(),
    "class_counts_by_split": split_table.drop(columns="crop").to_dict(orient="index"),
    "group_parsing": GROUP_REPORT, "download": DOWNLOAD_REPORT, "hygiene": hygiene, "leakage_audit": LEAK,
    "train_images": int(len(ROWS["train"])), "val_images": int(len(ROWS["val"])), "test_images": int(len(ROWS["test"])),
    "label_regimes": {r: len(DRAWS[r]) for r in REGIMES},
    "cache": {"path": str(cache_path), "resolution": cfg.cache_res},
}
save_json(DATASET_AUDIT, OUT / "stats" / "dataset_audit.json")

GATE_M1 = {"leaking_groups": LEAK["groups_spanning_splits"], "classes": K,
           "classes_without_train": [c for c, n in zip(CLASSES, np.bincount(Y["train"], minlength=K)) if n == 0],
           "classes_without_test": [c for c, n in zip(CLASSES, np.bincount(Y["test"], minlength=K)) if n == 0],
           "group_regex_match_rate": GROUP_REPORT.get("match_rate")}
_reasons = []
if GATE_M1["leaking_groups"] > 0:
    _reasons.append(f"{GATE_M1['leaking_groups']} groups span several splits")
if GATE_M1["classes_without_train"] or GATE_M1["classes_without_test"]:
    _reasons.append(f"classes without train/test images: {GATE_M1['classes_without_train']} / {GATE_M1['classes_without_test']}")
if GROUP_REPORT.get("match_rate") is not None and GROUP_REPORT["match_rate"] < 0.999:
    print(f"WARNING: {100 * (1 - GROUP_REPORT['match_rate']):.2f} % of file names used the fallback group id — check the examples above.")
if _reasons:
    save_json({**GATE_M1, "decision": "FAIL", "reasons": _reasons}, OUT / "stats" / "gate_Gminus1.json")
    raise RuntimeError("Gate -1 (data) FAILED: " + "; ".join(_reasons))
print("Gate -1 data checks passed (the ExG-position analysis follows).")


# ======================= notebook cell 18 =======================
@torch.no_grad()
def exg_patch_maps(cache_file, rows, batch=256):
    """ExG per patch on the centred evaluation view (cache resized to img_size) for the given cache rows."""
    arr = np.load(cache_file, mmap_mode="r")
    out = []
    for i in range(0, len(rows), batch):
        idx = np.asarray(rows[i:i + batch])
        order = np.argsort(idx, kind="stable")
        chunk = np.empty((len(idx),) + arr.shape[1:], dtype=np.uint8)
        chunk[order] = arr[idx[order]]
        x = torch.from_numpy(chunk).to(DEVICE).permute(0, 3, 1, 2).float().div_(255.0)
        if x.shape[-1] != cfg.img_size:
            x = F.interpolate(x, size=(cfg.img_size, cfg.img_size), mode="bilinear", antialias=True, align_corners=False)
        out.append(veg_patch_scores(x, cfg.patch_size, cfg.veg_index).cpu())
    return torch.cat(out).numpy().astype(np.float64)


def _corr(a, b):
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan"), float("nan")
    return float(pearsonr(a, b)[0]), float(spearmanr(a, b)[0])


M = exg_patch_maps(cache_path, ROWS["train"])                 # (N, grid, grid)
N_img, g = M.shape[0], M.shape[1]
rr, cc = np.meshgrid(np.arange(g), np.arange(g), indexing="ij")
mean_map = M.mean(0)
Mc = M - M.mean(axis=(1, 2), keepdims=True)                   # remove each image's overall greenness
Mz = Mc / (M.std(axis=(1, 2), keepdims=True) + 1e-6)          # per-image z-score: what veg_pos actually sees
eta_raw = float(mean_map.var() / M.var()) if M.var() > 0 else float("nan")
eta_within = float(Mc.mean(0).var() / Mc.var()) if Mc.var() > 0 else float("nan")
eta_z = float(Mz.mean(0).var() / Mz.var()) if Mz.var() > 0 else float("nan")
pr_row, sp_row = _corr(mean_map.ravel(), rr.ravel())
pr_col, sp_col = _corr(mean_map.ravel(), cc.ravel())
sub = np.random.default_rng(0).choice(N_img, min(N_img, 2000), replace=False)
pooled_row = _corr(Mz[sub].reshape(-1), np.broadcast_to(rr, Mz[sub].shape).reshape(-1))
EXG_POS = {
    "n_train_images": int(N_img), "grid": int(g), "veg_index": cfg.veg_index,
    "mean_exg_overall": float(M.mean()), "mean_exg_top_half": float(mean_map[: g // 2].mean()),
    "mean_exg_bottom_half": float(mean_map[g // 2:].mean()),
    "pearson_meanmap_vs_row": pr_row, "spearman_meanmap_vs_row": sp_row,
    "pearson_meanmap_vs_col": pr_col, "spearman_meanmap_vs_col": sp_col,
    "pooled_pearson_zexg_vs_row": pooled_row[0], "pooled_spearman_zexg_vs_row": pooled_row[1],
    "variance_explained_by_position_raw": eta_raw,
    "variance_explained_by_position_within_image": eta_within,
    "variance_explained_by_position_zscored": eta_z,
    "warn_threshold": cfg.pos_var_warn, "mean_map": mean_map.tolist(), "mean_z_map": Mz.mean(0).tolist(),
}
EXG_POS["position_warning"] = bool(eta_z == eta_z and eta_z > cfg.pos_var_warn)
save_json(EXG_POS, OUT / "stats" / "exg_position_analysis.json")

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
im0 = axes[0].imshow(mean_map, cmap="YlGn")
axes[0].set_title("Mean ExG per patch (training set)")
plt.colorbar(im0, ax=axes[0], fraction=0.046)
im1 = axes[1].imshow(Mz.mean(0), cmap="RdYlGn", vmin=-np.abs(Mz.mean(0)).max(), vmax=np.abs(Mz.mean(0)).max())
axes[1].set_title("Mean per-image z-scored ExG (what veg_pos ranks)")
plt.colorbar(im1, ax=axes[1], fraction=0.046)
axes[2].plot(mean_map.mean(1), np.arange(g), marker="o", label="row profile (top = 0)")
axes[2].plot(mean_map.mean(0), np.arange(g), marker="s", label="column profile")
axes[2].invert_yaxis()
axes[2].set_xlabel("mean ExG")
axes[2].set_ylabel("row / column index")
axes[2].legend()
axes[2].set_title(f"position explains {100 * eta_z:.1f} % of z-scored ExG variance")
for ax in axes[:2]:
    ax.set_xlabel("column")
    ax.set_ylabel("row")
plt.tight_layout()
plt.savefig(OUT / "figures" / "exg_position_heatmap.png", dpi=150)
plt.show()
plt.close(fig)

print(f"ExG vs row: Pearson {pr_row:+.2f}, Spearman {sp_row:+.2f} | vs column: Pearson {pr_col:+.2f}, Spearman {sp_col:+.2f}")
print(f"Share of ExG variance explained by patch position: raw {100 * eta_raw:.1f} % | within image {100 * eta_within:.1f} % | "
      f"z-scored {100 * eta_z:.1f} %")
if EXG_POS["position_warning"]:
    print(f"WARNING: patch position explains {100 * eta_z:.1f} % (> {100 * cfg.pos_var_warn:.0f} %) of the z-scored greenness. "
          "veg_pos is then partly a spatial prior; the pos_matched arm is what separates the two.")
save_json({**GATE_M1, "exg_position_done": True, "position_warning": EXG_POS["position_warning"], "decision": "PASS"},
          OUT / "stats" / "gate_Gminus1.json")
print("Gate -1: PASS")


# ======================= notebook cell 20 =======================
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def normalize(x01):
    return (x01 - MEAN.to(x01)) / STD.to(x01)


class CachedImages:
    """The uint8 cache goes into RAM if it is smaller than 35 % of system RAM, otherwise it is memory-mapped."""

    def __init__(self, path):
        arr = np.load(path, mmap_mode="r")
        size_gib = arr.nbytes / 1024**3
        self.in_ram = size_gib < 0.35 * RAM_GIB
        self.arr = np.ascontiguousarray(arr) if self.in_ram else arr
        print(f"Image cache: {arr.shape}, {size_gib:.2f} GiB vs 35 % of {RAM_GIB:.1f} GiB RAM = {0.35 * RAM_GIB:.2f} GiB -> "
              f"{'loaded into RAM' if self.in_ram else 'memory-mapped from disk'}")

    def fetch(self, idx):
        order = np.argsort(idx, kind="stable")
        data = self.arr[idx[order]]
        out = np.empty_like(data)
        out[order] = data
        return torch.from_numpy(out)


def to_device_float(u8):
    if DEVICE.type == "cuda":
        u8 = u8.pin_memory().to(DEVICE, non_blocking=True)
    return u8.permute(0, 3, 1, 2).float().div_(255.0)


def random_resized_crop(x, out_size, scale, gen, hflip=True, ratio=(3 / 4, 4 / 3)):
    b, _, h, w = x.shape
    dev = x.device
    area = torch.empty(b, device=dev).uniform_(scale[0], scale[1], generator=gen) * h * w
    ar = torch.exp(torch.empty(b, device=dev).uniform_(math.log(ratio[0]), math.log(ratio[1]), generator=gen))
    cw = torch.sqrt(area * ar).clamp(8, w)
    ch = torch.sqrt(area / ar).clamp(8, h)
    x0 = torch.rand(b, device=dev, generator=gen) * (w - cw)
    y0 = torch.rand(b, device=dev, generator=gen) * (h - ch)
    boxes = torch.stack([torch.arange(b, device=dev, dtype=x.dtype), x0, y0, x0 + cw, y0 + ch], 1)
    out = roi_align(x, boxes, (out_size, out_size), spatial_scale=1.0, sampling_ratio=2, aligned=True)
    if hflip:
        flip = torch.rand(b, device=dev, generator=gen) < 0.5
        out = torch.where(flip.view(b, 1, 1, 1), out.flip(3), out)
    return out


class GPULoader:
    """Yields (images in [0,1] on DEVICE, cache rows). Deterministic given (seed, epoch): every arm with the same seed
    sees the same images in the same order with the same crops (paired design). augment=False shuffles but returns the
    centred view (resized to out_size, or untouched if out_size is None) — used by the LeJEPA arm and the priors."""

    def __init__(self, cache, rows, batch_size, train, out_size, seed=0, drop_last=True, crop_scale=None, augment=None):
        self.cache, self.rows, self.train, self.out_size, self.seed = cache, np.asarray(rows), train, out_size, seed
        self.bs = min(batch_size, len(self.rows))
        self.drop_last = drop_last and train
        self.crop_scale = crop_scale or cfg.crop_scale
        self.augment = train if augment is None else augment
        self.epoch = 0

    def set_epoch(self, e):
        self.epoch = e

    def __len__(self):
        n = len(self.rows)
        return n // self.bs if self.drop_last else math.ceil(n / self.bs)

    def __iter__(self):
        rows = np.random.default_rng(self.seed * 100003 + self.epoch).permutation(self.rows) if self.train else self.rows
        gen = torch.Generator(device=DEVICE)
        gen.manual_seed(self.seed * 7919 + self.epoch)
        for b in range(len(self)):
            idx = rows[b * self.bs:(b + 1) * self.bs]
            x = to_device_float(self.cache.fetch(idx))
            if self.augment:
                x = random_resized_crop(x, self.out_size, self.crop_scale, gen, cfg.hflip)
            elif self.out_size is not None and x.shape[-1] != self.out_size:
                x = F.interpolate(x, size=(self.out_size, self.out_size), mode="bilinear", antialias=True, align_corners=False)
            yield x, idx


CACHE = CachedImages(cache_path)
ROW_PATHS = cache_meta["path"].tolist()


def rows_tag(rows):
    """Hash of the image paths behind a set of cache rows (machine-independent)."""
    return strings_hash(sorted(ROW_PATHS[i] for i in np.asarray(rows)))


TRAIN_TAG = rows_tag(ROWS["train"])


# ======================= notebook cell 22 =======================
def sincos_pos_embed(dim, gh, gw):
    gy, gx = np.meshgrid(np.arange(gh, dtype=np.float64), np.arange(gw, dtype=np.float64), indexing="ij")

    def emb1d(d, pos):
        omega = 1.0 / 10000 ** (np.arange(d // 2, dtype=np.float64) / (d / 2))
        out = np.einsum("m,d->md", pos.reshape(-1), omega)
        return np.concatenate([np.sin(out), np.cos(out)], 1)

    return torch.from_numpy(np.concatenate([emb1d(dim // 2, gy), emb1d(dim // 2, gx)], 1)).float().unsqueeze(0)


def gather_tokens(x, idx):
    return torch.gather(x, 1, idx.unsqueeze(-1).expand(-1, -1, x.size(-1)))


class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        b, n, c = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)
        x = F.scaled_dot_product_attention(q, k, v)
        return self.proj(x.transpose(1, 2).reshape(b, n, c))


class MLP(nn.Module):
    def __init__(self, dim, ratio=4):
        super().__init__()
        self.fc1, self.act, self.fc2 = nn.Linear(dim, dim * ratio), nn.GELU(), nn.Linear(dim * ratio, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.norm1, self.attn = nn.LayerNorm(dim, eps=1e-6), Attention(dim, heads)
        self.norm2, self.mlp = nn.LayerNorm(dim, eps=1e-6), MLP(dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


def _init_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


def _rescale_blocks(blocks):  # I-JEPA's depth-dependent rescaling of residual projections
    for i, blk in enumerate(blocks):
        blk.attn.proj.weight.data.div_(math.sqrt(2.0 * (i + 1)))
        blk.mlp.fc2.weight.data.div_(math.sqrt(2.0 * (i + 1)))


class ViT(nn.Module):
    """Fixed 2-D sin-cos positions, no CLS token, final LayerNorm. Inputs of another size (LeJEPA local crops) get
    sin-cos positions for their own grid."""

    def __init__(self, img_size, patch, dim, depth, heads, grad_ckpt=False):
        super().__init__()
        self.grid, self.patch, self.grad_ckpt, self.dim = img_size // patch, patch, grad_ckpt, dim
        self.patch_embed = nn.Conv2d(3, dim, patch, patch)
        self.register_buffer("pos_embed", sincos_pos_embed(dim, self.grid, self.grid), persistent=False)
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self._other_pos = {}
        self.apply(_init_weights)
        _rescale_blocks(self.blocks)

    def _pos(self, gh, gw, device):
        if gh == self.grid and gw == self.grid:
            return self.pos_embed
        key = (gh, gw, str(device))
        if key not in self._other_pos:
            self._other_pos[key] = sincos_pos_embed(self.dim, gh, gw).to(device)
        return self._other_pos[key]

    def forward(self, x, keep_idx=None):
        t = self.patch_embed(x).flatten(2).transpose(1, 2)
        t = t + self._pos(x.shape[-2] // self.patch, x.shape[-1] // self.patch, x.device)
        if keep_idx is not None:
            t = gather_tokens(t, keep_idx)
        for blk in self.blocks:
            if self.grad_ckpt and self.training and torch.is_grad_enabled():
                t = torch.utils.checkpoint.checkpoint(blk, t, use_reentrant=False)
            else:
                t = blk(t)
        return self.norm(t)


class Predictor(nn.Module):
    def __init__(self, grid, enc_dim, dim, depth, heads):
        super().__init__()
        self.embed = nn.Linear(enc_dim, dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.register_buffer("pos_embed", sincos_pos_embed(dim, grid, grid), persistent=False)
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.proj = nn.Linear(dim, enc_dim)
        self.apply(_init_weights)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        _rescale_blocks(self.blocks)

    def forward(self, ctx, ctx_idx, tgt_idx):
        b = ctx.shape[0]
        pos = self.pos_embed.expand(b, -1, -1)
        x = self.embed(ctx) + gather_tokens(pos, ctx_idx)
        m = self.mask_token.expand(b, tgt_idx.shape[1], -1) + gather_tokens(pos, tgt_idx)
        z = torch.cat([x, m], 1)
        for blk in self.blocks:
            z = blk(z)
        return self.proj(self.norm(z[:, -tgt_idx.shape[1]:]))


class SupervisedViT(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.encoder = ViT(cfg.img_size, cfg.patch_size, cfg.enc_dim, cfg.enc_depth, cfg.enc_heads, cfg.grad_ckpt)
        self.head = nn.Linear(cfg.enc_dim, n_classes)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        return self.head(self.encoder(x).mean(1))


def make_encoder():
    return ViT(cfg.img_size, cfg.patch_size, cfg.enc_dim, cfg.enc_depth, cfg.enc_heads, cfg.grad_ckpt)


def make_predictor():
    return Predictor(cfg.img_size // cfg.patch_size, cfg.enc_dim, cfg.pred_dim, cfg.pred_depth, cfg.pred_heads)


_e, _p = make_encoder(), make_predictor()
print(f"Encoder parameters: {sum(p.numel() for p in _e.parameters()) / 1e6:.2f} M | "
      f"predictor: {sum(p.numel() for p in _p.parameters()) / 1e6:.2f} M")
del _e, _p


# ======================= notebook cell 24 =======================
PLACEMENTS = {"ijepa": "uniform", "veg_pos": "veg_pos", "veg_neg": "veg_neg", "pos_matched": "prior", "attn": "attn"}
PLACEMENT_SIGN = {"veg_pos": 1.0, "veg_neg": -1.0, "attn": 1.0}
LOCAL_ASPECT = (3 / 4, 4 / 3)


def block_size(scale, aspect, gh, gw, gen):
    """Stage A's _size: scale and log-aspect drawn from the CPU generator, rounded to whole patches."""
    s = torch.empty(1).uniform_(scale[0], scale[1], generator=gen).item()
    a = math.exp(torch.empty(1).uniform_(math.log(aspect[0]), math.log(aspect[1]), generator=gen).item())
    area = s * gh * gw
    h = int(round(math.sqrt(area * a)))
    w = int(round(math.sqrt(area / a)))
    return min(max(h, 1), gh), min(max(w, 1), gw)


def enumerate_block_sizes(scale, aspect, gh, gw, n=301):
    """Every block size reachable by block_size (dense sweep over scale x log-aspect, same rounding and clipping)."""
    sizes = set()
    for s in np.linspace(scale[0], scale[1], n):
        area = s * gh * gw
        for la in np.linspace(math.log(aspect[0]), math.log(aspect[1]), n):
            a = math.exp(la)
            h, w = int(round(math.sqrt(area * a))), int(round(math.sqrt(area / a)))
            sizes.add((min(max(h, 1), gh), min(max(w, 1), gw)))
    return sorted(sizes)


def placement_probs(scores, h, w, sign, tau, eps):
    """p over the (nr x nc) valid top-left positions of an h x w block: (1-eps) softmax(sign z / tau) + eps uniform,
    where z is the per-image z-score of the block-averaged score map. tau = inf gives uniform."""
    bs = F.avg_pool2d(scores.unsqueeze(1).float(), (h, w), stride=1).flatten(1)
    n = bs.shape[1]
    if tau is None or math.isinf(tau):
        soft = torch.full_like(bs, 1.0 / n)
    else:
        z = (bs - bs.mean(1, keepdim=True)) / (bs.std(1, keepdim=True, unbiased=False) + 1e-6)
        soft = torch.softmax(sign * z / tau, 1)
    return (1 - eps) * soft + eps / n


def sample_positions(placement, b, h, w, gh, gw, scores, tau, eps, prior_fn, gen):
    """Top-left (row, col) per image. Returns (r, c, used_fallback)."""
    nr, nc = gh - h + 1, gw - w + 1
    if placement == "uniform" or nr * nc == 1:
        r = torch.randint(0, nr, (b,), device=DEVICE, generator=gen)
        c = torch.randint(0, nc, (b,), device=DEVICE, generator=gen)
        return r, c, False
    if placement == "prior":
        P = prior_fn(h, w)
        if P is None:   # should never happen: the prior covers every reachable block size (unit-tested)
            r = torch.randint(0, nr, (b,), device=DEVICE, generator=gen)
            c = torch.randint(0, nc, (b,), device=DEVICE, generator=gen)
            return r, c, True
        p = P.expand(b, -1)
    else:
        p = placement_probs(scores, h, w, PLACEMENT_SIGN[placement], tau, eps)
    flat = torch.multinomial(p, 1, generator=gen).squeeze(1)
    return flat // nc, flat % nc, False


class PriorLookup:
    def __init__(self, prior):
        self.prior = prior or {}
        self.cache = {}

    def __call__(self, h, w):
        if (h, w) not in self.prior:
            return None
        if (h, w) not in self.cache:
            self.cache[(h, w)] = torch.as_tensor(self.prior[(h, w)].reshape(1, -1), dtype=torch.float32, device=DEVICE)
        return self.cache[(h, w)]


class MaskGenerator:
    """4 target blocks (sizes shared by the batch, positions per image, placed by the arm's rule) and one context
    block minus the targets, equalised to the batch minimum. Context placement is uniform for every arm. Also
    accumulates the manipulation-check statistics (target ExG, target-centre histogram) without GPU syncs."""

    def __init__(self, arm, seed, tau=None, eps=None, prior=None):
        self.arm, self.placement = arm, PLACEMENTS[arm]
        self.gh = self.gw = cfg.img_size // cfg.patch_size
        self.n = self.gh * self.gw
        self.tau = cfg.veg_tau if tau is None else tau
        self.eps = cfg.veg_eps if eps is None else eps
        self.prior_fn = PriorLookup(prior)
        self.cpu_gen = torch.Generator().manual_seed(seed * 31 + 1)
        self.dev_gen = torch.Generator(device=DEVICE)
        self.dev_gen.manual_seed(seed * 31 + 2)
        self.rows = torch.arange(self.gh, device=DEVICE)
        self.cols = torch.arange(self.gw, device=DEVICE)
        self.centre_hist = torch.zeros(self.n, dtype=torch.float64, device=DEVICE)
        self.prior_fallbacks = 0

    def state(self):
        return {"cpu": self.cpu_gen.get_state(), "dev": self.dev_gen.get_state(), "centre_hist": self.centre_hist.cpu(),
                "prior_fallbacks": self.prior_fallbacks}

    def load(self, st):
        self.cpu_gen.set_state(st["cpu"])
        self.dev_gen.set_state(st["dev"])
        self.centre_hist = st["centre_hist"].to(DEVICE)
        self.prior_fallbacks = int(st["prior_fallbacks"])

    def _uniform(self, b, nr, nc):
        r = torch.randint(0, nr, (b,), device=DEVICE, generator=self.dev_gen)
        c = torch.randint(0, nc, (b,), device=DEVICE, generator=self.dev_gen)
        return r, c

    def _mask(self, r, c, h, w):
        mr = (self.rows[None] >= r[:, None]) & (self.rows[None] < (r + h)[:, None])
        mc = (self.cols[None] >= c[:, None]) & (self.cols[None] < (c + w)[:, None])
        return (mr[:, :, None] & mc[:, None, :]).flatten(1)

    def _idx(self, r, c, h, w):
        off = (torch.arange(h, device=DEVICE)[:, None] * self.gw + torch.arange(w, device=DEVICE)[None]).flatten()
        return (r * self.gw + c)[:, None] + off[None]

    @torch.no_grad()
    def __call__(self, x01, attn_scores=None):
        b = x01.shape[0]
        exg = veg_patch_scores(x01, cfg.patch_size, cfg.veg_index)
        scores = attn_scores if self.placement == "attn" else exg
        tgt_idx, union = [], torch.zeros(b, self.n, dtype=torch.bool, device=DEVICE)
        for _ in range(cfg.num_targets):
            h, w = block_size(cfg.target_scale, cfg.target_aspect, self.gh, self.gw, self.cpu_gen)
            r, c, fb = sample_positions(self.placement, b, h, w, self.gh, self.gw, scores, self.tau, self.eps,
                                        self.prior_fn, self.dev_gen)
            self.prior_fallbacks += b if fb else 0
            tgt_idx.append(self._idx(r, c, h, w))
            union |= self._mask(r, c, h, w)
            centre = (r + h // 2) * self.gw + (c + w // 2)
            self.centre_hist += torch.bincount(centre, minlength=self.n).double()
        hc, wc = block_size(cfg.context_scale, (1.0, 1.0), self.gh, self.gw, self.cpu_gen)
        ctx = torch.zeros(b, self.n, dtype=torch.bool, device=DEVICE)
        for _ in range(10):  # re-sample context blocks that keep too few tokens
            need = ctx.sum(1) < cfg.min_context
            if not need.any():
                break
            r, c = self._uniform(b, self.gh - hc + 1, self.gw - wc + 1)
            ctx = torch.where(need[:, None], self._mask(r, c, hc, wc) & ~union, ctx)
        bad = ctx.sum(1) < cfg.min_context
        ctx[bad] = ~union[bad]                       # fall back to "everything except the targets"
        empty = ctx.sum(1) == 0
        ctx[empty] = True                            # degenerate case: allow all tokens
        n_keep = int(ctx.sum(1).min().item())
        if cfg.ctx_tokens_cap:
            n_keep = max(1, min(n_keep, int(cfg.ctx_tokens_cap)))
        noise = torch.rand(b, self.n, device=DEVICE, generator=self.dev_gen).masked_fill(~ctx, -1.0)
        ctx_idx = noise.topk(n_keep, dim=1).indices.sort(1).values
        e = exg.flatten(1).float()
        ez = (e - e.mean(1, keepdim=True)) / (e.std(1, keepdim=True, unbiased=False) + 1e-6)
        u = union.float()
        cmask = torch.zeros_like(u).scatter_(1, ctx_idx, 1.0)
        st = {"ctx_tokens": n_keep, "tgt_tokens": sum(t.shape[1] for t in tgt_idx),
              "tgt_exg": ((e * u).sum(1) / u.sum(1).clamp(min=1)).mean(),
              "tgt_exg_z": ((ez * u).sum(1) / u.sum(1).clamp(min=1)).mean(),
              "ctx_exg": ((e * cmask).sum(1) / cmask.sum(1).clamp(min=1)).mean(),
              "all_exg": e.mean(), "degenerate_ctx": empty.float().sum()}
        return ctx_idx, tgt_idx, st


def prior_key(kind):
    key = dict(kind=kind, img_size=cfg.img_size, patch=cfg.patch_size, cache_res=cfg.cache_res, veg_index=cfg.veg_index,
               tau=cfg.veg_tau, eps=cfg.veg_eps, n_images=cfg.prior_images, seed=cfg.prior_seed, train=TRAIN_TAG)
    if kind == "target":
        key.update(scale=cfg.target_scale, aspect=cfg.target_aspect, crop_scale=cfg.crop_scale, hflip=cfg.hflip,
                   batch=cfg.batch_size)
    else:
        key.update(scale=cfg.lejepa_local_scale, aspect=LOCAL_ASPECT, batch=cfg.batch_size)
    return json.loads(json.dumps(key, default=str))


def prior_hash(prior):
    h = hashlib.sha1()
    for s in sorted(prior):
        h.update(f"{s[0]}x{s[1]}".encode())
        h.update(np.ascontiguousarray(prior[s], dtype=np.float64).tobytes())
    return h.hexdigest()[:12]


@torch.no_grad()
def estimate_position_prior(kind="target"):
    """pos_matched prior: for every reachable block size, the average over PRIOR_IMAGES training crops (same GPU
    augmentation as training, fixed generator) of the full veg_pos probability map, including the eps mixture.
    Depends only on data and config, so all seeds share it; its hash enters the run config hash."""
    pdir = OUT / "priors"
    prefix = "pos_prior" if kind == "target" else "lejepa_local_prior"
    key = prior_key(kind)
    gh = gw = cfg.img_size // cfg.patch_size
    sizes = enumerate_block_sizes(key["scale"], key["aspect"], gh, gw)
    meta_path = pdir / f"{prefix}_meta.json"
    if meta_path.exists():
        meta = load_json(meta_path)
        files = [pdir / f"{prefix}_{h}x{w}.npy" for h, w in sizes]
        if meta.get("key") == key and all(f.exists() for f in files):
            prior = {s: np.load(f) for s, f in zip(sizes, files)}
            if prior_hash(prior) == meta.get("hash"):
                print(f"Position prior ({kind}) loaded: {len(prior)} block sizes, hash {meta['hash']}")
                return prior, meta["hash"]
        print(f"Position prior ({kind}) on disk does not match the current data/config: re-estimating.")
    t0 = time.time()
    loader = GPULoader(CACHE, ROWS["train"], cfg.batch_size, train=True, out_size=cfg.img_size, seed=cfg.prior_seed,
                       augment=(kind == "target"))
    acc = {s: torch.zeros((gh - s[0] + 1) * (gw - s[1] + 1), dtype=torch.float64, device=DEVICE) for s in sizes}
    n, epoch = 0, 0
    while n < cfg.prior_images:
        loader.set_epoch(epoch)
        for x01, _ in loader:
            x01 = x01[: cfg.prior_images - n]
            scores = veg_patch_scores(x01, cfg.patch_size, cfg.veg_index)
            for s in sizes:
                acc[s] += placement_probs(scores, s[0], s[1], 1.0, cfg.veg_tau, cfg.veg_eps).double().sum(0)
            n += x01.shape[0]
            if n >= cfg.prior_images:
                break
        epoch += 1
    prior = {}
    for s in sizes:
        P = (acc[s] / n).cpu().numpy()
        prior[s] = (P / P.sum()).reshape(gh - s[0] + 1, gw - s[1] + 1)
        save_npy(prior[s], pdir / f"{prefix}_{s[0]}x{s[1]}.npy")
    ph = prior_hash(prior)
    save_json({"key": key, "sizes": [list(s) for s in sizes], "hash": ph, "images_used": n,
               "created": datetime.datetime.now().isoformat()}, meta_path)
    print(f"Position prior ({kind}) estimated from {n} training {'crops' if kind == 'target' else 'images'} for "
          f"{len(sizes)} block sizes {sizes} in {time.time() - t0:.1f}s; hash {ph}")
    return prior, ph


POS_PRIOR, POS_PRIOR_HASH = (estimate_position_prior("target") if ("pos_matched" in ARMS or RUN_MODE == "smoke")
                             else (None, None))


# ======================= notebook cell 25 =======================
def kl_div(p, q, eps=1e-9):
    p = np.asarray(p, float) + eps
    q = np.asarray(q, float) + eps
    p, q = p / p.sum(), q / q.sum()
    return float((p * np.log(p / q)).sum())


@torch.no_grad()
def mask_acceptance_check(arms, n_batches, seed=777):
    """Generate training masks for every arm on the same augmented batches; compare target-centre histograms and the
    mean ExG under the targets. veg_pos and pos_matched should have ~identical centre histograms (KL ~ 0) but differ
    in target ExG unless position and greenness are collinear (itself a finding)."""
    loader = GPULoader(CACHE, ROWS["train"], cfg.batch_size, train=True, out_size=cfg.img_size, seed=seed)
    gens = {a: MaskGenerator(a, 0, prior=POS_PRIOR) for a in arms}
    acc = {a: {k: 0.0 for k in ["tgt_exg", "tgt_exg_z", "ctx_exg", "all_exg", "ctx_tokens", "tgt_tokens"]} for a in arms}
    done, epoch = 0, 0
    while done < n_batches:
        loader.set_epoch(epoch)
        for x01, _ in loader:
            for a, g in gens.items():
                _, _, st = g(x01)
                for k in acc[a]:
                    acc[a][k] += float(st[k])
            done += 1
            if done >= n_batches:
                break
        epoch += 1
    hists = {a: g.centre_hist.cpu().numpy() for a, g in gens.items()}
    rows = []
    for a in arms:
        r = {"source": "pre-training check", "arm": a, "batches": done, **{k: v / done for k, v in acc[a].items()}}
        if "veg_pos" in hists:
            r["kl_centres_vs_veg_pos"] = kl_div(hists["veg_pos"], hists[a])
        rows.append(r)
    return pd.DataFrame(rows), hists


def run_mask_unit_tests():
    failures, report = [], []

    def check(name, ok, detail="", hard=True):
        tag = "PASS" if ok else ("FAIL" if hard else "WARN")
        report.append({"test": name, "result": tag, "detail": detail})
        print(f"  [{tag}] {name} {detail}")
        if hard and not ok:
            failures.append(name)

    gh = cfg.img_size // cfg.patch_size
    loader = GPULoader(CACHE, ROWS["train"], cfg.batch_size, train=True, out_size=cfg.img_size, seed=4242)
    x01, _ = next(iter(loader))
    b = x01.shape[0]
    test_arms = [a for a in KNOWN_ARMS if a != "pos_matched" or POS_PRIOR]
    attn_scores = torch.rand(b, gh, gh, device=DEVICE, generator=torch.Generator(device=DEVICE).manual_seed(5))
    for arm in test_arms:
        g = MaskGenerator(arm, 0, prior=POS_PRIOR)
        ctx_idx, tgt, st = g(x01, attn_scores=attn_scores)
        ok_shape = (ctx_idx.shape == (b, st["ctx_tokens"]) and ctx_idx.dtype == torch.long and len(tgt) == cfg.num_targets
                    and all(t.shape[0] == b and t.min() >= 0 and t.max() < gh * gh for t in tgt))
        check(f"{arm}: shapes and index ranges", ok_shape, f"ctx {tuple(ctx_idx.shape)}, targets {[tuple(t.shape) for t in tgt]}")
        check(f"{arm}: context indices sorted and unique", bool((ctx_idx[:, 1:] > ctx_idx[:, :-1]).all()))
        union = torch.zeros(b, gh * gh, dtype=torch.bool, device=DEVICE)
        for t in tgt:
            union.scatter_(1, t, True)
        overlap = torch.gather(union, 1, ctx_idx).sum(1)
        n_degenerate = int(st["degenerate_ctx"])
        check(f"{arm}: no context/target overlap", int((overlap > 0).sum()) <= n_degenerate,
              f"(rows with overlap {int((overlap > 0).sum())}, degenerate all-token fallbacks {n_degenerate})")
        g2 = MaskGenerator(arm, 0, prior=POS_PRIOR)
        ctx2, tgt2, _ = g2(x01, attn_scores=attn_scores)
        same = torch.equal(ctx_idx, ctx2) and all(torch.equal(a1, a2) for a1, a2 in zip(tgt, tgt2))
        check(f"{arm}: deterministic for a fixed seed", same)
    scores = veg_patch_scores(x01, cfg.patch_size, cfg.veg_index)
    sizes = enumerate_block_sizes(cfg.target_scale, cfg.target_aspect, gh, gh)
    max_err = 0.0
    for s in sizes:
        for sign in (1.0, -1.0):
            p = placement_probs(scores, s[0], s[1], sign, cfg.veg_tau, cfg.veg_eps)
            max_err = max(max_err, float((p.sum(1) - 1).abs().max()))
    check("placement probabilities sum to 1 (veg_pos and veg_neg)", max_err < 1e-4, f"(max |sum-1| = {max_err:.2e})")
    if POS_PRIOR:
        perr = max(abs(float(P.sum()) - 1) for P in POS_PRIOR.values())
        check("pos_matched prior maps sum to 1", perr < 1e-6, f"(max |sum-1| = {perr:.2e})")
        gen = torch.Generator().manual_seed(123)
        drawn = {block_size(cfg.target_scale, cfg.target_aspect, gh, gh, gen) for _ in range(20000)}
        missing = sorted(drawn - set(POS_PRIOR))
        check("prior covers every block size drawn in 20,000 samples", not missing, f"(missing {missing})")
    return failures, report


print("Mask-generator unit tests:")
MASK_FAILURES, MASK_REPORT = run_mask_unit_tests()

accept_arms = [a for a in ["ijepa", "veg_pos", "veg_neg", "pos_matched"] if a != "pos_matched" or POS_PRIOR]
ACCEPT, ACCEPT_HISTS = mask_acceptance_check(accept_arms, cfg.accept_batches)
print(f"\npos_matched acceptance check ({cfg.accept_batches} training batches, same crops for every arm):")
display(ACCEPT.round(4))
if POS_PRIOR and "veg_pos" in ACCEPT_HISTS:
    kl_pm = kl_div(ACCEPT_HISTS["veg_pos"], ACCEPT_HISTS["pos_matched"])
    kl_ij = kl_div(ACCEPT_HISTS["veg_pos"], ACCEPT_HISTS["ijepa"])
    thresh = 0.05 if RUN_MODE == "smoke" else 0.01
    ok = kl_pm < thresh
    print(f"KL(centres veg_pos || pos_matched) = {kl_pm:.4f} (should be ~0; threshold {thresh}) | "
          f"for reference KL(veg_pos || ijepa) = {kl_ij:.4f}")
    MASK_REPORT.append({"test": "pos_matched centre histogram matches veg_pos", "result": "PASS" if ok else "WARN",
                        "detail": f"KL {kl_pm:.4f}"})
    if not ok:
        print("  [WARN] the centre histograms differ more than expected — check the prior estimation.")
    d = ACCEPT.set_index("arm")["tgt_exg_z"]
    order_ok = d.get("veg_pos", 0) > d.get("ijepa", 0) > d.get("veg_neg", 0)
    print(f"Mean z-scored ExG under the targets: " + ", ".join(f"{a} {v:+.3f}" for a, v in d.items()) +
          ("  (expected order veg_pos > ijepa > veg_neg: OK)" if order_ok else "  [WARN] unexpected order"))
    gap_pm = d.get("pos_matched", 0) - d.get("ijepa", 0)
    gap_vp = d.get("veg_pos", 0) - d.get("ijepa", 0)
    if gap_vp > 0:
        print(f"pos_matched recovers {100 * gap_pm / gap_vp:.0f} % of veg_pos's target-greenness gain over ijepa through "
              "position alone (0 % = position carries no greenness; 100 % = position and greenness are collinear).")
save_csv(ACCEPT, OUT / "stats" / "manipulation_check_pretraining.csv")
save_csv(pd.DataFrame(MASK_REPORT), OUT / "stats" / "mask_unit_tests.csv")
if MASK_FAILURES:
    raise RuntimeError(f"Mask-generator unit tests failed: {MASK_FAILURES}")
print("All hard mask-generator unit tests passed.")


# ======================= notebook cell 27 =======================
def confusion(y, p, k):
    return np.bincount(np.asarray(y) * k + np.asarray(p), minlength=k * k).reshape(k, k)


def metrics_from_cm(cm):
    tp = np.diag(cm).astype(float)
    support, predicted = cm.sum(1), cm.sum(0)
    prec = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    rec = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    present = support > 0
    return {"acc": float(tp.sum() / max(1, cm.sum())), "macro_f1": float(f1[present].mean()),
            "bal_acc": float(rec[present].mean())}, f1


def param_groups(modules, wd=None):
    decay, no_decay = [], []
    for m in modules:
        for n, p in m.named_parameters():
            if not p.requires_grad:
                continue
            (no_decay if p.ndim < 2 or "mask_token" in n else decay).append(p)
    return [{"params": decay, "weight_decay": cfg.wd_start if wd is None else wd, "apply_wd": True},
            {"params": no_decay, "weight_decay": 0.0, "apply_wd": False}]


def lr_at(step, total, peak):
    warm = max(1, int(cfg.warmup_frac * total))
    if step < warm:
        return peak * (step + 1) / warm
    t = (step - warm) / max(1, total - warm)
    return cfg.min_lr + 0.5 * (peak - cfg.min_lr) * (1 + math.cos(math.pi * t))


def cos_ramp(step, total, start, end):
    t = min(1.0, step / max(1, total))
    return end + (start - end) * 0.5 * (1 + math.cos(math.pi * t))


def schedule_plan(n_rows, steps, batch):
    bs = min(batch, n_rows)
    spe = max(1, n_rows // bs)
    epochs = max(1, math.ceil(steps / spe))
    return {"batch": bs, "steps_per_epoch": spe, "epochs": epochs, "total_steps": epochs * spe, "target_steps": int(steps)}


def monitor_plan(epochs):
    """Monitoring and checkpoint frequency scale with the number of epochs (a 10 % data run has ~10x more epochs)."""
    rep_every = max(1, int(round(epochs / 100)))
    knn_every = max(cfg.knn_monitor_every, int(round(epochs / 20)))
    ckpt_every = max(cfg.ckpt_every, int(round(epochs / 20)))
    return rep_every, knn_every, ckpt_every


MONITOR_ROWS = ROWS["val"][:cfg.monitor_n] if len(ROWS["val"]) else ROWS["train"][:cfg.monitor_n]
KNN_MONITOR_ROWS = ROWS["train"][np.random.default_rng(0).permutation(len(ROWS["train"]))[:cfg.knn_monitor_train]]
REP_KEYS = ["rankme", "feat_std", "feat_norm", "top_sv_share"]


@torch.no_grad()
def representation_stats(encoder, rows, bs=128):
    was = encoder.training
    encoder.eval()
    feats = []
    for x01, _ in GPULoader(CACHE, rows, bs, train=False, out_size=cfg.img_size):
        with autocast_ctx():
            feats.append(encoder(normalize(x01)).mean(1).float())
    encoder.train(was)
    f = torch.cat(feats).cpu()
    sv = torch.linalg.svdvals(f - f.mean(0))  # centred, so a shared mean vector does not hide collapse
    p = sv / sv.sum() + 1e-7
    z = F.normalize(f, dim=1)
    return {"rankme": float(torch.exp(-(p * p.log()).sum())), "feat_std": float(z.std(0).mean()),
            "feat_norm": float(f.norm(dim=1).mean()), "top_sv_share": float(sv[0] / sv.sum())}


@torch.no_grad()
def online_knn_f1(encoder, train_rows, val_rows, bs=128):
    """Macro-F1 of a 20-NN classifier on validation images: progress monitor only. Labels are never used to choose
    checkpoints or settings."""
    if len(val_rows) == 0:
        return float("nan")
    was = encoder.training
    encoder.eval()
    feats = {}
    for name, rows in (("tr", train_rows), ("va", val_rows)):
        out = []
        for x01, _ in GPULoader(CACHE, rows, bs, train=False, out_size=cfg.img_size):
            with autocast_ctx():
                out.append(encoder(normalize(x01)).mean(1).float().cpu())
        feats[name] = torch.cat(out)
    encoder.train(was)
    ytr, yva = Y_ALL[train_rows], Y_ALL[val_rows]
    tr, va = F.normalize(feats["tr"], dim=1), F.normalize(feats["va"], dim=1)
    v, idx = (va @ tr.T).topk(min(20, len(tr)), dim=1)
    votes = torch.zeros(len(va), K).scatter_add_(1, torch.from_numpy(ytr)[idx], (v / cfg.knn_t).exp())
    return metrics_from_cm(confusion(yva, votes.argmax(1).numpy(), K))[0]["macro_f1"]


@torch.no_grad()
def first_batch_signature(loader):
    loader.set_epoch(0)
    x01, idx = next(iter(loader))
    return {"first_batch_idx_hash": strings_hash(np.asarray(idx).tolist()),
            "first_batch_mean": float(x01.float().mean()), "first_batch_std": float(x01.float().std())}


class JEPARun:
    """I-JEPA with an EMA target encoder. Same seed => same initial weights, same image order and crops for every arm
    (checked through init_hash / first-batch hashes). Budget in optimizer steps; schedules over total steps."""
    kind = "ijepa"

    def __init__(self, arm, seed, run_dir, steps, family="core", label=None, tau=None, eps=None, train_rows=None, variant=None):
        self.arm, self.seed, self.dir, self.family = arm, seed, Path(run_dir), family
        self.label, self.variant = label or arm, variant
        self.dir.mkdir(parents=True, exist_ok=True)
        self.placement = PLACEMENTS[arm]
        uses_score = self.placement in ("veg_pos", "veg_neg", "attn")
        self.tau = (cfg.veg_tau if tau is None else float(tau)) if uses_score else None
        self.eps = (cfg.veg_eps if eps is None else float(eps)) if uses_score else None
        self.prior, prior_h = self._prior()
        if self.placement == "prior" and not self.prior:
            raise RuntimeError("pos_matched needs the position prior (section 9).")
        self.train_rows = ROWS["train"] if train_rows is None else np.asarray(train_rows, dtype=np.int64)
        self.loader = self._make_loader(seed)
        self.plan = schedule_plan(len(self.train_rows), steps, self.loader.bs)
        assert self.plan["steps_per_epoch"] == len(self.loader)
        self.epochs, self.total = self.plan["epochs"], self.plan["total_steps"]
        self.rep_every, self.knn_every, self.ckpt_every = monitor_plan(self.epochs)
        self.hash = config_hash(cfg, self.kind if self.kind != "ijepa" else arm, seed, steps=steps, tau=self.tau, eps=self.eps,
                                prior=prior_h if self.placement == "prior" else None, data=rows_tag(self.train_rows),
                                extra=arm if self.kind != "ijepa" else None)
        seed_everything(seed)  # same initial weights for every arm with this seed (paired design)
        self._build_models()
        self.init_hash = self._init_hash()
        self._to_device()
        self.history, self.step, self.epochs_done, self.resumed_from = [], 0, 0, None
        self.pairing = {"seed": seed, "init_hash": self.init_hash, **first_batch_signature(self.loader)}

    # ---- hooks (overridden by the LeJEPA arm) ----
    def _prior(self):
        return POS_PRIOR, POS_PRIOR_HASH

    def _make_loader(self, seed):
        return GPULoader(CACHE, self.train_rows, cfg.batch_size, train=True, out_size=cfg.img_size, seed=seed)

    def _build_models(self):
        self.ctx_enc = make_encoder()
        self.tgt_enc = copy.deepcopy(self.ctx_enc)
        self.pred = make_predictor()

    def _init_hash(self):
        return state_hash(self.ctx_enc, self.pred)

    def _to_device(self):
        self.ctx_enc.to(DEVICE)
        self.tgt_enc.to(DEVICE).eval()
        for p in self.tgt_enc.parameters():
            p.requires_grad_(False)
        self.pred.to(DEVICE)
        self.peak_lr = cfg.blr * self.loader.bs / 256
        self.opt = torch.optim.AdamW(param_groups([self.ctx_enc, self.pred]), lr=self.peak_lr)
        self.scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda" and cfg.precision == "fp16")
        self.maskgen = MaskGenerator(self.arm, self.seed, tau=self.tau, eps=self.eps, prior=self.prior)
        self.params = [p for m in (self.ctx_enc, self.pred) for p in m.parameters() if p.requires_grad]
        self.cparams = list(self.ctx_enc.parameters())
        self.tparams = list(self.tgt_enc.parameters())

    def encoder_for_eval(self):
        return self.tgt_enc

    def set_train_mode(self):
        self.ctx_enc.train()
        self.pred.train()

    def centre_hist(self):
        return self.maskgen.centre_hist.cpu().numpy()

    def prior_fallbacks(self):
        return self.maskgen.prior_fallbacks

    def _models_state(self):
        return {"ctx_enc": self.ctx_enc.state_dict(), "tgt_enc": self.tgt_enc.state_dict(), "pred": self.pred.state_dict(),
                "maskgen": self.maskgen.state()}

    def _load_models(self, st):
        self.ctx_enc.load_state_dict(st["ctx_enc"])
        self.tgt_enc.load_state_dict(st["tgt_enc"])
        self.pred.load_state_dict(st["pred"])
        self.maskgen.load(st["maskgen"])

    # ---- checkpointing ----
    def state(self):
        return {**self._models_state(), "opt": self.opt.state_dict(), "scaler": self.scaler.state_dict(), "step": self.step,
                "epochs_done": self.epochs_done, "history": self.history, "cfg_hash": self.hash, "rng": get_rng_state(),
                "saved_at": datetime.datetime.now().isoformat()}

    def save(self):
        atomic_torch_save(self.state(), self.dir / "last.pt")

    def write_log(self):
        if self.history:
            save_csv(pd.DataFrame(self.history), self.dir / "train_log.csv")

    def try_resume(self):
        ck = self.dir / "last.pt"
        if not ck.exists():
            return
        st = safe_torch_load(ck, map_location="cpu")
        if st["cfg_hash"] != self.hash:
            raise RuntimeError(f"{ck} was written with a different configuration (hash {st['cfg_hash']} vs {self.hash}). "
                               "Move that run folder away to start fresh, or restore the old settings.")
        self._load_models(st)
        self.opt.load_state_dict(st["opt"])
        self.scaler.load_state_dict(st["scaler"])
        self.step, self.epochs_done, self.history = int(st["step"]), int(st["epochs_done"]), list(st["history"])
        set_rng_state(st["rng"])
        self.resumed_from = self.epochs_done
        self.write_log()   # drop log rows written after the last checkpoint
        print(f"  resumed {self.dir.name} at epoch {self.epochs_done}/{self.epochs} (step {self.step})")

    @torch.no_grad()
    def ema_update(self, m):
        torch._foreach_mul_(self.tparams, m)
        torch._foreach_add_(self.tparams, self.cparams, alpha=1 - m)

    def train_step(self, x01, idx=None):
        lr = lr_at(self.step, self.total, self.peak_lr)
        wd = cos_ramp(self.step, self.total, cfg.wd_start, cfg.wd_end)
        m = cfg.ema_start + (cfg.ema_end - cfg.ema_start) * min(1.0, self.step / max(1, self.total))
        for g in self.opt.param_groups:
            g["lr"] = lr
            if g["apply_wd"]:
                g["weight_decay"] = wd
        x = normalize(x01)
        with torch.no_grad(), autocast_ctx():
            h = self.tgt_enc(x)
        h = F.layer_norm(h.float(), (h.shape[-1],))
        attn_scores = None
        if self.placement == "attn":   # AttMask-style stand-in: distance of each target token from the image's mean token
            gh = cfg.img_size // cfg.patch_size
            attn_scores = (h - h.mean(1, keepdim=True)).norm(dim=-1).view(h.shape[0], gh, gh)
        ctx_idx, tgt_list, st = self.maskgen(x01, attn_scores=attn_scores)
        with autocast_ctx():
            z = self.ctx_enc(x, keep_idx=ctx_idx)
            num, den = 0.0, 0
            for t_idx in tgt_list:   # token-weighted loss over all predicted target tokens
                p = self.pred(z, ctx_idx, t_idx)
                t = gather_tokens(h, t_idx)
                num = num + F.smooth_l1_loss(p.float(), t, reduction="sum")
                den += t.numel()
            loss = num / den
        if not torch.isfinite(loss):
            self.opt.zero_grad(set_to_none=True)
            return None, st, (lr, wd, m)
        self.scaler.scale(loss).backward()
        if cfg.grad_clip:
            self.scaler.unscale_(self.opt)
            torch.nn.utils.clip_grad_norm_(self.params, cfg.grad_clip)
        self.scaler.step(self.opt)
        self.scaler.update()
        self.opt.zero_grad(set_to_none=True)
        self.ema_update(m)
        self.step += 1
        return loss.detach(), st, (lr, wd, m)

    def train(self, stop_after_epochs=None, verbose=True):
        if (self.dir / "encoder_final.pt").exists():
            print(f"  {self.dir.name}: pretraining already finished")
            return
        self.try_resume()
        print(f"  {self.dir.name}: {len(self.train_rows)} training images | batch {self.loader.bs} | "
              f"{self.plan['steps_per_epoch']} steps/epoch x {self.epochs} epochs = {self.total} steps "
              f"(target {self.plan['target_steps']}) | peak lr {self.peak_lr:.2e} | monitors every {self.rep_every} ep, "
              f"k-NN every {self.knn_every} ep, checkpoint every {self.ckpt_every} ep")
        print_every = max(1, self.epochs // 10)
        reserve_s = cfg.eval_reserve_min * 60
        for epoch in range(self.epochs_done, self.epochs):
            if self.history:
                avg = float(np.mean([r["epoch_time_s"] for r in self.history[-5:]]))
                if hours_left() * 3600 < 1.5 * avg + reserve_s:
                    self.save()
                    raise TimeBudgetExceeded(f"{self.dir.name}: stopped before epoch {epoch + 1}; checkpoint saved")
            self.loader.set_epoch(epoch)
            self.set_train_mode()
            if HAS_CUDA:
                torch.cuda.reset_peak_memory_stats()
            t0, loss_sum, n_ok, n_bad = time.time(), torch.zeros((), device=DEVICE), 0, 0
            acc, n_st = {}, 0
            sched = (float("nan"),) * 3
            try:
                for x01, idx in self.loader:
                    loss, st, sched = self.train_step(x01, idx)
                    n_st += 1
                    for k, v in st.items():
                        acc[k] = acc.get(k, 0.0) + (v.detach().float() if torch.is_tensor(v) else float(v))
                    if loss is None:
                        n_bad += 1
                        if n_bad > 20:
                            self.save()
                            raise RuntimeError("More than 20 non-finite losses in one epoch: lower blr or check the data.")
                        continue
                    loss_sum += loss
                    n_ok += 1
            except KeyboardInterrupt:
                self.save()
                print("Interrupted: checkpoint saved. Re-run to resume.")
                raise
            ep_time = time.time() - t0
            self.epochs_done = epoch + 1
            last = epoch + 1 == self.epochs
            row = {"epoch": epoch + 1, "step": self.step, "loss": float(loss_sum) / max(1, n_ok), "lr": sched[0], "wd": sched[1],
                   "ema": sched[2], "epoch_time_s": ep_time, "imgs_per_s": n_ok * self.loader.bs / max(ep_time, 1e-6),
                   "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if HAS_CUDA else 0.0, "nonfinite_steps": n_bad,
                   **{k: float(v) / max(1, n_st) for k, v in acc.items()}}
            row["prior_fallbacks"] = self.prior_fallbacks()
            if epoch % self.rep_every == 0 or last:
                row.update(representation_stats(self.encoder_for_eval(), MONITOR_ROWS))
            else:
                row.update({k: float("nan") for k in REP_KEYS})
            do_knn = epoch == 0 or (epoch + 1) % self.knn_every == 0 or last
            row["knn_val_f1"] = online_knn_f1(self.encoder_for_eval(), KNN_MONITOR_ROWS, ROWS["val"]) if do_knn else float("nan")
            self.history.append(row)
            self.write_log()
            if verbose and (epoch == 0 or (epoch + 1) % print_every == 0 or last):
                knn_txt = f"val kNN-F1 {100 * row['knn_val_f1']:.1f}% " if row["knn_val_f1"] == row["knn_val_f1"] else ""
                rank_txt = f"rank {row['rankme']:.1f} std {row['feat_std']:.4f} " if row["rankme"] == row["rankme"] else ""
                extra = (f"ctx {row['ctx_tokens']:.0f} tgt {row['tgt_tokens']:.0f} tok, tgt ExG z {row['tgt_exg_z']:+.2f}"
                         if "ctx_tokens" in row else
                         f"inv {row['inv_loss']:.4f} sigreg {row['sigreg_loss']:.4f}, local-crop ExG z {row['local_exg_z']:+.2f}"
                         if "inv_loss" in row else "")
                print(f"  {self.dir.name} ep {epoch + 1:4d}/{self.epochs} step {self.step} loss {row['loss']:.4f} {knn_txt}{rank_txt}"
                      f"| {row['imgs_per_s']:.0f} img/s, {row['peak_mem_gb']:.2f} GB {extra}")
            if row["rankme"] == row["rankme"] and row["rankme"] < 0.02 * cfg.enc_dim:
                print(f"  WARNING {self.dir.name}: effective rank {row['rankme']:.1f} is very low — possible collapse.")
            if (epoch + 1) % self.ckpt_every == 0 or last:
                self.save()
            if stop_after_epochs and self.epochs_done >= stop_after_epochs and not last:
                self.save()
                raise TimeBudgetExceeded(f"{self.dir.name}: deliberate stop after epoch {self.epochs_done} (resume test)")
        atomic_torch_save(self.encoder_for_eval().state_dict(), self.dir / "encoder_final.pt")
        save_npy(self.centre_hist(), self.dir / "target_centres.npy")
        save_json({"arm": self.arm, "label": self.label, "seed": self.seed, "cfg_hash": self.hash, **self.plan,
                   "train_minutes": sum(r["epoch_time_s"] for r in self.history) / 60}, self.dir / "train_done.json")


def resume_selftest():
    """Interrupt a tiny run after one epoch, resume it, and compare with an uninterrupted run (smoke mode only)."""
    root = OUT / "selftest"
    shutil.rmtree(root, ignore_errors=True)
    steps = 3 * schedule_plan(len(ROWS["train"]), 1, cfg.batch_size)["steps_per_epoch"]
    r1 = JEPARun("veg_pos", 99, root / "interrupted", steps, family="selftest")
    stopped = False
    try:
        r1.train(stop_after_epochs=1, verbose=False)
    except TimeBudgetExceeded:
        stopped = True
    del r1
    r1b = JEPARun("veg_pos", 99, root / "interrupted", steps, family="selftest")
    r1b.train(verbose=False)
    r2 = JEPARun("veg_pos", 99, root / "straight", steps, family="selftest")
    r2.train(verbose=False)
    a = safe_torch_load(root / "interrupted" / "encoder_final.pt")
    b = safe_torch_load(root / "straight" / "encoder_final.pt")
    diff = max(float((a[k].float() - b[k].float()).abs().max()) for k in a)
    rows = len(pd.read_csv(root / "interrupted" / "train_log.csv"))
    ok = stopped and r1b.resumed_from == 1 and rows == r1b.epochs
    exact = diff < 1e-5
    print(f"Resume self-test: stopped={stopped}, resumed at epoch {r1b.resumed_from}, log rows {rows}/{r1b.epochs}, "
          f"max |weight difference| vs uninterrupted run = {diff:.2e} -> "
          f"{'PASS' if ok and exact else ('PASS (resume works; tiny GPU non-determinism)' if ok else 'FAIL')}")
    if not ok:
        raise RuntimeError("Resume self-test failed")
    return {"stopped": stopped, "resumed_from": r1b.resumed_from, "max_weight_diff": diff, "exact": exact}


if RUN_MODE == "smoke":
    RESUME_TEST = resume_selftest()
    save_json(RESUME_TEST, OUT / "stats" / "resume_selftest.json")


# ======================= notebook cell 29 =======================
@torch.no_grad()
def extract_features(encoder, rows, size=None, norm=normalize):
    encoder.eval()
    out = []
    for x01, _ in GPULoader(CACHE, rows, cfg.eval_batch, train=False, out_size=size or cfg.img_size):
        with autocast_ctx():
            f = encoder(norm(x01))
        f = f.mean(1) if f.dim() == 3 else f
        out.append(f.float().cpu())
    return torch.cat(out).numpy()


def linear_probe(xtr, ytr, xva, yva, xte):
    """Logistic regression; C chosen by validation macro-F1. The scipy/sklearn 'Unknown solver options: iprint' message
    (version mismatch, Stage A problem 9) is dropped; convergence warnings are counted; every other warning is shown."""
    best, n_conv = (-1.0, None, None), 0
    for c in cfg.probe_C:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            clf = LogisticRegression(C=c, max_iter=cfg.probe_max_iter)
            clf.fit(xtr, ytr)
        for w in caught:
            msg = str(w.message)
            if "Unknown solver options" in msg and "iprint" in msg:
                continue
            if issubclass(w.category, ConvergenceWarning):
                n_conv += 1
                continue
            warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)
        f = metrics_from_cm(confusion(yva, clf.predict(xva), K))[0]["macro_f1"] if len(yva) else 0.0
        if f > best[0]:
            best = (f, c, clf)
    return best[2].predict(xte), best[1], best[0], n_conv


def knn_predict(xtr, ytr, xte):
    tr = F.normalize(torch.from_numpy(xtr), dim=1)
    te = F.normalize(torch.from_numpy(xte), dim=1)
    per_class = np.bincount(ytr, minlength=K)
    k = int(min(cfg.knn_k, max(1, per_class[per_class > 0].min())))  # 1-NN at 1-shot; never a global vote
    ytr_t = torch.from_numpy(ytr)
    preds = []
    for i in range(0, len(te), 1024):
        v, idx = (te[i:i + 1024] @ tr.T).topk(k, dim=1)
        votes = torch.zeros(len(v), K).scatter_add_(1, ytr_t[idx], (v / cfg.knn_t).exp())
        preds.append(votes.argmax(1))
    return torch.cat(preds).numpy()


def evaluate_features(feats):
    scaler = StandardScaler().fit(feats["train"])  # fitted on all training features; no labels involved
    xs = {s: scaler.transform(feats[s]) for s in feats}
    results, preds, per_class, n_conv = [], {}, {}, 0
    for regime in REGIMES:
        for d, idx in enumerate(DRAWS[regime]):
            ytr = Y["train"][idx]
            p, c, vf1, nc = linear_probe(xs["train"][idx], ytr, xs["val"], Y["val"], xs["test"])
            n_conv += nc
            m, f1c = metrics_from_cm(confusion(Y["test"], p, K))
            results.append({"eval": "probe", "regime": regime, "draw": d, "C": c, "val_macro_f1": vf1, **m})
            preds[f"probe/{regime}/{d}"] = p.astype(np.int16)
            if regime == "100pct":
                per_class["probe/100pct"] = f1c.tolist()
            pk = knn_predict(feats["train"][idx], ytr, feats["test"])
            m, f1k = metrics_from_cm(confusion(Y["test"], pk, K))
            results.append({"eval": "knn", "regime": regime, "draw": d, **m})
            preds[f"knn/{regime}/{d}"] = pk.astype(np.int16)
            if regime == "100pct":
                per_class["knn/100pct"] = f1k.tolist()
    return results, preds, per_class, {"probe_convergence_warnings": n_conv}


def feature_spectrum(x):
    x = torch.from_numpy(np.asarray(x, dtype=np.float32))
    sv = torch.linalg.svdvals(x - x.mean(0)).numpy()
    return sv / max(sv.sum(), 1e-12)


def save_eval(run_dir, arm, seed, results, preds, extra=None):
    np.savez_compressed(Path(run_dir) / "preds.npz", **preds)
    save_json({"arm": arm, "seed": seed, "profile": cfg.profile, "classes": CLASSES, "results": results, **(extra or {})},
              Path(run_dir) / "eval.json")


def eval_encoder_to_dir(enc, run_dir, arm, seed, extra, size=None):
    feats = {s: extract_features(enc, ROWS[s], size=size) for s in ["train", "val", "test"]}
    save_npy(feature_spectrum(feats["train"]), Path(run_dir) / "spectrum.npy")
    results, preds, per_class, info = evaluate_features(feats)
    save_eval(run_dir, arm, seed, results, preds, {**extra, "per_class_f1": per_class, **info})


def run_ssl_arm(arm, seed, steps=None, family="core", label=None, run_name=None, tau=None, train_rows=None, variant=None,
                run_cls=None):
    run_cls = run_cls or JEPARun
    steps = steps or cfg.target_steps
    label = label or arm
    run_dir = OUT / "runs" / (run_name or f"{label}_s{seed}")
    run = run_cls(arm, seed, run_dir, steps, family=family, label=label, tau=tau, train_rows=train_rows, variant=variant)
    prior_h = (run._prior()[1] if run.placement == "prior" else None)
    save_json({**asdict(cfg), "arm": arm, "label": label, "family": family, "variant": variant, "seed": seed, "kind": run.kind,
               "tau": run.tau, "eps": run.eps, "cfg_hash": run.hash, "plan": run.plan, "prior_hash": prior_h,
               "n_train_rows": int(len(run.train_rows)), "pairing": run.pairing}, run_dir / "config.json")
    save_json(run.pairing, run_dir / "pairing.json")
    run.train()
    enc = make_encoder().to(DEVICE)
    enc.load_state_dict(safe_torch_load(run_dir / "encoder_final.pt", map_location="cpu"))
    eval_encoder_to_dir(enc, run_dir, arm, seed, {"label": label, "family": family, "variant": variant, "kind": run.kind,
                                                   "plan": run.plan, "pairing": run.pairing, "cfg_hash": run.hash})
    if DELETE_LAST_PT_AFTER_EVAL and (run_dir / "last.pt").exists():
        (run_dir / "last.pt").unlink()
    del run, enc
    free_memory()


def run_random_init(seed):
    run_dir = OUT / "runs" / f"random_init_s{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)                 # identical to the SSL arms' starting weights for this seed
    enc, pred = make_encoder(), make_predictor()
    init_hash = state_hash(enc, pred)
    enc = enc.to(DEVICE)
    knn = online_knn_f1(enc, KNN_MONITOR_ROWS, ROWS["val"])
    rep = representation_stats(enc, MONITOR_ROWS)
    eval_encoder_to_dir(enc, run_dir, "random_init", seed, {"label": "random_init", "family": "baseline", "init_hash": init_hash,
                                                            "online_knn_val_f1": knn, "representation": rep})
    del enc, pred
    free_memory()


def run_dinov2(seed=0):
    run_dir = OUT / "runs" / f"dinov2_s{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(DEVICE).eval()
    except Exception as e:
        print(f"  DINOv2 unavailable (needs internet): {e}")
        return
    eval_encoder_to_dir(model, run_dir, "dinov2", seed, {"label": "dinov2", "family": "baseline", "note": "frozen DINOv2 ViT-S/14 at 224 px"},
                        size=224)
    del model
    free_memory()


@torch.no_grad()
def predict_classifier(model, rows):
    model.eval()
    out = []
    for x01, _ in GPULoader(CACHE, rows, cfg.eval_batch, train=False, out_size=cfg.img_size):
        with autocast_ctx():
            out.append(model(normalize(x01)).float().argmax(1).cpu())
    return torch.cat(out).numpy()


def train_supervised_regime(seed, regime, run_dir):
    """ViT from scratch on label draw 0 with augmentation and label smoothing; the epoch with the best validation
    macro-F1 is kept (validation selection is reported). Upper reference, NOT matched to the frozen probes."""
    ck = run_dir / f"sup_{regime}_last.pt"
    rows = ROWS["train"][DRAWS[regime][0]]
    seed_everything(seed)
    model = SupervisedViT(K).to(DEVICE)
    loader = GPULoader(CACHE, rows, cfg.batch_size, train=True, out_size=cfg.img_size, seed=seed, crop_scale=cfg.sup_crop_scale)
    total = cfg.sup_epochs * len(loader)
    peak = cfg.sup_blr * loader.bs / 256
    opt = torch.optim.AdamW(model.parameters(), lr=peak, weight_decay=cfg.sup_wd)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda" and cfg.precision == "fp16")
    start, step, best_f1, best_state, best_epoch = 0, 0, -1.0, None, 0
    if ck.exists():
        st = safe_torch_load(ck, map_location="cpu")
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        scaler.load_state_dict(st["scaler"])
        start, step, best_f1, best_state = int(st["epoch"]), int(st["step"]), float(st["best_f1"]), st["best_state"]
        best_epoch = int(st.get("best_epoch", 0))
        set_rng_state(st["rng"])
    y_all = torch.from_numpy(Y_ALL).to(DEVICE)

    def _save(epoch):
        atomic_torch_save({"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(), "epoch": epoch,
                           "step": step, "best_f1": best_f1, "best_state": best_state, "best_epoch": best_epoch,
                           "rng": get_rng_state()}, ck)

    for epoch in range(start, cfg.sup_epochs):
        if hours_left() * 3600 < cfg.eval_reserve_min * 60:
            _save(epoch)
            raise TimeBudgetExceeded(f"supervised {regime} seed {seed}: stopped at epoch {epoch}")
        loader.set_epoch(epoch)
        model.train()
        for x01, idx in loader:
            for g in opt.param_groups:
                g["lr"] = lr_at(step, total, peak)
            yb = y_all[torch.from_numpy(idx).to(DEVICE)]
            with autocast_ctx():
                loss = F.cross_entropy(model(normalize(x01)).float(), yb, label_smoothing=cfg.label_smoothing)
            if not torch.isfinite(loss):
                opt.zero_grad(set_to_none=True)
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip or 1e9)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            step += 1
        if (epoch + 1) % cfg.sup_eval_every == 0 or epoch + 1 == cfg.sup_epochs:
            pv = predict_classifier(model, ROWS["val"])
            f = metrics_from_cm(confusion(Y["val"], pv, K))[0]["macro_f1"]
            if f > best_f1:
                best_f1, best_epoch = f, epoch + 1
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            _save(epoch + 1)
    model.load_state_dict(best_state)
    return predict_classifier(model, ROWS["test"]), best_f1, best_epoch


def run_supervised(seed):
    run_dir = OUT / "runs" / f"supervised_s{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    results, preds = [], {}
    for regime in [r for r in cfg.sup_regimes if r in DRAWS]:
        p, vf1, be = train_supervised_regime(seed, regime, run_dir)
        m, _ = metrics_from_cm(confusion(Y["test"], p, K))
        results.append({"eval": "supervised", "regime": regime, "draw": 0, "val_macro_f1": vf1, "best_val_epoch": be,
                        "epochs": cfg.sup_epochs, **m})
        preds[f"supervised/{regime}/0"] = p.astype(np.int16)
        free_memory()
    save_eval(run_dir, "supervised", seed, results, preds,
              {"label": "supervised", "family": "baseline",
               "note": "upper reference, NOT matched: all weights trained with labels, augmentation, best-validation epoch"})


# ======================= notebook cell 31 =======================
def flop_split(arm="veg_pos", b=4):
    from torch.utils.flop_counter import FlopCounterMode
    enc, pred = make_encoder().to(DEVICE), make_predictor().to(DEVICE)
    mg = MaskGenerator(arm, 0, prior=POS_PRIOR)
    x01 = torch.rand(b, 3, cfg.img_size, cfg.img_size, device=DEVICE)
    ctx_idx, tgt_list, _ = mg(x01)
    x = normalize(x01)
    counts = {}
    with torch.no_grad():
        with FlopCounterMode(display=False) as fc:
            enc(x)
        counts["target_fwd"] = fc.get_total_flops()
        with FlopCounterMode(display=False) as fc:
            z = enc(x, keep_idx=ctx_idx)
        counts["context_fwd"] = fc.get_total_flops()
        with FlopCounterMode(display=False) as fc:
            for t in tgt_list:
                pred(z, ctx_idx, t)
        counts["predictor_fwd"] = fc.get_total_flops()
    total = counts["target_fwd"] + 3 * counts["context_fwd"] + 3 * counts["predictor_fwd"]  # backward ~ 2x forward
    del enc, pred
    free_memory()
    return {"gflops_per_image": total / b / 1e9, "predictor_share": 3 * counts["predictor_fwd"] / total,
            "context_share": 3 * counts["context_fwd"] / total, "target_share": counts["target_fwd"] / total}


def benchmark(arm="veg_pos", steps=8, warmup=3):
    bs = cfg.batch_size
    while True:
        try:
            cfg.batch_size = bs
            run = JEPARun(arm, 0, OUT / "benchmark_tmp", cfg.target_steps, family="benchmark")
            it = iter(run.loader)
            n_steps = min(len(run.loader), steps + warmup)
            wu = min(warmup, max(0, n_steps - 1))
            for i in range(n_steps):
                if i == wu:
                    if HAS_CUDA:
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                    t0 = time.time()
                x01, idx = next(it)
                run.train_step(x01, idx)
            if HAS_CUDA:
                torch.cuda.synchronize()
            ips = max(1, n_steps - wu) * run.loader.bs / (time.time() - t0)
            peak = torch.cuda.max_memory_allocated() / 1e9 if HAS_CUDA else 0.0
            del run, it
            free_memory()
            break
        except torch.cuda.OutOfMemoryError:
            free_memory()
            if bs <= 8:
                raise
            print(f"  out of memory at batch {bs}; trying {bs // 2}")
            bs //= 2
    shutil.rmtree(OUT / "benchmark_tmp", ignore_errors=True)
    return {"batch_size": bs, "imgs_per_s": ips, "peak_mem_gb": peak}


def eval_speed():
    enc = make_encoder().to(DEVICE)
    rows = ROWS["train"][: min(1024, len(ROWS["train"]))]
    extract_features(enc, rows[: min(64, len(rows))])   # warm-up
    t0 = time.time()
    extract_features(enc, rows)
    del enc
    free_memory()
    return len(rows) / max(1e-6, time.time() - t0)


def probe_seconds(n, d=None):
    """One logistic-regression fit (C = 1) on clustered random features of the real size: a rough stand-in."""
    d = d or cfg.enc_dim
    rng = np.random.default_rng(0)
    y = rng.integers(0, K, n)
    x = rng.normal(0, 0.6, (K, d))[y] + rng.normal(0, 1, (n, d))
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        LogisticRegression(C=1.0, max_iter=cfg.probe_max_iter).fit(x, y)
    return time.time() - t0


def lock_profile(bench):
    path = OUT / "locked_profile.json"
    current = {"profile": cfg.profile, "batch_size": cfg.batch_size, "precision": cfg.precision, "img_size": cfg.img_size,
               "enc_dim": cfg.enc_dim, "cache_res": cfg.cache_res, "gpu": GPU["name"]}
    if path.exists():
        locked = load_json(path)
        diff = {k: (locked.get(k), v) for k, v in current.items() if k != "gpu" and locked.get(k) != v}
        if diff and not ALLOW_PROFILE_CHANGE:
            raise RuntimeError(f"This output folder is locked to another profile: {diff}. Results from different profiles must "
                               "not be compared. Use a new OUTPUT_DIR, or set ALLOW_PROFILE_CHANGE = True deliberately.")
        if locked.get("gpu") != GPU["name"]:
            print(f"NOTE: locked on {locked.get('gpu')}, now running on {GPU['name']} (same profile; fine).")
    else:
        save_json({**current, "benchmark": bench, "locked_at": datetime.datetime.now().isoformat()}, path)


def project_hours(ips, eval_ips, probe_s):
    n_train, n_all = len(ROWS["train"]), sum(len(v) for v in ROWS.values())
    eval_s = n_all / eval_ips + probe_s

    def ssl_hours(steps, n_rows=n_train, batch=None, ips_=None):
        plan = schedule_plan(n_rows, steps, batch or cfg.batch_size)
        rep_every, knn_every, _ = monitor_plan(plan["epochs"])
        train_s = plan["total_steps"] * plan["batch"] / (ips_ or ips)
        mon_s = (plan["epochs"] / rep_every) * len(MONITOR_ROWS) / eval_ips + \
                (plan["epochs"] / knn_every + 1) * (len(KNN_MONITOR_ROWS) + len(ROWS["val"])) / eval_ips
        return (train_s + mon_s + eval_s) / 3600, plan

    core_h, core_plan = ssl_hours(cfg.target_steps)
    pilot_h, pilot_plan = ssl_hours(cfg.pilot_steps)
    sup_imgs = sum(len(DRAWS[r][0]) for r in cfg.sup_regimes if r in DRAWS)
    sup_h = (cfg.sup_epochs * sup_imgs / ips + (cfg.sup_epochs / cfg.sup_eval_every) * len(ROWS["val"]) / eval_ips) / 3600
    rand_h = eval_s / 3600
    n_core = len(SEEDS_RUN) * len(ARMS) / NUM_SHARDS
    n_rand = len(SEEDS_RUN) * ("random_init" in BASELINES) / NUM_SHARDS
    n_sup = len(SEEDS_RUN) * ("supervised" in BASELINES) / NUM_SHARDS
    n_t2 = len(TIER2_SEEDS_RUN)
    size_h = [ssl_hours(cfg.target_steps, n_rows=max(1, int(n_train * f)))[0] for f in cfg.size_fracs]
    rows = [("Gate 0 pilot (ijepa seed 0)", 1, pilot_h, f"{pilot_plan['total_steps']} steps, {pilot_plan['epochs']} epochs"),
            ("core SSL runs (arms x seeds)", n_core, core_h, f"{core_plan['total_steps']} steps, {core_plan['epochs']} epochs"),
            ("random_init evaluations", n_rand, rand_h, "evaluation only"),
            ("supervised baseline (per seed)", n_sup, sup_h, f"{cfg.sup_epochs} epochs on {[r for r in cfg.sup_regimes if r in DRAWS]}"),
            ("Tier 2: attn arm", n_t2, core_h, "if ENABLE_ATTN_ARM"),
            ("Tier 2: tau sweep", n_t2 * len(cfg.tau_sweep), core_h, "if ENABLE_TAU_SWEEP"),
            ("Tier 2: size axis", n_t2 * 2 * len(cfg.size_fracs), float(np.mean(size_h)) if size_h else 0.0, "if ENABLE_SIZE_AXIS"),
            ("Tier 2: LeJEPA arms [UNMEASURED: x3 guess]", n_t2 * 4, 3 * core_h, "measured in section 15d before it runs")]
    table = pd.DataFrame(rows, columns=["item", "runs", "hours_each", "note"])
    table["hours_total"] = table["runs"] * table["hours_each"]
    return table, core_plan, pilot_plan, {"eval_s_per_run": eval_s, "probe_s_per_run": probe_s}


bench_path, locked_path = OUT / "stats" / "benchmark.json", OUT / "locked_profile.json"
if locked_path.exists() and not ALLOW_PROFILE_CHANGE:
    cfg.batch_size = load_json(locked_path)["batch_size"]   # reuse the batch size every earlier run used
if RUN_MODE in ("pilot", "full") and bench_path.exists() and locked_path.exists():
    BENCH = load_json(bench_path)
    bench, fl = BENCH["benchmark"], BENCH["flops"]
    eval_ips, probe_s = BENCH["eval_imgs_per_s"], BENCH["probe_s_per_run"]
    measured_at = BENCH.get("measured_at")
    print(f"Re-using the benchmark from {BENCH.get('measured_at')} (set RUN_MODE = 'benchmark' to re-measure).")
else:
    bench = benchmark()
    cfg.batch_size = bench["batch_size"]
    fl = flop_split()
    eval_ips = eval_speed()
    n_tr = len(ROWS["train"])
    t_full, t_10 = probe_seconds(n_tr), probe_seconds(max(K, n_tr // 10))
    t_small = probe_seconds(min(50 * K, n_tr))
    n_shot = sum(1 for r in REGIMES if r.endswith("shot"))
    probe_s = len(cfg.probe_C) * (t_full + cfg.n_draws * t_10 + cfg.n_draws * n_shot * t_small)
    measured_at = datetime.datetime.now().isoformat()
cfg.batch_size = bench["batch_size"]
PROJ, CORE_PLAN, PILOT_PLAN, EVAL_COST = project_hours(bench["imgs_per_s"], eval_ips, probe_s)
print(f"Batch {bench['batch_size']} | {bench['imgs_per_s']:.0f} img/s (short benchmark; long runs are usually faster) | "
      f"peak GPU memory {bench['peak_mem_gb']:.2f} GB of {GPU['vram_gb']} GB | evaluation forward {eval_ips:.0f} img/s")
print(f"FLOPs per training image: {fl['gflops_per_image']:.1f} G | predictor share {100 * fl['predictor_share']:.0f}%")
print(f"Core run: {CORE_PLAN['steps_per_epoch']} steps/epoch x {CORE_PLAN['epochs']} epochs = {CORE_PLAN['total_steps']} steps | "
      f"pilot: {PILOT_PLAN['epochs']} epochs = {PILOT_PLAN['total_steps']} steps | "
      f"evaluation ≈ {EVAL_COST['eval_s_per_run'] / 60:.1f} min per run (probe part is a rough estimate)")
display(PROJ.round(2))
_core_total = PROJ.iloc[1:4]["hours_total"].sum()
print(f"Projected: Gate 0 ≈ {PROJ.iloc[0]['hours_total']:.1f} h | core grid incl. baselines on this shard ≈ {_core_total:.1f} h | "
      f"Tier 2 (only the sections you enable) up to ≈ {PROJ.iloc[4:]['hours_total'].sum():.1f} h")
if _core_total > cfg.max_session_hours:
    print(f"The queue needs more than one session ({cfg.max_session_hours:.0f} h): it stops cleanly and resumes next session.")
BENCH = {"benchmark": bench, "flops": fl, "eval_imgs_per_s": eval_ips, "probe_s_per_run": probe_s, "gpu": GPU,
         "profile": cfg.profile, "measured_at": measured_at, "core_plan": CORE_PLAN,
         "pilot_plan": PILOT_PLAN, "projection": PROJ.to_dict(orient="records")}
save_json(BENCH, bench_path)
if RUN_MODE != "benchmark":
    lock_profile(bench)


# ======================= notebook cell 33 =======================
SESSION_STOPPED = False
STRICT = RUN_MODE == "smoke"   # in smoke mode any failure stops the notebook, so bugs cannot hide


def run_queue(items, title):
    """items: list of (run_name, callable). Finished items (eval.json present) are skipped, unfinished ones resume from
    their checkpoint, a failing item is logged to error.txt and the queue moves on, the time budget stops the session."""
    global SESSION_STOPPED
    if SESSION_STOPPED:
        print(f"{title}: not started (the session time budget was reached earlier)")
        return
    t0 = time.time()
    for i, (name, fn) in enumerate(items):
        run_dir = OUT / "runs" / name
        if (run_dir / "eval.json").exists():
            print(f"[{i + 1}/{len(items)}] {name}: done, skipping")
            continue
        print(f"[{i + 1}/{len(items)}] {name}: starting ({budget_str()})")
        try:
            fn()
        except TimeBudgetExceeded as e:
            print(f"Session time budget reached — {e}. Re-run the notebook (Kaggle: attach this version's output as "
                  "RESUME_FROM_DIR) to continue.")
            SESSION_STOPPED = True
            break
        except Exception as e:
            if STRICT:
                raise
            run_dir.mkdir(parents=True, exist_ok=True)
            save_text(traceback.format_exc(), run_dir / "error.txt")
            print(f"  FAILED: {type(e).__name__}: {e} (traceback in {run_dir / 'error.txt'}); continuing")
            free_memory()
    print(f"{title}: queue pass finished in {(time.time() - t0) / 60:.1f} min")


def endpoint_of_run(name, ev="probe", regimes=None):
    """Mean over regimes of the mean over draws (test macro-F1) for one finished run."""
    regimes = [r for r in (regimes or cfg.primary_regimes) if r in REGIMES]
    d = pd.DataFrame(load_json(OUT / "runs" / name / "eval.json")["results"])
    d = d[(d["eval"] == ev) & (d["regime"].isin(regimes))]
    return float(d.groupby("regime")["macro_f1"].mean().mean()) if len(d) else float("nan")


def gate0_evaluate():
    pdir, rdir = OUT / "runs" / "pilot_ijepa_s0", OUT / "runs" / "random_init_s0"
    if not ((pdir / "eval.json").exists() and (rdir / "eval.json").exists()):
        return None
    log = pd.read_csv(pdir / "train_log.csv")
    k = log.dropna(subset=["knn_val_f1"])
    rk = log.dropna(subset=["rankme"])
    ri = load_json(rdir / "eval.json")
    ri_knn = float(ri["online_knn_val_f1"])
    knn_first, knn_last, knn_max = float(k["knn_val_f1"].iloc[0]), float(k["knn_val_f1"].iloc[-1]), float(k["knn_val_f1"].max())
    rank_min, floor = float(rk["rankme"].min()), cfg.rankme_floor_frac * cfg.enc_dim
    gain_pp = 100 * (knn_last - ri_knn)
    pass_knn, pass_rank = gain_pp >= cfg.g0_knn_margin_pp, rank_min >= floor
    decision = "PASS" if pass_knn and pass_rank else "FAIL"
    out = {"pilot_steps": int(log["step"].iloc[-1]), "pilot_epochs": int(log["epoch"].iloc[-1]),
           "online_knn_val_f1_first": knn_first, "online_knn_val_f1_last": knn_last, "online_knn_val_f1_max": knn_max,
           "random_init_online_knn_val_f1": ri_knn, "knn_gain_over_random_init_pp": gain_pp,
           "required_gain_pp": cfg.g0_knn_margin_pp, "rankme_min": rank_min, "rankme_floor": floor,
           "rankme_last": float(rk["rankme"].iloc[-1]), "pass_knn": bool(pass_knn), "pass_rankme": bool(pass_rank),
           "probe_endpoint_pilot": endpoint_of_run("pilot_ijepa_s0"), "probe_endpoint_random_init": endpoint_of_run("random_init_s0"),
           "knn_endpoint_pilot": endpoint_of_run("pilot_ijepa_s0", "knn"), "knn_endpoint_random_init": endpoint_of_run("random_init_s0", "knn")}
    out["decision"] = decision
    if decision == "PASS":
        out["recommendation"] = ("Training produces usable features. Set RUN_MODE = 'full' to run the core grid "
                                 f"({cfg.target_steps} steps per run).")
    else:
        why = []
        if not pass_knn:
            why.append(f"online k-NN gained only {gain_pp:+.1f} pp over random_init (needs {cfg.g0_knn_margin_pp})")
        if not pass_rank:
            why.append(f"RankMe fell to {rank_min:.1f} (< {floor:.1f})")
        out["recommendation"] = ("Gate 0 FAILED: " + "; ".join(why) + ". Do not run the grid as is. Options: enable the Tier-2 "
                                 "SIGReg / LeJEPA arm (ENABLE_LEJEPA = True), and/or lower blr (OVERRIDES={'blr': 2.5e-4}) or "
                                 "raise TARGET_STEPS; then re-run the pilot in a new OUTPUT_DIR.")
    if RUN_MODE == "smoke":
        out["decision"] = "SMOKE TEST — pipeline check only, meaningless: " + decision
    save_json(out, OUT / "stats" / "gate_G0.json")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].plot(log["step"], log["loss"])
    axes[0].set_title("Pilot JEPA loss")
    axes[1].plot(k["step"], 100 * k["knn_val_f1"], marker="o", label="pilot ijepa")
    axes[1].axhline(100 * ri_knn, color="grey", ls="--", label="random_init")
    axes[1].axhline(100 * ri_knn + cfg.g0_knn_margin_pp, color="red", ls=":", label="pass line")
    axes[1].set_title("Online k-NN macro-F1 on validation (%)")
    axes[1].legend()
    axes[2].plot(rk["step"], rk["rankme"])
    axes[2].axhline(floor, color="red", ls=":", label=f"floor {floor:.1f}")
    axes[2].set_title("Effective rank (RankMe) of target features")
    axes[2].legend()
    for ax in axes:
        ax.set_xlabel("optimizer step")
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "figures" / "gate0_curves.png", dpi=150)
    plt.show()
    plt.close(fig)
    return out


G0 = None
if RUN_MODE in ("pilot", "smoke"):
    run_queue([("random_init_s0", lambda: run_random_init(0)),
               ("pilot_ijepa_s0", lambda: run_ssl_arm("ijepa", 0, steps=cfg.pilot_steps, family="pilot", label="pilot_ijepa",
                                                       run_name="pilot_ijepa_s0"))], "Gate 0 pilot")
    G0 = gate0_evaluate()
    if G0:
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in G0.items()}, indent=2))
        print(f"\nGATE 0: {G0['decision']}\n{G0['recommendation']}")
elif (OUT / "stats" / "gate_G0.json").exists():
    G0 = load_json(OUT / "stats" / "gate_G0.json")
    print(f"Gate 0 (from {OUT / 'stats' / 'gate_G0.json'}): {G0['decision']}")

GRID_ALLOWED = True
if RUN_MODE == "full":
    if G0 is None:
        print("WARNING: Gate 0 has not been run in this OUTPUT_DIR (RUN_MODE = 'pilot'). The grid runs anyway; "
              "check the training curves before trusting the results.")
    elif G0["decision"] != "PASS" and not IGNORE_GATE0:
        GRID_ALLOWED = False
        print("Gate 0 did not pass, so the grid is NOT started. " + G0.get("recommendation", "") +
              " (Set IGNORE_GATE0 = True to run it anyway.)")


# ======================= notebook cell 35 =======================
def core_items():
    """Seed by seed, so a partial session still leaves complete seeds; then sharded across machines."""
    items = []
    for seed in SEEDS_RUN:
        for a in ARMS:
            items.append((f"{a}_s{seed}", lambda a=a, seed=seed: run_ssl_arm(a, seed)))
        if "random_init" in BASELINES:
            items.append((f"random_init_s{seed}", lambda seed=seed: run_random_init(seed)))
        if "supervised" in BASELINES:
            items.append((f"supervised_s{seed}", lambda seed=seed: run_supervised(seed)))
    return [it for i, it in enumerate(items) if i % NUM_SHARDS == SHARD_ID]


def check_pairing(labels, seeds, strict=False, include_random_init=True):
    """Same seed => identical initial weights, identical image order and (numerically) identical first-batch crops."""
    problems, checked = [], 0
    for s in seeds:
        infos = {lab: load_json(OUT / "runs" / f"{lab}_s{s}" / "pairing.json") for lab in labels
                 if (OUT / "runs" / f"{lab}_s{s}" / "pairing.json").exists()}
        ri = OUT / "runs" / f"random_init_s{s}" / "eval.json"
        if include_random_init and ri.exists() and infos:
            infos["random_init"] = {"init_hash": load_json(ri).get("init_hash")}
        if len(infos) < 2:
            continue
        ref_lab, ref = next(iter(infos.items()))
        for lab, inf in infos.items():
            checked += 1
            if inf.get("init_hash") != ref.get("init_hash"):
                problems.append(f"seed {s}: initial weights of {lab} differ from {ref_lab}")
            if "first_batch_idx_hash" in inf and "first_batch_idx_hash" in ref:
                if inf["first_batch_idx_hash"] != ref["first_batch_idx_hash"]:
                    problems.append(f"seed {s}: image order of {lab} differs from {ref_lab}")
                if abs(inf["first_batch_mean"] - ref["first_batch_mean"]) > 1e-3 or abs(inf["first_batch_std"] - ref["first_batch_std"]) > 1e-3:
                    problems.append(f"seed {s}: first-batch crops of {lab} differ from {ref_lab}")
    if problems:
        msg = "PAIRING CHECK FAILED:\n  " + "\n  ".join(problems)
        if strict:
            raise AssertionError(msg)
        print("!" * 100 + "\n" + msg + "\n" + "!" * 100)
    else:
        print(f"Pairing check: {checked} runs compared — same seed => identical initial weights, image order and first-batch crops.")
    return problems


if RUN_MODE in ("smoke", "full") and GRID_ALLOWED:
    run_queue(core_items(), "Core grid")
    PAIRING_PROBLEMS = check_pairing(ARMS, SEEDS_RUN, strict=RUN_MODE == "smoke")
elif RUN_MODE in ("benchmark", "pilot"):
    print(f"RUN_MODE = '{RUN_MODE}': the core grid is not run. Set RUN_MODE = 'full' to run it.")


# ======================= notebook cell 37 =======================
TIER2_ON = {"attn": ENABLE_ATTN_ARM or RUN_MODE == "smoke", "tau": ENABLE_TAU_SWEEP or RUN_MODE == "smoke",
            "size": ENABLE_SIZE_AXIS or RUN_MODE == "smoke", "lejepa": ENABLE_LEJEPA or RUN_MODE == "smoke"}
RUN_TIER2 = RUN_MODE in ("smoke", "full") and GRID_ALLOWED


def tau_label(t):
    return f"veg_pos@tau={t:g}"


def frac_label(arm, f):
    return f"{arm}@frac={f:g}"


def train_subset(frac, seed=777):
    """Fixed, class-stratified subset of the training pool; identical for every arm and seed."""
    rng = np.random.default_rng(seed + int(round(frac * 1000)))
    keep = []
    for c in range(K):
        ix = np.where(Y["train"] == c)[0]
        if len(ix):
            keep.append(rng.choice(ix, max(1, int(round(len(ix) * frac))), replace=False))
    return ROWS["train"][np.sort(np.concatenate(keep))]


def shard(items):
    return [it for i, it in enumerate(items) if i % NUM_SHARDS == SHARD_ID]


SIZE_SUBSETS = {f: train_subset(f) for f in cfg.size_fracs} if TIER2_ON["size"] else {}
if SIZE_SUBSETS:
    save_json({f"{f:g}": {"images": int(len(r)), "tag": rows_tag(r)} for f, r in SIZE_SUBSETS.items()},
              OUT / "stats" / "size_axis_subsets.json")

if RUN_TIER2:
    if TIER2_ON["attn"]:
        print("15a. attn arm — AttMask-style stand-in: targets placed on tokens far from the image's mean target-encoder token "
              "(no CLS attention exists in this ViT).")
        run_queue(shard([(f"attn_s{s}", lambda s=s: run_ssl_arm("attn", s)) for s in TIER2_SEEDS_RUN if "attn" not in ARMS]),
                  "Tier 2 attn")
    if TIER2_ON["tau"]:
        print(f"15b. tau sweep for veg_pos: new runs at tau {list(cfg.tau_sweep)}; tau 0.5 = core veg_pos, tau inf = core ijepa.")
        run_queue(shard([(f"veg_pos_tau{t:g}_s{s}",
                          lambda s=s, t=t: run_ssl_arm("veg_pos", s, family="tau", label=tau_label(t), run_name=f"veg_pos_tau{t:g}_s{s}",
                                                       tau=t, variant=t))
                         for s in TIER2_SEEDS_RUN for t in cfg.tau_sweep if not math.isclose(t, cfg.veg_tau)]), "Tier 2 tau sweep")
    if TIER2_ON["size"]:
        print(f"15c. data-size axis at a fixed budget of {cfg.target_steps} steps: fractions {list(cfg.size_fracs)} "
              f"({ {f'{f:g}': len(r) for f, r in SIZE_SUBSETS.items()} } images); 100 % = core runs.")
        run_queue(shard([(f"{a}_frac{f:g}_s{s}",
                          lambda s=s, f=f, a=a: run_ssl_arm(a, s, family="size", label=frac_label(a, f), run_name=f"{a}_frac{f:g}_s{s}",
                                                            train_rows=SIZE_SUBSETS[f], variant=f))
                         for s in TIER2_SEEDS_RUN for f in cfg.size_fracs for a in ["ijepa", "veg_pos"]]), "Tier 2 size axis")
    if not any(TIER2_ON[k] for k in ("attn", "tau", "size")):
        print("Tier 2 attn / tau sweep / size axis: all switched off (see the Quick configuration).")
else:
    print("Tier 2 sections are not run in this mode.")


# ======================= notebook cell 39 =======================
LEJEPA_ARMS = ["ijepa", "veg_pos", "veg_neg", "pos_matched"]   # "ijepa" = uniform local-crop placement


def lejepa_label(arm):
    return f"lejepa_{'uniform' if arm == 'ijepa' else arm}"


class SIGReg(nn.Module):
    """[UNVERIFIED] Sketched isotropic-Gaussian regulariser as recalled from LeJEPA's minimal implementation: project the
    embeddings on `slices` random unit directions and compare each 1-D empirical characteristic function with the
    standard normal one (Epps-Pulley statistic, trapezoid quadrature on [0, tmax], Gaussian window), times N."""

    def __init__(self, knots, t_max, slices):
        super().__init__()
        t = torch.linspace(0, t_max, knots, dtype=torch.float32)
        dt = t_max / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.slices = slices
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj, gen):
        # proj: (views, N, D) in float32
        A = torch.randn(proj.size(-1), self.slices, device=proj.device, generator=gen)
        A = A / A.norm(p=2, dim=0, keepdim=True)
        x_t = (proj @ A).unsqueeze(-1) * self.t                                   # (V, N, M, T)
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()  # (V, M, T)
        return ((err @ self.weights) * proj.size(-2)).mean()


class LeJEPANet(nn.Module):
    """Same ViT-Tiny backbone (created first after seeding, so its initial weights equal the I-JEPA arms' for the same
    seed) plus an MLP projector with BatchNorm [UNVERIFIED projector shape]."""

    def __init__(self):
        super().__init__()
        self.backbone = make_encoder()
        d, h, p = cfg.enc_dim, cfg.lejepa_proj_hidden, cfg.lejepa_proj_dim
        self.proj = nn.Sequential(nn.Linear(d, h), nn.BatchNorm1d(h), nn.GELU(), nn.Linear(h, h), nn.BatchNorm1d(h), nn.GELU(),
                                  nn.Linear(h, p))

    def forward(self, x):
        emb = self.backbone(x).mean(1)
        return emb, self.proj(emb)


class LocalCropSampler:
    """Placement analogue for LeJEPA: the arms act on the positions of the local crops (uniform / greenness-biased /
    anti-greenness / position-matched), with the same score, softmax/eps/tau and prior machinery as the masks."""

    def __init__(self, arm, seed, prior=None, tau=None, eps=None):
        self.arm, self.placement = arm, PLACEMENTS[arm]
        self.g = cfg.img_size // cfg.patch_size
        self.tau = cfg.veg_tau if tau is None else tau
        self.eps = cfg.veg_eps if eps is None else eps
        self.prior_fn = PriorLookup(prior)
        self.cpu_gen = torch.Generator().manual_seed(seed * 37 + 5)
        self.dev_gen = torch.Generator(device=DEVICE)
        self.dev_gen.manual_seed(seed * 37 + 6)
        self.centre_hist = torch.zeros(self.g * self.g, dtype=torch.float64, device=DEVICE)
        self.prior_fallbacks = 0
        self.rows = torch.arange(self.g, device=DEVICE)

    def state(self):
        return {"cpu": self.cpu_gen.get_state(), "dev": self.dev_gen.get_state(), "centre_hist": self.centre_hist.cpu(),
                "prior_fallbacks": self.prior_fallbacks}

    def load(self, st):
        self.cpu_gen.set_state(st["cpu"])
        self.dev_gen.set_state(st["dev"])
        self.centre_hist = st["centre_hist"].to(DEVICE)
        self.prior_fallbacks = int(st["prior_fallbacks"])

    @torch.no_grad()
    def __call__(self, base01):
        b, P, g, L = base01.shape[0], cfg.patch_size, self.g, cfg.lejepa_local_size
        exg = veg_patch_scores(base01, P, cfg.veg_index)
        e = exg.flatten(1).float()
        ez = (e - e.mean(1, keepdim=True)) / (e.std(1, keepdim=True, unbiased=False) + 1e-6)
        crops, ex, exz = [], torch.zeros((), device=DEVICE), torch.zeros((), device=DEVICE)
        for _ in range(cfg.lejepa_n_local):
            h, w = block_size(cfg.lejepa_local_scale, LOCAL_ASPECT, g, g, self.cpu_gen)
            r, c, fb = sample_positions(self.placement, b, h, w, g, g, exg, self.tau, self.eps, self.prior_fn, self.dev_gen)
            self.prior_fallbacks += b if fb else 0
            boxes = torch.stack([torch.arange(b, device=DEVICE), c * P, r * P, (c + w) * P, (r + h) * P], 1).to(base01.dtype)
            out = roi_align(base01, boxes, (L, L), spatial_scale=1.0, sampling_ratio=2, aligned=True)
            if cfg.hflip:
                flip = torch.rand(b, device=DEVICE, generator=self.dev_gen) < 0.5
                out = torch.where(flip.view(b, 1, 1, 1), out.flip(3), out)
            crops.append(out)
            mr = (self.rows[None] >= r[:, None]) & (self.rows[None] < (r + h)[:, None])
            mc = (self.rows[None] >= c[:, None]) & (self.rows[None] < (c + w)[:, None])
            m = (mr[:, :, None] & mc[:, None, :]).flatten(1).float()
            ex = ex + ((e * m).sum(1) / m.sum(1).clamp(min=1)).mean()
            exz = exz + ((ez * m).sum(1) / m.sum(1).clamp(min=1)).mean()
            self.centre_hist += torch.bincount((r + h // 2) * g + (c + w // 2), minlength=g * g).double()
        n = max(1, cfg.lejepa_n_local)
        return crops, {"local_exg": ex / n, "local_exg_z": exz / n, "all_exg": e.mean()}


class LeJEPARun(JEPARun):
    """[UNVERIFIED] LeJEPA-style training: no teacher, no stop-gradient; loss = (1 - lambda) * invariance (every view's
    projection to the mean projection of the global views) + lambda * SIGReg over all views."""
    kind = "lejepa"

    def _prior(self):
        return LEJEPA_PRIOR, LEJEPA_PRIOR_HASH

    def _make_loader(self, seed):
        return GPULoader(CACHE, self.train_rows, cfg.lejepa_batch_size, train=True, out_size=None, seed=seed, augment=False)

    def _build_models(self):
        self.net = LeJEPANet()

    def _init_hash(self):
        return state_hash(self.net.backbone)

    def _to_device(self):
        self.net.to(DEVICE)
        self.peak_lr = cfg.lejepa_blr * self.loader.bs / 256
        self.opt = torch.optim.AdamW(param_groups([self.net], wd=cfg.lejepa_wd), lr=self.peak_lr)
        self.scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda" and cfg.precision == "fp16")
        self.sampler = LocalCropSampler(self.arm, self.seed, prior=self.prior, tau=self.tau, eps=self.eps)
        self.sigreg = SIGReg(cfg.lejepa_knots, cfg.lejepa_tmax, cfg.lejepa_slices).to(DEVICE)
        self.view_gen = torch.Generator(device=DEVICE)
        self.view_gen.manual_seed(self.seed * 7919 + 11)
        self.params = [p for p in self.net.parameters() if p.requires_grad]

    def encoder_for_eval(self):
        return self.net.backbone

    def set_train_mode(self):
        self.net.train()

    def centre_hist(self):
        return self.sampler.centre_hist.cpu().numpy()

    def prior_fallbacks(self):
        return self.sampler.prior_fallbacks

    def _models_state(self):
        return {"net": self.net.state_dict(), "sampler": self.sampler.state(), "view_gen": self.view_gen.get_state()}

    def _load_models(self, st):
        self.net.load_state_dict(st["net"])
        self.sampler.load(st["sampler"])
        self.view_gen.set_state(st["view_gen"])

    def train_step(self, x, idx=None):
        lr = lr_at(self.step, self.total, self.peak_lr)
        for g in self.opt.param_groups:
            g["lr"] = lr
        b = x.shape[0]
        base = x if x.shape[-1] == cfg.img_size else F.interpolate(x, size=(cfg.img_size, cfg.img_size), mode="bilinear",
                                                                   antialias=True, align_corners=False)
        glob = [random_resized_crop(x, cfg.img_size, cfg.lejepa_global_scale, self.view_gen, cfg.hflip)
                for _ in range(cfg.lejepa_n_global)]
        loc, st = self.sampler(base)
        with autocast_ctx():
            _, pg = self.net(normalize(torch.cat(glob)))
            projs = [pg.float().view(cfg.lejepa_n_global, b, -1)]
            if loc:
                _, pl = self.net(normalize(torch.cat(loc)))
                projs.append(pl.float().view(len(loc), b, -1))
        allp = torch.cat(projs, 0)
        inv = (allp - projs[0].mean(0, keepdim=True)).square().mean()
        gen = torch.Generator(device=DEVICE)
        gen.manual_seed(int(self.seed * 1_000_003 + self.step))
        sig = self.sigreg(allp, gen)
        loss = (1 - cfg.lejepa_lambda) * inv + cfg.lejepa_lambda * sig
        st.update(inv_loss=inv.detach(), sigreg_loss=sig.detach())
        if not torch.isfinite(loss):
            self.opt.zero_grad(set_to_none=True)
            return None, st, (lr, cfg.lejepa_wd, float("nan"))
        self.scaler.scale(loss).backward()
        if cfg.grad_clip:
            self.scaler.unscale_(self.opt)
            torch.nn.utils.clip_grad_norm_(self.params, cfg.grad_clip)
        self.scaler.step(self.opt)
        self.scaler.update()
        self.opt.zero_grad(set_to_none=True)
        self.step += 1
        return loss.detach(), st, (lr, cfg.lejepa_wd, float("nan"))


def benchmark_lejepa(steps=6, warmup=2):
    bs = cfg.lejepa_batch_size
    while True:
        try:
            cfg.lejepa_batch_size = bs
            run = LeJEPARun("veg_pos", 0, OUT / "benchmark_lejepa_tmp", cfg.lejepa_steps, family="benchmark")
            it = iter(run.loader)
            n_steps = min(len(run.loader), steps + warmup)
            wu = min(warmup, max(0, n_steps - 1))
            for i in range(n_steps):
                if i == wu:
                    if HAS_CUDA:
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                    t0 = time.time()
                x, idx = next(it)
                run.train_step(x, idx)
            if HAS_CUDA:
                torch.cuda.synchronize()
            ips = max(1, n_steps - wu) * run.loader.bs / (time.time() - t0)
            peak = torch.cuda.max_memory_allocated() / 1e9 if HAS_CUDA else 0.0
            del run, it
            free_memory()
            break
        except torch.cuda.OutOfMemoryError:
            free_memory()
            if bs <= 8:
                raise
            print(f"  LeJEPA: out of memory at batch {bs}; trying {bs // 2}")
            bs //= 2
    shutil.rmtree(OUT / "benchmark_lejepa_tmp", ignore_errors=True)
    return {"batch_size": bs, "imgs_per_s": ips, "peak_mem_gb": peak}


# ======================= notebook cell 40 =======================
LEJEPA_PRIOR, LEJEPA_PRIOR_HASH = None, None
if RUN_TIER2 and TIER2_ON["lejepa"]:
    print("[UNVERIFIED] LeJEPA / SIGReg settings in use — check them against the paper (arXiv 2511.08544) and the official "
          "rbalestr-lab/lejepa code before reporting:")
    print({k: getattr(cfg, k) for k in LEJEPA_KEYS})
    if "pos_matched" in LEJEPA_ARMS:
        LEJEPA_PRIOR, LEJEPA_PRIOR_HASH = estimate_position_prior("lejepa_local")
    lb_path = OUT / "stats" / "benchmark_lejepa.json"
    if lb_path.exists() and RUN_MODE == "full":
        LB = load_json(lb_path)
        cfg.lejepa_batch_size = LB["benchmark"]["batch_size"]
        print(f"Re-using the LeJEPA benchmark from {LB['measured_at']}: batch {cfg.lejepa_batch_size}")
    else:
        lbench = benchmark_lejepa()
        cfg.lejepa_batch_size = lbench["batch_size"]
        lplan = schedule_plan(len(ROWS["train"]), cfg.lejepa_steps, cfg.lejepa_batch_size)
        per_run_h = lplan["total_steps"] * lplan["batch"] / lbench["imgs_per_s"] / 3600 + EVAL_COST["eval_s_per_run"] / 3600
        LB = {"benchmark": lbench, "plan": lplan, "hours_per_run": per_run_h,
              "relative_cost_vs_ijepa_per_image": bench["imgs_per_s"] / lbench["imgs_per_s"],
              "measured_at": datetime.datetime.now().isoformat()}
        save_json(LB, lb_path)
    n_runs = len(TIER2_SEEDS_RUN) * len(LEJEPA_ARMS)
    print(f"LeJEPA: batch {LB['benchmark']['batch_size']} | {LB['benchmark']['imgs_per_s']:.0f} img/s "
          f"({LB['relative_cost_vs_ijepa_per_image']:.1f}x the I-JEPA cost per image) | peak {LB['benchmark']['peak_mem_gb']:.2f} GB | "
          f"≈{LB['hours_per_run']:.2f} h per run, ≈{n_runs * LB['hours_per_run']:.1f} h for {n_runs} runs")
    run_queue(shard([(f"{lejepa_label(a)}_s{s}",
                      lambda s=s, a=a: run_ssl_arm(a, s, steps=cfg.lejepa_steps, family="lejepa", label=lejepa_label(a),
                                                   run_cls=LeJEPARun))
                     for s in TIER2_SEEDS_RUN for a in LEJEPA_ARMS]), "Tier 2 LeJEPA")
    PAIR_LJ = check_pairing([lejepa_label(a) for a in LEJEPA_ARMS], TIER2_SEEDS_RUN, strict=RUN_MODE == "smoke",
                            include_random_init=False)
else:
    print("Tier 2 SIGReg / LeJEPA: not run " + ("(ENABLE_LEJEPA = False)." if not TIER2_ON["lejepa"] else f"in RUN_MODE = '{RUN_MODE}'."))


# ======================= notebook cell 41 =======================
if "dinov2" in BASELINES and RUN_MODE == "full" and GRID_ALLOWED:
    print("15e. Frozen DINOv2 ViT-S/14 reference (needs internet; seed 0 only; evaluated at 224 px).")
    run_queue(shard([("dinov2_s0", lambda: run_dinov2(0))]), "DINOv2 reference")
else:
    print("DINOv2 reference: not run (add \"dinov2\" to BASELINES in full mode; it needs internet access).")


# ---- headless hook: shard workers and the pilot stop here; the final report pass runs everything below ----
if SKIP_REPORT:
    print("CJ_SKIP_REPORT=1: queue pass finished; reporting cells are skipped in this process.")
    import sys as _sys
    _sys.exit(0)

# ======================= notebook cell 43 =======================
def load_results():
    rows, preds, info = [], {}, {}
    for f in sorted((OUT / "runs").glob("*/eval.json")):
        ev = load_json(f)
        label, fam, seed = ev.get("label", ev["arm"]), ev.get("family", "core"), int(ev["seed"])
        for r in ev["results"]:
            rows.append({"label": label, "arm": ev["arm"], "family": fam, "variant": ev.get("variant"), "seed": seed, **r})
        pz = np.load(f.parent / "preds.npz")
        for key in pz.files:
            e, regime, d = key.split("/")
            preds[(label, e, regime, seed, int(d))] = pz[key].astype(np.int64)
        info[(label, seed)] = {"dir": f.parent, "family": fam, "per_class_f1": ev.get("per_class_f1", {}),
                               "convergence_warnings": ev.get("probe_convergence_warnings", 0)}
    return pd.DataFrame(rows), preds, info


def seed_level(res, ev):
    """Unit of analysis = seed: average the label draws within each (label, regime, seed)."""
    d = res[res["eval"] == ev]
    return d.groupby(["label", "family", "regime", "seed"], as_index=False)["macro_f1"].mean()


def endpoint_table(res, ev, regimes):
    s = seed_level(res, ev)
    s = s[s["regime"].isin(regimes)]
    n_reg = s.groupby(["label", "seed"])["regime"].nunique()
    e = s.groupby(["label", "family", "seed"], as_index=False)["macro_f1"].mean()
    e["complete"] = [n_reg.get((l, sd), 0) == len(regimes) for l, sd in zip(e["label"], e["seed"])]
    return e[e["complete"]].drop(columns="complete")


def fmt_cell(m, s, n):
    return f"{100 * m:.1f} ± {100 * (s if s == s else 0):.1f} (n={int(n)})"


def summary_table(res, ev, labels, regimes):
    s = seed_level(res, ev)
    s = s[s["label"].isin(labels)]
    if s.empty:
        return pd.DataFrame()
    agg = s.groupby(["label", "regime"])["macro_f1"].agg(["mean", "std", "count"]).reset_index()
    agg["cell"] = [fmt_cell(m, sd, c) for m, sd, c in zip(agg["mean"], agg["std"], agg["count"])]
    t = agg.pivot(index="label", columns="regime", values="cell")
    t = t.reindex(index=[l for l in labels if l in t.index], columns=[r for r in regimes if r in t.columns])
    e = endpoint_table(res, ev, PRIMARY_REGIMES)
    e = e[e["label"].isin(labels)].groupby("label")["macro_f1"].agg(["mean", "std", "count"])
    t["PRIMARY ENDPOINT"] = [fmt_cell(*e.loc[l]) if l in e.index else "" for l in t.index]
    return t


RES, PREDS, RUNINFO = load_results()
Y_TEST = np.load(OUT / "stats" / "y_test.npy")
PRIMARY_REGIMES = [r for r in cfg.primary_regimes if r in REGIMES]
present = set(RES["label"]) if len(RES) else set()
CORE_LABELS = [a for a in ARMS + ["attn"] if a in present]
BASE_LABELS = [b for b in ["random_init", "dinov2"] if b in present]
TAU_LABELS = [tau_label(t) for t in cfg.tau_sweep if tau_label(t) in present]
SIZE_LABELS = [frac_label(a, f) for f in cfg.size_fracs for a in ["ijepa", "veg_pos"] if frac_label(a, f) in present]
LJ_LABELS = [lejepa_label(a) for a in LEJEPA_ARMS if lejepa_label(a) in present]
if len(RES):
    print(f"Finished runs: {RES.groupby('label')['seed'].nunique().to_dict()}")
    print(f"Primary endpoint = mean over {PRIMARY_REGIMES} of the mean over label draws; cells: mean ± sd over SEEDS (n = seeds).")
    for ev in ["probe", "knn"]:
        t = summary_table(RES, ev, CORE_LABELS + BASE_LABELS, REGIMES)
        if len(t):
            print(f"\nTest macro-F1 (%), {'linear probe' if ev == 'probe' else 'k-NN'} — core arms and baselines")
            display(t)
    sup = RES[RES["eval"] == "supervised"]
    if len(sup):
        print("\nSupervised ViT-Tiny from scratch (draw 0) — UPPER REFERENCE, NOT MATCHED (all weights trained with labels, "
              "augmentation, epoch chosen on validation):")
        display(sup.groupby("regime").agg(macro_f1_mean=("macro_f1", "mean"), macro_f1_sd=("macro_f1", "std"),
                                          seeds=("seed", "nunique"), best_val_epoch_mean=("best_val_epoch", "mean"),
                                          val_macro_f1_mean=("val_macro_f1", "mean")).mul([100, 100, 1, 1, 100]).round(1))
    for title, labs in [("Gate-0 pilot", ["pilot_ijepa"]), ("Tier 2 tau sweep", ["ijepa", "veg_pos"] + TAU_LABELS),
                        ("Tier 2 data-size axis", ["ijepa", "veg_pos"] + SIZE_LABELS), ("Tier 2 LeJEPA", LJ_LABELS)]:
        if any(l in present for l in labs) and (title == "Gate-0 pilot" or len(labs) > 2 or title == "Tier 2 LeJEPA"):
            t = summary_table(RES, "probe", [l for l in labs if l in present], REGIMES)
            if len(t) and (title != "Tier 2 tau sweep" or TAU_LABELS) and (title != "Tier 2 data-size axis" or SIZE_LABELS):
                print(f"\n{title} — linear probe macro-F1 (%)")
                display(t)
    allsum = []
    for ev in ["probe", "knn"]:
        s = seed_level(RES, ev).groupby(["label", "family", "regime"])["macro_f1"].agg(["mean", "std", "count"]).reset_index()
        s.insert(0, "eval", ev)
        allsum.append(s)
    save_csv(pd.concat(allsum), OUT / "stats" / "summary_macro_f1.csv")
    save_csv(RES, OUT / "stats" / "all_results_long.csv")
    n_conv = sum(v["convergence_warnings"] for v in RUNINFO.values())
    print(f"\nLogistic-regression convergence warnings across all runs: {n_conv} (max_iter = {cfg.probe_max_iter})")
else:
    print("No finished runs yet.")

logs = []
for f in sorted((OUT / "runs").glob("*/train_log.csv")):
    conf_p = f.parent / "config.json"
    if not conf_p.exists():
        continue
    conf = load_json(conf_p)
    d = pd.read_csv(f)
    d["label"], d["arm"], d["seed"], d["family"] = conf["label"], conf["arm"], conf["seed"], conf["family"]
    logs.append(d)
LOGS = pd.concat(logs, ignore_index=True) if logs else pd.DataFrame()
if len(LOGS) and "ctx_tokens" in LOGS.columns:
    core_logs = LOGS[LOGS["label"].isin(CORE_LABELS)]
    if len(core_logs):
        tok = core_logs.groupby("label")[["ctx_tokens", "tgt_tokens"]].mean().reindex(CORE_LABELS)
        tok["ctx_tokens_min_epoch"] = core_logs.groupby("label")["ctx_tokens"].min()
        print("\nMean tokens per image (context differs across arms when targets overlap more or less; CTX_TOKENS_CAP equalises it):")
        display(tok.round(1))
        save_csv(tok.reset_index(), OUT / "stats" / "token_budget.csv")
        man = core_logs.groupby("label")[["tgt_exg", "tgt_exg_z", "ctx_exg", "all_exg"]].mean().reindex(CORE_LABELS).reset_index()
        man.insert(0, "source", "training logs (mean over epochs and seeds)")
        man = man.rename(columns={"label": "arm"})
        both = pd.concat([man, ACCEPT], ignore_index=True)
        print("Manipulation check — mean ExG under the target blocks (z = relative to the image's own patches):")
        display(both.round(4))
        save_csv(both, OUT / "stats" / "manipulation_check.csv")


# ======================= notebook cell 45 =======================
PIDX = {}
for (_lab, _e, _r, _s, _d) in PREDS:
    PIDX.setdefault((_lab, _e, _r), {}).setdefault(_s, set()).add(_d)


def wilcoxon_exact(d, alternative="two-sided"):
    """Exact signed-rank test by enumerating all sign patterns of the ranks (zero differences dropped, tied ranks
    averaged). Version-independent (scipy's exact branch differs between releases)."""
    d = np.asarray(d, float)
    d = d[np.abs(d) > 1e-12]
    n = len(d)
    if n == 0:
        return 1.0, 0.0
    r = rankdata(np.abs(d))
    w = float(r[d > 0].sum())
    if n <= 16:
        signs = ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1).astype(float)
        dist = signs @ r
        tol = 1e-9
        if alternative == "greater":
            p = np.mean(dist >= w - tol)
        elif alternative == "less":
            p = np.mean(dist <= w + tol)
        else:
            c = r.sum() / 2
            p = np.mean(np.abs(dist - c) >= abs(w - c) - tol)
    else:
        from scipy.stats import norm
        mu, sd = n * (n + 1) / 4, math.sqrt((r ** 2).sum() / 4)
        zv = (w - mu) / sd
        p = norm.sf(zv) if alternative == "greater" else norm.cdf(zv) if alternative == "less" else 2 * norm.sf(abs(zv))
    return float(min(1.0, p)), w


def signflip_p(d, alternative="two-sided"):
    """Exact paired sign-flip permutation test on the mean difference."""
    d = np.asarray(d, float)
    n = len(d)
    if n == 0 or n > 16:
        return float("nan")
    signs = 1 - 2 * ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1)
    means = (signs * d).mean(1)
    obs, tol = d.mean(), 1e-12
    if alternative == "greater":
        return float(np.mean(means >= obs - tol))
    if alternative == "less":
        return float(np.mean(means <= obs + tol))
    return float(np.mean(np.abs(means) >= abs(obs) - tol))


def macro_f1_batch(y, P):
    """Macro-F1 of many prediction vectors (rows of P) against y, in one bincount."""
    M = P.shape[0]
    idx = (np.arange(M, dtype=np.int64)[:, None] * (K * K) + y[None, :].astype(np.int64) * K + P.astype(np.int64)).ravel()
    cm = np.bincount(idx, minlength=M * K * K).reshape(M, K, K)
    tp = np.diagonal(cm, axis1=1, axis2=2).astype(float)
    denom = cm.sum(2) + cm.sum(1)
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    present = np.bincount(y, minlength=K) > 0
    return f1[:, present].mean(1)


def collect(labels, ev, regimes, only_seeds=None):
    """Stack the test predictions of `labels` for the seeds and label draws they all share."""
    seeds = None
    for lab in labels:
        for r in regimes:
            ss = set(PIDX.get((lab, ev, r), {}))
            seeds = ss if seeds is None else seeds & ss
    if only_seeds is not None and seeds is not None:
        seeds &= set(only_seeds)
    seeds = sorted(seeds or [])
    if not seeds:
        return None
    draws = {}
    for r in regimes:
        common = None
        for lab in labels:
            for s in seeds:
                ds = PIDX[(lab, ev, r)][s]
                common = set(ds) if common is None else common & ds
        if not common:
            return None
        draws[r] = sorted(common)
    dmax = max(len(v) for v in draws.values())
    index = np.full((len(labels), len(regimes), len(seeds), dmax), -1, dtype=np.int64)
    arr = []
    for li, lab in enumerate(labels):
        for ri, r in enumerate(regimes):
            for si, s in enumerate(seeds):
                for di, d in enumerate(draws[r]):
                    index[li, ri, si, di] = len(arr)
                    arr.append(PREDS[(lab, ev, r, s, d)])
    return {"labels": list(labels), "ev": ev, "regimes": list(regimes), "seeds": seeds,
            "nD": np.array([len(draws[r]) for r in regimes]), "index": index, "P": np.stack(arr).astype(np.int16)}


def analyse_group(labels, ev, regimes, B, seed=0, verbose=True, only_seeds=None):
    """Point estimates and hierarchical bootstrap (seeds -> label draws within seed -> test images shared by all arms,
    regimes and draws) for every label of a group, per regime and for the primary endpoint."""
    labels = [l for l in labels if any(k[0] == l for k in PIDX)]
    if len(labels) < 2:
        return None
    C = collect(labels, ev, regimes, only_seeds)
    if C is None:
        return None
    idx, valid = C["index"], C["index"] >= 0
    safe = np.where(valid, idx, 0)
    L, R, S, _ = idx.shape
    prim = [i for i, r in enumerate(regimes) if r in PRIMARY_REGIMES]

    def per_seed(f1):
        Fv = np.where(valid, f1[safe], np.nan)
        return np.nanmean(Fv, axis=3)   # (L, R, S)

    seed_reg = per_seed(macro_f1_batch(Y_TEST, C["P"]))
    seed_end = seed_reg[:, prim, :].mean(1) if prim else None
    rng = np.random.default_rng(seed)
    n = len(Y_TEST)
    boot_reg = np.empty((B, L, R))
    t0 = time.time()
    for b in range(B):
        ti = rng.integers(0, n, n)
        f1 = macro_f1_batch(Y_TEST[ti], C["P"][:, ti])
        Fv = np.where(valid, f1[safe], np.nan)
        ss = rng.integers(0, S, S)
        for ri in range(R):
            nd = C["nD"][ri]
            ds = rng.integers(0, nd, (S, nd))
            boot_reg[b, :, ri] = Fv[:, ri, ss[:, None], ds].mean(axis=(1, 2))
    if verbose:
        print(f"  bootstrap {ev}: {L} arms x {R} regimes x {S} seeds, B = {B}, {time.time() - t0:.0f}s")
    return {**C, "seed_reg": seed_reg, "seed_end": seed_end, "boot_reg": boot_reg,
            "boot_end": boot_reg[:, :, prim].mean(2) if prim else None}


def compare(G, a, b, regime=None):
    if G is None or a not in G["labels"] or b not in G["labels"]:
        return None
    ia, ib = G["labels"].index(a), G["labels"].index(b)
    if regime is None:
        if G["seed_end"] is None:
            return None
        vs, vb = G["seed_end"], G["boot_end"]
    else:
        if regime not in G["regimes"]:
            return None
        ri = G["regimes"].index(regime)
        vs, vb = G["seed_reg"][:, ri, :], G["boot_reg"][:, :, ri]
    d_seed, d_boot = vs[ia] - vs[ib], vb[:, ia] - vb[:, ib]
    q = 100 * (1 - cfg.equiv_ci) / 2
    lo95, hi95 = np.percentile(d_boot, [2.5, 97.5])
    lo90, hi90 = np.percentile(d_boot, [q, 100 - q])
    pg, w = wilcoxon_exact(d_seed, "greater")
    pl, _ = wilcoxon_exact(d_seed, "less")
    p2, _ = wilcoxon_exact(d_seed, "two-sided")
    return {"eval": G["ev"], "comparison": f"{a} - {b}", "a": a, "b": b, "regime": regime or "PRIMARY ENDPOINT",
            "diff_pp": 100 * float(d_seed.mean()), "ci95_lo_pp": 100 * lo95, "ci95_hi_pp": 100 * hi95,
            f"ci{int(100 * cfg.equiv_ci)}_lo_pp": 100 * lo90, f"ci{int(100 * cfg.equiv_ci)}_hi_pp": 100 * hi90,
            "n_seeds": len(d_seed), "seeds_a_better": int((d_seed > 0).sum()), "W_plus": w,
            "wilcoxon_p_greater": pg, "wilcoxon_p_less": pl, "wilcoxon_p_two_sided": p2,
            "perm_p_greater": signflip_p(d_seed, "greater"), "perm_p_two_sided": signflip_p(d_seed, "two-sided"),
            "seed_diffs_pp": [round(100 * float(x), 3) for x in d_seed]}


CI_E = f"ci{int(100 * cfg.equiv_ci)}"


def flags(row, holm_p=None):
    m = cfg.equiv_margin_pp
    out = {"equivalent": bool(row[f"{CI_E}_lo_pp"] > -m and row[f"{CI_E}_hi_pp"] < m),
           "lower": bool(row["ci95_hi_pp"] < 0), "ci_excludes_0": bool(row["ci95_lo_pp"] > 0 or row["ci95_hi_pp"] < 0)}
    out["greater_confirmed"] = bool(holm_p is not None and holm_p < cfg.alpha and row["ci95_lo_pp"] > 0)
    if holm_p is None:   # exploratory: descriptive only
        s = "CI above 0" if row["ci95_lo_pp"] > 0 else "CI below 0" if row["ci95_hi_pp"] < 0 else "CI includes 0"
        out["status"] = s + (", within margin" if out["equivalent"] else "")
    else:
        out["status"] = ("greater (confirmed)" if out["greater_confirmed"] else "equivalent" if out["equivalent"]
                         else "lower" if out["lower"] else "inconclusive")
    return out


# self-check of the exact test against scipy (no ties, no zeros)
try:
    from scipy.stats import wilcoxon as _scipy_wilcoxon
    _d = np.array([0.8, -0.3, 1.2, 0.5, 2.0, -0.1])
    for _alt in ["greater", "two-sided"]:
        _mine, _ref = wilcoxon_exact(_d, _alt)[0], _scipy_wilcoxon(_d, alternative=_alt, method="exact").pvalue
        assert abs(_mine - _ref) < 1e-9, (_alt, _mine, _ref)
    print("Exact Wilcoxon implementation agrees with scipy on a tie-free example.")
except AssertionError as e:
    raise RuntimeError(f"Exact Wilcoxon self-check failed: {e}")
except Exception as e:
    print(f"(scipy cross-check of the exact Wilcoxon skipped: {e})")

PRIMARY_ARMS = [a for a in ["ijepa", "veg_pos", "veg_neg", "pos_matched"] if a in ARMS]
GROUPS, comp_rows = {}, []
if len(RES) and {"ijepa", "veg_pos"} <= set(RES["label"]):
    print(f"Hierarchical bootstrap (B = {cfg.bootstrap_B}) for the core arms + random_init:")
    for ev in ["probe", "knn"]:
        GROUPS[ev] = analyse_group(PRIMARY_ARMS + ["random_init"], ev, REGIMES, cfg.bootstrap_B, seed=11 if ev == "probe" else 12)
    if "attn" in CORE_LABELS:
        GROUPS["attn_probe"] = analyse_group(["attn", "ijepa"], "probe", PRIMARY_REGIMES, cfg.bootstrap_B, seed=13)
        GROUPS["attn_knn"] = analyse_group(["attn", "ijepa"], "knn", PRIMARY_REGIMES, cfg.bootstrap_B, seed=14)

    # ---- primary, pre-registered family ----
    prim = [compare(GROUPS["probe"], a, b) for a, b in PRIMARY_PAIRS]
    prim = [r for r in prim if r is not None]
    alt_col = "wilcoxon_p_greater" if cfg.primary_alternative == "greater" else "wilcoxon_p_two_sided"
    hp = holm([r[alt_col] for r in prim])
    for r, h in zip(prim, hp):
        r.update(family="PRIMARY (pre-registered, Holm)", holm_p=float(h), test_used=alt_col, **flags(r, float(h)))
    if len(prim) < len(PRIMARY_PAIRS):
        print(f"WARNING: only {len(prim)} of the {len(PRIMARY_PAIRS)} pre-registered comparisons could be computed "
              "(missing arms or seeds); Holm is applied to the available ones.")
    comp_rows += prim

    # ---- secondary / exploratory ----
    for ev in ["probe", "knn"]:
        G = GROUPS.get(ev)
        pairs = (PRIMARY_PAIRS if ev == "knn" else []) + [p for p in SECONDARY_PAIRS if p[0] != "attn"] + \
                [(a, "random_init") for a in PRIMARY_ARMS]
        for a, b in pairs:
            r = compare(G, a, b)
            if r:
                r.update(family="exploratory", holm_p=float("nan"), test_used="", **flags(r))
                comp_rows.append(r)
        for regime in REGIMES:
            for a, b in PRIMARY_PAIRS + [p for p in SECONDARY_PAIRS if p[0] != "attn"]:
                r = compare(G, a, b, regime)
                if r:
                    r.update(family="exploratory (per regime)", holm_p=float("nan"), test_used="", **flags(r))
                    comp_rows.append(r)
        Ga = GROUPS.get(f"attn_{ev}")
        r = compare(Ga, "attn", "ijepa")
        if r:
            r.update(family="exploratory (Tier 2 attn)", holm_p=float("nan"), test_used="", **flags(r))
            comp_rows.append(r)

COMP = pd.DataFrame(comp_rows)
INTERP = {"row": "not enough results", "text": "The primary comparisons need finished ijepa, veg_pos, veg_neg and pos_matched runs."}
if len(COMP):
    save_csv(COMP.drop(columns=["seed_diffs_pp"]).assign(seed_diffs_pp=[json.dumps(x) for x in COMP["seed_diffs_pp"]]),
             OUT / "stats" / "paired_comparisons.csv")
    eq = COMP[COMP["regime"] == "PRIMARY ENDPOINT"][["eval", "comparison", "family", "diff_pp", f"{CI_E}_lo_pp", f"{CI_E}_hi_pp",
                                                     "ci95_lo_pp", "ci95_hi_pp", "equivalent"]].copy()
    eq["margin_pp"] = cfg.equiv_margin_pp
    eq["ci95_inside_margin"] = (eq["ci95_lo_pp"] > -cfg.equiv_margin_pp) & (eq["ci95_hi_pp"] < cfg.equiv_margin_pp)
    save_csv(eq, OUT / "stats" / "equivalence.csv")
    show = ["comparison", "diff_pp", "ci95_lo_pp", "ci95_hi_pp", f"{CI_E}_lo_pp", f"{CI_E}_hi_pp", "n_seeds", "seeds_a_better",
            "wilcoxon_p_greater", "wilcoxon_p_two_sided", "holm_p", "perm_p_two_sided", "status"]
    P1 = COMP[COMP["family"].str.startswith("PRIMARY")]
    print(f"\nPRIMARY ANALYSIS (pre-registered): linear-probe macro-F1 averaged over {PRIMARY_REGIMES}, unit = seed, "
          f"Holm over {len(P1)} comparisons, test = exact Wilcoxon ({cfg.primary_alternative}), alpha = {cfg.alpha}, "
          f"equivalence margin ±{cfg.equiv_margin_pp} pp ({int(100 * cfg.equiv_ci)}% CI)")
    display(P1[show].round(4))
    print("\nExploratory, primary endpoint (no claims):")
    display(COMP[(COMP["family"] == "exploratory") | (COMP["family"] == "exploratory (Tier 2 attn)")][["eval"] + show[:-3] + ["status"]].round(3))
    n_int = len(COMP)
    print(f"\n{n_int} confidence intervals were computed in total ({len(P1)} confirmatory). At 95 % about "
          f"{0.05 * n_int:.1f} of them are expected to exclude zero by chance alone when there is no effect; "
          f"{int(COMP['ci_excludes_0'].sum())} do.")

    # ---- interpretation table ----
    st = {r["b"]: r for _, r in P1.iterrows()}
    if {"ijepa", "pos_matched", "veg_neg"} <= set(st):
        g = {k: bool(v["greater_confirmed"]) for k, v in st.items()}
        e = {k: bool(v["equivalent"]) for k, v in st.items()}
        lo = {k: bool(v["lower"]) for k, v in st.items()}
        pm = COMP[(COMP["eval"] == "probe") & (COMP["comparison"] == "pos_matched - ijepa") & (COMP["regime"] == "PRIMARY ENDPOINT")]
        pm_pos = bool(len(pm) and pm.iloc[0]["ci95_lo_pp"] > 0)
        if g["ijepa"] and g["pos_matched"] and g["veg_neg"]:
            INTERP = {"row": 1, "text": "veg_pos beats ijepa, pos_matched and veg_neg: greenness-guided placement helps (positive result)."}
        elif g["ijepa"] and not g["pos_matched"] and pm_pos:
            INTERP = {"row": 2, "text": "veg_pos ~ pos_matched > ijepa: the effect is a spatial prior, not greenness" +
                      (" (veg_pos and pos_matched equivalent within the margin)." if e["pos_matched"] else
                       " (veg_pos vs pos_matched not shown equivalent: report as 'mostly' a spatial prior).")}
        elif all(e.values()):
            INTERP = {"row": 3, "text": "veg_pos ~ ijepa ~ others within the equivalence margin: placement does not matter at this "
                                        "scale; the training regime dominates (negative result)." +
                                        (" Note: veg_pos - ijepa is also detectably below zero (small)." if lo["ijepa"] else "")}
        elif lo["ijepa"]:
            INTERP = {"row": 4, "text": "veg_pos < ijepa: greenness-biased placement hurts."}
        else:
            parts = [f"veg_pos - {k}: {v['status']} ({v['diff_pp']:+.2f} pp, 95% CI [{v['ci95_lo_pp']:+.2f}, {v['ci95_hi_pp']:+.2f}])"
                     for k, v in st.items()]
            INTERP = {"row": 5, "text": "No row of the pre-registered table applies cleanly (mixed / inconclusive). " + "; ".join(parts)}
    if RUN_MODE == "smoke":
        INTERP["text"] = "SMOKE TEST — generated data, meaningless: " + INTERP["text"]
    print("\n" + "=" * 100)
    print(f"INTERPRETATION (pre-registered table, row {INTERP['row']}): {INTERP['text']}")
    print("=" * 100)

# ---- seed-level primary endpoint table ----
if len(RES):
    rows = []
    for ev in ["probe", "knn"]:
        s = seed_level(RES, ev)
        s = s[s["regime"].isin(PRIMARY_REGIMES)]
        wide = s.pivot_table(index=["label", "family", "seed"], columns="regime", values="macro_f1").reset_index()
        e = endpoint_table(RES, ev, PRIMARY_REGIMES).rename(columns={"macro_f1": "primary_endpoint"})
        wide = wide.merge(e, on=["label", "family", "seed"], how="left")
        wide.insert(0, "eval", ev)
        rows.append(wide)
    SEED_LEVEL = pd.concat(rows, ignore_index=True).rename(columns={"label": "arm"})
    save_csv(SEED_LEVEL, OUT / "stats" / "seed_level_primary.csv")

# ---- per-class F1 differences at 100 % labels (exploratory) ----
PER_CLASS = pd.DataFrame()
if len(RES):
    pcs = []
    for a, b in [("veg_pos", "ijepa"), ("veg_pos", "pos_matched")]:
        seeds = sorted({s for (l, s) in RUNINFO if l == a} & {s for (l, s) in RUNINFO if l == b})
        seeds = [s for s in seeds if "probe/100pct" in RUNINFO[(a, s)]["per_class_f1"] and "probe/100pct" in RUNINFO[(b, s)]["per_class_f1"]]
        if not seeds:
            continue
        D = np.array([np.array(RUNINFO[(a, s)]["per_class_f1"]["probe/100pct"]) - np.array(RUNINFO[(b, s)]["per_class_f1"]["probe/100pct"])
                      for s in seeds])
        rng = np.random.default_rng(21)
        boots = np.array([D[rng.integers(0, len(seeds), len(seeds))].mean(0) for _ in range(cfg.bootstrap_B)])
        lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
        for k, c in enumerate(CLASSES):
            pcs.append({"comparison": f"{a} - {b}", "class": c, "diff_pp": 100 * D[:, k].mean(), "ci95_lo_pp": 100 * lo[k],
                        "ci95_hi_pp": 100 * hi[k], "n_seeds": len(seeds)})
    PER_CLASS = pd.DataFrame(pcs)
    if len(PER_CLASS):
        save_csv(PER_CLASS, OUT / "stats" / "per_class_delta.csv")

# ---- Tier 2 analyses (exploratory) ----
TIER2_TABLES = {}
if TAU_LABELS and {"ijepa", "veg_pos"} <= present:
    G = analyse_group(["ijepa", "veg_pos"] + TAU_LABELS, "probe", PRIMARY_REGIMES, cfg.bootstrap_B, seed=31,
                      only_seeds=TIER2_SEEDS_RUN)
    rows = []
    for lab, tau in [(tau_label(t), t) for t in cfg.tau_sweep] + [("veg_pos", cfg.veg_tau)]:
        r = compare(G, lab, "ijepa")
        if r:
            rows.append({"tau": tau, **{k: r[k] for k in ["diff_pp", "ci95_lo_pp", "ci95_hi_pp", "n_seeds", "wilcoxon_p_two_sided"]}})
    if G:
        rows.append({"tau": float("inf"), "diff_pp": 0.0, "ci95_lo_pp": 0.0, "ci95_hi_pp": 0.0, "n_seeds": len(G["seeds"]),
                     "wilcoxon_p_two_sided": float("nan")})
    TIER2_TABLES["tau_sweep"] = pd.DataFrame(rows).sort_values("tau")
if SIZE_LABELS and {"ijepa", "veg_pos"} <= present:
    rows = []
    for f in list(cfg.size_fracs) + [1.0]:
        a, b = (frac_label("veg_pos", f), frac_label("ijepa", f)) if f < 1 else ("veg_pos", "ijepa")
        G = analyse_group([a, b], "probe", PRIMARY_REGIMES, cfg.bootstrap_B, seed=41, verbose=False, only_seeds=TIER2_SEEDS_RUN)
        r = compare(G, a, b)
        if r:
            rows.append({"train_fraction": f, "n_images": len(SIZE_SUBSETS[f]) if f < 1 else len(ROWS["train"]),
                         **{k: r[k] for k in ["diff_pp", "ci95_lo_pp", "ci95_hi_pp", "n_seeds", "wilcoxon_p_two_sided"]}})
    TIER2_TABLES["size_axis"] = pd.DataFrame(rows)
if LJ_LABELS:
    G = analyse_group(["ijepa", "veg_pos"] + LJ_LABELS, "probe", PRIMARY_REGIMES, cfg.bootstrap_B, seed=51,
                      only_seeds=TIER2_SEEDS_RUN)
    rows = []
    for a in LJ_LABELS:
        if a != "lejepa_uniform":
            r = compare(G, a, "lejepa_uniform")
            if r:
                rows.append({"comparison": r["comparison"], **{k: r[k] for k in ["diff_pp", "ci95_lo_pp", "ci95_hi_pp", "n_seeds",
                                                                                   "wilcoxon_p_two_sided"]}})
    if G and {"lejepa_veg_pos", "lejepa_uniform"} <= set(G["labels"]):
        L_ = G["labels"]
        i1, i2, i3, i4 = L_.index("veg_pos"), L_.index("ijepa"), L_.index("lejepa_veg_pos"), L_.index("lejepa_uniform")
        d_seed = (G["seed_end"][i1] - G["seed_end"][i2]) - (G["seed_end"][i3] - G["seed_end"][i4])
        d_boot = (G["boot_end"][:, i1] - G["boot_end"][:, i2]) - (G["boot_end"][:, i3] - G["boot_end"][:, i4])
        lo, hi = np.percentile(d_boot, [2.5, 97.5])
        rows.append({"comparison": "H4: (veg_pos - ijepa) - (lejepa_veg_pos - lejepa_uniform)", "diff_pp": 100 * d_seed.mean(),
                     "ci95_lo_pp": 100 * lo, "ci95_hi_pp": 100 * hi, "n_seeds": len(d_seed),
                     "wilcoxon_p_two_sided": wilcoxon_exact(d_seed, "two-sided")[0]})
    TIER2_TABLES["lejepa"] = pd.DataFrame(rows)
for name, t in TIER2_TABLES.items():
    if len(t):
        print(f"\nTier 2 ({name}), exploratory — linear-probe primary endpoint, differences in pp:")
        display(t.round(3))
        save_csv(t, OUT / "stats" / f"tier2_{name}.csv")

G1 = {"decision_table_row": INTERP["row"], "interpretation": INTERP["text"],
      "primary": COMP[COMP["family"].str.startswith("PRIMARY")].drop(columns=["seed_diffs_pp"]).to_dict(orient="records") if len(COMP) else [],
      "primary_seed_diffs_pp": {r["comparison"]: r["seed_diffs_pp"] for _, r in COMP.iterrows() if r["family"].startswith("PRIMARY")} if len(COMP) else {},
      "seeds_completed": {a: int(RES[RES["label"] == a]["seed"].nunique()) for a in PRIMARY_ARMS} if len(RES) else {},
      "seeds_planned": len(SEEDS_RUN), "intervals_computed": int(len(COMP)),
      "expected_chance_exclusions": 0.05 * len(COMP), "preregistration": str(OUT / "stats" / "preregistration.json"),
      "computed_at": datetime.datetime.now().isoformat()}
if len(RES) and any(v < len(SEEDS_RUN) for v in G1["seeds_completed"].values()) and RUN_MODE == "full":
    G1["interpretation"] = f"PROVISIONAL (not all {len(SEEDS_RUN)} seeds finished): " + G1["interpretation"]
    print(f"NOTE: provisional — seeds finished per arm: {G1['seeds_completed']}")
if len(COMP):
    save_json(G1, OUT / "stats" / "gate_G1_stageB.json")
else:
    print("Primary analysis not computed yet (needs finished core runs of ijepa and veg_pos at least).")


# ======================= notebook cell 47 =======================
COLORS = {"ijepa": "#1f77b4", "veg_pos": "#2ca02c", "veg_neg": "#d62728", "pos_matched": "#9467bd", "attn": "#ff7f0e",
          "random_init": "#7f7f7f", "supervised": "#8c564b", "dinov2": "#17becf"}
FIG = OUT / "figures"


def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=150)
    plt.show()
    plt.close(fig)


# ---- label efficiency (probe, k-NN) with hierarchical-bootstrap bands ----
for ev in ["probe", "knn"]:
    G = GROUPS.get(ev)
    if not G:
        continue
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for li, lab in enumerate(G["labels"]):
        mean = 100 * G["seed_reg"][li].mean(1)
        lo, hi = np.percentile(100 * G["boot_reg"][:, li, :], [2.5, 97.5], axis=0)
        ax.plot(range(len(G["regimes"])), mean, marker="o", label=lab, color=COLORS.get(lab))
        ax.fill_between(range(len(G["regimes"])), lo, hi, alpha=0.12, color=COLORS.get(lab))
    sup = RES[RES["eval"] == "supervised"] if len(RES) else pd.DataFrame()
    if ev == "probe" and len(sup):
        for r, v in sup.groupby("regime")["macro_f1"].mean().items():
            if r in G["regimes"]:
                ax.scatter(G["regimes"].index(r), 100 * v, marker="*", s=160, color=COLORS["supervised"],
                           label="supervised (upper ref., not matched)" if r == sup["regime"].iloc[0] else None, zorder=5)
    ax.set_xticks(range(len(G["regimes"])))
    ax.set_xticklabels(G["regimes"])
    for r in PRIMARY_REGIMES:
        if r in G["regimes"]:
            ax.axvspan(G["regimes"].index(r) - 0.3, G["regimes"].index(r) + 0.3, color="gold", alpha=0.12)
    ax.set_ylabel("Test macro-F1 (%)")
    ax.set_title(f"Label efficiency — {'linear probe' if ev == 'probe' else 'k-NN'} (mean over {len(G['seeds'])} seeds, "
                 "95% hierarchical bootstrap; shaded = primary regimes)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    savefig(fig, f"label_efficiency_{ev}.png")

# ---- forest plot of all endpoint comparisons ----
if len(COMP):
    sub = COMP[COMP["regime"] == "PRIMARY ENDPOINT"].copy()
    fig, ax = plt.subplots(figsize=(9, 0.8 + 0.42 * len(sub)))
    ax.axvspan(-cfg.equiv_margin_pp, cfg.equiv_margin_pp, color="grey", alpha=0.12, label=f"±{cfg.equiv_margin_pp} pp margin")
    for y, (_, r) in enumerate(sub.iterrows()):
        primary = r["family"].startswith("PRIMARY")
        ax.errorbar(r["diff_pp"], y, xerr=[[r["diff_pp"] - r["ci95_lo_pp"]], [r["ci95_hi_pp"] - r["diff_pp"]]], fmt="o",
                    color=COLORS.get(r["a"], "k"), capsize=3, ms=8 if primary else 5, lw=2.2 if primary else 1)
    ax.axvline(0, color="k", lw=1, ls="--")
    ax.set_yticks(range(len(sub)))
    ax.set_yticklabels([("★ " if f.startswith("PRIMARY") else "") + f"{c} [{e}]" for c, e, f in
                        zip(sub["comparison"], sub["eval"], sub["family"])], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Difference in primary-endpoint macro-F1 (pp), 95% hierarchical bootstrap CI")
    ax.set_title("Paired comparisons (★ = pre-registered primary family; others exploratory)")
    ax.legend(loc="lower right", fontsize=8)
    savefig(fig, "primary_endpoint_forest.png")

# ---- seed-level points ----
G = GROUPS.get("probe")
if G and G["seed_end"] is not None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(G["labels"]))
    for si, s in enumerate(G["seeds"]):
        ax.plot(x, 100 * G["seed_end"][:, si], color="grey", alpha=0.35, lw=1)
        ax.scatter(x, 100 * G["seed_end"][:, si], c=[COLORS.get(l, "k") for l in G["labels"]], s=30, zorder=3)
        ax.annotate(f"s{s}", (x[-1] + 0.05, 100 * G["seed_end"][-1, si]), fontsize=7, color="grey")
    ax.scatter(x, 100 * G["seed_end"].mean(1), marker="_", s=900, color="k", zorder=4, label="mean over seeds")
    ax.set_xticks(x)
    ax.set_xticklabels(G["labels"])
    ax.set_ylabel(f"Primary endpoint: probe macro-F1 (%) over {PRIMARY_REGIMES}")
    ax.set_title("Seed-level primary endpoint (lines connect runs that share a seed)")
    ax.grid(alpha=0.3)
    ax.legend()
    savefig(fig, "seed_level_dots.png")

# ---- training curves ----
if len(LOGS):
    L = LOGS[LOGS["label"].isin(CORE_LABELS)]
    if len(L):
        panels = [("loss", "JEPA loss"), ("knn_val_f1", "Online k-NN macro-F1 on validation"), ("rankme", "RankMe (target features)"),
                  ("feat_std", "Per-dim std of normalised features"), ("tgt_exg_z", "Target ExG (z, manipulation check)")]
        fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 3.8))
        first = min(L["seed"])
        for (lab, seed), d in L.groupby(["label", "seed"]):
            kw = dict(color=COLORS.get(lab), alpha=1.0 if seed == first else 0.3, label=lab if seed == first else None)
            for ax, (col, _) in zip(axes, panels):
                dd = d.dropna(subset=[col])
                ax.plot(dd["step"], dd[col] * (100 if col == "knn_val_f1" else 1), marker="." if col == "knn_val_f1" else None, **kw)
        for ax, (_, t) in zip(axes, panels):
            ax.set_title(t, fontsize=9)
            ax.set_xlabel("optimizer step")
            ax.grid(alpha=0.3)
        axes[0].legend(fontsize=8)
        savefig(fig, "training_curves.png")

# ---- feature spectrum ----
spec = {}
for lab in CORE_LABELS + BASE_LABELS:
    arrs = [np.load(p) for p in sorted((OUT / "runs").glob(f"{lab}_s*/spectrum.npy"))]
    if arrs:
        n = min(len(a) for a in arrs)
        spec[lab] = np.mean([a[:n] for a in arrs], 0)
if spec:
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for lab, sv in spec.items():
        ax.semilogy(np.arange(1, len(sv) + 1), sv, label=lab, color=COLORS.get(lab))
    ax.set_xlabel("component")
    ax.set_ylabel("normalised singular value (train features)")
    ax.set_title("Feature spectrum (mean over seeds)")
    ax.grid(alpha=0.3)
    ax.legend()
    savefig(fig, "feature_spectrum.png")

# ---- per-class differences ----
if len(PER_CLASS):
    comps = list(dict.fromkeys(PER_CLASS["comparison"]))
    fig, ax = plt.subplots(figsize=(max(8, 0.7 * K * len(comps)), 4.2))
    w = 0.8 / len(comps)
    for j, c in enumerate(comps):
        d = PER_CLASS[PER_CLASS["comparison"] == c]
        xs = np.arange(len(d)) + j * w
        ax.bar(xs, d["diff_pp"], w, label=c, alpha=0.8)
        ax.errorbar(xs, d["diff_pp"], yerr=[d["diff_pp"] - d["ci95_lo_pp"], d["ci95_hi_pp"] - d["diff_pp"]], fmt="none", color="k", capsize=2)
    ax.axhline(0, color="k", lw=1)
    ax.set_xticks(np.arange(K) + w * (len(comps) - 1) / 2)
    ax.set_xticklabels([f"{c}\n{LUCAS_CROP_NAMES.get(c, '')}" for c in CLASSES], fontsize=7)
    ax.set_ylabel("F1 difference (pp), probe at 100 % labels")
    ax.set_title("Per-class F1 differences (exploratory; 95% CI over seeds)")
    ax.legend()
    savefig(fig, "per_class_delta.png")

# ---- confusion matrix of the best arm ----
if len(RES):
    p100 = RES[(RES["eval"] == "probe") & (RES["regime"] == "100pct") & (RES["label"].isin(CORE_LABELS + BASE_LABELS))]
    if len(p100):
        best = p100.groupby("label")["macro_f1"].mean().idxmax()
        s0 = int(sorted(p100[p100["label"] == best]["seed"])[0])
        cm = confusion(Y_TEST, PREDS[(best, "probe", "100pct", s0, 0)], K)
        cmn = cm / np.maximum(1, cm.sum(1, keepdims=True))
        fig, ax = plt.subplots(figsize=(min(14, 2 + 0.55 * K), min(12, 1.5 + 0.5 * K)))
        im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(K))
        ax.set_yticks(range(K))
        ax.set_xticklabels(CLASSES, rotation=90, fontsize=7)
        ax.set_yticklabels([f"{c} {LUCAS_CROP_NAMES.get(c, '')}" for c in CLASSES], fontsize=7)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(f"Confusion matrix (row-normalised) — {best}, seed {s0}, probe, all labels")
        plt.colorbar(im, ax=ax, fraction=0.046)
        savefig(fig, "confusion_best.png")

# ---- manipulation check: target ExG, centre heat maps, example masks ----
man_arms = [a for a in ["ijepa", "veg_pos", "veg_neg", "pos_matched"] if a in ACCEPT_HISTS]
if man_arms:
    gh = cfg.img_size // cfg.patch_size
    n_ex = min(8, len(ROWS["val"]))
    x_ex, _ = next(iter(GPULoader(CACHE, ROWS["val"][:n_ex], n_ex, train=False, out_size=cfg.img_size)))
    fig = plt.figure(figsize=(2.1 * max(n_ex, 4), 2.3 * (len(man_arms) + 2)), constrained_layout=True)
    gs = fig.add_gridspec(len(man_arms) + 2, max(n_ex, len(man_arms) + 1))
    ax = fig.add_subplot(gs[0, : max(2, n_ex // 2)])
    tr = LOGS[LOGS["label"].isin(man_arms)].groupby(["label", "seed"])["tgt_exg_z"].mean() if len(LOGS) and "tgt_exg_z" in LOGS else pd.Series(dtype=float)
    for i, a in enumerate(man_arms):
        pre = float(ACCEPT.set_index("arm").loc[a, "tgt_exg_z"])
        ax.bar(i, pre, color=COLORS.get(a), alpha=0.5, label="pre-training check" if i == 0 else None)
        if a in tr.index.get_level_values(0):
            pts = tr.loc[a].values
            ax.scatter(np.full(len(pts), i), pts, color="k", s=14, zorder=3, label="training logs (per seed)" if i == 0 else None)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(man_arms)))
    ax.set_xticklabels(man_arms)
    ax.set_ylabel("mean z-scored ExG\nunder the targets")
    ax.set_title("(a) target greenness by arm")
    ax.legend(fontsize=7)
    for i, a in enumerate(man_arms):
        files = sorted((OUT / "runs").glob(f"{a}_s*/target_centres.npy"))
        hist = np.sum([np.load(f) for f in files], 0) if files else ACCEPT_HISTS[a]
        axh = fig.add_subplot(gs[1, i])
        axh.imshow(np.asarray(hist).reshape(gh, gh) / max(1e-9, np.sum(hist)), cmap="magma")
        axh.set_title(f"(b) centres: {a}", fontsize=8)
        axh.axis("off")
    img = x_ex.permute(0, 2, 3, 1).cpu().numpy()
    for row, a in enumerate(man_arms):
        mg = MaskGenerator(a, 0, prior=POS_PRIOR)
        ctx_idx, tgt, _ = mg(x_ex)
        lab = np.zeros((n_ex, gh * gh))
        lab[np.arange(n_ex)[:, None], ctx_idx.cpu().numpy()] = 1
        for t in tgt:
            lab[np.arange(n_ex)[:, None], t.cpu().numpy()] = 2
        for j in range(n_ex):
            axi = fig.add_subplot(gs[row + 2, j])
            axi.imshow(img[j])
            ov = np.zeros((gh, gh, 4))
            m = lab[j].reshape(gh, gh)
            ov[m == 1] = [0.1, 0.4, 1.0, 0.35]
            ov[m == 2] = [1.0, 0.1, 0.1, 0.45]
            axi.imshow(ov, extent=(0, cfg.img_size, cfg.img_size, 0), interpolation="nearest")
            axi.axis("off")
            if j == 0:
                axi.set_title(f"(c) {a}: blue = context, red = targets", fontsize=7, loc="left")
    fig.savefig(FIG / "manipulation_check.png", dpi=130)
    plt.show()
    plt.close(fig)

# ---- Tier 2 figures ----
t = TIER2_TABLES.get("tau_sweep")
if t is not None and len(t):
    fig, ax = plt.subplots(figsize=(6, 4))
    xs = [str(v) if math.isfinite(v) else "inf\n(= ijepa)" for v in t["tau"]]
    ax.errorbar(range(len(t)), t["diff_pp"], yerr=[t["diff_pp"] - t["ci95_lo_pp"], t["ci95_hi_pp"] - t["diff_pp"]], fmt="o-", capsize=3)
    ax.axhline(0, color="k", ls="--", lw=1)
    ax.set_xticks(range(len(t)))
    ax.set_xticklabels(xs)
    ax.set_xlabel("tau (softmax temperature)")
    ax.set_ylabel("veg_pos - ijepa (pp), primary endpoint")
    ax.set_title("Tier 2: dose-response in tau (exploratory)")
    ax.grid(alpha=0.3)
    savefig(fig, "tau_sweep.png")
t = TIER2_TABLES.get("size_axis")
if t is not None and len(t):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(t["n_images"], t["diff_pp"], yerr=[t["diff_pp"] - t["ci95_lo_pp"], t["ci95_hi_pp"] - t["diff_pp"]], fmt="o-", capsize=3)
    ax.set_xscale("log")
    ax.axhline(0, color="k", ls="--", lw=1)
    ax.set_xlabel(f"pretraining images (fixed budget of {cfg.target_steps} steps)")
    ax.set_ylabel("veg_pos - ijepa (pp), primary endpoint")
    ax.set_title("Tier 2: data-size axis (exploratory)")
    ax.grid(alpha=0.3)
    savefig(fig, "size_axis.png")
print("Figures saved to", FIG.resolve(), ":", sorted(p.name for p in FIG.glob("*.png")))


# ======================= notebook cell 49 =======================
PT = OUT / "paper_tables"


def latex_escape(s):
    s = str(s)
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"), ("_", r"\_"),
                 ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")]:
        s = s.replace(a, b)
    return s.replace("±", r"$\pm$").replace("−", "-").replace("≈", r"$\approx$")


def write_table(df, name, caption):
    """CSV + a booktabs LaTeX table (no jinja2 dependency)."""
    if df is None or not len(df):
        return
    df = df.reset_index() if df.index.name or not isinstance(df.index, pd.RangeIndex) else df
    save_csv(df, PT / f"{name}.csv")
    cols = list(df.columns)
    lines = [r"\begin{table}[t]", r"\centering", r"\small", rf"\caption{{{latex_escape(caption)}}}", rf"\label{{tab:{name}}}",
             r"\begin{tabular}{l" + "r" * (len(cols) - 1) + "}", r"\toprule", " & ".join(latex_escape(c) for c in cols) + r" \\",
             r"\midrule"]
    for _, r in df.iterrows():
        cells = [f"{v:.2f}" if isinstance(v, (float, np.floating)) else str(v) for v in r.values]
        lines.append(" & ".join(latex_escape(c) for c in cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    save_text("\n".join(lines) + "\n", PT / f"{name}.tex")


if len(RES):
    main = summary_table(RES, "probe", CORE_LABELS + BASE_LABELS, REGIMES)
    sup = RES[RES["eval"] == "supervised"]
    if len(sup) and len(main):
        srow = {r: fmt_cell(g["macro_f1"].mean(), g["macro_f1"].std(), g["seed"].nunique()) for r, g in sup.groupby("regime")}
        main.loc["supervised (upper ref., not matched)"] = [srow.get(c, "") for c in main.columns]
    write_table(main, "main_results_probe", "Linear-probe test macro-F1 (%), mean ± sd over pretraining seeds (label draws "
                f"averaged within seed). Primary endpoint = mean over {', '.join(PRIMARY_REGIMES)}.")
    write_table(summary_table(RES, "knn", CORE_LABELS + BASE_LABELS, REGIMES), "main_results_knn",
                "k-NN test macro-F1 (%), mean ± sd over pretraining seeds.")
if len(COMP):
    P1 = COMP[COMP["family"].str.startswith("PRIMARY")]
    pc = pd.DataFrame({"comparison": P1["comparison"], "delta (pp)": P1["diff_pp"],
                       "95% CI": [f"[{a:+.2f}, {b:+.2f}]" for a, b in zip(P1["ci95_lo_pp"], P1["ci95_hi_pp"])],
                       f"{int(100 * cfg.equiv_ci)}% CI": [f"[{a:+.2f}, {b:+.2f}]" for a, b in zip(P1[f"{CI_E}_lo_pp"], P1[f"{CI_E}_hi_pp"])],
                       "seeds +": [f"{a}/{n}" for a, n in zip(P1["seeds_a_better"], P1["n_seeds"])],
                       f"Wilcoxon p ({cfg.primary_alternative})": P1[P1["test_used"].iloc[0]] if len(P1) else [],
                       "Holm p": P1["holm_p"], "perm. p (2-sided)": P1["perm_p_two_sided"], "result": P1["status"]})
    write_table(pc, "primary_comparisons", "Pre-registered primary comparisons (linear-probe primary endpoint, unit = seed, "
                f"exact Wilcoxon signed-rank, Holm over 3, equivalence margin ±{cfg.equiv_margin_pp} pp).")
    ex = COMP[COMP["regime"] == "PRIMARY ENDPOINT"]
    ex = ex[~ex["family"].str.startswith("PRIMARY")]
    write_table(pd.DataFrame({"eval": ex["eval"], "comparison": ex["comparison"], "delta (pp)": ex["diff_pp"],
                              "95% CI": [f"[{a:+.2f}, {b:+.2f}]" for a, b in zip(ex["ci95_lo_pp"], ex["ci95_hi_pp"])],
                              "n seeds": ex["n_seeds"], "note": ex["status"]}),
                "exploratory_comparisons", "Exploratory comparisons on the primary endpoint (no multiplicity correction; no claims).")
abl = []
for name, t in TIER2_TABLES.items():
    if len(t):
        t = t.copy()
        t.insert(0, "ablation", name)
        abl.append(t)
if abl:
    write_table(pd.concat(abl, ignore_index=True), "ablations_tier2", "Tier-2 ablations (exploratory): differences in pp.")
for f, name, cap in [(OUT / "stats" / "token_budget.csv", "token_budget", "Mean context / target tokens per image by arm."),
                     (OUT / "stats" / "manipulation_check.csv", "manipulation_check", "Mean ExG under the target blocks by arm.")]:
    if f.exists():
        write_table(pd.read_csv(f).round(4), name, cap)
print("Paper tables:", sorted(p.name for p in PT.glob("*")))

bundle = OUT / "results_bundle.zip"
skip_dirs = {"cache", "synthetic_data", "selftest", "benchmark_tmp", "benchmark_lejepa_tmp"}
with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as z:
    for f in OUT.rglob("*"):
        if not f.is_file() or f == bundle or skip_dirs & set(f.relative_to(OUT).parts) or f.suffix == ".tmp":
            continue
        if f.suffix == ".pt" and not ZIP_CHECKPOINTS:
            continue
        z.write(f, f.relative_to(OUT))
done = len(list((OUT / "runs").glob("*/eval.json")))
print(f"Bundle: {bundle} ({bundle.stat().st_size / 1e6:.1f} MB) | finished runs in this folder: {done}")
print(f"Session time used: {(time.time() - SESSION_START) / 3600:.2f} h | free disk: {free_disk_gib(OUT):.1f} GiB")
