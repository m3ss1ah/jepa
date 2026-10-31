import json
from pathlib import Path

def patch_notebook(nb_path):
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
        
    print(f"Patching {nb_path}...")
    
    # 1. Patch Cell 2 (Configuration cell)
    cell2_code = '''RUN_MODE = "full"        # "smoke" = generated data, whole pipeline in minutes · "benchmark" = measure speed only · "full" = real runs
PROFILE = "auto"          # "auto" or one of: "laptop_4gb", "laptop_4gb_vits", "kaggle_t4", "kaggle_t4_fast", "kaggle_p100", "big_gpu", "cpu_debug"
DATA_ROOT = "./lucas_data"
METADATA_CSV = "./lucas_data/metadata.csv"          # optional CSV with columns path,label[,group][,split]; paths may be relative to DATA_ROOT
OUTPUT_DIR = None         # None = ./outputs on a laptop, /kaggle/working/outputs on Kaggle
RESUME_FROM_DIR = None    # Kaggle: a previous session's outputs attached as input, e.g. "/kaggle/input/<notebook-output>/outputs"

ARMS = ["ijepa", "veg", "ms", "veg_ms"]     # the Stage A factorial
BASELINES = ["random_init", "supervised"]   # add "dinov2" if internet access is available (frozen DINOv2 ViT-S/14 reference)
SEEDS = [0, 1, 2]                           # pretraining seeds (smoke mode uses [0, 1])
SHARD_ID, NUM_SHARDS = 0, 1                 # split the run queue across machines / teammates' accounts
OVERRIDES = {}                              # e.g. {"epochs": 300} · applied on top of the hardware profile
ALLOW_PROFILE_CHANGE = False                # keep False: all arms of one comparison must run on the same profile
ZIP_CHECKPOINTS = False                     # include *.pt checkpoints in the downloadable results zip'''

    for c in nb["cells"]:
        if c["cell_type"] == "code" and 'RUN_MODE = ' in "".join(c.get("source", [])) and 'METADATA_CSV' in "".join(c.get("source", [])):
            c["source"] = [line + "\n" for line in cell2_code.split("\n")]
            # remove trailing newline from last line
            if c["source"]:
                c["source"][-1] = c["source"][-1].rstrip("\n")
            break

    # 2. Patch Cell 10 (LUCAS data setup)
    cell10_code = '''import urllib.request, re, random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

LUCAS_CROP_NAMES = {
    "B11": "Common wheat",
    "B12": "Durum wheat",
    "B13": "Barley",
    "B14": "Rye",
    "B15": "Oats",
    "B16": "Maize",
    "B21": "Potatoes",
    "B22": "Sugar beet",
    "B31": "Sunflower",
    "B32": "Rapeseed",
    "B33": "Soya",
    "B55": "Temporary grassland"
}

def setup_lucas_data(dest_dir, limit_per_class=100, seed=42):
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    meta_path = dest_dir / "metadata.csv"
    manifest_path = dest_dir / "dataset_manifest.csv"
    
    if meta_path.exists() and manifest_path.exists():
        print("LUCAS data already prepared at", dest_dir)
        return
        
    base_url = 'https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/DRLL/LUCASvision/'
    classes = ['B11', 'B12', 'B13', 'B14', 'B15', 'B16', 'B21', 'B22', 'B31', 'B32', 'B33', 'B55']
    rng = random.Random(seed)
    
    all_links = {}
    for cls in classes:
        cls_dir = dest_dir / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        try:
            html = urllib.request.urlopen(f"{base_url}{cls}/").read().decode('utf-8')
        except Exception as e:
            print(f"Failed to access {cls}: {e}")
            continue
        links = re.findall(r'href="([^"]+\\.jpg)"', html)
        if len(links) > limit_per_class:
            links = sorted(links)
            links = rng.sample(links, limit_per_class)
        all_links[cls] = links
        
    rows = []
    manifest_rows = []
    def download_file(args):
        cls, filename = args
        url = f"{base_url}{cls}/{filename}"
        filepath = dest_dir / cls / filename
        if not filepath.exists():
            try:
                urllib.request.urlretrieve(url, filepath)
            except Exception as e:
                return None, None
        # Parse LUCAS filename format: LUCAS<YEAR>_<POINTID>_<VIEW>.jpg
        m = re.match(r"^LUCAS(\\d{4})_(\\d+)_([A-Za-z0-9]+)\\.jpg$", filename, re.I)
        if m:
            year, point_id, view = m.group(1), m.group(2), m.group(3)
        else:
            parts = filename.split('_')
            year = parts[0].replace('LUCAS', '') if len(parts) > 0 else 'unknown'
            point_id = parts[1] if len(parts) > 1 else filename.split('.')[0]
            view = parts[2].split('.')[0] if len(parts) > 2 else 'Cover'
            
        group_id = str(point_id)
        meta_row = {"path": f"{cls}/{filename}", "label": cls, "group": group_id}
        manifest_row = {
            "source_url": url,
            "filename": filename,
            "path": f"{cls}/{filename}",
            "crop_label": cls,
            "crop_name": LUCAS_CROP_NAMES.get(cls, "Unknown"),
            "point_id": point_id,
            "view": view,
            "survey_year": int(year) if year.isdigit() else year,
            "group": group_id
        }
        return meta_row, manifest_row

    tasks = [(c, f) for c, files in all_links.items() for f in files]
    print(f"Downloading/verifying {len(tasks)} images from LUCASvision...")
    with ThreadPoolExecutor(20) as ex:
        results = list(ex.map(download_file, tasks))
        
    for meta_r, mani_r in results:
        if meta_r:
            rows.append(meta_r)
            manifest_rows.append(mani_r)
            
    df = pd.DataFrame(rows)
    df.to_csv(meta_path, index=False)
    df_mani = pd.DataFrame(manifest_rows)
    df_mani.to_csv(manifest_path, index=False)
    print(f"Prepared verified LUCAS dataset with {len(df)} images and {df['group'].nunique()} unique survey points.")

if RUN_MODE != "smoke":
    setup_lucas_data("./lucas_data", limit_per_class=100, seed=cfg.split_seed if 'cfg' in globals() else 42)'''

    for c in nb["cells"]:
        if c["cell_type"] == "code" and 'def setup_lucas_data' in "".join(c.get("source", [])):
            c["source"] = [line + "\n" for line in cell10_code.split("\n")]
            if c["source"]:
                c["source"][-1] = c["source"][-1].rstrip("\n")
            break

    # 3. Patch Cell 11 (discover_dataset, build_cache, make_or_load_splits, _assert_no_split_overlap)
    for c in nb["cells"]:
        src = "".join(c.get("source", []))
        if c["cell_type"] == "code" and 'def discover_dataset' in src and 'def make_or_load_splits' in src:
            # We will rewrite the cell with robust group derivation, cache synchronization, and assertions
            lines = c["source"]
            new_code = '''IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SPLIT_ALIASES = {"train": "train", "training": "train", "val": "val", "valid": "val", "validation": "val",
                 "test": "test", "testing": "test"}


def make_synthetic_dataset(root, n_classes=6, per_class=50, res=96, seed=0):
    """Generated 'crop' images (green striped patches on soil) with site groups and 3 planted duplicates.
    Used only by the smoke test, to exercise every code path without real data."""
    root = Path(root)
    if (root / "metadata.csv").exists():
        return root / "metadata.csv"
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:res, 0:res] / res
    rows = []
    for c in range(n_classes):
        d = root / f"crop_{c}"
        d.mkdir(parents=True, exist_ok=True)
        theta, freq = np.pi * c / n_classes, 3 + 2 * (c % 3)
        for i in range(per_class):
            img = np.ones((res, res, 3)) * (np.array([0.45, 0.35, 0.25]) + rng.normal(0, 0.03, 3))
            img += rng.normal(0, 0.04, (res, res, 3))
            stripes = np.sin(2 * np.pi * freq * (xx * np.cos(theta) + yy * np.sin(theta)) + rng.uniform(0, 6.28)) > (0.3 if c % 2 else -0.3)
            cx, cy, r = rng.uniform(0.3, 0.7), rng.uniform(0.3, 0.7), rng.uniform(0.3, 0.5)
            veg = stripes & (((xx - cx) ** 2 + (yy - cy) ** 2) < r ** 2)
            img[veg] = np.array([0.15, 0.50 + 0.08 * (c % 2), 0.12]) + rng.normal(0, 0.05, (veg.sum(), 3))
            path = d / f"img_{i:03d}.png"
            Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(path)
            rows.append({"path": str(path.relative_to(root)), "label": f"crop_{c}", "group": f"site_{c}_{i // 3}"})
    for k in range(3):  # planted duplicates in different groups: the hygiene check must merge them
        src = root / rows[k * 7]["path"]
        dst = src.parent / f"dup_{k}.png"
        shutil.copy(src, dst)
        rows.append({"path": str(dst.relative_to(root)), "label": rows[k * 7]["label"], "group": f"dupgroup_{k}"})
    pd.DataFrame(rows).to_csv(root / "metadata.csv", index=False)
    return root / "metadata.csv"


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
    """Finds the shallowest folder whose sub-folders contain images (class folders or train/test folders)."""
    base = Path(base)
    if not base.exists():
        return None
    for depth in range(max_depth):
        for d in sorted(p for p in base.glob("/".join(["*"] * (depth + 1))) if p.is_dir()):
            subs = [s for s in d.iterdir() if s.is_dir()]
            names = {s.name.lower() for s in subs}
            if "train" in names and ("test" in names or "testing" in names):
                return d
            if sum(_has_images(s) for s in subs[:200]) >= 2:
                return d
    return None


def list_images(folder):
    return sorted(str(p) for p in Path(folder).rglob("*") if p.suffix.lower() in IMG_EXT)


def discover_dataset(data_root, metadata_csv=None):
    data_root = Path(data_root) if data_root else None
    if metadata_csv:
        df = pd.read_csv(metadata_csv)
        if not {"path", "label"} <= set(df.columns):
            raise ValueError("METADATA_CSV needs at least the columns: path, label")
        base = data_root or Path(metadata_csv).parent
        df["path"] = [p if os.path.isabs(p) else str(base / p) for p in df["path"]]
    else:
        if data_root is None or not data_root.exists():
            raise FileNotFoundError(f"DATA_ROOT not found: {data_root}. Set DATA_ROOT in the configuration cell.")
        subdirs = {d.name.lower(): d for d in data_root.iterdir() if d.is_dir()}
        split_dirs = {SPLIT_ALIASES[k]: v for k, v in subdirs.items() if k in SPLIT_ALIASES}
        rows = []
        if "train" in split_dirs and "test" in split_dirs:
            for split, sd in split_dirs.items():
                for cd in sorted(x for x in sd.iterdir() if x.is_dir()):
                    rows += [dict(path=p, label=cd.name, split=split) for p in list_images(cd)]
        else:
            for cd in sorted(x for x in data_root.iterdir() if x.is_dir()):
                rows += [dict(path=p, label=cd.name) for p in list_images(cd)]
        df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No images found. Expected class sub-folders (or train/test folders containing class sub-folders).")
    if "group" not in df.columns:
        # Derive group from LUCAS survey-point ID if filename matches LUCAS pattern
        lucas_pat = re.compile(r"LUCAS\d{4}_(\d+)_[A-Za-z0-9]+\.jpg", re.I)
        groups = []
        for p in df["path"]:
            fname = Path(p).name
            m = lucas_pat.match(fname)
            groups.append(m.group(1) if m else str(p))
        if any(lucas_pat.match(Path(p).name) for p in df["path"]):
            df["group"] = groups
            print(f"Derived group column from verified LUCAS survey-point identifiers ({df['group'].nunique()} unique point groups across {len(df)} images).")
        else:
            print("NOTE: no 'group' column (site / field ID). Each image is treated as its own group; if several photos "
                  "come from the same field, provide a METADATA_CSV with a group column to avoid train/test leakage.")
            df["group"] = df["path"]
    df["label"] = df["label"].astype(str)
    df["group"] = df["group"].astype(str)
    return df.reset_index(drop=True)


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
    cache_dir = Path(out_dir) / "cache"
    img_path, meta_path = cache_dir / f"images_{res}.npy", cache_dir / f"images_{res}.csv"
    if img_path.exists() and meta_path.exists():
        meta = pd.read_csv(meta_path, keep_default_na=False)
        if list(meta["path"]) == list(df["path"]):
            print(f"Image cache found: {img_path} ({len(meta)} images)")
            # Synchronize metadata columns from df (especially 'group' and 'label') so stale cached metadata is not reused!
            for col in df.columns:
                meta[col] = df[col].values
            meta.to_csv(meta_path, index=False)
            return img_path, meta
        print("Dataset file list changed: rebuilding the image cache.")
    n = len(df)
    tmp = cache_dir / f"images_{res}.tmp.npy"
    arr = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=(n, res, res, 3))
    hashes, ok = [0] * n, np.zeros(n, dtype=bool)
    t0 = time.time()
    workers = min(8, (os.cpu_count() or 2))
    with ThreadPoolExecutor(workers) as ex:
        for i, (a, h, good) in enumerate(ex.map(lambda p: _process_image(p, res), df["path"])):
            if good:
                arr[i], hashes[i], ok[i] = a, h, True
            if (i + 1) % max(1, n // 10) == 0:
                print(f"  cached {i + 1}/{n} images ({time.time() - t0:.0f}s)")
    arr.flush()
    del arr
    os.replace(tmp, img_path)
    meta = df.copy()
    meta["dhash"] = [f"{h:016x}" for h in hashes]
    meta["ok"] = ok
    meta.to_csv(meta_path, index=False)
    print(f"Image cache written: {img_path} | unreadable images: {(~ok).sum()}")
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
    """Finds exact and near-duplicate images (dHash Hamming distance <= max_dist) and merges their groups,
    so duplicates can never end up in different splits. Returns meta with a 'group_final' column and a report."""
    meta = meta[meta["ok"].astype(bool)].reset_index(drop=True).copy()
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
    uf_grp = UnionFind(meta["group"].unique().tolist())
    for c in dup_clusters:
        g = meta["group"].iloc[c].tolist()
        for x in g[1:]:
            uf_grp.union(g[0], x)
    meta["group_final"] = [uf_grp.find(g) for g in meta["group"]]
    conflicting = [c for c in dup_clusters if meta["label"].iloc[c].nunique() > 1]
    report = {"images_ok": n, "duplicate_clusters": len(dup_clusters), "images_in_duplicate_clusters": int(sum(len(c) for c in dup_clusters)),
              "duplicate_clusters_with_conflicting_labels": len(conflicting), "near_duplicate_check": near_checked,
              "groups_before": int(meta["group"].nunique()), "groups_after": int(meta["group_final"].nunique())}
    return meta, report


def group_stratified_split(meta, val_frac, test_frac, seed, carve_from=None):
    """Assigns whole groups to splits, class by class, so class proportions are roughly kept and no group
    (site / field / duplicate cluster) spans two splits. With carve_from='train', only a validation split is
    carved out of the existing train split."""
    rng = np.random.default_rng(seed)
    pool = meta if carve_from is None else meta[meta["split"] == carve_from]
    g = pool.groupby("group_final").agg(label=("label", lambda s: s.mode().iloc[0]), size=("label", "size")).reset_index()
    order = [("test", test_frac), ("val", val_frac)] if carve_from is None else [("val", val_frac)]
    assign = {}
    for lab, sub in g.groupby("label"):
        sub = sub.iloc[rng.permutation(len(sub))]
        if len(sub) < 3:  # too few groups to split this class: keep it in train only
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
        if all(d != "train" for _, d in labs):  # guarantee at least one training group per class
            assign[labs[-1][0]] = "train"
    out = meta.copy()
    if carve_from is None:
        out["split"] = out["group_final"].map(assign).fillna("train")
    else:
        m = out["split"] == carve_from
        out.loc[m, "split"] = out.loc[m, "group_final"].map(assign).fillna(carve_from)
    return out


def _assert_no_split_overlap(splits_df):
    """Explicit assertions that train, validation and test have no overlapping group IDs or point IDs."""
    for grp_col in ["group_final", "group"]:
        if grp_col in splits_df.columns:
            tr = set(splits_df.loc[splits_df["split"] == "train", grp_col])
            va = set(splits_df.loc[splits_df["split"] == "val", grp_col])
            te = set(splits_df.loc[splits_df["split"] == "test", grp_col])
            assert not (tr & va), f"Leakage detected! Train and Val share {grp_col}: {tr & va}"
            assert not (tr & te), f"Leakage detected! Train and Test share {grp_col}: {tr & te}"
            assert not (va & te), f"Leakage detected! Val and Test share {grp_col}: {va & te}"
    print("PASS: Verified zero group/point overlap across train, val, and test splits.")


def make_or_load_splits(meta, out_dir, force=False):
    path = Path(out_dir) / "splits.csv"
    if path.exists() and not force:
        s = pd.read_csv(path, keep_default_na=False)
        paths_match = sorted(s["path"]) == sorted(meta["path"])
        groups_match = ("group_final" in s.columns and set(s["group_final"]) == set(meta["group_final"]) and
                        "group" in s.columns and set(s["group"]) == set(meta["group"]))
        if paths_match and groups_match:
            print(f"Splits loaded from {path}")
            _assert_no_split_overlap(s)
            return s
        else:
            print(f"Existing {path} is stale (paths_match={paths_match}, groups_match={groups_match}). Regenerating splits...")
    if "split" in meta.columns and meta["split"].astype(str).isin(["train", "val", "test"]).all():
        s = meta.copy()
        s["split"] = s["split"].astype(str)
        if not (s["split"] == "val").any():
            s = group_stratified_split(s, cfg.val_frac / (1 - cfg.test_frac), 0.0, cfg.split_seed, carve_from="train")
        leak = s.groupby("group_final")["split"].nunique()
        if (leak > 1).any():
            print(f"WARNING: {(leak > 1).sum()} groups (duplicates or same site) span the predefined splits — the "
                  "predefined test set leaks. Consider re-splitting by removing the train/test folder structure.")
        print("Using the dataset's predefined train/test split (validation carved from train if missing).")
    else:
        s = group_stratified_split(meta, cfg.val_frac, cfg.test_frac, cfg.split_seed)
    classes = sorted(s["label"].unique())
    s["label_idx"] = s["label"].map({c: i for i, c in enumerate(classes)})
    
    # Invalidate stale label_draws when splits change
    draws_path = Path(out_dir) / "label_draws.json"
    if draws_path.exists():
        try:
            draws_path.unlink()
        except Exception:
            pass

    _assert_no_split_overlap(s)
    s.to_csv(path, index=False)
    print(f"Splits written to {path}")
    return s


def make_or_load_draws(y_train, n_classes, out_dir, seed=1234):
    """Fixed label subsets per regime: k images per class (k-shot) or p% per class; saved so every arm and seed
    is evaluated on exactly the same labelled images."""
    path = Path(out_dir) / "label_draws.json"
    if path.exists():
        d = load_json(path)
        return {k: [np.array(x, dtype=np.int64) for x in v] for k, v in d.items()}
    rng = np.random.default_rng(seed)
    counts = np.bincount(y_train, minlength=n_classes)
    by_class = [np.where(y_train == c)[0] for c in range(n_classes)]
    draws = {}
    for k in cfg.shots:
        if (counts >= k).mean() < 0.8:
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
    save_json({k: [x.tolist() for x in v] for k, v in draws.items()}, path)
    return draws


def regime_order(names):
    shots = sorted([n for n in names if n.endswith("shot")], key=lambda s: int(s[:-4]))
    pcts = sorted([n for n in names if n.endswith("pct")], key=lambda s: int(s[:-3]))
    return shots + pcts'''
            c["source"] = [line + "\n" for line in new_code.split("\n")]
            if c["source"]:
                c["source"][-1] = c["source"][-1].rstrip("\n")
            break

    # 4. Patch Cell 13 (Execute discovery, cache, split, assertions)
    cell13_code = '''if RUN_MODE == "smoke":
    METADATA_CSV_USED = make_synthetic_dataset(OUT / "synthetic_data")
    DATA_ROOT_USED = OUT / "synthetic_data"
else:
    METADATA_CSV_USED = METADATA_CSV
    DATA_ROOT_USED = Path(DATA_ROOT) if DATA_ROOT else (autodetect_data_root() if IS_KAGGLE else None)
    if DATA_ROOT_USED is None and not METADATA_CSV_USED:
        raise FileNotFoundError("Set DATA_ROOT (or METADATA_CSV) in the configuration cell. On Kaggle, attach the dataset via 'Add Data'.")
    print("Data root:", DATA_ROOT_USED)

raw = discover_dataset(DATA_ROOT_USED, METADATA_CSV_USED)
cache_path, cache_meta = build_cache(raw, OUT, cfg.cache_res)
meta_ok, hygiene = hygiene_and_groups(cache_meta, cfg.dedup_hamming)
save_json(hygiene, OUT / "stats" / "hygiene_report.json")
splits = make_or_load_splits(meta_ok, OUT)

# Explicit assertions that train, val, and test have no overlapping group IDs
_assert_no_split_overlap(splits)

# map split rows back to rows of the image cache
row_of_path = {p: i for i, p in enumerate(cache_meta["path"])}
splits["cache_row"] = splits["path"].map(row_of_path)
CLASSES = sorted(splits["label"].unique())
K = len(CLASSES)
ROWS = {s: splits.loc[splits["split"] == s, "cache_row"].to_numpy(np.int64) for s in ["train", "val", "test"]}
Y = {s: splits.loc[splits["split"] == s, "label_idx"].to_numpy(np.int64) for s in ["train", "val", "test"]}
Y_ALL = np.full(len(cache_meta), -1, dtype=np.int64)
for s in ROWS:
    Y_ALL[ROWS[s]] = Y[s]
DRAWS = make_or_load_draws(Y["train"], K, OUT)
REGIMES = regime_order(list(DRAWS))
np.save(OUT / "stats" / "y_test.npy", Y["test"])

print("Hygiene:", hygiene)
summary = splits.groupby(["label", "split"]).size().unstack(fill_value=0)[["train", "val", "test"]]
print(f"{K} classes | train {len(ROWS['train'])} | val {len(ROWS['val'])} | test {len(ROWS['test'])} | chance = {100 / K:.1f}%")
display(summary)
print("Label regimes:", {r: len(DRAWS[r]) for r in REGIMES})'''

    for c in nb["cells"]:
        if c["cell_type"] == "code" and 'raw = discover_dataset(' in "".join(c.get("source", [])) and not 'def discover_dataset' in "".join(c.get("source", [])):
            c["source"] = [line + "\n" for line in cell13_code.split("\n")]
            if c["source"]:
                c["source"][-1] = c["source"][-1].rstrip("\n")
            break

    # 5. Patch Cell 14 (Audit)
    cell14_code = '''import json
import datetime

LUCAS_CROP_NAMES = {
    "B11": "Common wheat",
    "B12": "Durum wheat",
    "B13": "Barley",
    "B14": "Rye",
    "B15": "Oats",
    "B16": "Maize",
    "B21": "Potatoes",
    "B22": "Sugar beet",
    "B31": "Sunflower",
    "B32": "Rapeseed",
    "B33": "Soya",
    "B55": "Temporary grassland"
}

class_split_counts = splits.groupby(["label", "split"]).size().unstack(fill_value=0)[["train", "val", "test"]].to_dict(orient="index")
class_summary = {
    cls: {
        "crop_name": LUCAS_CROP_NAMES.get(cls, "Unknown"),
        "train": counts.get("train", 0),
        "val": counts.get("val", 0),
        "test": counts.get("test", 0),
        "total": sum(counts.values())
    }
    for cls, counts in class_split_counts.items()
}

tr_g = set(splits.loc[splits["split"] == "train", "group_final"])
va_g = set(splits.loc[splits["split"] == "val", "group_final"])
te_g = set(splits.loc[splits["split"] == "test", "group_final"])

audit = {
    "dataset_source": "LUCASvision (JRC)",
    "official_url": "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/DRLL/LUCASvision/",
    "preparation_date": datetime.datetime.now().isoformat(),
    "total_images_discovered": len(raw),
    "total_usable_images": int(meta_ok['ok'].sum()),
    "unreadable_or_corrupt_images": int((~cache_meta['ok']).sum()),
    "number_of_classes": K,
    "class_names": CLASSES,
    "label_mapping": LUCAS_CROP_NAMES,
    "class_counts_after_split": class_summary,
    "images_per_class": meta_ok['label'].value_counts().to_dict(),
    "number_of_unique_groups": int(meta_ok['group_final'].nunique()),
    "images_per_group_summary": meta_ok.groupby('group_final').size().describe().to_dict(),
    "duplicate_images_removed": hygiene['images_in_duplicate_clusters'],
    "missing_labels": int(meta_ok['label'].isna().sum()),
    "missing_group_ids": int(meta_ok['group_final'].isna().sum()),
    "train_images": len(ROWS['train']),
    "val_images": len(ROWS['val']),
    "test_images": len(ROWS['test']),
    "train_groups": int(splits[splits['split'] == 'train']['group_final'].nunique()),
    "val_groups": int(splits[splits['split'] == 'val']['group_final'].nunique()),
    "test_groups": int(splits[splits['split'] == 'test']['group_final'].nunique()),
    "group_overlap_verification": {
        "train_val_overlap": len(tr_g & va_g),
        "train_test_overlap": len(tr_g & te_g),
        "val_test_overlap": len(va_g & te_g),
        "verified_zero_leakage": (len(tr_g & va_g) == 0 and len(tr_g & te_g) == 0 and len(va_g & te_g) == 0)
    }
}

with open(OUT / "stats" / "dataset_audit.json", "w") as f:
    json.dump(audit, f, indent=4)

print("Dataset Audit:")
print(json.dumps(audit, indent=2))'''

    for c in nb["cells"]:
        if c["cell_type"] == "code" and 'dataset_audit.json' in "".join(c.get("source", [])):
            c["source"] = [line + "\n" for line in cell14_code.split("\n")]
            if c["source"]:
                c["source"][-1] = c["source"][-1].rstrip("\n")
            break

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"Successfully patched {nb_path}.")

if __name__ == "__main__":
    patch_notebook("Crop_JEPA_StageA.ipynb")
    patch_notebook("Crop_JEPA_StageA_Final.ipynb")
