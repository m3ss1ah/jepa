#!/usr/bin/env python
"""Download ALL LUCASVision images (12 crop classes) from the JRC server into CJ_DATA_ROOT.
Uses the notebook's own downloader: resumable, atomic (no half-written files), retried, verified.
Re-running only fetches what is missing. Exit code 0 = dataset complete."""
import os, sys, re, json, time, random, datetime, urllib.request, urllib.parse, shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd

DATA_ROOT = os.environ.get("CJ_DATA_ROOT", "./lucas_data")
CACHE_RES = int(os.environ.get("CJ_CACHE_RES", 192))
DOWNLOAD_THREADS = 20
ALLOW_INCOMPLETE_DOWNLOAD = False


def _retry_io(fn, tries=8, wait=0.5):
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
        json.dump(obj, f, indent=2, default=str)
    _atomic_replace(tmp, path)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_csv(df, path, **kw):
    path = Path(path)
    tmp = _tmp_path(path)
    df.to_csv(tmp, index=kw.pop("index", False), **kw)
    _atomic_replace(tmp, path)


def free_disk_gib(path="."):
    p = Path(path).resolve()
    while not p.exists() and p != p.parent:
        p = p.parent
    return shutil.disk_usage(p).free / 1024**3


# ---- verbatim from notebook cell 12 ----
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




if __name__ == "__main__":
    print(f"Downloading LUCASVision into {Path(DATA_ROOT).resolve()} (free disk {free_disk_gib(DATA_ROOT):.1f} GiB)")
    last = None
    for attempt in range(1, 6):
        try:
            selected, report = setup_lucas_data(DATA_ROOT, None, DOWNLOAD_THREADS, seed=42, cache_res=CACHE_RES)
            n_listed = report["listed_total"]
            n_disk = sum(v["on_disk"] for v in report["per_class"].values())
            print(f"COMPLETE: {n_disk}/{n_listed} images on disk, {report['failed']} failures.")
            sys.exit(0 if n_disk == n_listed and report["failed"] == 0 else 2)
        except Exception as e:
            last = e
            print(f"[attempt {attempt}/5] {type(e).__name__}: {e}")
            if "Not enough disk space" in str(e):
                break
            time.sleep(30 * attempt)
    print("DOWNLOAD FAILED:", last)
    sys.exit(1)
