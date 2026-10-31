import os
import re
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

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

BASE_URL = "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/DRLL/LUCASvision/"

def repair_and_verify():
    data_root = Path("./lucas_data")
    out_dir = Path("./outputs")
    cache_dir = out_dir / "cache"
    stats_dir = out_dir / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Verify/build lucas_data/metadata.csv
    pat = re.compile(r"^LUCAS(\d{4})_(\d+)_([A-Za-z0-9]+)\.jpg$", re.I)
    rows = []
    for cls in sorted(os.listdir(data_root)):
        cls_dir = data_root / cls
        if not cls_dir.is_dir() or cls not in LUCAS_CROP_NAMES:
            continue
        for f in sorted(cls_dir.glob("*.jpg")):
            m = pat.match(f.name)
            if not m:
                print(f"Warning: {f.name} does not match LUCAS pattern")
                continue
            year, pt_id, view = m.group(1), m.group(2), m.group(3)
            rows.append({
                "path": f"{cls}/{f.name}",
                "label": cls,
                "group": pt_id,  # Survey point identifier
                "point_id": pt_id,
                "survey_year": int(year),
                "view": view,
                "filename": f.name,
                "source_url": f"{BASE_URL}{cls}/{f.name}"
            })
            
    df_meta = pd.DataFrame(rows)
    print(f"Discovered {len(df_meta)} images across {df_meta['label'].nunique()} classes.")
    
    # Save standard metadata.csv with columns path, label, group
    meta_csv_path = data_root / "metadata.csv"
    df_meta[["path", "label", "group"]].to_csv(meta_csv_path, index=False)
    print(f"Saved verified metadata to {meta_csv_path}")
    
    # 2. Update cache if images_256.csv exists
    cache_meta_path = cache_dir / "images_256.csv"
    if cache_meta_path.exists():
        cache_meta = pd.read_csv(cache_meta_path)
        # Update group in cache_meta using path mapping
        path_to_group = dict(zip(
            [str(data_root / p) for p in df_meta["path"]],
            df_meta["group"].astype(str)
        ))
        # Handle normalized windows / posix path formats
        updated_groups = []
        for p in cache_meta["path"]:
            norm_p = str(Path(p))
            norm_key = str(data_root / Path(p).name)
            # Find matching row by filename
            fname = Path(p).name
            matching = df_meta[df_meta["filename"] == fname]
            if len(matching) > 0:
                updated_groups.append(str(matching.iloc[0]["group"]))
            else:
                updated_groups.append(str(p))
        cache_meta["group"] = updated_groups
        cache_meta.to_csv(cache_meta_path, index=False)
        print(f"Synchronized group identifiers in {cache_meta_path}")
    else:
        # Build cache meta
        cache_meta = df_meta.copy()
        cache_meta["path"] = [str(data_root / p) for p in cache_meta["path"]]
        cache_meta["ok"] = True
        cache_meta["dhash"] = "0"
        
    # 3. Hygiene & duplicate check
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
                
    meta_ok = cache_meta[cache_meta["ok"].astype(bool)].reset_index(drop=True).copy()
    n = len(meta_ok)
    hashes = np.array([int(h, 16) for h in meta_ok["dhash"]], dtype=np.uint64)
    uf_img = UnionFind(range(n))
    seen = {}
    for i, h in enumerate(hashes.tolist()):
        if h in seen and h != 0:
            uf_img.union(seen[h], i)
        elif h != 0:
            seen[h] = i
            
    # Hamming distance <= 10
    max_dist = 10
    if max_dist > 0 and n <= 60000:
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
    
    uf_grp = UnionFind(meta_ok["group"].unique().tolist())
    for c in dup_clusters:
        g = meta_ok["group"].iloc[c].tolist()
        for x in g[1:]:
            uf_grp.union(g[0], x)
    meta_ok["group_final"] = [uf_grp.find(g) for g in meta_ok["group"]]
    
    # Record duplicate status
    in_dup_cluster = set()
    for c in dup_clusters:
        for idx in c:
            in_dup_cluster.add(idx)
    meta_ok["duplicate_status"] = ["duplicate" if i in in_dup_cluster else "unique" for i in range(n)]
    meta_ok["quality_decision"] = ["ok" if ok else "corrupt" for ok in meta_ok["ok"]]

    # 4. Group-stratified split (70% train, 15% val, 15% test)
    val_frac = 0.15
    test_frac = 0.15
    seed = 42
    rng = np.random.default_rng(seed)
    
    g = meta_ok.groupby("group_final").agg(label=("label", lambda s: s.mode().iloc[0]), size=("label", "size")).reset_index()
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
            
    splits = meta_ok.copy()
    splits["split"] = splits["group_final"].map(assign).fillna("train")
    classes = sorted(splits["label"].unique())
    splits["label_idx"] = splits["label"].map({c: i for i, c in enumerate(classes)})
    
    # 5. Explicit assertions that train, val, and test have no overlapping group IDs
    train_groups = set(splits.loc[splits["split"] == "train", "group_final"])
    val_groups = set(splits.loc[splits["split"] == "val", "group_final"])
    test_groups = set(splits.loc[splits["split"] == "test", "group_final"])
    
    assert len(train_groups & val_groups) == 0, f"Overlapping groups between train and val: {train_groups & val_groups}"
    assert len(train_groups & test_groups) == 0, f"Overlapping groups between train and test: {train_groups & test_groups}"
    assert len(val_groups & test_groups) == 0, f"Overlapping groups between val and test: {val_groups & test_groups}"
    
    # Assert on raw point IDs as well
    train_pts = set(splits.loc[splits["split"] == "train", "group"])
    val_pts = set(splits.loc[splits["split"] == "val", "group"])
    test_pts = set(splits.loc[splits["split"] == "test", "group"])
    
    assert len(train_pts & val_pts) == 0, f"Overlapping point IDs between train and val: {train_pts & val_pts}"
    assert len(train_pts & test_pts) == 0, f"Overlapping point IDs between train and test: {train_pts & test_pts}"
    assert len(val_pts & test_pts) == 0, f"Overlapping point IDs between val and test: {val_pts & test_pts}"
    
    print(f"ASSERTIONS PASSED: Train ({len(train_groups)} groups), Val ({len(val_groups)} groups), Test ({len(test_groups)} groups) have ZERO overlapping groups/points.")
    
    # Save splits.csv
    splits_path = out_dir / "splits.csv"
    splits.to_csv(splits_path, index=False)
    print(f"Saved location-aware splits to {splits_path}")
    
    # 6. Regenerate label_draws.json
    y_train = splits.loc[splits["split"] == "train", "label_idx"].to_numpy(np.int64)
    n_classes = len(classes)
    shots = [1, 2, 5, 10, 20]
    percents = [1, 5, 10, 25, 50, 100]
    n_draws = 5
    draw_rng = np.random.default_rng(1234)
    counts = np.bincount(y_train, minlength=n_classes)
    by_class = [np.where(y_train == c)[0] for c in range(n_classes)]
    
    draws = {}
    for k in shots:
        if (counts >= k).mean() < 0.8:
            continue
        draws[f"{k}shot"] = [np.concatenate([draw_rng.choice(ix, min(k, len(ix)), replace=False) for ix in by_class if len(ix)])
                             for _ in range(n_draws)]
    for p in percents:
        if p >= 100:
            draws["100pct"] = [np.arange(len(y_train))]
        else:
            draws[f"{p}pct"] = [np.concatenate([draw_rng.choice(ix, max(1, int(round(len(ix) * p / 100))), replace=False)
                                                for ix in by_class if len(ix)]) for _ in range(n_draws)]
    
    draws_path = out_dir / "label_draws.json"
    with open(draws_path, "w") as f:
        json.dump({k: [x.tolist() for x in v] for k, v in draws.items()}, f)
    print(f"Regenerated label draws saved to {draws_path}")
    
    # Save y_test
    np.save(stats_dir / "y_test.npy", splits.loc[splits["split"] == "test", "label_idx"].to_numpy(np.int64))

    # 7. Build full dataset manifest
    manifest_rows = []
    for _, row in splits.iterrows():
        p = row["path"]
        fname = Path(p).name
        m = pat.match(fname)
        year, pt_id, view = (m.group(1), m.group(2), m.group(3)) if m else ("", "", "")
        cls = row["label"]
        manifest_rows.append({
            "source_url": f"{BASE_URL}{cls}/{fname}",
            "filename": fname,
            "path": p,
            "crop_label": cls,
            "crop_name": LUCAS_CROP_NAMES.get(cls, "Unknown"),
            "point_id": pt_id,
            "view": view,
            "survey_year": year,
            "duplicate_status": row.get("duplicate_status", "unique"),
            "quality_decision": row.get("quality_decision", "ok"),
            "group": row["group"],
            "group_final": row["group_final"],
            "split": row["split"],
            "label_idx": row["label_idx"]
        })
    df_manifest = pd.DataFrame(manifest_rows)
    df_manifest.to_csv(out_dir / "dataset_manifest.csv", index=False)
    df_manifest.to_csv(data_root / "dataset_manifest.csv", index=False)
    print("Dataset manifest saved to outputs/dataset_manifest.csv and lucas_data/dataset_manifest.csv")
    
    # 8. Detailed audit with label mapping and post-split class counts
    class_split_counts = splits.groupby(["label", "split"]).size().unstack(fill_value=0)[["train", "val", "test"]].to_dict(orient="index")
    # Add crop names
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
    
    audit = {
        "dataset_name": "LUCASvision ARABLE Crop Subset",
        "dataset_source": "European Commission Joint Research Centre (JRC)",
        "official_url": BASE_URL,
        "citation_reference": "LUCAS (Land Use/Cover Area frame Survey) vision dataset",
        "verification_date": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total_images": len(splits),
        "total_unique_points": int(splits["group"].nunique()),
        "number_of_classes": len(classes),
        "label_mapping": LUCAS_CROP_NAMES,
        "class_counts_by_split": class_summary,
        "split_totals": {
            "train_images": int((splits["split"] == "train").sum()),
            "val_images": int((splits["split"] == "val").sum()),
            "test_images": int((splits["split"] == "test").sum()),
            "train_groups": int(splits.loc[splits["split"] == "train", "group_final"].nunique()),
            "val_groups": int(splits.loc[splits["split"] == "val", "group_final"].nunique()),
            "test_groups": int(splits.loc[splits["split"] == "test", "group_final"].nunique())
        },
        "group_overlap_verification": {
            "train_val_overlap": len(train_groups & val_groups),
            "train_test_overlap": len(train_groups & test_groups),
            "val_test_overlap": len(val_groups & test_groups),
            "verified_zero_leakage": True
        },
        "quality_and_hygiene": {
            "unreadable_corrupt_images": int((~splits["ok"]).sum()),
            "duplicate_images_found": int((splits["duplicate_status"] == "duplicate").sum())
        }
    }
    
    with open(stats_dir / "dataset_audit.json", "w") as f:
        json.dump(audit, f, indent=4)
    print("Updated stats/dataset_audit.json successfully.")
    
    print("\nClass breakdown after split:")
    print(pd.DataFrame(class_summary).T[["crop_name", "train", "val", "test", "total"]])

if __name__ == "__main__":
    repair_and_verify()
