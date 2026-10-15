import nbformat

with open('Crop_JEPA_StageA.ipynb', 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

setup_code = """
import urllib.request, re, random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

def setup_lucas_data(dest_dir, limit_per_class=100, seed=42):
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    meta_path = dest_dir / "metadata.csv"
    
    if meta_path.exists():
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
    def download_file(args):
        cls, filename = args
        url = f"{base_url}{cls}/{filename}"
        filepath = dest_dir / cls / filename
        if not filepath.exists():
            try:
                urllib.request.urlretrieve(url, filepath)
            except Exception as e:
                return None
        parts = filename.split('_')
        group_id = parts[1] if len(parts) > 1 else filename
        return {"path": f"{cls}/{filename}", "label": cls, "group": group_id}

    tasks = [(c, f) for c, files in all_links.items() for f in files]
    print(f"Downloading {len(tasks)} images from LUCASvision...")
    with ThreadPoolExecutor(20) as ex:
        results = list(ex.map(download_file, tasks))
        
    for r in results:
        if r:
            rows.append(r)
            
    df = pd.DataFrame(rows)
    df.to_csv(meta_path, index=False)
    print(f"Prepared LUCAS dataset with {len(df)} images.")

if RUN_MODE != "smoke":
    setup_lucas_data("./lucas_data", limit_per_class=100, seed=cfg.split_seed if 'cfg' in globals() else 42)
"""

audit_code = """
import json

audit = {
    "dataset_source": "LUCASvision (JRC)",
    "official_url": "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/DRLL/LUCASvision/",
    "preparation_date": datetime.datetime.now().isoformat(),
    "total_images_discovered": len(raw),
    "total_usable_images": int(meta_ok['ok'].sum()),
    "unreadable_or_corrupt_images": int((~cache_meta['ok']).sum()),
    "number_of_classes": K,
    "class_names": CLASSES,
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
    "test_groups": int(splits[splits['split'] == 'test']['group_final'].nunique())
}

with open(OUT / "stats" / "dataset_audit.json", "w") as f:
    json.dump(audit, f, indent=4)

print("Dataset Audit:")
print(json.dumps(audit, indent=2))
"""

for c in nb.cells:
    if c.cell_type == 'code' and 'RUN_MODE = ' in c.source:
        c.source = c.source.replace('RUN_MODE = "smoke"', 'RUN_MODE = "benchmark"')
        c.source = c.source.replace('DATA_ROOT = None', 'DATA_ROOT = "./lucas_data"\nMETADATA_CSV = "./lucas_data/metadata.csv"')
        break

# Insert setup code before the discover_dataset cell
insert_setup_idx = -1
insert_audit_idx = -1
for i, c in enumerate(nb.cells):
    if c.cell_type == 'code' and 'discover_dataset(' in c.source:
        insert_setup_idx = i
        insert_audit_idx = i + 1
        break

if insert_setup_idx != -1:
    nb.cells.insert(insert_setup_idx, nbformat.v4.new_code_cell(setup_code))
    # Note: after insertion, the discover_dataset cell moves to insert_setup_idx + 1
    # So we want to insert audit AFTER discover_dataset, which means insert_setup_idx + 2
    nb.cells.insert(insert_setup_idx + 2, nbformat.v4.new_code_cell(audit_code))

with open('Crop_JEPA_StageA.ipynb', 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)
