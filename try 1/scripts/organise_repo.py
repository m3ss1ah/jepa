import shutil, os
from pathlib import Path

root = Path('.')

# 1. Create clean folder structure
for d in ['scripts', 'notebooks', 'docs']:
    (root / d).mkdir(exist_ok=True)

# 2. Move scratch scripts to scripts/
scripts = ['edit_nb_lucas.py', 'fetch_lucas.py', 'fetch_lucas2.py',
           'fix_nb.py', 'inspect_nb.py', 'to_full.py', 'notebook_code.py']
for s in scripts:
    p = root / s
    if p.exists():
        shutil.move(str(p), str(root / 'scripts' / s))

# 3. Move all .docx files to docs/
for f in root.glob('*.docx'):
    shutil.move(str(f), str(root / 'docs' / f.name))

# 4. Move intermediate/executed notebooks to notebooks/
for nb in ['Crop_JEPA_StageA_Executed.ipynb',
           'Crop_JEPA_StageA_Benchmark.ipynb',
           'Crop_JEPA_StageA_Full_Resumed.ipynb']:
    p = root / nb
    if p.exists():
        shutil.move(str(p), str(root / 'notebooks' / nb))

# Also move the organise script itself to scripts
# (don't recurse into outputs/lucas_data)
print("Repo organised successfully.")
print("Directories created: scripts/, notebooks/, docs/")
