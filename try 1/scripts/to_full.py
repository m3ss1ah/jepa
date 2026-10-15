import nbformat

with open('Crop_JEPA_StageA.ipynb', 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

for c in nb.cells:
    if c.cell_type == 'code' and 'RUN_MODE = ' in c.source:
        c.source = c.source.replace('RUN_MODE = "benchmark"', 'RUN_MODE = "full"')

with open('Crop_JEPA_StageA.ipynb', 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)
