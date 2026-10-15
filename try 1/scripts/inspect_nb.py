import nbformat

with open('Crop_JEPA_StageA.ipynb', 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

# Print first few lines of each cell to find where to insert
for i, cell in enumerate(nb.cells):
    if cell.cell_type == 'code':
        print(f"Cell {i}: {cell.source[:100]}")
