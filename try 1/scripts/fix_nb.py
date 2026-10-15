import nbformat

with open('Crop_JEPA_StageA.ipynb', 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

# Find the audit cell and remove it
audit_idx = -1
for i, c in enumerate(nb.cells):
    if c.cell_type == 'code' and 'dataset_audit.json' in c.source:
        audit_idx = i
        break

audit_cell = nb.cells.pop(audit_idx)

# Find the cell that executes `raw = discover_dataset`
exec_idx = -1
for i, c in enumerate(nb.cells):
    if c.cell_type == 'code' and 'raw = discover_dataset(' in c.source and not 'def discover_dataset' in c.source:
        exec_idx = i
        break

# Insert the audit cell after the execution cell
nb.cells.insert(exec_idx + 1, audit_cell)

with open('Crop_JEPA_StageA.ipynb', 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)
