import nbformat as nbf

# Read the existing notebook
with open('randeng_pegasus_finetuning.ipynb', 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

# Update the data path in the notebook
for cell in nb.cells:
    if cell.cell_type == 'code':
        cell.source = cell.source.replace("'../data/lcsts/lcsts_data.json'", "'data/lcsts/lcsts_data.json'")

# Write the updated notebook
with open('randeng_pegasus_finetuning.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
