"""Из эталонного ноутбука делает студенческую версию.

Удаляет code-ячейки, начинающиеся с `# [SOLUTION]`, и markdown-ячейки, начинающиеся с `<!-- AUTHOR -->`.
Raw-ячейки `# [STUDENT]` превращает в code-ячейки без маркера. Чистит outputs и execution_count.
"""
import sys

import nbformat
from nbformat.v4 import new_code_cell

SOURCE = sys.argv[1] if len(sys.argv) > 1 else "sft_function_calling_hw_solution.ipynb"
TARGET = sys.argv[2] if len(sys.argv) > 2 else "sft_function_calling_hw_student.ipynb"

notebook = nbformat.read(SOURCE, as_version=4)
cells = []
for cell in notebook.cells:
    if cell.cell_type == "code" and cell.source.startswith("# [SOLUTION]"):
        continue
    if cell.cell_type == "markdown" and cell.source.startswith("<!-- AUTHOR -->"):
        continue
    if cell.cell_type == "raw" and cell.source.startswith("# [STUDENT]"):
        cell = new_code_cell(cell.source.split("\n", 1)[1])
    if cell.cell_type == "code":
        cell.outputs = []
        cell.execution_count = None
    cells.append(cell)
notebook.cells = cells

for cell in notebook.cells:
    assert "[SOLUTION]" not in cell.source and "<!-- AUTHOR -->" not in cell.source
nbformat.validate(notebook)
nbformat.write(notebook, TARGET)
print(f"{TARGET}: {len(cells)} ячеек, заглушек: {sum('NotImplementedError' in c.source for c in cells)}")
