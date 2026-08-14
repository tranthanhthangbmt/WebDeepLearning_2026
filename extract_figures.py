import fitz  # PyMuPDF
import json
import os

with open('figure_mapping.json', 'r') as f:
    mapping = json.load(f)

pdf_path = 'TaiLieu/Figures/global-figures.pdf'
out_dir = 'TaiLieu/Figures/Images'

pdf = fitz.open(pdf_path)

# Extract chapters 13 to 29
target_chapters = [f"{i}." for i in range(13, 30)]

for fig_num, page_index in mapping.items():
    if any(fig_num.startswith(ch) for ch in target_chapters):
        # page_index is 1-based from our mapping, fitz is 0-based
        page = pdf[page_index - 1]
        
        # Matrix to increase resolution. zoom=2 means ~144 DPI
        zoom = 2.0
        mat = fitz.Matrix(zoom, zoom)
        
        pix = page.get_pixmap(matrix=mat, alpha=False)
        out_path = os.path.join(out_dir, f'figure_{fig_num}.jpg')
        pix.save(out_path)
        print(f"Extracted {fig_num} to {out_path}")

pdf.close()
print("Done extracting chapters 13 to 29.")
