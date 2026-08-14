import fitz
import re

pdf_path = 'TaiLieu/Figures/global-figures.pdf'
pdf = fitz.open(pdf_path)

multi_fig_pages = []

for i in range(len(pdf)):
    page = pdf[i]
    text = page.get_text("text")
    # Find all "Figure X.Y"
    matches = re.findall(r'Figure\s+(\d+\.\d+)', text)
    # Deduplicate while preserving order
    seen = set()
    figs = []
    for m in matches:
        if m not in seen:
            seen.add(m)
            figs.append(m)
            
    if len(figs) > 1:
        # Check their positions
        pos = []
        for f in figs:
            rects = page.search_for(f"Figure {f}")
            if rects:
                pos.append((f, rects[0]))
        multi_fig_pages.append((i, pos))

print(f"Found {len(multi_fig_pages)} pages with multiple figures:")
for p, pos in multi_fig_pages:
    print(f"Page {p+1}:")
    for f, r in pos:
        print(f"  {f} at Y={r.y0:.1f} to {r.y1:.1f}, X={r.x0:.1f}")

pdf.close()
