import fitz
import re
import os

pdf_path = 'TaiLieu/Figures/global-figures.pdf'
out_dir = 'TaiLieu/Figures/Images'
os.makedirs(out_dir, exist_ok=True)

pdf = fitz.open(pdf_path)
zoom = 2.0
mat = fitz.Matrix(zoom, zoom)

for i in range(len(pdf)):
    page = pdf[i]
    text = page.get_text("text")
    blocks = page.get_text("blocks")
    seen = set()
    figs = []
    pos = []
    
    for b in blocks:
        text_block = b[4]
        # Match only at the very beginning of the block
        m = re.match(r'^Figure\s+(\d+\.\d+)\b', text_block)
        if m:
            f_num = m.group(1)
            if f_num not in seen:
                seen.add(f_num)
                figs.append(f_num)
                pos.append((f_num, fitz.Rect(b[:4])))
            
    if not figs:
        continue
        
    if len(figs) == 1:
        # Only 1 figure, save the whole page
        f_num = figs[0]
        pix = page.get_pixmap(matrix=mat, alpha=False)
        out_path = os.path.join(out_dir, f'figure_{f_num}.jpg')
        pix.save(out_path)
        print(f"Extracted {f_num} (Full Page) to {out_path}")
        continue
        
    # Sort by Y-coordinate
    pos = sorted(pos, key=lambda x: x[1].y1)
    
    slices = []
    current_slice_figs = [pos[0][0]]
    current_slice_y_end = pos[0][1].y1
    
    for j in range(1, len(pos)):
        f_num, rect = pos[j]
        if abs(rect.y1 - current_slice_y_end) < 50:
            # Side-by-side figure
            current_slice_figs.append(f_num)
            current_slice_y_end = max(current_slice_y_end, rect.y1)
        else:
            # Finish previous slice
            slices.append((current_slice_figs, current_slice_y_end + 35))
            current_slice_figs = [f_num]
            current_slice_y_end = rect.y1
            
    # Add the last slice (goes to bottom of page)
    slices.append((current_slice_figs, page.rect.height))
    
    y_start = 0
    for slice_figs, y_end in slices:
        # Create clip rect
        clip_rect = fitz.Rect(0, y_start, page.rect.width, y_end)
        pix = page.get_pixmap(matrix=mat, clip=clip_rect, alpha=False)
        
        for f_num in slice_figs:
            out_path = os.path.join(out_dir, f'figure_{f_num}.jpg')
            pix.save(out_path)
            print(f"Extracted {f_num} (Sliced Y:{y_start:.1f}-{y_end:.1f}) to {out_path}")
            
        y_start = y_end

pdf.close()
print("Done extracting and slicing all figures!")
