import os
import re

global_figures_path = r'd:\DongAUniversity\TÀI LIỆU DẠY HỌC_2024-2025\Trí tuệ nhân tạo_UDA_2025\webTTNT_2026\TaiLieu\Figures\global-figures.htm'
html_dir = r'd:\DongAUniversity\TÀI LIỆU DẠY HỌC_2024-2025\Trí tuệ nhân tạo_UDA_2025\webTTNT_2026\TaiLieu\ebooks_Chapters_Vi'

# Step 1: Extract mapping of Figure X.Y -> image filename
with open(global_figures_path, 'r', encoding='utf-8', errors='ignore') as f:
    global_html = f.read()

# Find all Figure X.Y
matches = re.finditer(r'(?:Figure|Hình)\s+(\d+)\.(\d+)', global_html, flags=re.IGNORECASE)
figures_map = {}

for m in matches:
    ch_num = int(m.group(1))
    fig_num = int(m.group(2))
    key = f'{ch_num}.{fig_num}'
    if key in figures_map:
        continue
    
    # search backwards for the nearest <img>
    search_area = global_html[:m.start()]
    img_matches = list(re.finditer(r'<img[^>]*src=[\x22\x27]([^>]+)[\x22\x27][^>]*>', search_area, flags=re.IGNORECASE))
    
    # Check the last few images before the caption
    for img in reversed(img_matches):
        src = img.group(1)
        if 'global-figures_files' in src:
            filename = src.split('/')[-1]
            # avoid tiny images like spacers
            if 'image003.gif' in filename or 'image021.gif' in filename or 'image025.gif' in filename:
                pass
            figures_map[key] = filename
            break

print(f"Found {len(figures_map)} figures mapped to images.")

# Step 2: Insert into chapter HTMLs
for i in range(1, 11):
    filepath = os.path.join(html_dir, f'chapter_{i:02d}_vi.html')
    if not os.path.exists(filepath):
        continue
        
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # FIRST, strip out ALL previously injected figure-containers
    # We use a regex that matches from <div class="figure-container"... to </div>
    # Using re.DOTALL to match across newlines
    content = re.sub(r'\n*<div class="figure-container".*?</div>\n*', '\n', content, flags=re.DOTALL)
    
    # THEN, find the TRUE captions: paragraph start followed by Hình X.Y
    # \d+ matches the digits. (?:Hình|Figure) matches either.
    matches = list(re.finditer(r'<p[^>]*>(?:<[^>]+>|\s)*?(?:Hình|Figure)\s+(\d+)\.(\d+)', content, flags=re.IGNORECASE))
    
    # Iterate backwards so string modifications don't mess up indices
    for m in reversed(matches):
        ch_num = int(m.group(1))
        fig_num = int(m.group(2))
        key = f'{ch_num}.{fig_num}'
        
        if key in figures_map:
            img_filename = figures_map[key]
            img_tag = f'\n<div class="figure-container" style="text-align: center; margin: 20px 0;">\n  <img src="../Figures/global-figures_files/{img_filename}" style="max-width: 100%; border: 1px solid #ccc; padding: 5px;">\n</div>\n'
            
            content = content[:m.start()] + img_tag + content[m.start():]

    if original_content != content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Fixed chapter_{i:02d}_vi.html")
    else:
        print(f"No changes for chapter_{i:02d}_vi.html")
