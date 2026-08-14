import os
import glob
import re

base_dir = 'TaiLieu/ebooks_Chapters_Vi3'

def process_chapter(filepath, chapter_num):
    if not os.path.exists(filepath):
        return
        
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    inserted_figs = set()
    
    def replacer(match):
        full_match = match.group(0)
        fig_num = match.group(2) # e.g. "13.1"
        
        # Check if the image exists
        img_path = f'TaiLieu/Figures/Images/figure_{fig_num}.jpg'
        if not os.path.exists(img_path):
            return full_match # No image, do nothing
            
        if fig_num in inserted_figs:
            return full_match # Already inserted, keep as is
            
        inserted_figs.add(fig_num)
        
        rel_img = f'../../Figures/Images/figure_{fig_num}.jpg'
            
        img_html = f'''
<div class="figure-container" style="text-align: center; margin: 2rem 0;">
  <img src="{rel_img}" style="max-width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);" alt="Hình {fig_num}">
</div>
'''
        return img_html + full_match
        
    pattern = re.compile(rf'((?:<p>\s*)?<(?:strong|em|b)>Hình\s*({chapter_num}\.\d+))', re.IGNORECASE)
    
    new_content = pattern.sub(replacer, content)
    
    if new_content != content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Updated {filepath} with {len(inserted_figs)} images: {inserted_figs}")
    else:
        print(f"No changes made to {filepath}")

for ch in range(24, 30):
    # check chapter_XX_vi.html first
    pattern_vi = os.path.join(base_dir, f"Chapter_{ch}*", f"chapter_{ch}_vi.html")
    files_vi = glob.glob(pattern_vi)
    
    # check index.html second
    pattern_idx = os.path.join(base_dir, f"Chapter_{ch}*", "index.html")
    files_idx = glob.glob(pattern_idx)
    
    all_files = files_vi + files_idx
    if all_files:
        for f in all_files:
            process_chapter(f, str(ch))
    else:
        print(f"Chapter {ch} HTML not found")
