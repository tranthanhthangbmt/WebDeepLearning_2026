import os
import glob
import re

def inject_chapter(filepath, chapter_num):
    if not os.path.exists(filepath):
        print(f"Not found: {filepath}")
        return
        
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    inserted_figs = set()
    
    def replacer(match):
        full_match = match.group(0)
        fig_num = match.group(3) # Group 3 is num
        
        # Check if the image exists
        img_path = f'TaiLieu/Figures/Images/figure_{fig_num}.jpg'
        if not os.path.exists(img_path):
            return full_match # No image, do nothing
            
        if fig_num in inserted_figs:
            return full_match # Already inserted, keep as is
            
        inserted_figs.add(fig_num)
        
        # Calculate relative path
        if 'ebooks_Chapters_Vi3' in filepath:
            rel_img = f'../../Figures/Images/figure_{fig_num}.jpg'
        else:
            rel_img = f'../Figures/Images/figure_{fig_num}.jpg'
            
        img_html = f'''
<div class="figure-container" style="text-align: center; margin: 2rem 0;">
  <img src="{rel_img}" style="max-width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);" alt="Hình {fig_num}">
</div>
'''
        return img_html + full_match
        
    # Regex: optional <p>, strong/em/b tag, then "Hình" or "Figure", then number.
    pattern = re.compile(rf'((?:<p>\s*)?<(?:strong|em|b|i)>(Hình|Figure)\s*({chapter_num}\.\d+))', re.IGNORECASE)
    
    new_content = pattern.sub(replacer, content)
    
    if new_content != content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Updated {filepath} with {len(inserted_figs)} images")
    else:
        print(f"No changes made to {filepath}")

# Chapter 11
inject_chapter('TaiLieu/ebooks_Chapters_Vi/chapter_11_vi.html', '11')

# Chapters 12 to 23
base_dir = 'TaiLieu/ebooks_Chapters_Vi3'
for ch in range(12, 24):
    pattern_vi = os.path.join(base_dir, f"Chapter_{ch}*", f"chapter_{ch}_vi.html")
    files_vi = glob.glob(pattern_vi)
    if files_vi:
        for f in files_vi:
            inject_chapter(f, str(ch))
    else:
        print(f"Chapter {ch} HTML not found")
