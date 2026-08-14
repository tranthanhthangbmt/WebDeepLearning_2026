import os
import re

# Paths to the chapters
chapter_11_path = 'TaiLieu/ebooks_Chapters_Vi/chapter_11_vi.html'
chapter_12_path = 'TaiLieu/ebooks_Chapters_Vi3/Chapter_12_Quantifying uncertainty/chapter_12_vi.html'

def process_chapter(filepath, chapter_num):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return
        
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    inserted_figs = set()
    
    def replacer(match):
        full_match = match.group(0)
        fig_num = match.group(2) # match.group(2) is the number, like '11.1'
        
        # Check if the image exists
        img_path = f'TaiLieu/Figures/Images/figure_{fig_num}.jpg'
        if not os.path.exists(img_path):
            return full_match # No image, do nothing
            
        if fig_num in inserted_figs:
            return full_match # Already inserted, keep as is
            
        inserted_figs.add(fig_num)
        
        if "ebooks_Chapters_Vi3" in filepath:
            rel_img = f'../../Figures/Images/figure_{fig_num}.jpg'
        else:
            rel_img = f'../Figures/Images/figure_{fig_num}.jpg'
            
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


process_chapter(chapter_11_path, "11")
process_chapter(chapter_12_path, "12")
