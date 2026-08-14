import os
import glob
import re

base_dir = 'TaiLieu/ebooks_Chapters_Vi3'

def clean_chapter(filepath):
    if not os.path.exists(filepath):
        return
        
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Regex to match the injected figure block
    pattern = re.compile(
        r'<div class="figure-container"[^>]*>\s*<img src="../../Figures/Images/figure_\d+\.\d+\.jpg"[^>]*>\s*</div>\n?',
        re.IGNORECASE
    )
    
    new_content = pattern.sub('', content)
    
    if new_content != content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Cleaned {filepath}")
    else:
        print(f"Already clean: {filepath}")

for ch in range(11, 30):
    pattern_vi = os.path.join(base_dir, f"Chapter_{ch}*", f"chapter_{ch}_vi.html")
    for f in glob.glob(pattern_vi):
        clean_chapter(f)
        
    pattern_idx = os.path.join(base_dir, f"Chapter_{ch}*", "index.html")
    for f in glob.glob(pattern_idx):
        clean_chapter(f)

print("Finished cleaning HTML files.")
