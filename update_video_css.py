import os
import re

video_dir = 'TaiLieu/Video'

def process_file(html_path):
    with open(html_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # We want to replace 'padding: 0 10px;' with 'padding: 0 60px; justify-content: center;' inside #control-bar
    # Let's find #control-bar block
    control_bar_match = re.search(r'#control-bar\s*{[^}]+}', content)
    if not control_bar_match:
        return False
        
    old_block = control_bar_match.group(0)
    if 'padding: 0 60px;' in old_block or 'padding: 0 80px;' in old_block:
        return False # Already updated
        
    new_block = old_block.replace('padding: 0 10px;', 'padding: 0 80px; justify-content: center;')
    
    if new_block != old_block:
        new_content = content.replace(old_block, new_block)
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        return True
    return False

count = 0
if os.path.exists(video_dir):
    for chapter_folder in os.listdir(video_dir):
        if chapter_folder.startswith('Chapter_'):
            html_path = os.path.join(video_dir, chapter_folder, 'index.html')
            if os.path.exists(html_path):
                if process_file(html_path):
                    print(f"Updated {html_path}")
                    count += 1
print(f"Done. Updated {count} files.")
