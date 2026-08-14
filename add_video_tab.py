import os
import re

d = 'chapters'
for f in os.listdir(d):
    if not (f.startswith('chapter_') and f.endswith('.md')): continue
    ch_match = re.search(r'chapter_(\d+)_', f)
    if not ch_match: continue
    ch_num_str = ch_match.group(1)
    
    path = os.path.join(d, f)
    with open(path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    if '#### **Video**' in content:
        print(f'Already has Video tab: {f}')
        continue
    
    pattern = r'(#### \*\*Slide\*\*.*?)(?=\n#### \*\*)'
    replacement = r'\1\n\n#### **Video**\n\n<div class="pdf-container" style="margin-bottom: 20px;">\n  <iframe src="video/Chapter' + ch_num_str + r'/index.html" width="100%" height="100%"></iframe>\n</div>\n'
    
    new_content, count = re.subn(pattern, replacement, content, flags=re.DOTALL, count=1)
    
    if count > 0:
        with open(path, 'w', encoding='utf-8') as file:
            file.write(new_content)
        print(f'Updated {f}')
    else:
        print(f'Failed to update {f}')
