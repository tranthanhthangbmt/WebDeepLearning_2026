import os
import re

docs_dir = 'Chapters'

def process_file(md_path):
    with open(md_path, 'r', encoding='utf-8') as f:
        content = f.read()

    matches = re.findall(r'(<iframe[^>]*src="TaiLieu/Video/[^>]*></iframe>)', content)
    changed = False
    for match in set(matches):
        if '<div class="video-mobile-wrapper">' not in content[content.find(match)-40:content.find(match)]:
            content = content.replace(match, f'<div class="video-mobile-wrapper">\n{match}\n</div>')
            changed = True

    if changed:
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    return False

count = 0
if os.path.exists(docs_dir):
    for md_filename in os.listdir(docs_dir):
        if md_filename.startswith('chapter_') and md_filename.endswith('.md'):
            md_path = os.path.join(docs_dir, md_filename)
            if process_file(md_path):
                print(f"Wrapped video in {md_filename}")
                count += 1
print(f"Done. Wrapped {count} files.")
