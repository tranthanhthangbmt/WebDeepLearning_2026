import os
import re

docs_dir = 'Chapters'

def process_file(md_path):
    with open(md_path, 'r', encoding='utf-8') as f:
        content = f.read()

    start_idx = content.find('#### ** 💻 Luyện tập **')
    if start_idx == -1:
        return False
        
    next_tab_match = re.search(r'\n#### \*\*|\n<!-- tabs:end -->', content[start_idx + len('#### ** 💻 Luyện tập **'):])
    
    if not next_tab_match:
        return False
        
    end_idx = start_idx + len('#### ** 💻 Luyện tập **') + next_tab_match.start()
    
    # Extract the block
    luyen_tap_block = content[start_idx:end_idx].strip()
    
    # If the next match is already <!-- tabs:end -->, it is already at the end.
    if content[end_idx:].lstrip().startswith('<!-- tabs:end -->'):
        return False
        
    # Remove the block from original position
    new_content = content[:start_idx].rstrip() + '\n\n' + content[end_idx:].lstrip()
    
    tabs_end_idx = new_content.rfind('<!-- tabs:end -->')
    if tabs_end_idx != -1:
        final_content = new_content[:tabs_end_idx].rstrip() + '\n\n' + luyen_tap_block + '\n\n' + new_content[tabs_end_idx:]
    else:
        final_content = new_content.rstrip() + '\n\n' + luyen_tap_block + '\n\n<!-- tabs:end -->\n'
        
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(final_content)
        
    return True

for md_filename in os.listdir(docs_dir):
    if md_filename.startswith('chapter_') and md_filename.endswith('.md'):
        md_path = os.path.join(docs_dir, md_filename)
        if process_file(md_path):
            print(f"Moved tab in {md_filename}")

print("Done moving tabs.")
