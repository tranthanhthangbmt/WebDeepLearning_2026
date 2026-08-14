import os
import re

filepath = r'd:\DongAUniversity\TÀI LIỆU DẠY HỌC_2024-2025\Trí tuệ nhân tạo_UDA_2025\webTTNT_2026\TaiLieu\Stuart J. Russell, Peter Norvig - Artificial Intelligence_ A Modern Approach, Global Edition-Pearson (2021)_Vietnamses.htm'
output_dir = r'd:\DongAUniversity\TÀI LIỆU DẠY HỌC_2024-2025\Trí tuệ nhân tạo_UDA_2025\webTTNT_2026\TaiLieu\ebooks_Chapters_Vi'

with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
    html_content = f.read()

style_match = re.search(r'(?i)<head>.*?(<style.*?</style>).*?</head>', html_content, flags=re.DOTALL)
style_content = style_match.group(1) if style_match else ''

matches = list(re.finditer(r'<h[123][^>]*>(.*?)</h[123]>', html_content, flags=re.IGNORECASE | re.DOTALL))
ch_indexes = {}

for m in matches:
    text = re.sub(r'<[^>]+>', '', m.group(1)).strip()
    match_ch = re.search(r'^Chương\s+(\d+)', text, flags=re.IGNORECASE)
    if match_ch:
        ch_num = int(match_ch.group(1))
        # only keep the first occurrence of each chapter heading (in case it appears in TOC, which we can skip by index > 5%)
        if ch_num not in ch_indexes and m.start() > len(html_content) * 0.05:
            ch_indexes[ch_num] = m.start()

found_ch = sorted(ch_indexes.keys())
print(f'Processing {len(found_ch)} chapters.')
for ch in found_ch[:10]:
    print(f'Ch {ch} at idx {ch_indexes[ch]}')

for i, ch_num in enumerate(found_ch):
    if ch_num > 10:
        continue
        
    start_search = ch_indexes[ch_num]
    # The heading tag itself starts at `start_search`, we should just use this exactly!
    # Wait, there might be some <p> immediately before it that belongs to it? No, usually a heading is the start.
    # However, to be safe, maybe we should just split EXACTLY at `start_search`? Yes!
    start_idx = start_search
    
    if i < len(found_ch) - 1:
        end_idx = ch_indexes[found_ch[i+1]]
    else:
        end_idx = len(html_content)
        
    chapter_html = html_content[start_idx:end_idx]
    full_html = '<html><head><meta charset=\"utf-8\">\n' + style_content + '\n</head><body>\n' + chapter_html + '\n</body></html>'
    
    out_filename = os.path.join(output_dir, f'chapter_{ch_num:02d}_vi.html')
    with open(out_filename, 'w', encoding='utf-8') as f:
        f.write(full_html)
    print(f'Saved Chapter {ch_num} to {out_filename}')
    
print('Done!')
