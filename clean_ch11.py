import re

filepath = r'd:\DongAUniversity\TÀI LIỆU DẠY HỌC_2024-2025\Trí tuệ nhân tạo_UDA_2025\webTTNT_2026\TaiLieu\ebooks_Chapters_Vi\chapter_11_vi.html'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

pattern = re.compile(
    r'<div class="figure-container"[^>]*>\s*<img src="\.\./Figures/Images/figure_\d+\.\d+\.jpg"[^>]*>\s*</div>\n?',
    re.IGNORECASE
)

new_content = pattern.sub('', content)

if new_content != content:
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(new_content)
    print('Cleaned Chapter 11')
else:
    print('Already clean: Chapter 11')
