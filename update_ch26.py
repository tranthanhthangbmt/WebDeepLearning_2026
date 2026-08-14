import os
import re

md_path = r'D:\DongAUniversity\TÀI LIỆU DẠY HỌC_2024-2025\Trí tuệ nhân tạo_UDA_2025\webTTNT_2026\chapters\chapter_26_robotics.md'
html_path = r'D:\DongAUniversity\TÀI LIỆU DẠY HỌC_2024-2025\Trí tuệ nhân tạo_UDA_2025\webTTNT_2026\TaiLieu\ebooks_Chapters_Vi3\Chapter_26_Robotics\chapter_26_vi.html'

with open(html_path, 'r', encoding='utf-8') as f:
    html_content = f.read()

with open(md_path, 'r', encoding='utf-8') as f:
    md_content = f.read()

pattern = re.compile(r'(#### \*\*Tiếng Việt\*\*).*?(#### \*\*Tiếng Anh\*\*)', re.DOTALL)

def replacer(match):
    return match.group(1) + '\n\n' + html_content + '\n\n' + match.group(2)

new_md_content = pattern.sub(replacer, md_content)

with open(md_path, 'w', encoding='utf-8') as f:
    f.write(new_md_content)

print("Updated chapter_26_robotics.md")
