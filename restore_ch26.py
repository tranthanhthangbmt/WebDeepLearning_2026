import re

md_path = r'D:\DongAUniversity\TÀI LIỆU DẠY HỌC_2024-2025\Trí tuệ nhân tạo_UDA_2025\webTTNT_2026\chapters\chapter_26_robotics.md'

with open(md_path, 'r', encoding='utf-8') as f:
    md_content = f.read()

# Replace the Tiếng Việt section which currently contains the raw HTML
pattern = re.compile(r'(#### \*\*Tiếng Việt\*\*).*?(#### \*\*Tiếng Anh\*\*)', re.DOTALL)

replacement = r'''\1
<div class="pdf-container">
  <iframe src="TaiLieu/ebooks_Chapters_Vi3/Chapter_26_Robotics/chapter_26_vi.html?v=1" width="100%" height="100%"></iframe>
</div>

\2'''

new_md_content = pattern.sub(replacement, md_content)

with open(md_path, 'w', encoding='utf-8') as f:
    f.write(new_md_content)

print("Restored iframe in chapter_26_robotics.md")
