import os
import re

filepath = r'd:\DongAUniversity\TÀI LIỆU DẠY HỌC_2024-2025\Trí tuệ nhân tạo_UDA_2025\webTTNT_2026\TaiLieu\Stuart J. Russell, Peter Norvig - Artificial Intelligence_ A Modern Approach, Global Edition-Pearson (2021)_Vietnamses.htm'

with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
    html_content = f.read()

matches = list(re.finditer(r'<img[^>]*src=[\x22\x27]([^\x22\x27]+)[\x22\x27][^>]*>', html_content, flags=re.IGNORECASE))
for m in matches[:5]:
    print(m.group(0))
