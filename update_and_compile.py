import glob
import re
import os
import subprocess

tex_files = glob.glob('d:/DongAUniversity/TÀI LIỆU DẠY HỌC_2024-2025/Trí tuệ nhân tạo_UDA_2025/webTTNT_2026/TaiLieu/slide_4th/Chapter*_4th.tex')

for filepath in tex_files:
    basename = os.path.basename(filepath)
    # Extract chapter number, e.g. "Chapter04_4th.tex" -> "4"
    match = re.search(r'Chapter(\d+)', basename)
    if not match:
        continue
    
    chapter_num = int(match.group(1))
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # If there is already a \renewcommand{\thefigure}, remove it or replace it
    if r'\renewcommand{\thefigure}' in content:
        content = re.sub(r'\\renewcommand\{\\thefigure\}\{[^\}]+\}\n?', '', content)
        
    replacement = f'\\renewcommand{{\\thefigure}}{{{chapter_num}.\\arabic{{figure}}}}\n\\begin{{document}}'
    content = content.replace(r'\begin{document}', replacement)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f'Updated {basename} with chapter {chapter_num}')
    
    # Compile the file
    print(f'Compiling {basename}...')
    result = subprocess.run(
        ['pdflatex', '-interaction=nonstopmode', basename],
        cwd='d:/DongAUniversity/TÀI LIỆU DẠY HỌC_2024-2025/Trí tuệ nhân tạo_UDA_2025/webTTNT_2026/TaiLieu/slide_4th',
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    if result.returncode == 0:
        print(f'Successfully compiled {basename}')
    else:
        print(f'Error compiling {basename} (exit code {result.returncode})')

print('All files processed.')
