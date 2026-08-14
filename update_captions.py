import glob
import re
import os
import subprocess

filepaths = glob.glob('TaiLieu/slide_4th/Chapter*_4th.tex')

for path in filepaths:
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Add \usepackage{caption} and \captionsetup{font=LARGE} to preamble
    # We can insert it after \usepackage{graphicx}
    if r'\usepackage{caption}' not in content:
        content = content.replace(r'\usepackage{graphicx}', r'\usepackage{graphicx}' + '\n' + r'\usepackage{caption}' + '\n' + r'\captionsetup{font=LARGE}')
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f'Updated {path}')
        
        # Compile
        print(f'Compiling {os.path.basename(path)}...')
        result = subprocess.run(
            ['pdflatex', '-interaction=nonstopmode', os.path.basename(path)],
            cwd='d:/DongAUniversity/TÀI LIỆU DẠY HỌC_2024-2025/Trí tuệ nhân tạo_UDA_2025/webTTNT_2026/TaiLieu/slide_4th',
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        if result.returncode == 0:
            print(f'Successfully compiled {os.path.basename(path)}')
        else:
            print(f'Error compiling {os.path.basename(path)} (exit code {result.returncode})')
    else:
        # If it's already there but maybe not LARGE, let's replace it
        new_content = re.sub(r'\\captionsetup\{font=.*?\}', r'\\captionsetup{font=LARGE}', content)
        if new_content != content:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            print(f'Compiling {os.path.basename(path)}...')
            result = subprocess.run(
                ['pdflatex', '-interaction=nonstopmode', os.path.basename(path)],
                cwd='d:/DongAUniversity/TÀI LIỆU DẠY HỌC_2024-2025/Trí tuệ nhân tạo_UDA_2025/webTTNT_2026/TaiLieu/slide_4th',
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )

print('All files processed.')
