import glob
import re
import os
import subprocess

filepaths = glob.glob('TaiLieu/slide_4th/Chapter*_4th.tex')

for path in filepaths:
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split by \heading, but keep the \heading in the chunks
    parts = re.split(r'(\\heading\{.*?\}\n)', content)
    
    # parts[0] is preamble before first \heading
    new_parts = [parts[0]]
    
    # Iterate over heading and content pairs
    # parts[1] is \heading, parts[2] is content, parts[3] is \heading, etc.
    for i in range(1, len(parts), 2):
        heading = parts[i]
        sec_content = parts[i+1] if i+1 < len(parts) else ""
        
        # Count figures in this section
        figures = re.findall(r'\\begin\{figure\}.*?\\end\{figure\}', sec_content, flags=re.DOTALL)
        
        if len(figures) == 1:
            # We have exactly one figure. Apply the hack to it.
            def process_figure(match):
                fig_content = match.group(0)
                
                # Replace \includegraphics options
                # Handle cases with or without [...]
                fig_content = re.sub(r'\\includegraphics(?:\[.*?\])?', r'\\includegraphics[width=1.0\\textwidth,height=0.85\\textheight,keepaspectratio]', fig_content)
                
                # Check if \vspace*{-2.0in} is already there (we added it manually to chapter 3)
                if r'\vspace*{-2.0in}' not in fig_content:
                    fig_content = fig_content.replace(r'\end{figure}', '  \\vspace*{-2.0in}\n\\end{figure}')
                    
                return fig_content
                
            sec_content = re.sub(r'\\begin\{figure\}.*?\\end\{figure\}', process_figure, sec_content, flags=re.DOTALL)
            
        new_parts.append(heading)
        new_parts.append(sec_content)
        
    new_content = "".join(new_parts)
    
    if new_content != content:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f'Updated {path}')
        
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
        # even if not updated, let's compile if we want to ensure pdfs are there
        pass
        
print('All files processed.')
