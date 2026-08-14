import os
import glob
import re

def fix_headings():
    # Fix style file
    sty_path = r"D:\DongAUniversity\TÀI LIỆU DẠY HỌC_2024-2025\Trí tuệ nhân tạo_UDA_2025\webTTNT_2026\TaiLieu\slide\aima2e-slides.sty"
    with open(sty_path, "r", encoding="utf-8") as f:
        sty_content = f.read()
        
    sty_content = sty_content.replace(r"\def\heading#1{\newpage", r"\def\heading#1{\clearpage")
    sty_content = sty_content.replace(r"\def\phantomheading#1{\newpage", r"\def\phantomheading#1{\clearpage")
    sty_content = sty_content.replace(r"\def\noheading{%"+"\n"+r"\newpage", r"\def\noheading{%"+"\n"+r"\clearpage")
    
    with open(sty_path, "w", encoding="utf-8") as f:
        f.write(sty_content)
        
    print("Fixed aima2e-slides.sty")

    # Fix tex files
    tex_files = glob.glob(r"D:\DongAUniversity\TÀI LIỆU DẠY HỌC_2024-2025\Trí tuệ nhân tạo_UDA_2025\webTTNT_2026\TaiLieu\slide_4th\*.tex")
    count = 0
    for tex_file in tex_files:
        with open(tex_file, "r", encoding="utf-8") as f:
            content = f.read()
            
        new_content = content.replace(r"\renewcommand{\heading}[1]{\newpage", r"\renewcommand{\heading}[1]{\clearpage")
        
        if new_content != content:
            with open(tex_file, "w", encoding="utf-8") as f:
                f.write(new_content)
            count += 1
            
    print(f"Fixed {count} .tex files in slide_4th")

if __name__ == "__main__":
    fix_headings()
