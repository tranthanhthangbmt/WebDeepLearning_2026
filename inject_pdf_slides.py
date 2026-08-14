import os
import re
import glob

def inject_pdf_slides():
    chapters_dir = r'd:\DongAUniversity\TÀI LIỆU DẠY HỌC_2024-2025\Trí tuệ nhân tạo_UDA_2025\webTTNT_2026\chapters'
    slides_pdf_dir = r'd:\DongAUniversity\TÀI LIỆU DẠY HỌC_2024-2025\Trí tuệ nhân tạo_UDA_2025\webTTNT_2026\TaiLieu\slide'
    
    chapter_files = glob.glob(os.path.join(chapters_dir, 'chapter_*.md'))
    
    for chapter_file in chapter_files:
        basename = os.path.basename(chapter_file)
        match = re.search(r'chapter_(\d+)_', basename)
        if not match:
            continue
        chapter_num = match.group(1)
        
        # Determine the correct PDF to embed
        # Some chapters have a,b (like chapter04a.pdf, chapter04b.pdf)
        # We will embed all matching pdfs
        pdf_pattern = os.path.join(slides_pdf_dir, f'chapter{chapter_num}*.pdf')
        pdf_files = sorted(glob.glob(pdf_pattern))
        
        if not pdf_files:
            continue
            
        replacement = '#### **Slide**\n\n'
        for pfile in pdf_files:
            p_basename = os.path.basename(pfile)
            pdf_path = f"TaiLieu/slide/{p_basename}"
            replacement += f'<div class="pdf-container" style="margin-bottom: 20px;">\n  <iframe src="{pdf_path}" width="100%" height="100%"></iframe>\n</div>\n\n'
                
        with open(chapter_file, 'r', encoding='utf-8') as cf:
            content = cf.read()
            
        # We need to replace everything after #### **Slide** until the next #### ** or <!-- tabs:end -->
        pattern = re.compile(r'(#### \*\*Slide\*\*\n)(.*?)(?=\n#### \*\*|\n<!-- tabs:end -->)', re.DOTALL)
        
        if pattern.search(content):
            # We replace the whole section starting from #### **Slide**
            new_content = pattern.sub(replacement.replace('\\', '\\\\'), content)
            
            with open(chapter_file, 'w', encoding='utf-8') as cf:
                cf.write(new_content)
            print(f"Injected PDF slides into {basename}")
        else:
            print(f"Could not find #### **Slide** section in {basename}")

if __name__ == '__main__':
    inject_pdf_slides()
