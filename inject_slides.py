import os
import re
import glob

def inject_slides():
    chapters_dir = r'd:\DongAUniversity\TÀI LIỆU DẠY HỌC_2024-2025\Trí tuệ nhân tạo_UDA_2025\webTTNT_2026\chapters'
    slides_md_dir = r'd:\DongAUniversity\TÀI LIỆU DẠY HỌC_2024-2025\Trí tuệ nhân tạo_UDA_2025\webTTNT_2026\TaiLieu\slide_md'
    
    # Custom mappings for chapters from chapter 20 onwards due to edition differences
    custom_mappings = {
        '20': [],                 # Chapter 20 (Knowledge in Learning) has no slides
        '21': ['chapter20a.md'],  # Chapter 21 (Learning Probabilistic Models) <- Statistical Learning
        '22': ['chapter20b.md'],  # Chapter 22 (Deep Learning) <- Neural Networks
        '24': ['chapter22.md'],   # Chapter 24 (Natural Language Processing) <- Communication and Language
        '26': ['chapter25.md'],   # Chapter 26 (Robotics) <- Robotics
        '27': ['chapter24.md']    # Chapter 27 (Computer Vision) <- Vision
    }
    
    chapter_files = glob.glob(os.path.join(chapters_dir, 'chapter_*.md'))
    
    for chapter_file in chapter_files:
        basename = os.path.basename(chapter_file)
        # Extract chapter number, e.g., '01' from 'chapter_01_introduction.md'
        match = re.search(r'chapter_(\d+)_', basename)
        if not match:
            continue
        chapter_num = match.group(1)
        
        slide_files = []
        if chapter_num in custom_mappings:
            for sname in custom_mappings[chapter_num]:
                spath = os.path.join(slides_md_dir, sname)
                if os.path.exists(spath):
                    slide_files.append(spath)
        else:
            # Find matching slides by chapter number prefix
            slide_pattern = os.path.join(slides_md_dir, f'chapter{chapter_num}*.md')
            slide_files = sorted(glob.glob(slide_pattern))
            
        # Read all matched slide files
        if slide_files:
            slide_content = ''
            for sfile in slide_files:
                with open(sfile, 'r', encoding='utf-8') as sf:
                    slide_content += sf.read() + '\n\n'
        else:
            slide_content = '*(Chưa có slide)*\n'
                
        # Read the chapter file
        with open(chapter_file, 'r', encoding='utf-8') as cf:
            content = cf.read()
            
        # Replace the content between #### **Slide** and the next #### ** or <!-- tabs:end -->
        pattern = re.compile(r'(#### \*\*Slide\*\*\n)(.*?)(?=\n#### \*\*|\n<!-- tabs:end -->)', re.DOTALL)
        
        if pattern.search(content):
            new_content = pattern.sub(r'\1' + '\n' + slide_content.replace('\\', '\\\\') + '\n', content)
            
            with open(chapter_file, 'w', encoding='utf-8') as cf:
                cf.write(new_content)
            print(f"Injected slides into {basename}")
        else:
            print(f"Could not find #### **Slide** section in {basename}")

if __name__ == '__main__':
    inject_slides()
