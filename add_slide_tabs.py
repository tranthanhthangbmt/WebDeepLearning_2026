import os
import re
import glob

def add_slide_tabs():
    chapters_dir = r'd:\DongAUniversity\TÀI LIỆU DẠY HỌC_2024-2025\Môn Học sâu_UDA\WebDeepLearning_2026\chapters'
    
    chapter_files = glob.glob(os.path.join(chapters_dir, 'chapter_*.md'))
    
    for chapter_file in chapter_files:
        basename = os.path.basename(chapter_file)
        
        # Only process chapter_01.md, chapter_20.md (exclude _en.md or other files)
        match = re.fullmatch(r'chapter_(\d{2})\.md', basename)
        if not match:
            continue
            
        chapter_num = match.group(1)
        
        # We want to inject:
        # #### **Slide**
        # 
        # <div class="pdf-container" style="margin-bottom: 20px; width: 100%; height: 85vh;">
        #   <iframe src="TaiLieu/slideDL/ChapterXX.pdf" width="100%" height="100%" style="border: none;"></iframe>
        # </div>
        #
        # <!-- tabs:end -->
        
        with open(chapter_file, 'r', encoding='utf-8') as cf:
            content = cf.read()
            
        if '#### **Slide**' in content:
            print(f"Slide tab already exists in {basename}")
            continue
            
        slide_tab = f"""#### **Slide**

<div class="pdf-container" style="margin-bottom: 20px; width: 100%; height: 85vh;">
  <iframe src="TaiLieu/slideDL/Chapter{chapter_num}.pdf" width="100%" height="100%" style="border: none;"></iframe>
</div>

<!-- tabs:end -->"""

        new_content = content.replace('<!-- tabs:end -->', slide_tab)
        
        if new_content != content:
            with open(chapter_file, 'w', encoding='utf-8') as cf:
                cf.write(new_content)
            print(f"Injected Slide tab into {basename}")
        else:
            print(f"Could not find <!-- tabs:end --> in {basename}")

if __name__ == '__main__':
    add_slide_tabs()
