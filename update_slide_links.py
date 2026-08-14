import glob
import re
import os

def update_links():
    md_files = glob.glob(r"D:\DongAUniversity\TÀI LIỆU DẠY HỌC_2024-2025\Trí tuệ nhân tạo_UDA_2025\webTTNT_2026\chapters\chapter_*.md")
    
    for file_path in md_files:
        filename = os.path.basename(file_path)
        m = re.match(r"chapter_(\d{2})", filename)
        if not m: continue
        
        chapter_num = m.group(1)
        ch_int = int(chapter_num)
        
        if ch_int <= 23:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                
            new_pdf = f"TaiLieu/slide_4th/Chapter{chapter_num}_4th.pdf"
            
            # Thay thế tất cả các link slide cũ bằng link slide_4th mới
            new_content = re.sub(
                r'TaiLieu/slide/chapter\w*\.pdf',
                new_pdf,
                content
            )
            
            # Xử lý đặc biệt cho chương 20 (có 2 iframe slide cũ 20a và 20b)
            if ch_int == 20:
                # Tìm block div thứ 2 và xóa nó đi
                # Dạng:
                # <div class="pdf-container" style="margin-bottom: 20px;">
                #   <iframe src="TaiLieu/slide_4th/Chapter20_4th.pdf" width="100%" height="100%"></iframe>
                # </div>
                pattern_duplicate = (
                    r'(<div class="pdf-container" style="margin-bottom: 20px;">\s*'
                    r'<iframe src="TaiLieu/slide_4th/Chapter20_4th\.pdf".*?</iframe>\s*'
                    r'</div>\s*)'
                    r'<div class="pdf-container" style="margin-bottom: 20px;">\s*'
                    r'<iframe src="TaiLieu/slide_4th/Chapter20_4th\.pdf".*?</iframe>\s*'
                    r'</div>'
                )
                new_content = re.sub(pattern_duplicate, r'\1', new_content, flags=re.DOTALL)
                
            if content != new_content:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(new_content)
                print(f"Updated {filename}")

if __name__ == "__main__":
    update_links()
