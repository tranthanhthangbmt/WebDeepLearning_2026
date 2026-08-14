import os
from PyPDF2 import PdfReader

pdf_path = r"TaiLieu\ebooks_Chapters_Vi3\Chapter_22_Deep Learning.pdf"
output_dir = r"TaiLieu\ebooks_Chapters_Vi3\Chapter_22_Deep Learning"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

reader = PdfReader(pdf_path)
total_pages = len(reader.pages)

# 39 pages total. We want 13 parts, so 3 pages per part.
pages_per_part = 3
num_parts = 13

for part in range(1, num_parts + 1):
    start_page = (part - 1) * pages_per_part
    end_page = start_page + pages_per_part
    
    # If it's the last part, make sure to read until the very end
    if part == num_parts:
        end_page = total_pages
        
    text_content = ""
    for i in range(start_page, end_page):
        if i < total_pages:
            page = reader.pages[i]
            text_content += page.extract_text() + "\n"
            
    output_file = os.path.join(output_dir, f"part{part}_22.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(text_content)
        
    print(f"Created {output_file} (Pages {start_page+1} to {end_page})")

print("Extraction completed!")
