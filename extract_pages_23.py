import os
import PyPDF2

def extract_pages(pdf_path, output_dir, chunks):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        num_pages = len(reader.pages)
        print(f"Total pages: {num_pages}")
        
        for part_idx, (start_page, end_page) in enumerate(chunks):
            # end_page is inclusive based on index
            part_text = []
            for i in range(start_page, min(end_page + 1, num_pages)):
                text = reader.pages[i].extract_text()
                if text:
                    part_text.append(text)
            
            output_path = os.path.join(output_dir, f"part{part_idx + 1}_23.txt")
            with open(output_path, 'w', encoding='utf-8') as out_file:
                out_file.write("\n\n".join(part_text))
            print(f"Wrote {output_path} (pages {start_page} to {end_page})")

if __name__ == "__main__":
    pdf_path = "TaiLieu/ebooks_Chapters_Vi3/Chapter_23_Reinforcement Learning.pdf"
    output_dir = "TaiLieu/ebooks_Chapters_Vi3/Chapter_23_Reinforcement Learning"
    
    # Define chunks (start_page, end_page) 0-indexed
    # Pages: 34
    chunks = [
        (0, 2),    # Part 1: pages 0-2
        (3, 5),    # Part 2: pages 3-5
        (6, 8),    # Part 3: pages 6-8
        (9, 11),   # Part 4: pages 9-11
        (12, 14),  # Part 5: pages 12-14
        (15, 17),  # Part 6: pages 15-17
        (18, 20),  # Part 7: pages 18-20
        (21, 23),  # Part 8: pages 21-23
        (24, 26),  # Part 9: pages 24-26
        (27, 29),  # Part 10: pages 27-29
        (30, 31),  # Part 11: pages 30-31
        (32, 33),  # Part 12: pages 32-33
    ]
    
    extract_pages(pdf_path, output_dir, chunks)
