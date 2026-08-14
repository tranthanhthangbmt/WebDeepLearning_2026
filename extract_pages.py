import sys
import PyPDF2

def extract_pages(pdf_path, start_page, end_page, out_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        # start_page and end_page are 1-indexed, inclusive
        for i in range(start_page - 1, end_page):
            text += reader.pages[i].extract_text() + '\n\n'
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f'Extracted pages {start_page} to {end_page} to {out_path}')

if __name__ == '__main__':
    extract_pages(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
