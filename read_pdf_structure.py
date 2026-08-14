import PyPDF2

def extract_text(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        num_pages = len(reader.pages)
        print(f'Total pages: {num_pages}')
        text = ''
        for i in range(num_pages):
            page_text = reader.pages[i].extract_text()
            print(f'--- PAGE {i+1} ---')
            print(page_text[:200].replace('\n', ' '))

extract_text('TaiLieu/ebooks_Chapters_Vi3/Chapter_21_Learning Probalilistic models.pdf')
