import os
import urllib.request
import urllib.parse
import json
import time
import PyPDF2

def translate_via_api(text):
    if not text.strip():
        return text
    query = text.strip()
    url = "https://translate.googleapis.com/translate_a/single?client=gtx&sl=en&tl=vi&dt=t&q=" + urllib.parse.quote(query)
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=15) as response:
                res = json.loads(response.read().decode('utf-8'))
            translated_parts = [part[0] for part in res[0] if part[0]]
            translated_text = "".join(translated_parts)
            lead_space = text[:len(text) - len(text.lstrip())]
            trail_space = text[len(text.rstrip()):]
            return lead_space + translated_text + trail_space
        except Exception as e:
            print(f"Translation error on attempt {attempt+1}: {e}")
            time.sleep(2 ** attempt)
    return text # Fallback to original text if failed

def extract_and_translate(pdf_path, md_path):
    print(f"Reading {pdf_path}...")
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        num_pages = len(reader.pages)
        
        full_markdown = "# Chapter 1: What is deep learning?\n\n"
        full_markdown += "*(Bản dịch tự động từ PDF)*\n\n"
        
        for i in range(num_pages):
            print(f"Processing page {i+1}/{num_pages}...")
            page_text = reader.pages[i].extract_text()
            if not page_text:
                continue
                
            # Split text by double newlines or single newlines based on structure
            paragraphs = page_text.split('\n\n')
            if len(paragraphs) == 1:
                # Fallback to single newline split if PyPDF2 didn't preserve paragraph breaks well
                # We'll group lines together if they don't end with a period
                lines = page_text.split('\n')
                paragraphs = []
                current_p = ""
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    current_p += line + " "
                    if line.endswith('.') or line.endswith(':') or line.endswith('?'):
                        paragraphs.append(current_p.strip())
                        current_p = ""
                if current_p:
                    paragraphs.append(current_p.strip())
            
            for p in paragraphs:
                p = p.strip()
                if not p: continue
                # Basic cleaning
                p = p.replace('- ', '')
                
                # Identify if heading (rough heuristic: short and Title Case, no period at end)
                is_heading = len(p) < 60 and not p.endswith('.') and p.istitle()
                
                print(f"Translating paragraph of {len(p)} chars...")
                translated = translate_via_api(p)
                
                if is_heading:
                    full_markdown += f"## {translated}\n\n"
                else:
                    full_markdown += f"{translated}\n\n"
            
            # Save progress per page
            with open(md_path, 'w', encoding='utf-8') as md_file:
                md_file.write(full_markdown)
                
            time.sleep(0.5) # rate limiting
            
    print(f"Done! Saved to {md_path}")

if __name__ == '__main__':
    pdf = 'Chapters/Chapter_01.pdf'
    md = 'chapters/chapter_01_what_is_deep_learning.md'
    os.makedirs('chapters', exist_ok=True)
    extract_and_translate(pdf, md)
