import os
import urllib.request
import urllib.parse
import json
import time
import fitz

def is_code(text):
    # Heuristics to detect python code block
    code_indicators = ['>>> ', 'import ', 'def ', ' = ', 'keras.', 'layers.', 'np.', 'tf.', 'model.', 'return ', 'assert ', 'plt.']
    symbol_density = sum(text.count(c) for c in ['(', ')', '[', ']', '{', '}', '=', '+', '-', '*', '/']) / max(len(text), 1)
    
    if any(ind in text for ind in code_indicators):
        return True
    if symbol_density > 0.15: # High density of symbols might indicate code or math formula
        return True
    return False

def translate_via_api(text):
    if not text.strip():
        return text
        
    if is_code(text):
        print("Skipping translation for code block...")
        return f"```python\n{text.strip()}\n```"

    # Replace newlines with spaces for normal text so translator handles it better
    query = text.strip().replace('\n', ' ')
    url = "https://translate.googleapis.com/translate_a/single?client=gtx&sl=en&tl=vi&dt=t&q=" + urllib.parse.quote(query)
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=15) as response:
                res = json.loads(response.read().decode('utf-8'))
            translated_parts = [part[0] for part in res[0] if part[0]]
            translated_text = "".join(translated_parts)
            return translated_text + "\n"
        except Exception as e:
            print(f"Translation error on attempt {attempt+1}: {e}")
            time.sleep(2 ** attempt)
    return text + "\n" # Fallback to original text if failed

def extract_and_translate(pdf_path, md_path):
    print(f"Reading {pdf_path}...")
    doc = fitz.open(pdf_path)
    num_pages = len(doc)
    
    full_markdown = "# Chapter 2: The mathematical building blocks of neural networks\n\n"
    full_markdown += "*(Bản dịch tự động từ PDF sử dụng công nghệ nhận diện Code Blocks mới)*\n\n"
    
    for i in range(num_pages):
        print(f"Processing page {i+1}/{num_pages}...")
        page = doc[i]
        blocks = page.get_text("blocks")
        
        for block in blocks:
            # block[6] == 0 means it's a text block
            if block[6] != 0:
                continue
                
            p = block[4].strip()
            if not p: continue
            
            # Simple cleaning for bullet points
            if p.startswith('¡'):
                p = '- ' + p[1:].strip()
            
            # Identify if heading (rough heuristic: short and Title Case, no period at end)
            is_heading = len(p) < 60 and not p.endswith('.') and p.istitle() and '\n' not in p
            
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
    pdf = 'Chapters/Chapter_02.pdf'
    md = 'chapters/chapter_02_the_mathematical_building_blocks.md'
    os.makedirs('chapters', exist_ok=True)
    extract_and_translate(pdf, md)
