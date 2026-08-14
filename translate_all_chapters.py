import os
import urllib.request
import urllib.parse
import json
import time

def translate_via_api(text):
    if not text.strip():
        return text

    query = text.replace('\n', ' ')
    url = "https://translate.googleapis.com/translate_a/single?client=gtx&sl=en&tl=vi&dt=t&q=" + urllib.parse.quote(query)
    for attempt in range(6):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=10) as response:
                res = json.loads(response.read().decode('utf-8'))
            translated_parts = [part[0] for part in res[0] if part[0]]
            return "".join(translated_parts)
        except Exception as e:
            print(f"Translate err: {e}")
            time.sleep(2 ** attempt)
    return text

def translate_block(block):
    block = block.strip()
    if not block:
        return ""
        
    # Code block
    if block.startswith('```'):
        return block
        
    # Image or figure link
    if block.startswith('![') or block.startswith('[Figure'):
        return block
        
    # Heading
    if block.startswith('#'):
        hashes = ""
        for char in block:
            if char == '#': hashes += '#'
            else: break
        text_part = block[len(hashes):].strip()
        translated = translate_via_api(text_part)
        return f"{hashes} {translated}"
        
    # Lists
    if block.startswith('* ') or block.startswith('- ') or block.startswith('> '):
        lines = block.split('\n')
        translated_lines = []
        for line in lines:
            line_stripped = line.strip()
            if line_stripped.startswith('* '):
                t = translate_via_api(line_stripped[2:])
                translated_lines.append(f"* {t}")
            elif line_stripped.startswith('- '):
                t = translate_via_api(line_stripped[2:])
                translated_lines.append(f"- {t}")
            elif line_stripped.startswith('> '):
                t = translate_via_api(line_stripped[2:])
                translated_lines.append(f"> {t}")
            else:
                # Continuation of list item
                if line_stripped:
                    t = translate_via_api(line_stripped)
                    translated_lines.append(t)
        return "\n".join(translated_lines)
        
    # HTML blocks
    if block.startswith('<'):
        return block

    # Normal paragraph
    return translate_via_api(block)

def process_chapter(en_file, out_file):
    with open(en_file, 'r', encoding='utf-8') as f:
        content = f.read()
        
    raw_blocks = content.split('\n\n')
    blocks = []
    
    in_code = False
    current_code_block = []
    
    for rb in raw_blocks:
        if '```' in rb:
            count = rb.count('```')
            if count % 2 != 0:
                if not in_code:
                    in_code = True
                    current_code_block.append(rb)
                else:
                    in_code = False
                    current_code_block.append(rb)
                    blocks.append("\n\n".join(current_code_block))
                    current_code_block = []
            else:
                if in_code:
                    current_code_block.append(rb)
                else:
                    blocks.append(rb)
        else:
            if in_code:
                current_code_block.append(rb)
            else:
                blocks.append(rb)
                
    if current_code_block:
        blocks.append("\n\n".join(current_code_block))
        
    translated_blocks = []
    total = len(blocks)
    
    for i, b in enumerate(blocks):
        if i % 10 == 0:
            print(f"  Translating block {i+1}/{total}")
        t = translate_block(b)
        translated_blocks.append(t)
        # Sleep slightly if we called the API (heuristic)
        if not b.startswith('```') and not b.startswith('![') and not b.startswith('<') and len(b.strip()) > 10:
            time.sleep(0.5)
            
    translated_full = "\n\n".join(translated_blocks)
    
    final_output = f"<!-- tabs:start -->\n\n#### **Tiếng Anh (English)**\n\n{content}\n\n#### **Tiếng Việt (Vietnamese)**\n\n{translated_full}\n\n<!-- tabs:end -->\n"
    
    with open(out_file, 'w', encoding='utf-8') as f:
        f.write(final_output)

def main():
    print("Starting translation process for 20 chapters...")
    for i in range(1, 21):
        en_file = f"chapters/chapter_{i:02d}_en.md"
        out_file = f"chapters/chapter_{i:02d}.md"
        if not os.path.exists(en_file):
            print(f"File {en_file} not found, skipping.")
            continue
            
        print(f"\nProcessing Chapter {i}...")
        process_chapter(en_file, out_file)
        print(f"Finished Chapter {i}")
        
    print("All chapters translated and saved!")

if __name__ == '__main__':
    main()
