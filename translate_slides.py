import os
import re
import urllib.request
import urllib.parse
import json
import time
import glob

# Constants
SLIDE_DIR = r"d:\DongAUniversity\TÀI LIỆU DẠY HỌC_2024-2025\Trí tuệ nhân tạo_UDA_2025\webTTNT_2026\TaiLieu\slide"
BACKUP_DIR = r"d:\DongAUniversity\TÀI LIỆU DẠY HỌC_2024-2025\Trí tuệ nhân tạo_UDA_2025\webTTNT_2026\TaiLieu\slide_en_backup"

# Translate API function
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
    raise Exception("Failed to translate text after 5 attempts")

# Balance brace extractor
def get_brace_block(text, keyword, start_idx):
    idx = text.find(keyword, start_idx)
    if idx == -1:
        return None
    brace_start = idx + len(keyword) - 1
    brace_count = 0
    for i in range(brace_start, len(text)):
        if text[i] == '{':
            brace_count += 1
        elif text[i] == '}':
            brace_count -= 1
            if brace_count == 0:
                return {
                    'start_idx': idx,
                    'brace_start': brace_start,
                    'end_idx': i,
                    'content': text[brace_start+1:i]
                }
    return None

def tokenize_and_placeholder(text):
    placeholders = {}
    placeholder_idx = 1000
    
    i = 0
    n = len(text)
    result = []
    
    ignored_keywords = [
        r'\code{', r'\fig{', r'\file{', r'\epsffile{', r'\input{', 
        r'\vspace{', r'\vspace*{', r'\hspace{', r'\hspace*{', 
        r'\documentclass{', r'\usepackage{', r'\begin{', r'\end{',
        r'\mat{', r'\cal{', r'\bbox{', r'\label{', r'\ref{', r'\cite{', 
        r'\epsfflex{', r'\FigBox{', r'\graphbox{', r'\figbox{', 
        r'\twofig{', r'\twograph{', r'\threefig{', r'\threegraph{',
        r'\item{'
    ]
    
    translatable_keywords = [
        r'\heading{', r'\pheading{', r'\defn{', r'\emph{', r'\note{', 
        r'\quote{', r'\centerline{',
        r'\txm{', r'\txg{', r'\txb{', r'\txc{', r'\txr{', r'\txv{', r'\txk{', r'\txy{', r'\txw{', r'\txR{',
        r'\underline{', r'\bf{', r'\it{', r'\sc{', r'\black{', r'\text{', r'\mbox{', r'\u{', r'\q{'
    ]
    
    simple_macros = [
        r'\al', r'\nl', r'\nnl', r'\blob', r'\sf', r'\\', r'\noindent', r'\quad', r'\qquad',
        r'\mysum', r'\myint', r'\myprod', r'\pref', r'\indiff', r'\prefeq', r'\lequiv', r'\implies', 
        r'\impliessymbol', r'\lequivsymbol', r'\xor', r'\All', r'\Exi', r'\Exii', r'\union', 
        r'\intersection', r'\emptyset', r'\emptylist', r'\Parents', r'\parents', r'\Children', 
        r'\children', r'\MarkovBlanket', r'\markovBlanket', r'\tick', r'\cross', r'\smiley', r'\frowny',
        r'\DollarSign', r'\reals', r'\cspace', r'\espace', r'\wspace', r'\co', r'\fp'
    ]

    while i < n:
        # 1. Check comments (ignore \%)
        if text[i] == '%' and (i == 0 or text[i-1] != '\\'):
            eol = text.find('\n', i)
            if eol == -1:
                eol = n
            comment_text = text[i:eol]
            pid = f"[I{placeholder_idx}]"
            placeholder_idx += 1
            placeholders[pid] = comment_text
            result.append(pid)
            i = eol
            continue
            
        # 2a. Check display math \[ ... \]
        if text[i:i+2] == '\\[':
            end_math = text.find('\\]', i+2)
            if end_math == -1:
                end_math = n
            else:
                end_math += 2
            math_text = text[i:end_math]
            pid = f"[I{placeholder_idx}]"
            placeholder_idx += 1
            placeholders[pid] = math_text
            result.append(pid)
            i = end_math
            continue

        # 2b. Check verbatim and math environments
        env_match = re.match(r'\\begin\{(verbatim|eqnarray\**|equation\**|math|displaymath|align\**|gather\**)\}', text[i:i+30])
        if env_match:
            env_name = env_match.group(1)
            end_tag = f"\\end{{{env_name}}}"
            end_idx = text.find(end_tag, i)
            if end_idx == -1:
                end_idx = n
            else:
                end_idx += len(end_tag)
            env_text = text[i:end_idx]
            pid = f"[I{placeholder_idx}]"
            placeholder_idx += 1
            placeholders[pid] = env_text
            result.append(pid)
            i = end_idx
            continue

        # 2c. Check math block $$...$$
        if text[i:i+2] == '$$':
            end_math = text.find('$$', i+2)
            if end_math == -1:
                end_math = n
            else:
                end_math += 2
            math_text = text[i:end_math]
            pid = f"[I{placeholder_idx}]"
            placeholder_idx += 1
            placeholders[pid] = math_text
            result.append(pid)
            i = end_math
            continue
            
        # 3. Check math block $...$
        if text[i] == '$':
            end_math = -1
            for k in range(i+1, n):
                if text[k] == '$' and text[k-1] != '\\':
                    end_math = k + 1
                    break
            if end_math == -1:
                end_math = n
            math_text = text[i:end_math]
            pid = f"[I{placeholder_idx}]"
            placeholder_idx += 1
            placeholders[pid] = math_text
            result.append(pid)
            i = end_math
            continue
            
        # 4. Check titleslide
        if text[i:i+12] == r'\titleslide{':
            block1 = get_brace_block(text, r'\titleslide{', i)
            if block1:
                next_start = block1['end_idx'] + 1
                while next_start < n and text[next_start].isspace():
                    next_start += 1
                if next_start < n and text[next_start] == '{':
                    block2 = get_brace_block(text, '{', next_start - 1)
                    if block2:
                        pid_start = f"[S{placeholder_idx}]"
                        pid_mid = f"[M{placeholder_idx}]"
                        pid_end = f"[E{placeholder_idx}]"
                        placeholder_idx += 1
                        
                        placeholders[pid_start] = r"\titleslide{"
                        placeholders[pid_mid] = "}{"
                        placeholders[pid_end] = "}"
                        
                        sub1_tok, sub1_phs = tokenize_and_placeholder(block1['content'])
                        sub2_tok, sub2_phs = tokenize_and_placeholder(block2['content'])
                        placeholders.update(sub1_phs)
                        placeholders.update(sub2_phs)
                        
                        result.append(pid_start + sub1_tok + pid_mid + sub2_tok + pid_end)
                        i = block2['end_idx'] + 1
                        continue
            
        # 5. Check ignored keywords with braces
        matched_ignored = False
        for kw in ignored_keywords:
            if text[i:i+len(kw)] == kw:
                block = get_brace_block(text, kw, i)
                if block:
                    ignored_text = text[block['start_idx']:block['end_idx']+1]
                    pid = f"[I{placeholder_idx}]"
                    placeholder_idx += 1
                    placeholders[pid] = ignored_text
                    result.append(pid)
                    i = block['end_idx'] + 1
                    matched_ignored = True
                    break
        if matched_ignored:
            continue
            
        # 6. Check translatable keywords with braces
        matched_trans = False
        for kw in translatable_keywords:
            if text[i:i+len(kw)] == kw:
                block = get_brace_block(text, kw, i)
                if block:
                    pid_start = f"[S{placeholder_idx}]"
                    pid_end = f"[E{placeholder_idx}]"
                    placeholder_idx += 1
                    
                    placeholders[pid_start] = kw
                    placeholders[pid_end] = "}"
                    
                    sub_tokenized, sub_phs = tokenize_and_placeholder(block['content'])
                    placeholders.update(sub_phs)
                    
                    result.append(pid_start + sub_tokenized + pid_end)
                    i = block['end_idx'] + 1
                    matched_trans = True
                    break
        if matched_trans:
            continue
            
        # 7. Check simple macros
        matched_simple = False
        for macro in simple_macros:
            kw_len = len(macro)
            if text[i:i+kw_len] == macro:
                # check word boundary
                if macro[-1].isalpha() and i+kw_len < n and text[i+kw_len].isalpha():
                    continue
                pid = f"[I{placeholder_idx}]"
                placeholder_idx += 1
                placeholders[pid] = macro
                result.append(pid)
                i += kw_len
                matched_simple = True
                break
        if matched_simple:
            continue
            
        # 8. If nothing matched, consume one character
        result.append(text[i])
        i += 1
        
    return "".join(result), placeholders

def restore_placeholders(text, placeholders):
    text = re.sub(r'\[\s*([I|S|E|M])\s*(\d+)\s*\]', r'[\1\2]', text)
    
    max_loops = len(placeholders) + 10
    while max_loops > 0:
        found_placeholder = False
        for pid, val in placeholders.items():
            if pid in text:
                text = text.replace(pid, val)
                found_placeholder = True
        if not found_placeholder:
            break
        max_loops -= 1
    return text

def translate_slide_block(slide_content):
    tokenized_text, placeholders = tokenize_and_placeholder(slide_content)
    try:
        translated_tokenized = translate_via_api(tokenized_text)
    except Exception as e:
        print("API Translation failed, skipping translation for this block. Error:", e)
        return slide_content
    translated_content = restore_placeholders(translated_tokenized, placeholders)
    return translated_content

def translate_file(filepath):
    print(f"\n==================================================")
    print(f"Translating file: {os.path.basename(filepath)}")
    
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
        
    pattern = re.compile(r'(%+\s*Slide\s*%+)', re.IGNORECASE)
    parts = pattern.split(content)
    
    translated_parts = []
    total_parts = len(parts)
    
    for idx, part in enumerate(parts):
        if pattern.match(part):
            translated_parts.append(part)
        else:
            if r'\documentclass' in part or (idx == 0 and r'\begin{document}' in part):
                print(f"Part {idx+1}/{total_parts}: Preamble (skipping translation)")
                translated_parts.append(part)
            elif not part.strip():
                translated_parts.append(part)
            else:
                print(f"Part {idx+1}/{total_parts}: Translating slide block...")
                translated_block = translate_slide_block(part)
                translated_parts.append(translated_block)
                time.sleep(0.5)
                
    translated_content = "".join(translated_parts)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(translated_content)
    print(f"Successfully translated and saved {os.path.basename(filepath)}")

def main():
    if not os.path.exists(BACKUP_DIR):
        print(f"Creating backup directory at {BACKUP_DIR}...")
        os.makedirs(BACKUP_DIR, exist_ok=True)
        tex_files = glob.glob(os.path.join(SLIDE_DIR, "*.tex"))
        for tf in tex_files:
            dest = os.path.join(BACKUP_DIR, os.path.basename(tf))
            with open(tf, 'r', encoding='utf-8', errors='ignore') as src_f:
                with open(dest, 'w', encoding='utf-8') as dest_f:
                    dest_f.write(src_f.read())
        print("Backup completed.")
    else:
        print("Backup directory already exists. Skipping backup.")

    tex_files = sorted(glob.glob(os.path.join(SLIDE_DIR, "*.tex")))
    print(f"Found {len(tex_files)} .tex files to translate.")
    
    for tf in tex_files:
        translate_file(tf)
        
    print("\nAll slide files translated successfully!")

if __name__ == '__main__':
    main()
