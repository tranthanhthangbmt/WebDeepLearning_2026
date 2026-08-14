import os
import re
import glob

def clean_tex(content):
    # Remove comments
    content = re.sub(r'(?<!\\)%.*', '', content)
    
    # Replace LaTeX quotes with standard quotes
    content = content.replace("``", '"')
    content = content.replace("''", '"')
    
    # Process \code{...} blocks
    def replace_code_blocks(text):
        idx = 0
        while True:
            idx = text.find(r'\code{', idx)
            if idx == -1:
                break
            brace_count = 0
            start_content = idx + 6
            end_idx = -1
            for i in range(start_content, len(text)):
                if text[i] == '{':
                    brace_count += 1
                elif text[i] == '}':
                    if brace_count == 0:
                        end_idx = i
                        break
                    brace_count -= 1
            if end_idx != -1:
                code = text[start_content:end_idx]
                code = code.replace(r'{\ts}', '')
                code = code.replace(r'{\ac}', ', ')
                code = code.replace(r'\,', ', ')
                code = re.sub(r'\\prog{([^}]+)}', r'\1', code)
                code = re.sub(r'\\noprog{([^}]+)}', r'\1', code)
                code = re.sub(r'\\var{([^}]+)}', r'\1', code)
                code = re.sub(r'\\key{([^}]+)}', r'\1', code)
                code = re.sub(r'\\setq{([^}]+)}{([^}]+)}', r'\1 <- \2', code)
                code = re.sub(r'\\func{([^}]+)}{([^}]+)}{([^}]+)}', r'function \1(\2) returns \3', code)
                code = re.sub(r'\\func{([^}]+)}{([^}]+)}', r'function \1(\2)', code)
                code = re.sub(r'\\firstinputs{([^}]+)}{([^}]+)}', r'  inputs: \1, \2', code)
                code = re.sub(r'\\inputs{([^}]+)}{([^}]+)}', r'  inputs: \1, \2', code)
                code = re.sub(r'\\firstlocal{([^}]+)}{([^}]+)}', r'  local: \1, \2', code)
                code = re.sub(r'\\local{([^}]+)}{([^}]+)}', r'  local: \1, \2', code)
                code = re.sub(r'\\firststatic{([^}]+)}{([^}]+)}', r'  static: \1, \2', code)
                code = re.sub(r'\\static{([^}]+)}{([^}]+)}', r'  static: \1, \2', code)
                code = re.sub(r'\\setq{([^}]+)}', r'\1', code)
                code = code.replace(r'$\infty$', 'infinity')
                code = code.replace(r'$\neq$', '!=')
                code = code.replace(r'\infty', 'infinity')
                code = code.replace(r'\neq', '!=')
                code = code.replace(r'\bodysep', '\n')
                code = code.replace(r'\subfnsep', '\n')
                code = code.replace(r'\action', 'action')
                code = re.sub(r'\\phantom{[^}]+}', '        ', code)
                code = re.sub(r'{\\it\s+([^}]+)}', r'*\1*', code)
                code = re.sub(r'{\\tt\s+([^}]+)}', r'`\1`', code)
                code = re.sub(r'\\cl{([^}]+)}{([^}]+)}{([^}]+)}', r'\1 --\2--> \3', code)
                
                new_block = '\n```text\n' + code.strip() + '\n```\n'
                text = text[:idx] + new_block + text[end_idx+1:]
                idx += len(new_block)
            else:
                idx += 6
        return text

    content = replace_code_blocks(content)

    # Replace \blob with -
    content = content.replace(r'\blob', '-')
    
    # Replace {\tt text} with `text`
    content = re.sub(r'{\\tt\s+([^}]+)}', r'`\1`', content)
    
    # Process verbatim blocks
    def process_verbatim(match):
        return '\n```text\n' + match.group(1).strip() + '\n```\n'
    content = re.sub(r'\\begin{verbatim}(.*?)\\end{verbatim}', process_verbatim, content, flags=re.DOTALL)
    
    # Process tabular and mytabular before changing \\
    def process_tabular(match):
        inner = match.group(1)
        
        # If it's a layout tabular containing images or minipages, just dissolve it
        if r'\epsffile' in inner or r'\fig' in inner or r'\begin{minipage}' in inner:
            return '\n' + inner.replace('&', ' ') + '\n'
            
        # remove hline, tabhead, tabtop, tabbot
        inner = re.sub(r'\\hline', '', inner)
        inner = re.sub(r'\\tabhead', '', inner)
        inner = re.sub(r'\\tabtop', '', inner)
        inner = re.sub(r'\\tabbot', '', inner)
        inner = inner.replace(r'\newline', ' ')
        
        rows = inner.split(r'\\')
        md_rows = []
        for row in rows:
            row = row.strip()
            if not row: continue
            cols = row.split('&')
            clean_cols = [' '.join(col.strip().split()) for col in cols]
            md_rows.append('| ' + ' | '.join(clean_cols) + ' |')
            
        if not md_rows: return ''
        num_cols = md_rows[0].count('|') - 1
        header_sep = '|' + '|'.join(['---'] * num_cols) + '|'
        empty_header = '|' + '|'.join([' &nbsp; '] * num_cols) + '|'
        return '\n\n' + empty_header + '\n' + header_sep + '\n' + '\n'.join(md_rows) + '\n\n'

    content = re.sub(r'\\begin{tabular}{[^\n]+}(.*?)\\end{tabular}', process_tabular, content, flags=re.DOTALL)
    content = re.sub(r'\\begin{mytabular}{[^\n]+}(.*?)\\end{mytabular}', process_tabular, content, flags=re.DOTALL)

    # Replace \al, \nl, \nnl with newlines and spaces
    content = content.replace(r'\al', '\n  ')
    content = content.replace(r'\nl', '\n    ')
    content = content.replace(r'\nnl', '\n      ')
    content = content.replace(r'\\', '\n')
    
    # Remove LaTeX preamble commands
    content = re.sub(r'\\documentstyle(?:\[[^\]]*\])?{[^}]+}', '', content)
    content = re.sub(r'\\documentclass(?:\[[^\]]*\])?{[^}]+}', '', content)
    
    # Replace \u{text} with <u>text</u>
    content = re.sub(r'\\u{([^}]+)}', r'<u>\1</u>', content)
    
    # Replace \q{text} with <u>text</u>??
    content = re.sub(r'\\q{([^}]+)}', r'<u>\1</u>??', content)
    
    # Replace \verb|text| with `text`
    content = re.sub(r'\\verb\|([^|]+)\|', r'`\1`', content)
    
    # Replace {\em text} or \emph{text} with *text*
    content = re.sub(r'{\\em\s+([^}]+)}', r'*\1*', content)
    content = re.sub(r'\\emph{([^}]+)}', r'*\1*', content)
    
    # Replace {\sc text} with **text**
    content = re.sub(r'{\\sc\s+([^}]+)}', r'**\1**', content)
    
    # Replace \fnvar{text} with text
    content = re.sub(r'\\fnvar{([^}]+)}', r'\1', content)
    
    # Replace layout spacing macros
    content = content.replace(r'\quad', ' &nbsp;&nbsp; ')
    content = content.replace(r'\qquad', ' &nbsp;&nbsp;&nbsp;&nbsp; ')
    
    # Replace ~ (non-breaking space) with normal space
    content = re.sub(r'(?<!\\)~', ' ', content)
    
    def replace_fig(match):
        full_match = match.group(0)
        # Search for basename.ps
        m = re.search(r'([\w\+\-]+)\.ps', full_match)
        if m:
            base = m.group(1)
            return f"![Hình ảnh](../TaiLieu/slide_md/figures/{base}.png)"
        return f"[Hình ảnh: {full_match}]"
    
    # Replace \fig{...} taking nested braces into account
    content = re.sub(r'\\fig{[^{}]*(?:{[^{}]*}[^{}]*)*\}', replace_fig, content)
    # Handle the specific typo `\fig{\file{figures]{turing.ps}}`
    content = re.sub(r'\\fig{\\file{figures\]{[^}]+\.ps}}', replace_fig, content)
    content = re.sub(r'\\epsffile{[^{}]*(?:{[^{}]*}[^{}]*)*\}', replace_fig, content)
    
    # Replace \titleslide{text1}{text2}
    content = re.sub(r'\\titleslide{([^}]+)}{([^}]+)}', r'# \1\n\n## \2', content)
    
    # Replace \heading{text}
    content = re.sub(r'\\heading{([^}]+)}', r'\n---\n## \1\n', content)
    content = re.sub(r'\\pheading{([^}]+)}', r'\n---\n## \1\n', content)
    
    # Math symbols from aima-slides.sty that might be commonly used
    content = content.replace(r'\pv', 'P')
    content = content.replace(r'\qv', 'Q')
    
    # Clean up standard LaTeX environments that we don't render well
    content = re.sub(r'\\begin{document}', '', content)
    content = re.sub(r'\\end{document}', '', content)
    content = re.sub(r'\\begin{huge}', '', content)
    content = re.sub(r'\\end{huge}', '', content)
    content = re.sub(r'\\begin{LARGE}', '', content)
    content = re.sub(r'\\end{LARGE}', '', content)
    content = re.sub(r'\\begin{center}', '', content)
    content = re.sub(r'\\end{center}', '', content)
    content = re.sub(r'\\begin{tabular}{[^}]+}', '', content)
    content = re.sub(r'\\end{tabular}', '', content)
    content = re.sub(r'\\begin{minipage}(?:\[[^\]]*\])?{[^}]+}', '', content)
    content = re.sub(r'\\end{minipage}', '', content)

    
    content = re.sub(r'\\sf', '', content)
    
    # Remove layout macros
    content = re.sub(r'\\vspace\*{[^}]+}', '', content)
    content = re.sub(r'\\vspace{[^}]+}', '', content)
    content = re.sub(r'\\noindent', '', content)
    content = re.sub(r'\\epsfxsize=?[0-9.]*(?:\\[a-zA-Z]+)?', '', content)
    content = re.sub(r'\\centerline{([^{}]*(?:{[^{}]*}[^{}]*)*)\}', r'\1', content)
    content = content.replace(r'\hfill', '')
    content = re.sub(r'\\hbox{([^{}]*(?:{[^{}]*}[^{}]*)*)\}', r'\1', content)
    content = re.sub(r'\\input{[^}]+}', '', content) # Just remove inputs for now if they are missing
    
    # Clean up multiple newlines
    content = re.sub(r'\n{3,}', '\n\n', content)
    
    return content.strip()

def resolve_inputs(content, base_dir):
    # Resolve \file{dir}{filename} to dir/filename
    content = re.sub(r'\\file{([^}]+)}{([^}]+)}', r'\1/\2', content)
    
    def replace_input(match):
        rel_path = match.group(1).strip()
        if not rel_path.endswith('.tex'):
            rel_path += '.tex'
        full_path = os.path.join(base_dir, rel_path)
        if os.path.exists(full_path):
            with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                return resolve_inputs(f.read(), base_dir)
        else:
            return f"<!-- Missing input: {rel_path} -->"
    return re.sub(r'\\input{([^}]+)}', replace_input, content)

def process_slides(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    tex_files = glob.glob(os.path.join(input_dir, 'chapter*.tex'))
    
    for tex_file in tex_files:
        basename = os.path.basename(tex_file)
        name, _ = os.path.splitext(basename)
        md_file = os.path.join(output_dir, f'{name}.md')
        
        try:
            with open(tex_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading {tex_file}: {e}")
            continue
            
        content = resolve_inputs(content, input_dir)
        md_content = clean_tex(content)
        
        try:
            with open(md_file, 'w', encoding='utf-8') as f:
                f.write(md_content)
            print(f"Generated {md_file}")
        except Exception as e:
            print(f"Error writing {md_file}: {e}")

if __name__ == '__main__':
    input_dir = r'd:\DongAUniversity\TÀI LIỆU DẠY HỌC_2024-2025\Trí tuệ nhân tạo_UDA_2025\webTTNT_2026\TaiLieu\slide'
    output_dir = r'd:\DongAUniversity\TÀI LIỆU DẠY HỌC_2024-2025\Trí tuệ nhân tạo_UDA_2025\webTTNT_2026\TaiLieu\slide_md'
    print(f"Converting .tex files in {input_dir} to .md in {output_dir}...")
    process_slides(input_dir, output_dir)
    print("Done!")
