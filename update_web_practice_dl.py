import os
import re

notebook_dir = 'codeJupyterNotebook/deep-learning-with-python-notebooks-master'
docs_dir = 'Chapters'
repo_base_url = 'https://colab.research.google.com/github/tranthanhthangbmt/WebDeepLearning_2026/blob/main/codeJupyterNotebook/deep-learning-with-python-notebooks-master/'

notebooks = {}

if os.path.exists(notebook_dir):
    for filename in sorted(os.listdir(notebook_dir)):
        if not filename.endswith('.ipynb'):
            continue
        
        match = re.search(r'chapter_?(\d+)', filename, re.IGNORECASE)
        if match:
            chapter = int(match.group(1))
            if '_en' in filename.lower():
                lang = 'EN'
            elif '_vn' in filename.lower():
                lang = 'VN'
            else:
                lang = 'VN'
            
            if chapter not in notebooks:
                notebooks[chapter] = {'VN': [], 'EN': []}
            notebooks[chapter][lang].append(filename)

def generate_notebook_html(filename, index=None, chapter=None):
    colab_url = repo_base_url + filename
    download_url = f"codeJupyterNotebook/deep-learning-with-python-notebooks-master/{filename}"
    
    # clean name
    clean_name = re.sub(r'_(VN|EN|vn|en)\.ipynb$', '', filename)
    clean_name = clean_name.replace('.ipynb', '')
    name_display = clean_name.replace('_', ' ').replace('-', ' ').title()
    prefix = ""
    if index is not None and index > 0 and chapter is not None:
        prefix = f"{chapter:02d}.{index} - "
        
    return f'''    <li style="margin-bottom: 15px; padding: 15px; background: white; border-radius: 6px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
      <strong style="font-size:16px;">{prefix}{name_display}</strong><br>
      <div style="margin-top: 10px; display: flex; gap: 10px; flex-wrap: wrap;">
        <a href="{colab_url}" target="_blank" style="background: #fbbc04; color: #fff; padding: 6px 14px; border-radius: 6px; text-decoration: none; font-size: 14px; font-weight: 600; box-shadow: 0 2px 4px rgba(251,188,4,0.3);">🔥 Mở trên Google Colab</a>
        <a href="{download_url}" download style="background: #1a73e8; color: white; padding: 6px 14px; border-radius: 6px; text-decoration: none; font-size: 14px; font-weight: 600; box-shadow: 0 2px 4px rgba(26,115,232,0.3);">💾 Tải file .ipynb về máy</a>
      </div>
    </li>'''

for chapter, data in notebooks.items():
    vn_files = data['VN']
    en_files = data['EN']
    
    # Fallback
    if not vn_files and en_files:
        vn_files = en_files
    if not en_files and vn_files:
        en_files = vn_files
        
    vn_html = '\n'.join([generate_notebook_html(f, i+1 if len(vn_files)>1 else None, chapter) for i, f in enumerate(vn_files)])
    en_html = '\n'.join([generate_notebook_html(f, i+1 if len(en_files)>1 else None, chapter) for i, f in enumerate(en_files)])
    
    chapter_str = f'{chapter:02d}'
    
    # Now find the markdown files that might contain the practice block.
    # It could be chapter_XX.md or chapter_XX_...md
    
    for md_filename in os.listdir(docs_dir):
        if md_filename.startswith(f'chapter_{chapter_str}') and md_filename.endswith('.md'):
            md_path = os.path.join(docs_dir, md_filename)
            
            with open(md_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            pattern = re.compile(r'<div class="practice-container".*?</ul>\s*</div>', re.DOTALL)
            
            new_block = f'''<div class="practice-container" style="background: #f8faff; border: 1px solid #cce0ff; border-radius: 8px; padding: 20px; margin-top: 15px;">
  <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 15px;">
    <h3 style="margin:0; color: #1a73e8; display:flex; align-items:center; gap:8px;">🚀 Bài tập Thực hành Jupyter Notebook</h3>
    <div class="lang-toggle" style="display:flex; gap: 5px;">
      <button id="btn-vn" onclick="togglePracticeLang('VN')" style="opacity: 1; cursor:pointer; background:white; border:1px solid #cce0ff; border-radius:4px; padding:5px 10px; font-weight:600; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">🇻🇳 VN</button>
      <button id="btn-en" onclick="togglePracticeLang('EN')" style="opacity: 0.5; cursor:pointer; background:white; border:1px solid #cce0ff; border-radius:4px; padding:5px 10px; font-weight:600; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">🇬🇧 EN</button>
    </div>
  </div>
  <p style="margin-bottom: 15px;">Dưới đây là sổ tay (notebook) chứa mã nguồn Python thực hành cho chương này. Bạn có thể mở trực tiếp trên Google Colab để chạy thử nghiệm, hoặc tải file về máy.</p>
  <ul id="notebook-list-VN" style="display:block; list-style-type: none; padding-left: 0;">
{vn_html}
  </ul>
  <ul id="notebook-list-EN" style="display:none; list-style-type: none; padding-left: 0;">
{en_html}
  </ul>
</div>'''

            if pattern.search(content):
                new_content = pattern.sub(new_block, content, count=1)
                with open(md_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                print(f'Updated {md_filename}')

print("Done updating markdown files.")
