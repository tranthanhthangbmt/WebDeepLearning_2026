import os
import re

# Repo URL for Google Colab links. Assuming user's github matches the ML one.
repo_url = "tranthanhthangbmt/WebDeepLearning_2026"
notebook_dir = os.path.join("TaiLieu", "NotebookJupyter")
chapters_dir = "Chapters"

# Get all notebooks
notebooks = [f for f in os.listdir(notebook_dir) if f.endswith('.ipynb')]

for nb in notebooks:
    # Example format: chapter02_mathematical-building-blocks.ipynb
    match = re.match(r'chapter(\d+)_?(.*)\.ipynb', nb)
    if not match:
        continue
    
    ch_num_str = match.group(1) # e.g. '02'
    title_raw = match.group(2) # e.g. 'mathematical-building-blocks'
    
    if not title_raw:
        title = f"Bài tập thực hành Chương {int(ch_num_str)}"
    else:
        title = title_raw.replace('-', ' ').title()
    
    # Path to corresponding chapter MD file
    ch_file = f"chapter_{ch_num_str}.md"
    ch_path = os.path.join(chapters_dir, ch_file)
    
    if not os.path.exists(ch_path):
        print(f"File {ch_path} not found.")
        continue
        
    with open(ch_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    if '#### ** 💻 Luyện tập **' in content or '#### **💻 Luyện tập**' in content:
        print(f"Chapter {ch_num_str} already has a practice tab.")
        continue
        
    # Build the practice tab HTML
    practice_html = f"""#### ** 💻 Luyện tập **

<div class="practice-container" style="background: #f8faff; border: 1px solid #cce0ff; border-radius: 8px; padding: 20px; margin-top: 15px;">
  <h3 style="margin-top:0; color: #1a73e8; display:flex; align-items:center; gap:8px;">🚀 Bài tập Thực hành Jupyter Notebook</h3>
  <p>Dưới đây là sổ tay (notebook) chứa mã nguồn Python thực hành cho chương này. Bạn có thể mở trực tiếp trên Google Colab để chạy thử nghiệm, hoặc tải file về máy.</p>
  <ul style="list-style-type: none; padding-left: 0;">
    <li style="margin-bottom: 15px; padding: 15px; background: white; border-radius: 6px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
      <strong style="font-size:16px;">{title}</strong><br>
      <div style="margin-top: 10px; display: flex; gap: 10px; flex-wrap: wrap;">
        <a href="https://colab.research.google.com/github/{repo_url}/blob/main/TaiLieu/NotebookJupyter/{nb}" target="_blank" style="background: #fbbc04; color: #fff; padding: 6px 14px; border-radius: 6px; text-decoration: none; font-size: 14px; font-weight: 600; box-shadow: 0 2px 4px rgba(251,188,4,0.3);">🔥 Mở trên Google Colab</a>
        <a href="TaiLieu/NotebookJupyter/{nb}" download style="background: #1a73e8; color: white; padding: 6px 14px; border-radius: 6px; text-decoration: none; font-size: 14px; font-weight: 600; box-shadow: 0 2px 4px rgba(26,115,232,0.3);">💾 Tải file .ipynb về máy</a>
      </div>
    </li>
  </ul>
</div>

"""

    # Insert before <!-- tabs:end -->
    if '<!-- tabs:end -->' in content:
        new_content = content.replace('<!-- tabs:end -->', practice_html + '<!-- tabs:end -->')
        with open(ch_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Updated {ch_path}")
    else:
        print(f"Could not find <!-- tabs:end --> in {ch_path}")
