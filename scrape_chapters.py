import os
import requests
from bs4 import BeautifulSoup
import markdownify

base_url = "https://deeplearningwithpython.io"
index_url = f"{base_url}/chapters/"

def download_image(img_url):
    # img_url e.g. /images/ch01/ai-ml-dl.07201556.png
    local_path = img_url.lstrip('/')
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    full_url = base_url + img_url
    
    if not os.path.exists(local_path):
        try:
            r = requests.get(full_url)
            r.raise_for_status()
            with open(local_path, 'wb') as f:
                f.write(r.content)
            print(f"Downloaded image: {local_path}")
        except Exception as e:
            print(f"Failed to download image {full_url}: {e}")
            
    # Docsify uses relative paths from root, so simply "images/..." is perfect
    return local_path

def scrape():
    print("Fetching index page...")
    r = requests.get(index_url)
    r.encoding = 'utf-8'
    r.raise_for_status()
    soup = BeautifulSoup(r.text, 'html.parser')
    
    chapter_links = []
    for a in soup.select('.posts-list .post-preview a'):
        href = a.get('href')
        if href:
            chapter_links.append(href)
            
    os.makedirs('chapters', exist_ok=True)
    
    sidebar_lines = []
    sidebar_lines.append("\n- **Sách gốc tiếng Anh (English Original)**\n")
    
    for idx, link in enumerate(chapter_links):
        ch_num = idx + 1
        print(f"Scraping chapter {ch_num}: {link}")
        full_url = base_url + link
        ch_r = requests.get(full_url)
        ch_r.encoding = 'utf-8'
        ch_r.raise_for_status()
        ch_soup = BeautifulSoup(ch_r.text, 'html.parser')
        
        title_element = ch_soup.select_one('.title h1')
        title = title_element.text if title_element else f"Chapter {ch_num}"
        
        article = ch_soup.find('article')
        if not article:
            print(f"No article found for Chapter {ch_num}.")
            continue
            
        # Download images and replace src
        for img in article.find_all('img'):
            src = img.get('src')
            if src and src.startswith('/images/'):
                local_img_path = download_image(src)
                img['src'] = '../' + local_img_path
                
        # Remove empty contents div at top (like TOC) to avoid clutter, optional
        toc = article.find('div', class_='contents')
        if toc:
            toc.decompose()
                
        # Convert to markdown
        md_content = markdownify.markdownify(str(article), heading_style="ATX", code_language="python")
        
        md_title = f"# Chapter {ch_num}: {title}\n\n"
        final_md = md_title + md_content
        
        filename = f"chapter_{ch_num:02d}_en.md"
        filepath = os.path.join('chapters', filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(final_md)
            
        sidebar_lines.append(f"  - [Chapter {ch_num}: {title}](chapters/{filename})\n")
        
    print("Scraping finished. Appending to _sidebar.md...")
    with open('_sidebar.md', 'a', encoding='utf-8') as f:
        f.writelines(sidebar_lines)
    print("Done! All 20 chapters downloaded.")

if __name__ == '__main__':
    scrape()
