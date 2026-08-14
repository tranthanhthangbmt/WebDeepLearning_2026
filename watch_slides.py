import os
import time
import subprocess
from pathlib import Path

def get_tex_mtimes(folder):
    mtimes = {}
    for f in Path(folder).glob('*.tex'):
        mtimes[str(f)] = f.stat().st_mtime
    return mtimes

def main():
    folder = os.path.join('TaiLieu', 'slide')
    print(f"Watching {folder} for changes in .tex files...")
    last_mtimes = get_tex_mtimes(folder)
    
    try:
        while True:
            time.sleep(1)
            current_mtimes = get_tex_mtimes(folder)
            
            for f, mtime in current_mtimes.items():
                if f not in last_mtimes or mtime > last_mtimes[f]:
                    print(f"\n[{time.strftime('%H:%M:%S')}] Detected change in {f}")
                    last_mtimes[f] = mtime
                    
                    filename = os.path.basename(f)
                    
                    # 1. Compile to PDF using latexmk
                    print(f"Compiling {filename} to PDF using pdflatex...")
                    try:
                        subprocess.run(
                            ['latexmk', '-pdf', '-shell-escape', '-interaction=nonstopmode', filename],
                            cwd=folder,
                            check=False
                        )
                    except Exception as e:
                        print(f"LaTeX compile error: {e}")
                    
                    # 2. Update markdown and web
                    print(f"Updating web content...")
                    try:
                        subprocess.run(['python', 'parse_tex_slides.py'], check=False)
                        subprocess.run(['python', 'inject_slides.py'], check=False)
                    except Exception as e:
                        print(f"Web update error: {e}")
                        
                    print(f"[{time.strftime('%H:%M:%S')}] Done processing {f}.")
                    
            last_mtimes = current_mtimes
            
    except KeyboardInterrupt:
        print("\nStopped watching.")

if __name__ == '__main__':
    main()
