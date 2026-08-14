import os
import subprocess
import glob

def convert_figures():
    input_dir = r"d:\DongAUniversity\TÀI LIỆU DẠY HỌC_2024-2025\Trí tuệ nhân tạo_UDA_2025\webTTNT_2026\TaiLieu\slide\figures"
    output_dir = r"d:\DongAUniversity\TÀI LIỆU DẠY HỌC_2024-2025\Trí tuệ nhân tạo_UDA_2025\webTTNT_2026\TaiLieu\slide_md\figures"
    os.makedirs(output_dir, exist_ok=True)
    
    ps_files = glob.glob(os.path.join(input_dir, "*.ps"))
    print(f"Found {len(ps_files)} .ps files.")
    
    for ps_file in ps_files:
        basename = os.path.basename(ps_file)
        png_name = basename.replace(".ps", ".png")
        out_file = os.path.join(output_dir, png_name)
        
        # Check if already converted
        if os.path.exists(out_file):
            continue
            
        cmd = [
            "gswin64c", "-dSAFER", "-dBATCH", "-dNOPAUSE", "-dEPSCrop",
            "-sDEVICE=pngalpha", "-r150", f"-sOutputFile={out_file}", ps_file
        ]
        print(f"Converting {basename}...")
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            print(f"Error converting {basename}: {e}")
            
    print("All conversions completed.")

if __name__ == '__main__':
    convert_figures()
