import os

# Đường dẫn thư mục chứa các file part
input_dir = r"d:\DongAUniversity\TÀI LIỆU DẠY HỌC_2024-2025\Trí tuệ nhân tạo_UDA_2025\webTTNT_2026\TaiLieu\ebooks_Chapters_Vi3\Chapter_21_Learning Probalilistic models"
output_file = r"d:\DongAUniversity\TÀI LIỆU DẠY HỌC_2024-2025\Trí tuệ nhân tạo_UDA_2025\webTTNT_2026\TaiLieu\ebooks_Chapters_Vi3\Chapter_21_Learning Probalilistic models\chapter_21_vi.html"

# Mã HTML bao bọc
html_template = """<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Chương 21: Learning Probabilistic Models</title>
    <script type="text/javascript" id="MathJax-script" async
      src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
    </script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1000px;
            margin: 0 auto;
            padding: 20px;
        }
        h1, h2, h3, h4 {
            color: #2c3e50;
            margin-top: 1.5em;
        }
        p {
            margin-bottom: 1em;
            text-align: justify;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            margin-bottom: 1em;
        }
        table, th, td {
            border: 1px solid #ddd;
        }
        th, td {
            padding: 8px;
            text-align: left;
        }
        th {
            background-color: #f2f2f2;
        }
        .figure {
            margin: 2em 0;
            text-align: center;
        }
        .figure img {
            max-width: 100%;
            height: auto;
        }
        .caption {
            font-style: italic;
            color: #555;
            margin-top: 0.5em;
        }
        blockquote {
            border-left: 4px solid #ccc;
            padding-left: 10px;
            margin-left: 0;
            font-style: italic;
            color: #666;
        }
        code {
            background-color: #f4f4f4;
            padding: 2px 4px;
            border-radius: 4px;
            font-family: Consolas, monospace;
        }
    </style>
</head>
<body>
<h1>Chương 21: Học các mô hình xác suất (Learning Probabilistic Models)</h1>
"""

html_footer = """
</body>
</html>
"""

# Gộp các file part lại với nhau
with open(output_file, 'w', encoding='utf-8') as f_out:
    f_out.write(html_template)
    
    for i in range(1, 11):
        part_file = os.path.join(input_dir, f"part{i}_21.html")
        if os.path.exists(part_file):
            with open(part_file, 'r', encoding='utf-8') as f_in:
                f_out.write(f"<!-- BEGIN PART {i} -->\n")
                f_out.write(f_in.read() + "\n")
                f_out.write(f"<!-- END PART {i} -->\n\n")
        else:
            print(f"Warning: {part_file} not found!")

    f_out.write(html_footer)

print(f"Successfully created {output_file}")
