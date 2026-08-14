# Kế hoạch Tạo Slide Chương 14: Phân loại Văn bản (Text Classification)

## 1. Phân tích Hiện trạng Chương 14
- **Nội dung cốt lõi:** Dẫn dắt sinh viên vào lĩnh vực Xử lý Ngôn ngữ Tự nhiên (NLP). Khởi đầu từ việc biến văn bản thô thành số (Text Vectorization Pipeline). Phân tích 2 trường phái biểu diễn văn bản: Tập hợp không thứ tự (Bag-of-Words / N-grams) và Biểu diễn có thứ tự kết hợp Không gian ngữ nghĩa (Word Embeddings).
- **Tình trạng hình ảnh:** Chỉ có **6 hình ảnh**. Bao gồm: Sơ đồ Pipeline xử lý text, Biểu đồ Accuracy của Bag-of-Words, Biểu diễn Sparse vs Dense, Không gian Vector từ vựng, Sơ đồ Lớp Embedding (Từ điển Lookup), và Kiến trúc CBOW (Continuous Bag-of-Words).
- **Thách thức:** Số lượng hình ảnh khá ít (chỉ 6 hình) so với các chương trước. Để đảm bảo tối thiểu **15 slides** và duy trì độ sâu học thuật, tôi sẽ phải phân tích thật sâu các khái niệm Toán học và Đại số tuyến tính đứng sau NLP: 
  - Quy trình Token hóa (Tokenization).
  - Thuật toán TF-IDF cho Bag-of-Words.
  - Sự khác biệt Toán học giữa One-hot Vector và Dense Vector (Embedding).
  - Không gian Vector (Vector Space) thể hiện quan hệ ngữ nghĩa (Ví dụ kinh điển: $King - Man + Woman = Queen$).

## 2. Cấu trúc Nội dung Chi tiết (Dự kiến 16-18 Slides)
- **Phần 1: Giới thiệu Xử lý Ngôn ngữ Tự nhiên - NLP (3 slides)**
  - Lịch sử NLP: Từ Rule-based (Hệ chuyên gia) sang Machine Learning.
  - Máy tính "đọc" hiểu văn bản như thế nào? (Bài toán Vector hóa).
- **Phần 2: Chuỗi quy trình Tiền xử lý (Text Pipeline) (4 slides)**
  - Ba bước bất di bất dịch: Chuẩn hóa (Standardization) $\rightarrow$ Tách từ (Tokenization) $\rightarrow$ Đánh chỉ mục (Indexing). Hình: `text-pipeline.c09bbad6.png`.
  - Phân tích chi tiết từng bước.
- **Phần 3: Phương pháp biểu diễn Tập hợp từ (Bag-of-Words) (4 slides)**
  - Mất đi yếu tố "Thứ tự thời gian" của câu. Dữ liệu trở thành một "Túi từ".
  - Thuật toán N-grams (Bigrams, Trigrams) vớt vát lại thứ tự cục bộ. Cân số hóa bằng TF-IDF.
  - Trực quan hóa kết quả phân loại IMDB. Hình: `bag-of-words-acc.533d9a5b.png`.
- **Phần 4: Word Embeddings - Trái tim của NLP Hiện đại (6 slides)**
  - Khác biệt giữa One-hot Encoding (Rời rạc, trực giao) và Embeddings (Dày đặc, liên tục). Hình: `word-representations.b71fcc82.png` và `word-embeddings.1bc937b3.png`.
  - Lớp Nhúng (Embedding Layer) hoạt động như một bảng Từ điển (Lookup Table). Hình: `embedding-dictionary.80faa429.png`.
  - Không gian Vector Ngữ nghĩa: Khoảng cách hình học (Cosine Similarity) phản ánh độ tương đồng ý nghĩa.
  - Thuật toán Word2Vec: Mô hình CBOW (Continuous Bag-of-Words) dự đoán từ mục tiêu dựa trên ngữ cảnh. Hình: `cbow.01aaf529.png`.
- **Tổng kết (1 slide)**

## 3. Các bước Triển khai
1. **Viết mã LaTeX:** Khởi tạo `TaiLieu/slideDL/Chapter14.tex`.
2. **Biên dịch:** Chạy `pdflatex` 2 lần liên tiếp.
3. **Hoàn thiện Tasks:** Báo cáo kết quả và cập nhật hệ thống.
