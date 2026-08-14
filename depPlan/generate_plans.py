import os

base_dir = r"D:\DongAUniversity\TÀI LIỆU DẠY HỌC_2024-2025\Trí tuệ nhân tạo_UDA_2025\webTTNT_2026\depPlan"

plans = {
    "03": {
        "title": "Giải quyết vấn đề bằng Tìm kiếm (Solving Problems by Searching)",
        "content": """1. **3.1 Các tác nhân giải quyết vấn đề (10 phút)**
   - Vấn đề là gì, Goal formulation.
2. **3.2 Các bài toán ví dụ (15 phút)**
   - Toy problems (8-puzzle, 8-queens) và Real-world problems (Định tuyến, Robot...).
3. **3.3 Thuật toán Tìm kiếm (15 phút)**
   - Cây tìm kiếm, đồ thị, trạng thái biên (frontier).
4. **3.4 Chiến lược Tìm kiếm Không thông tin (25 phút)**
   - Tìm kiếm theo chiều rộng (BFS), chiều sâu (DFS), sâu lặp dần (IDS), chi phí cực tiểu (UCS).
5. **3.5 Tìm kiếm Có thông tin (Heuristic) (20 phút)**
   - Tham lam (Greedy), Tìm kiếm A* (A* Search).
6. **3.6 Hàm Heuristic (5 phút)**
   - Đặc tính admissibility và consistency."""
    },
    "04": {
        "title": "Tìm kiếm trong Môi trường phức tạp (Search in Complex Environments)",
        "content": """1. **4.1 Tìm kiếm Cục bộ và Bài toán tối ưu hóa (30 phút)**
   - Leo đồi (Hill-climbing), Luyện kim nhân tạo (Simulated Annealing), Thuật toán Di truyền (Genetic Algorithms).
2. **4.2 Tìm kiếm Cục bộ trong Không gian Liên tục (15 phút)**
   - Gradient descent cơ bản.
3. **4.3 Tác nhân với Hành động Không tất định (15 phút)**
   - Tìm kiếm cây AND-OR (AND-OR search trees).
4. **4.4 Tìm kiếm với Quan sát Một phần (15 phút)**
   - Không gian niềm tin (Belief states).
5. **4.5 Môi trường chưa biết & Khám phá Trực tuyến (15 phút)**
   - Tác nhân trực tuyến (Online Search Agents)."""
    },
    "05": {
        "title": "Bài toán Thỏa mãn Ràng buộc (Constraint Satisfaction Problems)",
        "content": """1. **5.1 Định nghĩa CSP (15 phút)**
   - Biến (Variables), Miền giá trị (Domains), Ràng buộc (Constraints). Lấy ví dụ tô màu bản đồ.
2. **5.2 Suy diễn CSP: Lan truyền Ràng buộc (25 phút)**
   - Node consistency, Arc consistency (AC-3), Path consistency.
3. **5.3 Tìm kiếm Quay lui (Backtracking) cho CSP (25 phút)**
   - Lựa chọn biến (MRV, Degree heuristic), Lựa chọn giá trị (LCV), Kiểm tra trước (Forward checking).
4. **5.4 Tìm kiếm Cục bộ cho CSP (15 phút)**
   - Heuristic Min-conflicts.
5. **5.5 Cấu trúc bài toán (10 phút)**
   - Bài toán đồ thị độc lập và Cây ràng buộc."""
    },
    "06": {
        "title": "Tìm kiếm Đối kháng và Trò chơi (Adversarial Search and Games)",
        "content": """1. **6.1 Trò chơi và Lý thuyết Trò chơi (10 phút)**
   - Môi trường Multi-agent cạnh tranh, Zero-sum games.
2. **6.2 Quyết định tối ưu: Thuật toán Minimax (25 phút)**
   - Đánh giá giá trị node, Cây Minimax.
3. **6.3 Cắt tỉa Alpha-Beta (25 phút)**
   - Hiệu năng cắt tỉa và cách loại bỏ các nhánh không cần thiết.
4. **6.4 Quyết định thời gian thực không hoàn hảo (15 phút)**
   - Hàm đánh giá Heuristic, Giới hạn độ sâu (Depth limit).
5. **6.5 - 6.7 Trò chơi ngẫu nhiên & Khuyết thiếu thông tin (15 phút)**
   - Trò chơi có yếu tố xác suất (Expectiminimax)."""
    },
    "07": {
        "title": "Tác nhân Logic (Logical Agents)",
        "content": """1. **7.1 Tác nhân dựa trên Tri thức (10 phút)**
   - Cơ sở tri thức (Knowledge Base - KB), TELL và ASK.
2. **7.2 Thế giới Wumpus (Wumpus World) (15 phút)**
   - Mô tả môi trường: Vàng, Hố, Quái vật Wumpus.
3. **7.3 Logic học cơ bản (15 phút)**
   - Cú pháp (Syntax), Ngữ nghĩa (Semantics), Tính hệ quả (Entailment).
4. **7.4 Logic Mệnh đề (Propositional Logic) (15 phút)**
   - Các phép toán (AND, OR, NOT, IMPLIES, EQUIV).
5. **7.5 Chứng minh định lý Mệnh đề (20 phút)**
   - Suy diễn logic (Inference), Phân giải (Resolution), Forward/Backward chaining.
6. **7.6 - 7.7 Tác nhân Logic Mệnh đề (15 phút)**
   - SAT solvers và cách áp dụng vào Wumpus World."""
    },
    "08": {
        "title": "Logic Bậc nhất (First-Order Logic)",
        "content": """1. **8.1 Hạn chế của Logic Mệnh đề (10 phút)**
   - Tại sao cần biểu diễn đối tượng, quan hệ thay vì chỉ sự kiện?
2. **8.2 Cú pháp và Ngữ nghĩa của Logic Bậc nhất (30 phút)**
   - Đối tượng (Objects), Quan hệ (Relations), Hàm (Functions). Lượng từ (Quantifiers: For All, Exists).
3. **8.3 Sử dụng Logic Bậc nhất (25 phút)**
   - Các ví dụ biểu diễn: Quan hệ gia đình, Tập hợp, Danh sách.
4. **8.4 Kỹ nghệ Tri thức trong FOL (25 phút)**
   - Quy trình Kỹ nghệ Tri thức (Knowledge Engineering)."""
    },
    "09": {
        "title": "Suy diễn trong Logic Bậc nhất (Inference in First-Order Logic)",
        "content": """1. **9.1 Tri thức mệnh đề và Bậc nhất (15 phút)**
   - Loại bỏ lượng từ tồn tại (Skolemization), Đại diện toàn cục.
2. **9.2 Hợp nhất và Lược đồ nâng (Unification and Lifting) (20 phút)**
   - Quy tắc hợp nhất các biến logic (Unification algorithm).
3. **9.3 Liên kết Thuận (Forward Chaining) (20 phút)**
   - Suy luận diễn dịch theo hướng dữ liệu.
4. **9.4 Liên kết Ngược (Backward Chaining) (20 phút)**
   - Suy luận theo hướng mục tiêu (Goal-directed), Ứng dụng trong Prolog.
5. **9.5 Phân giải FOL (Resolution) (15 phút)**
   - Đưa về dạng chuẩn tắc (CNF) trong FOL."""
    },
    "10": {
        "title": "Biểu diễn Tri thức (Knowledge Representation)",
        "content": """1. **10.1 Kỹ thuật Tri thức Thực thể (Ontological Engineering) (15 phút)**
   - Các mức trừu tượng, phân cấp thực thể.
2. **10.2 Phân loại và Đối tượng (Categories and Objects) (20 phút)**
   - Mạng ngữ nghĩa (Semantic Networks). Subclass, Khung (Frames).
3. **10.3 Biểu diễn Sự kiện (Events) (20 phút)**
   - Tính thời gian, hoàn cảnh, sự biến đổi theo thời gian.
4. **10.4 Thực thể niềm tin (Mental Events) (20 phút)**
   - Tri thức về tri thức, mô hình hóa suy nghĩ của tác nhân khác.
5. **10.5 Hệ thống suy diễn cho Danh mục (15 phút)**
   - Các hệ thống quản lý tri thức, Logic mô tả (Description Logics)."""
    }
}

template = """# Goal Description

Tiếp nối chuỗi bài giảng, mục tiêu là tạo file trình chiếu LaTeX (`Chapter{num}_4th.tex`) cho **Chương {num}: {title}** dựa trên nội dung tiếng Việt của AIMA Phiên bản thứ 4. Các slide sẽ kế thừa tuỳ chỉnh `\parbox`, tự động xử lý hình ảnh và ngắt dòng hợp lý. File kế hoạch này được lưu tại `depPlan/newSlide_ch{num}_4th.md`.

## User Review Required

> [!WARNING]
> Chương {num} chứa các phần yêu cầu biểu diễn sơ đồ/đồ thị thuật toán rất nhiều. Tôi sẽ ưu tiên sử dụng `figure_{num}.x.jpg` (sơ đồ đồ thị) thay vì nhét chữ mô tả thuật toán. Thầy đồng ý với cách trình bày tối ưu trực quan này chứ?

## Proposed Changes

### 1. Cấu trúc và Phân bổ Bài giảng (90 phút)

{content}

### 2. Chi tiết các file sẽ thao tác

#### [NEW] D:\DongAUniversity\TÀI LIỆU DẠY HỌC_2024-2025\Trí tuệ nhân tạo_UDA_2025\webTTNT_2026\TaiLieu\slide_4th\Chapter{num}_4th.tex
- Tạo mã nguồn LaTeX cho bài giảng Chương {num}.
- Sử dụng cấu trúc `\heading` có parbox. Đảm bảo ảnh từ thư mục `Figures/Images` (ví dụ `figure_{num}.x.jpg`) hiển thị với `[width=0.95\textwidth, height=0.65\textheight, keepaspectratio]`.

#### [NEW] D:\DongAUniversity\TÀI LIỆU DẠY HỌC_2024-2025\Trí tuệ nhân tạo_UDA_2025\webTTNT_2026\depPlan\newSlide_ch{num}_4th.md
- File kế hoạch lưu trữ nội dung phân bổ bài giảng cho Chương {num}.

## Verification Plan

### Manual Verification
- Chạy `pdflatex Chapter{num}_4th.tex` để sinh PDF.
- Đối chiếu PDF: Mọi đồ thị, biểu thức toán học và sơ đồ hiển thị tràn viền, rõ ràng.
"""

for num, data in plans.items():
    filepath = os.path.join(base_dir, f"newSlide_ch{num}_4th.md")
    content = template.format(num=num, title=data["title"], content=data["content"])
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

print("Generated all 8 markdown plan files successfully!")
