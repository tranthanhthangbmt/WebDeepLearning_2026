# Kế hoạch Xây dựng Slide Chương 17: RA QUYẾT ĐỊNH ĐA TÁC TỬ (Multiagent Decision Making)

## 1. Mục tiêu và Tiêu chuẩn
- **Mục tiêu:** Sinh viên hiểu và nắm vững lý thuyết trò chơi (Game Theory), lý thuyết đa tác tử từ cơ bản đến phức tạp (Hợp tác, Không hợp tác, Đấu giá, Biểu quyết).
- **Yêu cầu kỹ thuật LaTeX:** 
  - Đảm bảo biên dịch hoàn hảo với `pdflatex`.
  - Biểu diễn Toán học: Hàm Lợi ích (Payoff / Utility), Ma trận thưởng phạt (Payoff matrix), và Cây trò chơi mở rộng (Extensive form games) sẽ được thiết kế gọn gàng.
  - Sử dụng layout `aima2e-slides.sty` một cách đồng nhất với 16 chương trước.

## 2. Cấu trúc Nội dung dự kiến (Mapping Sections)

### Trang Tiêu đề
- **Tiêu đề chính:** CHƯƠNG 17: RA QUYẾT ĐỊNH ĐA TÁC TỬ
- **Tiêu đề phụ:** Trí tuệ nhân tạo - Artificial Intelligence
- **Nội dung Chương:** Liệt kê các đề mục từ 17.1 đến 17.4.

### 17.1 Thuộc tính của môi trường Đa Tác Tử (Properties of Multiagent Environments)
- Sự khác biệt căn bản giữa Một người ra quyết định và Nhiều người cùng ra quyết định (Tính cạnh tranh và phụ thuộc lẫn nhau).
- Lập kế hoạch đa tác tử (Multiagent planning) và bài toán Hợp tác - Phối hợp (Cooperation and coordination).

### 17.2 Lý thuyết trò chơi Không Hợp Tác (Non-Cooperative Game Theory)
- **Trò chơi Dạng Chuẩn (Normal form games):** Mỗi người ra một nước đi duy nhất, ẩn giấu thông tin. (Ví dụ: Song đề Tù nhân - Prisoner's Dilemma).
- Khái niệm cân bằng: **Cân bằng Nash (Nash Equilibrium)** và Chiến lược ưu thế (Dominant Strategy).
- Phúc lợi xã hội (Social welfare) và tính tối ưu Pareto (Pareto optimality).
- Trò chơi lặp lại (Repeated games) và Chiến lược ăn miếng trả miếng (Tit-for-tat).
- **Trò chơi Dạng Mở rộng (Extensive form games):** Biểu diễn dạng cây (Tree) cho trò chơi đánh theo lượt (Cờ vua, Cờ caro). Tính toán Cân bằng hoàn hảo hệ thống con (Subgame perfect equilibrium).

### 17.3 Lý Thuyết Trò Chơi Hợp Tác (Cooperative Game Theory)
- Thành lập các Liên minh (Coalitions).
- Giá trị Shapley (Shapley Value): Cách phân chia lợi ích công bằng nhất khi làm việc nhóm dựa trên đóng góp biên.
- Cấu trúc liên minh, chiến lược và độ phức tạp tính toán trong game hợp tác.

### 17.4 Đưa ra các quyết định tập thể (Making Collective Decisions)
- **Mạng hợp đồng (Contract net):** Giao thức chia nhỏ nhiệm vụ và đấu thầu giữa các tác nhân phần mềm.
- **Đấu giá (Auctions):** Phân bổ tài nguyên khan hiếm. Phân tích các loại đấu giá (Đấu giá Anh, Đấu giá Hà Lan, Đấu giá Vickrey/Kín giá thứ hai).
- **Bỏ phiếu (Voting):** Định lý bất khả thi của Arrow (Arrow's Impossibility Theorem) - Không có hệ thống bầu cử nào là hoàn hảo tuyệt đối.
- Thương lượng (Bargaining) giữa các tác tử.

### Tóm tắt Chương 17
- AI đa tác tử là mô hình chân thực nhất của xã hội loài người. Lý thuyết trò chơi giúp AI biết cách thỏa hiệp, cạnh tranh hoặc hợp tác để đạt lợi ích cao nhất một cách toán học.

## 3. Kế hoạch Hiện thực hóa (Thực thi)

1. Tinh gọn lý thuyết: Game Theory có rất nhiều thuật ngữ kinh tế học (Nash Equilibrium, Pareto, Vickrey Auction), tôi sẽ Việt hóa một cách cẩn thận kèm tiếng Anh trong ngoặc kép để sinh viên tra cứu.
2. Thiết kế Ma trận Payoff bằng môi trường `tabular` của LaTeX để thể hiện trực quan trò chơi "Song đề tù nhân".
3. Lên phương án chèn hình ảnh minh họa cho Cây trò chơi mở rộng và Giao thức Mạng hợp đồng.
4. Biên dịch thử với `pdflatex` để rà soát lỗi tràn lề.

---
**Ghi chú:** Kế hoạch Chương 17 hoàn tất. Tôi sẽ tiếp tục xuất bản kế hoạch cho Chương 18 ngay lập tức.
