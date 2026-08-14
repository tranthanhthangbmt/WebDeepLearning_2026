# Kế hoạch Xây dựng Slide Chương 26: ROBOTICS (Robotics)

## 1. Mục tiêu và Tiêu chuẩn
- **Mục tiêu:** Cung cấp cái nhìn tổng quan về lĩnh vực Robotics trong Trí tuệ nhân tạo, bao gồm phần cứng robot, nhận thức (perception), lập kế hoạch chuyển động (planning and control), và tương tác giữa người với robot.
- **Yêu cầu kỹ thuật LaTeX:** 
  - Biên dịch với `pdflatex` (chuẩn slide học thuật đại học).
  - Có các định nghĩa rõ ràng, công thức toán học về không gian trạng thái, chuyển động, bộ lọc hạt (particle filters) cho định vị, và hàm giá trị trong học tăng cường.
  - Sử dụng block, in đậm các thuật ngữ quan trọng.
  - Sẵn sàng khoảng trống (placeholder) hoặc code chèn ảnh minh họa cho các loại robot, cảm biến và không gian cấu hình (configuration space).
  - Đảm bảo số lượng slide $\geq$ 15 trang, tùy độ chi tiết để trình bày vừa đủ trong một buổi học.

## 2. Cấu trúc Nội dung dự kiến (Mapping Sections)

### Trang Tiêu đề
- **Tiêu đề chính:** CHƯƠNG 26: ROBOTICS (ROBOT)
- **Tiêu đề phụ:** Trí tuệ nhân tạo - Artificial Intelligence
- **Nội dung Chương:** Tóm tắt các nội dung chính: Phần cứng, Nhận thức, Lập kế hoạch & Điều khiển, Học tăng cường, Ứng dụng.

### 26.1 Robots (Giới thiệu về Robot)
- Định nghĩa: Robot là các tác tử vật lý (physical agents) thực hiện nhiệm vụ bằng cách thao tác trên thế giới thực.
- Phân biệt tác tử phần mềm (software agents) và robot.

### 26.2 Robot Hardware (Phần cứng Robot)
- **26.2.1 Phân loại robot theo phần cứng:**
  - Manipulators (Cánh tay robot): Cố định, dùng trong công nghiệp.
  - Mobile robots (Robot di động): Di chuyển trong môi trường (bánh xe, chân, bay, bơi).
- **26.2.2 Cảm biến (Sensing the world):**
  - Passive sensors (Cảm biến bị động): Camera.
  - Active sensors (Cảm biến chủ động): Lidar, Radar, Sonar, Time-of-flight camera.
  - Proprioceptive sensors (Cảm biến nội cảm): Đo trạng thái bên trong robot (động cơ, khớp nối).
- **26.2.3 Tạo chuyển động (Producing motion):**
  - Actuators (Cơ cấu chấp hành): Động cơ điện, hệ thống thủy lực, khí nén.

### 26.3 What kind of problem is robotics solving? (Robot đang giải quyết bài toán gì?)
- Bài toán cốt lõi: Làm sao ánh xạ từ lịch sử cảm biến sang hành động cơ cấu chấp hành trong một môi trường vật lý, liên tục, không chắc chắn và động.

### 26.4 Robotic Perception (Nhận thức của Robot)
- **26.4.1 Định vị và lập bản đồ (Localization and mapping):**
  - **Localization:** Cập nhật niềm tin (belief state) về vị trí hiện tại dựa trên dữ liệu cảm biến. Thường dùng **Mô hình Markov Ẩn (HMM)** hoặc **Bộ lọc hạt (Particle Filters / Monte Carlo Localization)**.
  - **Mapping:** Xây dựng bản đồ của môi trường. Bài toán **SLAM** (Simultaneous Localization and Mapping).
- **26.4.2 Các loại nhận thức khác:** Nhận diện vật thể, theo dõi con người.
- **26.4.3 Học có giám sát và không giám sát:** Áp dụng Deep Learning (CNN) để xử lý ảnh từ camera robot.

### 26.5 Planning and Control (Lập kế hoạch và Điều khiển)
- **26.5.1 Không gian cấu hình (Configuration space):**
  - Biểu diễn trạng thái robot thay vì không gian làm việc (workspace).
  - Kinematics (Động học) và Inverse Kinematics (Động học nghịch).
- **26.5.2 Lập kế hoạch chuyển động (Motion planning):**
  - Tìm đường đi (path) từ điểm bắt đầu đến điểm kết thúc, tránh chướng ngại vật (Free space vs. Obstacle space).
  - Thuật toán Probabilistic Roadmap (PRM), Rapidly-exploring Random Trees (RRT).
- **26.5.3 Điều khiển bám quỹ đạo (Trajectory tracking control):** PID controllers.
- **26.5.4 Điều khiển tối ưu (Optimal control):** Cực tiểu hóa cost function cho các chuyển động trơn tru.

### 26.6 Planning Uncertain Movements (Lập kế hoạch cho các chuyển động không chắc chắn)
- Lập kế hoạch với MDPs (Markov Decision Processes) và POMDPs để đối phó với nhiễu trong điều khiển và cảm biến.

### 26.7 Reinforcement Learning in Robotics (Học tăng cường trong Robot)
- Áp dụng RL (Reinforcement Learning) cho robot thật, đối mặt với "curse of dimensionality" và chi phí thu thập dữ liệu thế giới thực.
- **26.7.1 Khai thác mô hình (Exploiting models):** Model-based RL, học mô hình dynamics rồi lập kế hoạch.
- **26.7.2 Khai thác thông tin khác:** Imitation learning (Học bắt chước), Sim-to-Real transfer (Huấn luyện trong mô phỏng rồi chuyển sang robot thật).

### 26.8 Humans and Robots (Con người và Robot)
- **26.8.1 Sự phối hợp (Coordination):** Robot hoạt động an toàn và dự đoán được trong môi trường có con người.
- **26.8.2 Học những gì con người muốn (Learning to do what humans want):** Inverse Reinforcement Learning (IRL), Value Alignment.

### 26.9 Alternative Robotic Frameworks (Các khung robot thay thế)
- **26.9.1 Bộ điều khiển phản xạ (Reactive controllers):** Phản ứng trực tiếp từ cảm biến đến hành động mà không cần mô hình hóa môi trường (bỏ qua Planning).
- **26.9.2 Kiến trúc Subsumption:** Xây dựng các lớp hành vi từ cơ bản đến phức tạp (e.g., tránh vật cản -> đi tới đích).

### 26.10 Application Domains (Lĩnh vực ứng dụng)
- Công nghiệp và Nông nghiệp.
- Vận chuyển (Xe tự lái, máy bay không người lái giao hàng).
- Y tế (Robot phẫu thuật, hỗ trợ người khuyết tật).
- Khám phá không gian (Mars rovers).

### Tổng kết Chương 26
- Nhấn mạnh sự kết hợp giữa AI và cơ khí: Nhận thức (Perception) + Lập kế hoạch (Planning) + Hành động (Action). Robot đối mặt với môi trường vật lý liên tục, đầy nhiễu và không chắc chắn.

## 3. Kế hoạch Hiện thực hóa (Thực thi sinh Slide)

1. **Từ vựng chuyên ngành:** Sử dụng song ngữ (Tiếng Việt - Tiếng Anh) cho các thuật ngữ cốt lõi như Configuration space (Không gian cấu hình), Localization (Định vị), SLAM, Inverse Kinematics, PRM, RRT.
2. **Toán học và Công thức:** Đưa vào các biểu thức toán học như Bayes rule cho Localization, công thức tính DOF (Degrees of Freedom) hoặc hàm lỗi PID.
3. **Phân bổ trang:**
   - Mỗi tiểu mục (ví dụ 26.2, 26.4, 26.5) sẽ có từ 1-3 slide tùy lượng kiến thức.
   - Ước tính tổng số slide sẽ rơi vào khoảng **18 - 22 slides**.
4. **Hình ảnh minh họa:** Chừa các khối `\begin{figure}` trống hoặc mã ví dụ để thầy chèn ảnh (ví dụ: ảnh xe tự lái, sơ đồ PRM/RRT, hình ảnh cánh tay robot).
