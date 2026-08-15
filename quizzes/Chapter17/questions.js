const questions = [
  {
    type: "mcq",
    question: "Mục đích chính của việc học một 'không gian tiềm ẩn' (latent space) trong các mô hình tạo hình ảnh là gì?",
    options: [
      "Tìm một không gian vectơ chiều thấp liên tục, nơi mọi điểm đều có thể giải mã thành một hình ảnh hợp lệ trông giống như thật.",
      "Lọc bỏ tất cả các chi tiết thừa trong hình ảnh để máy tính không phải lưu trữ các điểm ảnh (pixel) màu đen trên màn hình.",
      "Tăng kích thước của hình ảnh lên gấp hàng nghìn lần để tạo ra không gian đủ rộng cho các thuật toán tìm kiếm cục bộ.",
      "Dịch trực tiếp các hình ảnh từ định dạng PNG sang JPEG mà không làm thay đổi phân bố xác suất gốc của dữ liệu."
    ],
    answer: 0,
    explanation: "Không gian tiềm ẩn (Latent space) là một không gian biểu diễn nén (chiều thấp) nhưng mang tính cấu trúc cao. Bằng cách lấy mẫu một điểm bất kỳ trong không gian này và đưa qua bộ giải mã, ta thu được một hình ảnh mới."
  },
  {
    type: "mcq",
    question: "Bộ mã hóa tự động biến thiên (VAE) khác biệt với bộ mã hóa tự động (Autoencoder) cổ điển ở điểm cốt lõi nào?",
    options: [
      "VAE ánh xạ đầu vào thành các tham số của một phân bố thống kê (trung bình và phương sai) thay vì một mã cố định.",
      "VAE không sử dụng mạng nơ-ron tích chập (CNN) mà thay thế hoàn toàn bằng mô hình Transformer nhiều lớp.",
      "VAE chỉ có thể xử lý được văn bản và không thể xử lý hình ảnh do giới hạn của hàm mất mát Cross-Entropy.",
      "VAE bỏ qua bộ giải mã (Decoder) và chỉ giữ lại bộ mã hóa (Encoder) để tối ưu hóa việc phân loại đối tượng."
    ],
    answer: 0,
    explanation: "Thay vì nén hình ảnh thành một vectơ cố định, VAE nén nó thành `z_mean` (trung bình) và `z_log_var` (phương sai) của một phân bố chuẩn, sau đó lấy mẫu ngẫu nhiên từ phân bố này."
  },
  {
    type: "mcq",
    question: "Trong VAE, thủ thuật lấy mẫu (sampling) từ phân bố chuẩn được thực hiện thông qua công thức `z = z_mean + exp(z_log_variance * 0.5) * epsilon`. Thành phần `epsilon` đóng vai trò gì?",
    options: [
      "Là một tensor chứa nhiễu ngẫu nhiên, giúp quá trình lấy mẫu có tính ngẫu nhiên nhưng vẫn giữ được khả năng tính toán đạo hàm ngược.",
      "Là hằng số học tập (learning rate) giúp mô hình giảm dần tốc độ hội tụ sau mỗi kỷ nguyên (epoch) để tránh overfitting.",
      "Là ma trận trọng số cuối cùng của lớp phân loại, quyết định hình ảnh được sinh ra sẽ thuộc nhãn nào (ví dụ: chó, mèo).",
      "Là giá trị trung bình cộng của toàn bộ tập dữ liệu huấn luyện, giúp mô hình giữ được sự cân bằng về độ sáng."
    ],
    answer: 0,
    explanation: "`epsilon` (được rút từ phân phối N(0,1)) tạo ra tính ngẫu nhiên (Reparameterization trick), buộc bộ giải mã phải học cách tạo ra hình ảnh hợp lệ cho mọi điểm xung quanh vùng `z_mean`."
  },
  {
    type: "mcq",
    question: "Hàm mất mát (Loss function) của VAE bao gồm hai thành phần: Tổn thất tái thiết (Reconstruction loss) và Tổn thất chính quy hóa. Thành phần chính quy hóa thường được tính bằng độ đo nào?",
    options: [
      "Phân kỳ Kullback–Leibler (KL Divergence), ép phân bố đầu ra của bộ mã hóa tiến gần đến một phân bố chuẩn tập trung quanh 0.",
      "Sai số toàn phương trung bình (Mean Squared Error), ép tất cả các điểm ảnh của ảnh đầu ra phải trùng khớp hoàn toàn với ảnh đầu vào.",
      "Hàm mất mát tương phản (Contrastive Loss), dùng để đẩy xa khoảng cách giữa các hình ảnh không cùng một lớp với nhau.",
      "Độ đo tương đồng Cosine (Cosine Similarity), dùng để tính góc lệch giữa vectơ đầu vào và vectơ của nhãn thực tế."
    ],
    answer: 0,
    explanation: "KL Divergence (Tổn thất KL) phạt mô hình nếu phân bố tiềm ẩn mà nó học được lệch quá xa so với phân phối chuẩn N(0,1), giúp không gian tiềm ẩn liên tục và tròn trịa."
  },
  {
    type: "mcq",
    question: "Ý tưởng cốt lõi (sự thấu hiểu quan trọng nhất) đằng sau các mô hình khuếch tán (Diffusion Models) là gì?",
    options: [
      "Nếu ta có thể huấn luyện một mô hình loại bỏ một lượng nhiễu nhỏ, ta có thể lặp lại nó nhiều lần để khử nhiễu thuần túy thành một bức ảnh thực tế.",
      "Ta chỉ cần phóng to hình ảnh nhiễu lên 100 lần, các chi tiết sắc nét sẽ tự động được phục hồi dựa trên bộ lọc song phương (bilateral filter).",
      "Mọi bức ảnh nhiễu đều là một bức tranh nghệ thuật, ta chỉ cần sử dụng GAN để gán cho nó một nhãn văn bản phù hợp.",
      "Sử dụng bộ nhớ đệm (Cache) để lưu trữ tất cả các hình ảnh trên Internet, sau đó tìm hình ảnh gần giống nhất với ảnh nhiễu."
    ],
    answer: 0,
    explanation: "Diffusion model hoạt động bằng cách khử nhiễu từng bước nhỏ (reverse diffusion). Bắt đầu từ nhiễu trắng hoàn toàn, sau hàng chục/trăm bước lặp, một hình ảnh sắc nét được hình thành."
  },
  {
    type: "mcq",
    question: "Mô hình mạng nơ-ron nào thường được sử dụng làm lõi để dự đoán lượng nhiễu trong từng bước của mô hình Khuếch tán (Diffusion)?",
    options: [
      "Mạng U-Net, với khả năng lấy mẫu xuống (downsampling) rồi lấy mẫu lên (upsampling) kết hợp với các kết nối dư (skip connections).",
      "Mạng Transformer thuần túy chỉ bao gồm các khối giải mã (Decoder-only), chuyên xử lý các dữ liệu chuỗi thời gian một chiều.",
      "Mạng Đồ thị (Graph Neural Network), chuyên xử lý các cấu trúc hình học phức tạp không đồng nhất như mạng xã hội.",
      "Mạng RNN hai chiều (Bi-directional RNN), dùng để ghi nhớ chuỗi pixel từ trái sang phải và từ phải sang trái."
    ],
    answer: 0,
    explanation: "U-Net là kiến trúc hoàn hảo cho tác vụ image-to-image (như khử nhiễu). Nó duy trì được chi tiết không gian cục bộ nhờ vào các kết nối tắt (skip connections) giữa các nhánh đối xứng."
  },
  {
    type: "mcq",
    question: "Khái niệm 'Thời gian khuếch tán' (Diffusion time) hay 'Lịch trình khuếch tán' (Diffusion schedule) dùng để chỉ điều gì?",
    options: [
      "Mối quan hệ xác định lượng nhiễu (noise rate) và lượng tín hiệu (signal rate) còn lại trong ảnh ở mỗi bước lặp của quá trình.",
      "Tổng thời gian mô hình mất để huấn luyện xong một kỷ nguyên (epoch), thường được đo bằng số giờ trên một chiếc GPU mạnh.",
      "Tốc độ mà ánh sáng truyền qua bộ mã hóa, quyết định tốc độ suy luận của mô hình trên các thiết bị di động.",
      "Lịch trình cập nhật trọng số của trình tối ưu hóa (Optimizer) từ tốc độ học cao xuống tốc độ học thấp (Decay)."
    ],
    answer: 0,
    explanation: "Lịch trình khuếch tán (VD: cosine schedule) quy định cách tín hiệu bị phá hủy dần và nhiễu tăng dần trong quá trình forward (từ t=0 đến t=1)."
  },
  {
    type: "mcq",
    question: "Trong kiến trúc U-Net được sử dụng cho mô hình Khuếch tán, tại sao các bước 'lấy mẫu xuống' (downsampling) lại dùng `strides` thay vì `MaxPooling`?",
    options: [
      "Vì stride tích chập giúp mô hình bảo toàn tốt hơn vị trí không gian của các đặc trưng (where things are), vốn rất quan trọng để tái tạo ảnh.",
      "Vì MaxPooling yêu cầu một thư viện đồ họa đặc biệt mà TensorFlow/Keras hiện tại không còn hỗ trợ ở các phiên bản mới.",
      "Vì strides có khả năng tự động thay đổi kích thước ma trận theo hình dạng tự do mà không cần tính toán bằng CPU.",
      "Vì lớp tích chập với strides sẽ hoàn toàn bỏ qua các pixel có giá trị bằng 0, giúp tiết kiệm bộ nhớ RAM khổng lồ."
    ],
    answer: 0,
    explanation: "Việc sử dụng MaxPooling thường làm mất đi thông tin vị trí không gian chính xác của đối tượng. Trong các bài toán tạo/tái tạo ảnh (như U-Net), Strided Convolutions được ưa chuộng hơn."
  },
  {
    type: "mcq",
    question: "Đầu ra của mô hình U-Net khử nhiễu trong vòng lặp của Diffusion Model là gì?",
    options: [
      "Một mặt nạ nhiễu dự đoán (Predicted noise mask), phần này sau đó được trừ ra khỏi ảnh nhiễu hiện tại.",
      "Xác suất của 1000 lớp nhãn hình ảnh, chỉ ra bức ảnh này là chó, mèo, ô tô hay các vật thể quen thuộc khác.",
      "Một chuỗi văn bản bằng tiếng Anh giải thích những gì mô hình đang nhìn thấy trong đống nhiễu loạn.",
      "Bức ảnh hoàn hảo sắc nét 100% chỉ sau đúng một lần chạy dự đoán duy nhất, bỏ qua toàn bộ vòng lặp."
    ],
    answer: 0,
    explanation: "Thay vì dự đoán trực tiếp hình ảnh sạch (vốn rất khó), mô hình U-Net học cách dự đoán lượng nhiễu (noise mask) đã được tiêm vào ở bước đó để thuật toán dần dần trừ nó đi."
  },
  {
    type: "mcq",
    question: "Mô hình chuyển văn bản thành hình ảnh (Text-to-Image) như Stable Diffusion kết nối văn bản với quá trình sinh ảnh bằng cách nào?",
    options: [
      "Sử dụng một bộ mã hóa văn bản (Text Encoder) chuyển đổi câu lệnh thành vectơ, sau đó tiêm (inject) vectơ này vào mạng U-Net khử nhiễu để định hướng nó.",
      "Dịch câu lệnh tiếng Anh sang ngôn ngữ máy nhị phân, sau đó in các ký tự số 0 và 1 trực tiếp lên các điểm ảnh trên bức tranh.",
      "So sánh độ dài của câu văn bản và tự động sinh ra một hình ảnh có kích thước tính bằng centimet tương đương với số từ.",
      "Tự động tìm kiếm câu lệnh trên Google Images và sao chép pixel của bức ảnh đầu tiên trả về."
    ],
    answer: 0,
    explanation: "Text Encoder (thường là CLIP) nhúng văn bản thành dạng vectơ tiềm ẩn ngữ nghĩa. Các vectơ này được đưa vào mô hình U-Net (qua cross-attention) để định hướng quá trình khử nhiễu tạo hình ảnh đúng như ý muốn."
  },
  {
    type: "mcq",
    question: "Kỹ thuật 'Dấu nhắc tiêu cực' (Negative prompt) trong tạo hình ảnh hoạt động dựa trên nguyên lý nào?",
    options: [
      "Trong lúc khử nhiễu, mô hình chủ động điều hướng quá trình ra xa khỏi vùng không gian tiềm ẩn khớp với mô tả của dấu nhắc tiêu cực.",
      "Mô hình tự động bôi đen (blackout) tất cả các vùng trên bức ảnh chứa các đối tượng được nhắc đến trong dấu nhắc tiêu cực.",
      "Mô hình sẽ sinh ra bức ảnh, sau đó dùng công cụ Photoshop tích hợp để cắt bỏ các vật thể không mong muốn một cách vật lý.",
      "Mô hình từ chối sinh ảnh và báo lỗi nếu người dùng nhập vào những từ nằm trong danh sách dấu nhắc tiêu cực bị cấm."
    ],
    answer: 0,
    explanation: "Classifier-Free Guidance cho phép kết hợp cả lời nhắc tích cực (kéo hình ảnh về phía nó) và tiêu cực (đẩy hình ảnh ra xa nó) trong quá trình cập nhật mỗi bước khử nhiễu."
  },
  {
    type: "mcq",
    question: "Nếu lấy mã hóa vectơ của hai câu lệnh 'một con mèo' và 'một con chó' rồi tính giá trị trung bình (interpolate), hình ảnh sinh ra từ vectơ này sẽ như thế nào?",
    options: [
      "Mô hình sẽ sinh ra một bức ảnh kết hợp mềm mại, nội suy giữa đặc điểm của mèo và chó nhờ tính liên tục của không gian tiềm ẩn.",
      "Mô hình sẽ chia màn hình ra làm hai nửa, bên trái hiện ảnh con mèo rõ nét, bên phải hiện ảnh con chó rõ nét.",
      "Mô hình sẽ báo lỗi không tương thích định dạng vì không thể cộng hai mảng chứa chuỗi ký tự (String) với nhau.",
      "Mô hình sẽ chỉ hiện ảnh đứt gãy hoàn toàn (nhiễu trắng) do vi phạm quy tắc định dạng của thư viện đồ họa GPU."
    ],
    answer: 0,
    explanation: "Vì không gian nhúng của Text Encoder (CLIP) và không gian tiềm ẩn của Diffusion đều có tính liên tục và ngữ nghĩa chặt chẽ, việc nội suy (interpolation) sẽ tạo ra các hình ảnh trung gian hợp lý."
  },
  {
    type: "mcq",
    question: "Tại sao các ảnh tạo bởi Stable Diffusion (VD: người phi hành gia cưỡi voi origami) đôi khi vẫn chứa lỗi hình học như 'voi có ba ngà' hay 'người có sáu ngón tay'?",
    options: [
      "Vì mô hình không thực sự hiểu kiến thức giải phẫu hay vật lý mà chỉ đang cố nội suy các pixel dựa trên phân phối dữ liệu đã học.",
      "Vì nhà phát triển cố tình lập trình các lỗi nhỏ này vào trong mô hình để chống giả mạo bản quyền tác phẩm nghệ thuật.",
      "Vì card đồ họa GPU của người dùng bị quá nhiệt trong quá trình kết xuất hình ảnh, làm cháy các điểm ảnh trên màn hình.",
      "Vì người dùng quên không thêm dấu phẩy (,) vào sau chữ 'ngà voi' trong câu lệnh mô tả tiếng Anh."
    ],
    answer: 0,
    explanation: "Mặc dù tạo ra kết quả ấn tượng, nhưng mô hình AI sinh ảnh thực chất chỉ là học phân phối pixel thống kê (pattern matching), chứ không được lập trình để mô phỏng thế giới vật lý thực 3D, nên dễ bị sai lệch về giải phẫu."
  },
  {
    type: "mcq",
    question: "Để cải thiện chất lượng (giảm nhiễu, tăng chi tiết) của ảnh sinh ra từ Stable Diffusion, bạn thường cần điều chỉnh tham số nào?",
    options: [
      "Tăng số bước khử nhiễu (num_steps/diffusion_steps), cho phép mô hình có nhiều lần lặp hơn để tinh chỉnh chi tiết.",
      "Tăng chỉ số Learning Rate lên cực đại (ví dụ 1.0) để ép mô hình học nhanh hơn trong một chu kỳ xung nhịp CPU.",
      "Chỉnh kích thước batch size xuống 1 để mô hình có thể dồn toàn bộ sự chú ý của GPU vào một hình ảnh duy nhất.",
      "Loại bỏ hoàn toàn Dấu nhắc tiêu cực (Negative prompt) để không làm mô hình bị phân tâm bởi các thông tin ngoài lề."
    ],
    answer: 0,
    explanation: "Trong suy luận (inference), việc chạy vòng lặp khử nhiễu với nhiều bước hơn (VD: từ 20 lên 50 steps) sẽ giúp bức ảnh chuyển đổi từ nhiễu sang ảnh sắc nét một cách mượt mà và chi tiết hơn, đánh đổi bằng thời gian sinh lâu hơn."
  },
  {
    type: "mcq",
    question: "Mô hình khuếch tán (Diffusion) khác VAE (Variational Autoencoder) ở ưu điểm nổi trội nhất nào đối với tác vụ sinh ảnh?",
    options: [
      "Mô hình khuếch tán tạo ra hình ảnh với độ trung thực (fidelity) cao hơn, sắc nét và chi tiết vượt trội so với VAE thường bị mờ.",
      "Mô hình khuếch tán chỉ cần huấn luyện mất vài giây trên một máy tính cá nhân không cần thẻ đồ họa rời GPU.",
      "Mô hình khuếch tán chỉ sử dụng các phép toán cộng trừ cơ bản mà không cần tính toán bất kỳ ma trận tích chập nào.",
      "Mô hình khuếch tán trực tiếp nén dữ liệu video thành file mp4 có dung lượng siêu nhỏ phục vụ cho việc truyền tải mạng."
    ],
    answer: 0,
    explanation: "VAE nổi bật ở cấu trúc tiềm ẩn rõ ràng, nhưng hình ảnh thường bị mờ (do bản chất của L2/KL loss). Diffusion models hiện nay thống trị vì khả năng sinh ra chi tiết ảnh vô cùng sắc nét và chân thực (SOTA)."
  },
  {
    type: "fill",
    question: "Tổn thất ______________ trong VAE (thường là Kullback–Leibler divergence) giúp điều chuẩn không gian tiềm ẩn, ép phân phối trở nên liên tục và mượt mà.",
    options: ["chính quy hóa", "chính quy", "KL", "Kullback-Leibler"],
    answer: "KL",
    explanation: "KL Loss (Kullback-Leibler) được dùng như một dạng chính quy hóa (regularization term) để ép phân bố mã hóa Z tiến về gần phân bố chuẩn N(0,1)."
  },
  {
    type: "fill",
    question: "Kỹ thuật dùng trong VAE: `z = z_mean + exp(z_log_var) * epsilon` được gọi là thủ thuật tham số hóa lại (______________ trick).",
    options: ["reparameterization", "Reparameterization"],
    answer: "reparameterization",
    explanation: "Reparameterization trick (thủ thuật tham số hóa lại) cho phép truyền ngược đạo hàm qua một quá trình lấy mẫu ngẫu nhiên."
  },
  {
    type: "fill",
    question: "Kiến trúc mạng tích chập thường được dùng để dự đoán mặt nạ nhiễu (noise mask) trong mô hình khuếch tán gọi là ______________.",
    options: ["U-Net", "unet", "Unet"],
    answer: "U-Net",
    explanation: "U-Net lấy tên theo hình dáng chữ U của kiến trúc: Downsampling -> Bottleneck -> Upsampling với skip connections."
  },
  {
    type: "fill",
    question: "Trong Diffusion Model, quá trình thêm dần nhiễu vào ảnh sạch (Forward process) làm ảnh mờ đi cho đến khi trở thành nhiễu trắng thuần túy. Quá trình mô hình làm sạch ảnh từ nhiễu gọi là khuếch tán ______________.",
    options: ["ngược", "reverse"],
    answer: "ngược",
    explanation: "Reverse diffusion (khuếch tán ngược) là quá trình mạng học cách khử nhiễu dần dần (từng bước) để khôi phục ảnh nguyên gốc."
  },
  {
    type: "fill",
    question: "Mô hình sinh hình ảnh tạo chấn động như Stable Diffusion, Midjourney thuộc họ mô hình ______________.",
    options: ["khuếch tán", "diffusion", "Khuếch tán"],
    answer: "khuếch tán",
    explanation: "Sự trỗi dậy của AI tạo ảnh từ 2022 hoàn toàn là nhờ sức mạnh của các mô hình khuếch tán (Diffusion Models)."
  },
  {
    type: "matching",
    question: "Ghép nối các thành phần của kiến trúc VAE:",
    options: [
      "Bộ mã hóa (Encoder)",
      "Bộ giải mã (Decoder)",
      "Không gian tiềm ẩn (Latent space)"
    ],
    answer: [
      "Ánh xạ hình ảnh đầu vào thành 2 tham số: giá trị trung bình (mean) và phương sai (variance).",
      "Ánh xạ một điểm vectơ từ không gian tiềm ẩn trở lại thành một lưới pixel (hình ảnh gốc).",
      "Không gian vectơ chiều thấp liên tục, nơi mà các khái niệm ngữ nghĩa của ảnh được tổ chức gọn gàng."
    ],
    explanation: "VAE gồm 3 phần chính: Encoder (dịch ảnh sang tham số), quá trình Sampling, và Decoder (phục hồi ảnh)."
  },
  {
    type: "matching",
    question: "Ghép nối các khái niệm trong mô hình Khuếch tán (Diffusion):",
    options: [
      "Lịch trình khuếch tán (Diffusion schedule)",
      "Mặt nạ nhiễu dự đoán (Predicted noise mask)",
      "Tín hiệu (Signal rate)"
    ],
    answer: [
      "Quy định tỷ lệ nhiễu và tín hiệu ở từng bước cụ thể của thời gian t.",
      "Đầu ra của mạng U-Net, dự đoán lượng nhiễu đã được cộng thêm vào ảnh ở bước t.",
      "Đại lượng đại diện cho phần hình ảnh gốc sạch (clean image) chưa bị phá hủy bởi nhiễu."
    ],
    explanation: "Quá trình khuếch tán là một sự pha trộn được kiểm soát giữa tín hiệu nguyên bản (signal) và nhiễu ngẫu nhiên (noise)."
  },
  {
    type: "matching",
    question: "Ghép nối các giai đoạn bên trong kiến trúc mạng U-Net:",
    options: [
      "Downsampling stage (Lấy mẫu xuống)",
      "Middle stage (Giai đoạn giữa)",
      "Upsampling stage (Lấy mẫu lên)"
    ],
    answer: [
      "Sử dụng Strides Conv2D để giảm kích thước bản đồ đặc trưng (ví dụ: 128x128 xuống 16x16).",
      "Xử lý các đặc trưng ngữ nghĩa mức cao nhất với kích thước bản đồ không đổi nhỏ nhất.",
      "Kết hợp nội dung từ các nhánh skip connections và khôi phục dần kích thước (lên 128x128)."
    ],
    explanation: "Kiến trúc chữ U của U-Net giúp nó nén ảnh để hiểu ngữ cảnh lớn, nhưng vẫn giữ được chi tiết tinh xảo nhờ skip connections."
  },
  {
    type: "matching",
    question: "So sánh 3 loại mô hình tạo hình ảnh (Generative Models):",
    options: [
      "VAE (Variational Autoencoder)",
      "Diffusion Models",
      "GAN (Generative Adversarial Networks)"
    ],
    answer: [
      "Mô hình nén dữ liệu vào phân bố chuẩn. Hình ảnh sinh ra liên tục, nhưng thường kém độ sắc nét.",
      "Mô hình khử nhiễu lặp vòng (Iterative Denoising). SOTA hiện nay, ảnh cực nét nhưng chạy chậm (nhiều bước).",
      "Gồm 2 mạng (Generator và Discriminator) đối kháng nhau. Sinh ảnh nhanh, đẹp nhưng khó huấn luyện (thường sụp đổ mode)."
    ],
    explanation: "Mỗi mô hình sinh đều có đánh đổi (trade-offs) về tốc độ, chất lượng và tính ổn định. Hiện tại Diffusion đang thống trị lĩnh vực 2D Image Generation."
  },
  {
    type: "matching",
    question: "Ghép nối các thành phần trong lệnh tạo văn bản thành hình (Text-to-Image Generation):",
    options: [
      "Prompt (Dấu nhắc tích cực)",
      "Negative Prompt (Dấu nhắc tiêu cực)",
      "Num Steps (Số bước khuếch tán)"
    ],
    answer: [
      "Câu lệnh mô tả những gì bạn muốn xuất hiện trong hình ảnh (VD: Astronaut riding an elephant).",
      "Câu lệnh mô tả những thứ bạn muốn mô hình tránh xa (VD: blurry, low quality, bad anatomy).",
      "Tham số quy định có bao nhiêu lần lặp khử nhiễu sẽ được thực hiện (VD: 20, 50 bước)."
    ],
    explanation: "Bằng cách kết hợp linh hoạt 3 tham số này, người dùng (AI Artist) có thể ép Stable Diffusion sinh ra các tác phẩm mỹ thuật đỉnh cao."
  },
  {
    type: "sorting",
    question: "Sắp xếp thứ tự luồng dữ liệu (Data flow) đi qua một mô hình VAE hoàn chỉnh để huấn luyện:",
    options: [
      "Ảnh đầu vào đi qua Encoder, được nén thành hai vectơ: Z_mean và Z_log_var.",
      "Sử dụng thủ thuật Reparameterization để lấy mẫu ngẫu nhiên một vectơ tiềm ẩn Z.",
      "Đưa vectơ Z đi qua Decoder (sử dụng Conv2DTranspose) để khôi phục lại ảnh.",
      "Tính toán Tổn thất tái thiết (Lỗi giữa ảnh gốc và ảnh khôi phục) + Tổn thất KL.",
      "Lan truyền ngược (Backpropagation) để cập nhật trọng số cho cả Encoder và Decoder."
    ],
    answer: [0, 1, 2, 3, 4],
    explanation: "Quá trình forward của VAE bao gồm Nén -> Lấy mẫu ngẫu nhiên -> Giải nén -> Tính loss -> Lan truyền ngược."
  },
  {
    type: "sorting",
    question: "Sắp xếp vòng lặp Khuếch tán ngược (Reverse Diffusion Generation Loop) sinh ảnh từ nhiễu:",
    options: [
      "Khởi tạo một tensor chứa Nhiễu ngẫu nhiên thuần túy (Pure Noise) với kích thước ảnh (VD: 512x512).",
      "Tính toán lượng nhiễu (noise rate) và tín hiệu (signal rate) cho Bước T hiện tại.",
      "Đưa tensor ảnh cùng với embeddings văn bản vào mạng U-Net để dự đoán mặt nạ nhiễu.",
      "Trừ lượng nhiễu dự đoán ra khỏi tensor ảnh để tạo ra ảnh sạch hơn một chút (ít nhiễu hơn).",
      "Lặp lại bước 2-4 cho đến khi Bước T = 0 (loại bỏ hoàn toàn nhiễu, ảnh hiện ra rõ nét)."
    ],
    answer: [0, 1, 2, 3, 4],
    explanation: "Quá trình khử nhiễu bắt đầu từ t=T (100% nhiễu) lùi dần về t=0 (0% nhiễu), ở mỗi nhịp U-Net sẽ cạo đi một lớp nhiễu mỏng."
  },
  {
    type: "sorting",
    question: "Sắp xếp các bước để thực hiện phép 'Nội suy không gian tiềm ẩn' (Latent Interpolation) giữa 2 câu lệnh A và B:",
    options: [
      "Tính toán vectơ nhúng văn bản (Text Embeddings) cho Câu lệnh A.",
      "Tính toán vectơ nhúng văn bản (Text Embeddings) cho Câu lệnh B.",
      "Tạo một chuỗi các vectơ trung gian nằm giữa A và B bằng phép tính trung bình trọng số (Lerp).",
      "Đưa từng vectơ trung gian kết hợp với cùng một ma trận nhiễu ban đầu vào mô hình Khuếch tán.",
      "Thu được chuỗi hình ảnh nội suy mượt mà, biến đổi dần từ ý tưởng A sang ý tưởng B."
    ],
    answer: [0, 1, 2, 3, 4],
    explanation: "Nhờ tính liên tục của Text Embeddings (CLIP), việc tính điểm giữa (ví dụ 50% A + 50% B) sẽ tạo ra một khái niệm kết hợp (ví dụ: lai giữa chó và mèo)."
  },
  {
    type: "sorting",
    question: "Sắp xếp quy trình làm việc (Forward pass) của mạng U-Net trong tác vụ Khử nhiễu:",
    options: [
      "Ảnh nhiễu được đưa vào Giai đoạn Lấy mẫu xuống (Downsampling stage) qua các lớp Conv2D strides=2.",
      "Kích thước giảm đi nhưng số lượng kênh (filters) tăng lên để trích xuất đặc trưng ngữ nghĩa.",
      "Đi qua Giai đoạn Giữa (Middle block) chứa các khối dư (residual block).",
      "Đưa vào Giai đoạn Lấy mẫu lên (Upsampling stage), kết hợp (Concatenate) với các đặc trưng nhánh tắt (Skip connections).",
      "Đầu ra cuối cùng là một mặt nạ nhiễu (Noise Mask) có kích thước y hệt ảnh đầu vào ban đầu."
    ],
    answer: [0, 1, 2, 3, 4],
    explanation: "Đây là cách luồng dữ liệu (tensor) chảy qua chữ U của mạng U-Net, đảm bảo mô hình vừa hiểu được cấu trúc tổng thể, vừa vẽ lại được chi tiết từng góc cạnh."
  },
  {
    type: "sorting",
    question: "Sắp xếp quy trình chuẩn bị tập dữ liệu (Dataset) để huấn luyện mô hình hình ảnh (VD: Oxford Flowers):",
    options: [
      "Tải file nén `.tgz` chứa hàng ngàn hình ảnh hoa từ Internet về máy tính nội bộ.",
      "Giải nén dữ liệu và phân loại thư mục.",
      "Dùng `image_dataset_from_directory` để tạo dataset lặp (iterable).",
      "Áp dụng cờ `crop_to_aspect_ratio=True` để resize ảnh về 128x128 mà không làm méo tỷ lệ (Aspect ratio).",
      "Chia ảnh thành các Batch (Lô) có kích thước bằng nhau (VD: 32 ảnh/batch) bằng hàm `.rebatch()`."
    ],
    answer: [0, 1, 2, 3, 4],
    explanation: "Tiền xử lý ảnh trong Keras/TensorFlow thường tuân thủ nguyên tắc: Tải dữ liệu -> Đọc qua tf.data -> Resize -> Batching."
  }
];

export default questions;
