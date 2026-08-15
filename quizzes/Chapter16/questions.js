const questions = [
  {
    type: "mcq",
    question: "Mô hình ngôn ngữ lớn (LLM - Large Language Model) thuộc loại mô hình học máy nào?",
    options: [
      "Mô hình tổng quát (Generative model) có khả năng sinh ra nội dung mới dựa trên phân bố xác suất học được.",
      "Mô hình phân biệt (Discriminative model) chuyên phân loại dữ liệu đầu vào thành các nhãn đã biết trước.",
      "Mô hình học tăng cường (Reinforcement model) tự động tối ưu hóa hành vi bằng cách tương tác với môi trường.",
      "Mô hình học không giám sát hoàn toàn, không có bất kỳ dạng nhãn nào trong toàn bộ quá trình huấn luyện."
    ],
    answer: 0,
    explanation: "LLM (như GPT, Llama, Gemma) là các mô hình tổng quát (Generative models), chuyên lấy mẫu từ không gian tiềm ẩn thống kê để sinh ra văn bản/nội dung mới."
  },
  {
    type: "mcq",
    question: "GPT (Generative Pretrained Transformer) khác với mô hình Transformer nguyên bản (từ bài báo Attention Is All You Need) ở điểm thiết kế cấu trúc nào?",
    options: [
      "Nó loại bỏ hoàn toàn bộ mã hóa (Encoder) và chỉ sử dụng các khối giải mã (Decoder-only).",
      "Nó chỉ sử dụng bộ mã hóa (Encoder-only) giống như BERT để hiểu ngữ cảnh hai chiều.",
      "Nó sử dụng RNN thay thế cho cơ chế Attention để tiết kiệm bộ nhớ khi xử lý chuỗi dài.",
      "Nó chèn thêm một lớp Tích chập (CNN) trước mỗi khối Attention để phân tích cục bộ."
    ],
    answer: 0,
    explanation: "GPT là kiến trúc Decoder-only. Nó loại bỏ hoàn toàn Encoder, điều này có nghĩa là thông tin chỉ có thể chảy từ trái sang phải (nhân quả/causal), không có cái nhìn hai chiều."
  },
  {
    type: "mcq",
    question: "Trong GPT, quá trình 'Tiền huấn luyện' (Pretraining) thực chất đang giải quyết bài toán cốt lõi nào?",
    options: [
      "Dự đoán mã thông báo tiếp theo (next-token prediction) trên một lượng khổng lồ dữ liệu văn bản thô.",
      "Dịch các câu từ ngôn ngữ này sang ngôn ngữ khác để xây dựng một từ điển đa ngôn ngữ toàn cầu.",
      "Phân tích cảm xúc của tất cả các bài viết trên mạng internet để lọc ra những nội dung độc hại.",
      "Học cách trả lời chính xác các câu hỏi kiến thức phổ thông được trích xuất từ bách khoa toàn thư."
    ],
    answer: 0,
    explanation: "Tiền huấn luyện GPT (Pretraining) sử dụng kỹ thuật học tự giám sát bằng cách đoán từ tiếp theo (next-token prediction). Nhờ lượng dữ liệu khổng lồ (hàng nghìn tỷ token), nó học được cấu trúc ngôn ngữ và kiến thức thế giới."
  },
  {
    type: "mcq",
    question: "Chiến lược lấy mẫu 'Tìm kiếm tham lam' (Greedy search) trong quá trình sinh văn bản hoạt động như thế nào?",
    options: [
      "Ở mỗi bước, nó luôn chọn mã thông báo có xác suất dự đoán cao nhất làm đầu ra tiếp theo.",
      "Nó sinh ra hàng nghìn chuỗi ngẫu nhiên, sau đó chọn chuỗi dài nhất để hiển thị cho người dùng.",
      "Nó tự động bỏ qua các từ như 'the', 'a', 'an' để làm cho câu văn ngắn gọn và xúc tích hơn.",
      "Nó chọn ngẫu nhiên một trong số mười từ có xác suất cao nhất ở mỗi bước sinh văn bản."
    ],
    answer: 0,
    explanation: "Tìm kiếm tham lam (Greedy search) luôn lấy mẫu token có xác suất cao nhất (argmax) ở mỗi bước. Điều này thường dẫn đến việc mô hình lặp đi lặp lại một cụm từ an toàn."
  },
  {
    type: "mcq",
    question: "Điều chỉnh tham số 'Nhiệt độ' (Temperature) ảnh hưởng như thế nào đến kết quả sinh văn bản của LLM?",
    options: [
      "Nhiệt độ cao làm phân bố xác suất phẳng hơn, giúp văn bản sáng tạo và đa dạng nhưng dễ trở nên vô nghĩa.",
      "Nhiệt độ thấp làm mô hình suy nghĩ lâu hơn, tốn nhiều CPU hơn nhưng văn bản sinh ra chính xác tuyệt đối.",
      "Nhiệt độ cao buộc mô hình chỉ chọn từ trong một từ điển giới hạn các thuật ngữ chuyên ngành học thuật.",
      "Nhiệt độ thấp tự động chuyển đổi tất cả các từ trong văn bản sang dạng chữ thường để dễ đọc hơn."
    ],
    answer: 0,
    explanation: "Temperature chia các logit trước khi đưa qua Softmax. Nhiệt độ cao (>1.0) san phẳng xác suất (tăng tính ngẫu nhiên). Nhiệt độ thấp (<1.0) làm xác suất cực đoan hơn (tăng tính an toàn, giống greedy)."
  },
  {
    type: "mcq",
    question: "Kỹ thuật 'Lấy mẫu Top-K' (Top-K sampling) kiểm soát tính ngẫu nhiên bằng cách nào?",
    options: [
      "Chỉ cho phép mô hình lấy mẫu ngẫu nhiên từ K mã thông báo có xác suất cao nhất, loại bỏ phần đuôi phân bố.",
      "Chỉ sinh ra tối đa K từ cho một câu trả lời, đảm bảo mô hình không tạo ra các đoạn văn quá dài dòng.",
      "Bắt buộc mô hình phải luôn sinh ra K từ khóa quan trọng mà người dùng đã chỉ định trong lời nhắc.",
      "Chia dữ liệu đầu vào thành K phần bằng nhau và sinh ra K luồng văn bản song song cùng một lúc."
    ],
    answer: 0,
    explanation: "Top-k sampling lọc bỏ tất cả các token có xác suất thấp, chỉ giữ lại top K token tiềm năng nhất, sau đó tính lại xác suất và lấy mẫu ngẫu nhiên từ K token này, giúp văn bản vừa đa dạng vừa hợp lý."
  },
  {
    type: "mcq",
    question: "Tại sao vòng lặp sinh văn bản (generation loop) cơ bản lại rất chậm nếu không sử dụng kỹ thuật 'Caching' (Bộ nhớ đệm)?",
    options: [
      "Mô hình phải tính toán lại toàn bộ thông tin của các từ trong quá khứ ở mỗi bước sinh một từ mới.",
      "Bộ vi xử lý GPU phải khởi động lại hoàn toàn bộ nhớ của mình mỗi khi sinh xong một mã thông báo.",
      "Quá trình này yêu cầu tải lại toàn bộ trọng số của mô hình từ ổ cứng lên RAM ở mỗi bước sinh từ.",
      "Thuật toán tìm kiếm tham lam yêu cầu phải duyệt qua toàn bộ từ điển tiếng Anh trước khi quyết định."
    ],
    answer: 0,
    explanation: "Nếu không có cơ chế lưu đệm Key-Value (KV Cache), ở mỗi bước sinh 1 từ mới, Transformer lại phải chạy tính toán Attention cho toàn bộ chuỗi từ đầu, dẫn đến độ phức tạp O(N^2)."
  },
  {
    type: "mcq",
    question: "Trong quá trình sinh văn bản của Transformer, ta có thể lưu đệm (cache) những thành phần nào để tăng tốc độ một cách tối đa?",
    options: [
      "Các vectơ Khóa (Key) và Giá trị (Value) của tất cả các lớp Attention cho các token trong quá khứ.",
      "Toàn bộ vectơ Truy vấn (Query) của từ cuối cùng để tái sử dụng cho câu hỏi tiếp theo của người dùng.",
      "Điểm số Softmax cuối cùng của tất cả các từ trong từ điển để không phải tính lại hàm mũ (exp).",
      "Vectơ nhúng vị trí (Positional Embedding) của từ cuối cùng nhân với ma trận trọng số chuẩn hóa."
    ],
    answer: 0,
    explanation: "Các vectơ Key và Value của các từ trong quá khứ không bao giờ thay đổi khi sinh thêm từ mới. Việc lưu đệm chúng (KV Cache) giúp mô hình chỉ cần tính Attention cho từ duy nhất vừa được sinh ra."
  },
  {
    type: "mcq",
    question: "Khái niệm 'Ảo giác' (Hallucination) trong các Mô hình Ngôn ngữ Lớn là gì?",
    options: [
      "Mô hình tạo ra các thông tin nghe có vẻ hợp lý và tự tin, nhưng hoàn toàn sai sự thật hoặc bịa đặt.",
      "Mô hình sinh ra các chuỗi ký tự vô nghĩa không thuộc bất kỳ ngôn ngữ nào của con người (như ngoại ngữ lạ).",
      "Mô hình bị quá tải bộ nhớ và bắt đầu trả về các phần tử ngẫu nhiên trong bộ nhớ RAM của máy chủ.",
      "Người dùng nhập vào các câu hỏi mẹo khiến mô hình từ chối trả lời vì vi phạm các quy tắc đạo đức."
    ],
    answer: 0,
    explanation: "Ảo giác xảy ra khi LLM đưa ra thông tin không có cơ sở thực tế (bịa đặt) nhưng lại với giọng điệu rất tự tin. Điều này xuất phát từ bản chất chỉ là 'đoán từ tiếp theo' của nó."
  },
  {
    type: "mcq",
    question: "Mục đích của việc 'Tinh chỉnh lệnh' (Instruction Fine-tuning) là gì?",
    options: [
      "Huấn luyện mô hình cách tuân thủ yêu cầu và định dạng của người dùng dưới dạng các cặp Hỏi-Đáp/Chỉ dẫn.",
      "Dạy cho mô hình những kiến thức vật lý và toán học mới nhất mà nó chưa từng thấy trong lúc tiền huấn luyện.",
      "Giảm dung lượng của mô hình từ hàng chục tỷ tham số xuống còn vài triệu tham số để chạy trên điện thoại.",
      "Ép mô hình phải quên đi các từ ngữ thô tục bằng cách xóa bỏ hoàn toàn chúng khỏi danh sách từ vựng."
    ],
    answer: 0,
    explanation: "Tiền huấn luyện chỉ giúp LLM sinh văn bản ngẫu nhiên. Tinh chỉnh lệnh (Instruction Tuning) sử dụng các mẫu [instruction] và [response] để dạy mô hình cư xử như một trợ lý (chatbot) biết nghe lời."
  },
  {
    type: "mcq",
    question: "LoRA (Low-Rank Adaptation) giải quyết thách thức lớn nào khi tinh chỉnh các Mô hình Ngôn ngữ Lớn?",
    options: [
      "Giảm yêu cầu về bộ nhớ RAM của GPU bằng cách đóng băng trọng số gốc và chỉ huấn luyện một số lượng nhỏ tham số mới.",
      "Loại bỏ hiện tượng ảo giác hoàn toàn bằng cách áp dụng các ràng buộc xếp hạng logic cấp thấp (Low-Rank Logic).",
      "Tăng tốc độ sinh từ của mô hình lên hàng nghìn lần nhờ vào việc rút gọn số lượng lớp Transformer bên trong.",
      "Cho phép mô hình học các ngôn ngữ mới mà không cần bất kỳ dữ liệu văn bản nào thông qua dịch thuật ma trận."
    ],
    answer: 0,
    explanation: "Tinh chỉnh toàn bộ mô hình (Full Fine-tuning) đòi hỏi bộ nhớ khổng lồ. LoRA đóng băng trọng số gốc và chèn thêm các ma trận bù đắp (update matrices) có hạng thấp (low-rank), giảm số tham số cần huấn luyện xuống hàng nghìn lần."
  },
  {
    type: "mcq",
    question: "Theo kỹ thuật LoRA, giả sử một lớp Dense có ma trận trọng số gốc kích thước 2048x2048, LoRA sẽ thêm vào hai ma trận A và B. Kích thước của chúng thường là bao nhiêu (với rank r=8)?",
    options: [
      "Ma trận A kích thước 2048x8 và ma trận B kích thước 8x2048.",
      "Cả hai ma trận A và B đều có kích thước siêu nhỏ là 8x8.",
      "Ma trận A kích thước 2048x2048 và ma trận B kích thước 8x8.",
      "Ma trận A kích thước 2048x8 và ma trận B kích thước 2048x8."
    ],
    answer: 0,
    explanation: "LoRA sử dụng hai ma trận để tạo ra ma trận cập nhật có cùng kích thước với trọng số gốc: `(2048 x 8) * (8 x 2048) = 2048 x 2048`. Việc này giảm số tham số từ ~4 triệu xuống còn khoảng ~32 ngàn."
  },
  {
    type: "mcq",
    question: "Tại sao trong mô hình GPT, các logit đầu ra lại thường được xử lý thông qua `SparseCategoricalCrossentropy(from_logits=True)` thay vì dùng trực tiếp Softmax ở lớp cuối?",
    options: [
      "Giữ nguyên các giá trị logit thô giúp quá trình tính toán loss ổn định hơn và có thể dùng trực tiếp để lấy mẫu nhiệt độ.",
      "Hàm Softmax không thể hoạt động được với các từ điển có kích thước lớn hơn 100,000 mã thông báo (token).",
      "Keras không hỗ trợ lớp Softmax độc lập khi huấn luyện các mô hình sử dụng kỹ thuật Precision kết hợp (Mixed Precision).",
      "Việc áp dụng Softmax sẽ làm cho tất cả các trọng số của mô hình tự động chuyển đổi thành các giá trị số nguyên."
    ],
    answer: 0,
    explanation: "Việc đưa thẳng logits thô (không chuẩn hóa) vào hàm loss (với cờ from_logits=True) mang lại sự ổn định về mặt số học. Hơn nữa, ở lúc sinh từ (inference), ta có thể áp dụng Temperature lên logit trước khi dùng Softmax."
  },
  {
    type: "mcq",
    question: "Việc sử dụng 'Độ chính xác hỗn hợp' (Mixed precision) như `mixed_float16` mang lại lợi ích chính nào khi huấn luyện mô hình Transformer lớn?",
    options: [
      "Tăng gấp đôi tốc độ tính toán và giảm một nửa dung lượng bộ nhớ yêu cầu mà gần như không giảm độ chính xác của mô hình.",
      "Tự động dịch văn bản tiếng Anh sang tiếng Pháp với độ chính xác cao ngay trong lúc mô hình đang được huấn luyện.",
      "Cho phép kết hợp cả số thực và văn bản vào cùng một luồng dữ liệu đầu vào duy nhất cho bộ mã hóa (Encoder).",
      "Giải quyết hiện tượng biến mất gradient hoàn toàn ở các mạng nơ-ron có độ sâu trên hàng nghìn lớp ẩn."
    ],
    answer: 0,
    explanation: "Mixed precision sử dụng float16 (16-bit) cho phần lớn các tính toán ma trận (giúp tăng tốc độ và giảm bộ nhớ VRAM) trong khi vẫn giữ float32 (32-bit) cho các trọng số tích lũy để bảo toàn tính ổn định số học."
  },
  {
    type: "mcq",
    question: "Khi huấn luyện trước (pretraining) một mô hình ngôn ngữ lớn (LLM), người ta thường nối hàng ngàn tài liệu thành một chuỗi duy nhất. Làm thế nào để mô hình biết một tài liệu đã kết thúc?",
    options: [
      "Chèn một mã thông báo đặc biệt, chẳng hạn như `<|endoftext|>`, vào giữa ranh giới của các tài liệu.",
      "Sử dụng một mạng nơ-ron thứ hai để phân đoạn các tài liệu và thông báo cho bộ giải mã (Decoder).",
      "Thêm vào một nghìn mã thông báo đệm (Padding tokens) toàn số 0 để làm khoảng cách giữa các bài báo.",
      "Bắt buộc mỗi tài liệu phải được viết bằng một ngôn ngữ khác nhau để mô hình tự động nhận diện."
    ],
    answer: 0,
    explanation: "Các LLM như GPT thường nối liên tục dữ liệu và chỉ phân cách các tài liệu bằng một token duy nhất (như `<|endoftext|>`). Bằng cách này, mô hình học được lúc nào chủ đề kết thúc và chuẩn bị bắt đầu một chủ đề hoàn toàn mới."
  },
  {
    type: "fill",
    question: "LLM được tạo ra chủ yếu nhờ kỹ thuật ______________ Transformer, tập trung hoàn toàn vào kiến trúc khối giải mã (Decoder-only).",
    options: ["GPT", "Generative Pretrained Transformer"],
    answer: "GPT",
    explanation: "GPT (Generative Pretrained Transformer) là họ mô hình tiên phong trong việc chứng minh sức mạnh của mô hình Decoder-only trên quy mô lớn."
  },
  {
    type: "fill",
    question: "Hiện tượng LLM sinh ra những thông tin tự tin nhưng hoàn toàn sai lệch thực tế được gọi là ______________.",
    options: ["ảo giác", "hallucination", "hallucinations"],
    answer: "ảo giác",
    explanation: "Ảo giác (Hallucination) là rủi ro lớn nhất khi sử dụng LLM cho các ứng dụng đòi hỏi tính chính xác tuyệt đối về kiến thức."
  },
  {
    type: "fill",
    question: "Để giảm bộ nhớ khi tinh chỉnh mô hình có hàng tỷ tham số, kỹ thuật ______________ (viết tắt) thêm vào các ma trận bù đắp có thứ hạng thấp.",
    options: ["LoRA"],
    answer: "LoRA",
    explanation: "LoRA (Low-Rank Adaptation) là kỹ thuật Parameter-Efficient Fine-Tuning (PEFT) phổ biến nhất hiện nay."
  },
  {
    type: "fill",
    question: "Trong sinh văn bản, tham số ______________ (Temperature) dùng để điều chỉnh mức độ san phẳng của phân bố xác suất Softmax.",
    options: ["Nhiệt độ", "nhiệt độ", "temperature"],
    answer: "Nhiệt độ",
    explanation: "Nhiệt độ (Temperature) kiểm soát sự cân bằng giữa tính ngẫu nhiên (sáng tạo) và tính chắc chắn (tham lam)."
  },
  {
    type: "fill",
    question: "Để cải thiện tốc độ vòng lặp sinh văn bản (autoregressive), người ta lưu lại kết quả tính toán trước đó của Khóa và Giá trị. Kỹ thuật này gọi là KV ______________.",
    options: ["Cache", "cache"],
    answer: "Cache",
    explanation: "KV Cache ngăn chặn việc Transformer phải tính toán lại Attention cho toàn bộ chuỗi từ ở mỗi bước."
  },
  {
    type: "matching",
    question: "Ghép nối các thuật ngữ với ý nghĩa tương ứng trong quá trình sinh văn bản:",
    options: [
      "Greedy Search",
      "Temperature = 0.1",
      "Temperature = 2.0",
      "Top-K Sampling"
    ],
    answer: [
      "Luôn chọn mã thông báo có xác suất lớn nhất (argmax).",
      "Làm sắc nét phân bố xác suất, hoạt động gần giống với Greedy.",
      "San phẳng phân bố xác suất, văn bản sinh ra rất hỗn loạn, thiếu ngữ pháp.",
      "Chỉ xét K mã thông báo tiềm năng nhất, loại bỏ xác suất của các từ còn lại."
    ],
    explanation: "Mỗi chiến lược điều chỉnh (sampling) có mục đích riêng để cân bằng độ sáng tạo và độ nhất quán của mô hình ngôn ngữ."
  },
  {
    type: "matching",
    question: "Ghép nối quá trình phát triển của họ mô hình GPT của OpenAI:",
    options: [
      "GPT-1 (2018)",
      "GPT-2 (2019)",
      "GPT-3 (2020)"
    ],
    answer: [
      "117 triệu tham số, chủ yếu được tinh chỉnh để làm classification.",
      "1.5 tỷ tham số, chứng minh khả năng zero-shot và few-shot learning cơ bản.",
      "175 tỷ tham số, có thể giải quyết các task phức tạp chỉ bằng cách viết prompt dài."
    ],
    explanation: "OpenAI đã giữ nguyên cấu trúc cốt lõi của GPT nhưng mở rộng quy mô (Scale) lên gấp 100 lần sau mỗi thế hệ, tạo ra sự nhảy vọt về chất lượng."
  },
  {
    type: "matching",
    question: "Ghép nối các thành phần trong phương pháp Tinh chỉnh lệnh (Instruction Fine-Tuning):",
    options: [
      "[instruction]",
      "[response]",
      "Sample Weight (Trọng số mẫu)"
    ],
    answer: [
      "Điểm đánh dấu bắt đầu của lời nhắc (prompt) hoặc câu hỏi của người dùng.",
      "Điểm đánh dấu bắt đầu phần trả lời mà mô hình bắt buộc phải học theo.",
      "Dùng để tắt tính Loss ở phần câu hỏi và khoảng trắng (padding), chỉ tính loss ở câu trả lời."
    ],
    explanation: "Instruction tuning biến một mô hình ngôn ngữ thô (base model) thành một chatbot (chat model) thông qua việc mớm cấu trúc hỏi-đáp."
  },
  {
    type: "matching",
    question: "Ghép nối các kỹ thuật tối ưu phần cứng cho LLM:",
    options: [
      "Mixed Precision (Float16)",
      "KV Cache",
      "LoRA"
    ],
    answer: [
      "Tăng tốc toán học ma trận và giảm 50% băng thông VRAM khi huấn luyện/suy luận.",
      "Thay đổi độ phức tạp thời gian khi sinh văn bản từ O(N^2) xuống mức gần tuyến tính.",
      "Đóng băng trọng số, giảm số lượng Optimizer States (trạng thái Adam) xuống 1000 lần."
    ],
    explanation: "Đây là 3 trụ cột kỹ thuật giúp việc huấn luyện và chạy mô hình ngôn ngữ lớn (LLM) trên máy tính thông thường hoặc server nhỏ trở nên khả thi."
  },
  {
    type: "matching",
    question: "Ghép nối các loại mô hình sinh theo định dạng của nó:",
    options: [
      "ChatGPT (OpenAI)",
      "Midjourney",
      "LLM Base Model (chưa qua instruction tuning)"
    ],
    answer: [
      "Mô hình LLM đã trải qua Instruction Tuning và RLHF để giao tiếp an toàn.",
      "Mô hình tổng quát để sinh hình ảnh từ lời nhắc văn bản (Text-to-Image).",
      "Một cỗ máy 'đoán từ tiếp theo' thuần túy, thường đóng vai trò như công cụ tự hoàn thành."
    ],
    explanation: "Mô hình sinh (Generative Models) có nhiều dạng: sinh chữ, sinh ảnh. LLM thô khác với Chatbot đã tinh chỉnh."
  },
  {
    type: "sorting",
    question: "Sắp xếp các bước cơ bản trong một vòng lặp sinh văn bản của LLM (LLM Generation Loop):",
    options: [
      "Người dùng cung cấp một chuỗi mồi (Prompt).",
      "Mô hình chạy Forward Pass để tính toán phân bố xác suất cho từ tiếp theo.",
      "Sử dụng chiến lược lấy mẫu (ví dụ Top-K hoặc Greedy) để chọn ra một mã thông báo (token).",
      "Thêm mã thông báo vừa được chọn vào cuối chuỗi mồi hiện tại.",
      "Lặp lại quá trình cho đến khi đạt độ dài tối đa hoặc xuất hiện token kết thúc."
    ],
    answer: [0, 1, 2, 3, 4],
    explanation: "Quá trình tự hồi quy (Autoregressive) diễn ra liên tục theo chu kỳ: Đọc ngữ cảnh -> Đoán xác suất -> Chọn 1 từ -> Cập nhật ngữ cảnh -> Lặp lại."
  },
  {
    type: "sorting",
    question: "Sắp xếp lịch sử phát triển của xử lý ngôn ngữ và mô hình sinh theo trình tự thời gian (từ cũ nhất đến mới nhất):",
    options: [
      "Thuật toán lặp LSTM ra đời.",
      "Kiến trúc Transformer (Attention Is All You Need) được giới thiệu.",
      "OpenAI phát hành GPT-1 (Mô hình ngôn ngữ tiền huấn luyện).",
      "Sự bùng nổ của ChatGPT với mô hình GPT-3.5/GPT-4."
    ],
    answer: [0, 1, 2, 3],
    explanation: "LSTM (1997) -> Transformer (2017) -> GPT-1 (2018) -> ChatGPT (Cuối 2022)."
  },
  {
    type: "sorting",
    question: "Sắp xếp các thao tác toán học trong lớp LoraLinear khi tính toán quá trình Forward Pass:",
    options: [
      "Nhận đầu vào X (ví dụ kích thước 2048).",
      "Nhân X với ma trận trọng số gốc cố định W để được kết quả Frozen.",
      "Nhân X với ma trận A (giảm chiều xuống Rank R).",
      "Nhân tiếp kết quả với ma trận B (phục hồi chiều lên 2048) để tạo bản Cập nhật (Update).",
      "Cộng kết quả Frozen và Update lại với nhau làm kết quả đầu ra cuối cùng."
    ],
    answer: [0, 1, 2, 3, 4],
    explanation: "Phương trình của LoRA là: h = W_0 * x + B * A * x (trong đó W_0 bị đóng băng, A và B có thể học được)."
  },
  {
    type: "sorting",
    question: "Sắp xếp các bước để thực hiện quá trình Tinh chỉnh Lệnh (Instruction Fine-Tuning) cho một mô hình LLM thô:",
    options: [
      "Chuẩn bị tập dữ liệu gồm các cặp Hướng dẫn (Instruction) và Câu trả lời (Response).",
      "Nối Hướng dẫn và Câu trả lời thành một chuỗi duy nhất, chèn các token đánh dấu (như [instruction], [response]).",
      "Tạo mảng Sample Weight: đặt trọng số bằng 0 cho phần Hướng dẫn và bằng 1 cho phần Câu trả lời.",
      "Tải một mô hình LLM đã tiền huấn luyện (Base Model).",
      "Huấn luyện mô hình (Fine-tune) để nó dự đoán từ tiếp theo trong phần Câu trả lời."
    ],
    answer: [0, 1, 2, 3, 4],
    explanation: "Để làm chatbot, ta không cần mô hình học lại câu hỏi, chỉ cần học cách sinh ra câu trả lời dựa trên câu hỏi. Ta mask (trọng số 0) phần prompt và tính loss ở phần response."
  },
  {
    type: "sorting",
    question: "Sắp xếp các mức độ của chiến lược Lấy mẫu (Sampling) theo sự gia tăng của 'tính ngẫu nhiên' (Từ an toàn nhất đến hỗn loạn nhất):",
    options: [
      "Tìm kiếm tham lam (Greedy Search).",
      "Nhiệt độ (Temperature) = 0.5.",
      "Nhiệt độ (Temperature) = 1.0 (Bình thường).",
      "Nhiệt độ (Temperature) = 2.0."
    ],
    answer: [0, 1, 2, 3],
    explanation: "Greedy là hoàn toàn không có tính ngẫu nhiên (chỉ chọn argmax). Khi Temperature tăng, xác suất càng phân tán, văn bản càng trở nên ngẫu nhiên và khó đoán."
  }
];

export default questions;
