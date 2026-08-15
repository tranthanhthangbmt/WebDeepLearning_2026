const questions = [
  {
    type: "mcq",
    question: "Mô hình ngôn ngữ (Language model) hoạt động dựa trên nguyên lý cơ bản nào?",
    options: [
      "Dự đoán mã thông báo tiếp theo dựa trên tất cả các mã thông báo đã xuất hiện trong quá khứ.",
      "Phân loại các câu văn thành các danh mục được xác định trước, như tích cực hoặc tiêu cực.",
      "Mã hóa câu thành một vectơ duy nhất và so sánh nó với toàn bộ văn bản trong cơ sở dữ liệu.",
      "Đếm tần suất xuất hiện của các từ để tạo thành ma trận túi từ (bag-of-words) thưa thớt."
    ],
    answer: 0,
    explanation: "Mô hình ngôn ngữ học phân bố xác suất `p(token | past_tokens)`, từ đó dự đoán từng mã thông báo tiếp theo để sinh văn bản."
  },
  {
    type: "mcq",
    question: "Trong bài toán sinh văn bản, thuật ngữ 'tự hồi quy' (autoregressive) có nghĩa là gì?",
    options: [
      "Đầu ra dự đoán ở một bước thời gian được đưa ngược lại làm đầu vào cho bước tiếp theo.",
      "Mạng nơ-ron tự động điều chỉnh số lượng lớp lặp lại (RNN) theo chiều dài chuỗi đầu vào.",
      "Mô hình sử dụng các hồi quy tuyến tính để tìm ra hàm mất mát tối ưu nhất khi huấn luyện.",
      "Văn bản được dịch ngược lại sang ngôn ngữ gốc để kiểm tra tính chính xác của mô hình."
    ],
    answer: 0,
    explanation: "Tự hồi quy (Autoregressive) ám chỉ việc mô hình sinh từng token một, và token vừa được sinh ra lại trở thành một phần ngữ cảnh đầu vào để sinh token tiếp theo."
  },
  {
    type: "mcq",
    question: "Tại sao trong mô hình sinh văn bản bằng RNN, việc sử dụng lớp `Bidirectional` lại làm hỏng quá trình huấn luyện?",
    options: [
      "Nó cung cấp cho mạng thông tin từ các mã thông báo tương lai, dẫn đến việc 'gian lận'.",
      "Kích thước của vectơ trạng thái sẽ tăng gấp đôi, làm cạn kiệt bộ nhớ GPU rất nhanh chóng.",
      "Mô hình không thể hiểu được các chuỗi văn bản quá dài, do lỗi biến mất gradient.",
      "Lớp Bidirectional yêu cầu phải có cơ chế chú ý (attention) đi kèm mới hoạt động được."
    ],
    answer: 0,
    explanation: "Trong mô hình ngôn ngữ (dự đoán token tiếp theo), RNN phải đi theo một chiều. Nếu dùng Bidirectional, mô hình sẽ nhìn thấy trước tương lai, khiến việc học mất đi ý nghĩa."
  },
  {
    type: "mcq",
    question: "Mô hình Sequence-to-Sequence (Seq2Seq) cổ điển giải quyết bài toán dịch máy như thế nào?",
    options: [
      "Bộ mã hóa nén câu nguồn thành một vectơ trạng thái, sau đó bộ giải mã dùng vectơ đó để sinh câu đích.",
      "Bộ mã hóa dịch từng từ của câu nguồn, sau đó bộ giải mã xếp chúng lại theo đúng ngữ pháp đích.",
      "Mô hình dự đoán cùng lúc toàn bộ các từ trong bản dịch dựa trên một ma trận xác suất khổng lồ.",
      "Sử dụng hai mạng lặp LSTM độc lập, một để dịch chủ ngữ, một để dịch vị ngữ và tân ngữ."
    ],
    answer: 0,
    explanation: "Mô hình seq2seq tiêu chuẩn sử dụng Bộ mã hóa (Encoder) để đọc toàn bộ chuỗi nguồn và mã hóa nó thành một state vector. State này được dùng làm trạng thái khởi tạo cho Bộ giải mã (Decoder)."
  },
  {
    type: "mcq",
    question: "Điểm yếu lớn nhất của mô hình dịch máy RNN Seq2Seq cổ điển là gì?",
    options: [
      "Bộ mã hóa phải nén toàn bộ câu nguồn dài vào một vectơ trạng thái có kích thước cố định.",
      "Không thể xử lý được các từ vựng chưa từng xuất hiện (Out-of-vocabulary) trong dữ liệu huấn luyện.",
      "Phải sử dụng phương pháp one-hot encoding khiến bộ nhớ quá tải khi kích thước từ vựng tăng.",
      "Các lớp LSTM không thể tính toán song song trên nhiều luồng dữ liệu cùng một lúc."
    ],
    answer: 0,
    explanation: "RNN cố gắng nhồi nhét tất cả ý nghĩa của câu nguồn (dù ngắn hay rất dài) vào một vectơ trạng thái cuối cùng, gây ra nút thắt cổ chai thông tin và dẫn đến việc 'quên' phần đầu câu."
  },
  {
    type: "mcq",
    question: "Cơ chế Chú ý (Attention mechanism) giải quyết vấn đề của Seq2Seq RNN như thế nào?",
    options: [
      "Cho phép bộ giải mã tập trung vào các phần khác nhau của câu nguồn tại mỗi bước dự đoán.",
      "Thay thế toàn bộ lớp RNN bằng mạng nơ-ron tích chập (CNN) để xử lý chuỗi nhanh hơn.",
      "Chia câu nguồn thành nhiều câu ngắn hơn để bộ mã hóa không bị quá tải bộ nhớ trạng thái.",
      "Tăng kích thước của vectơ trạng thái tỷ lệ thuận với số lượng từ vựng của câu văn nguồn."
    ],
    answer: 0,
    explanation: "Attention cho phép Decoder, tại mỗi bước dịch, tính điểm mức độ liên quan (score) để 'nhìn' trực tiếp vào các token quan trọng nhất trong chuỗi Encoder thay vì dựa vào một trạng thái nén duy nhất."
  },
  {
    type: "mcq",
    question: "Trong cơ chế Attention, khái niệm Query (Truy vấn) tương ứng với điều gì?",
    options: [
      "Biểu diễn của từ hiện tại mà bộ giải mã đang xử lý để dự đoán từ tiếp theo.",
      "Biểu diễn của từ ở câu nguồn mà bộ mã hóa vừa mới phân tích xong.",
      "Kết quả điểm số tương đồng giữa hai vectơ để quyết định trọng số chú ý.",
      "Vectơ tổng hợp được tạo ra sau khi tính tổng có trọng số các giá trị (values)."
    ],
    answer: 0,
    explanation: "Trong hệ thống dịch, 'Query' là trạng thái hiện tại của Decoder (từ đang cần dịch), 'Key' và 'Value' là các token từ câu nguồn do Encoder cung cấp."
  },
  {
    type: "mcq",
    question: "Mục đích của việc sử dụng 'Sự chú ý nhiều đầu' (Multi-head attention) là gì?",
    options: [
      "Cho phép mô hình đồng thời học nhiều loại mối quan hệ khác nhau (ví dụ: ngữ pháp, ngữ nghĩa).",
      "Giảm độ phức tạp tính toán của mô hình bằng cách chia nhỏ chuỗi đầu vào thành nhiều mảnh.",
      "Tránh lỗi biến mất gradient khi huấn luyện các mô hình biến áp có hàng trăm lớp kết nối.",
      "Kết hợp thông tin từ nhiều ngôn ngữ nguồn khác nhau để dịch sang một ngôn ngữ đích."
    ],
    answer: 0,
    explanation: "Sự chú ý nhiều đầu (Multi-head attention) chạy nhiều phép tính chú ý song song với các tham số khác nhau. Một 'đầu' có thể chú ý đến chủ ngữ, một 'đầu' khác chú ý đến thì của động từ, v.v."
  },
  {
    type: "mcq",
    question: "Sự tự chú ý (Self-attention) khác với cơ chế chú ý tiêu chuẩn ở điểm nào?",
    options: [
      "Tính toán sự liên quan giữa các token trong cùng một chuỗi với nhau thay vì giữa hai chuỗi.",
      "Tự động loại bỏ các từ không quan trọng ra khỏi câu trước khi tiến hành dịch máy.",
      "Chỉ sử dụng duy nhất một 'đầu' chú ý để tăng tốc độ tính toán cho các mô hình nhỏ.",
      "Yêu cầu dữ liệu phải được mã hóa trước bằng phương pháp túi từ (Bag-of-words)."
    ],
    answer: 0,
    explanation: "Trong Self-attention, Query, Key, và Value đều xuất phát từ cùng một chuỗi đầu vào. Nó giúp mô hình hiểu ngữ cảnh của một từ dựa trên tất cả các từ khác trong chính câu đó."
  },
  {
    type: "mcq",
    question: "Tại sao Transformer lại vượt trội hơn RNN trong việc xử lý văn bản dài?",
    options: [
      "Nó xem xét toàn bộ các mối quan hệ từ xa qua Self-attention thay vì truyền tuần tự qua các trạng thái.",
      "Nó tự động tóm tắt các đoạn văn dài thành các câu ngắn gọn trước khi xử lý chúng tiếp.",
      "Nó luôn sử dụng mạng tích chập 1 chiều (1D CNN) với kích thước cửa sổ rất lớn cho văn bản.",
      "Nó có khả năng nén các ma trận nhúng từ thành không gian một chiều để tính toán nhanh."
    ],
    answer: 0,
    explanation: "RNN bị giới hạn bởi tính chất tuần tự (khoảng cách càng xa, tín hiệu càng mờ). Transformer thông qua Self-attention có thể kết nối bất kỳ hai token nào trong câu bằng một phép tính trực tiếp duy nhất, không phụ thuộc khoảng cách."
  },
  {
    type: "mcq",
    question: "Theo bài báo 'Attention Is All You Need', yếu tố nào sau đây là BẮT BUỘC để bổ sung tính phi tuyến vào khối Transformer?",
    options: [
      "Một mạng chuyển tiếp (Feedforward Network) đi kèm sau mỗi lớp Self-attention.",
      "Việc loại bỏ cơ chế Dropout để tăng cường khả năng học sâu của mô hình.",
      "Sử dụng hàm kích hoạt Sigmoid ở lớp xuất ra cuối cùng của kiến trúc.",
      "Khởi tạo ngẫu nhiên toàn bộ trọng số của các truy vấn và từ khóa."
    ],
    answer: 0,
    explanation: "Lớp Attention thực chất chỉ thực hiện phép chiếu tuyến tính tổng hợp. Nếu không có mạng truyền thẳng (Feedforward Network - các lớp Dense) với các hàm kích hoạt (như ReLU), Transformer sẽ chỉ là một phép nhân ma trận tuyến tính khổng lồ."
  },
  {
    type: "mcq",
    question: "Vì Transformer không xử lý từ theo tuần tự giống như RNN, nó làm cách nào để biết thứ tự của các từ trong câu?",
    options: [
      "Bổ sung một vectơ 'Mã hóa vị trí' (Positional Encoding) vào biểu diễn của mỗi từ.",
      "Sử dụng một RNN nhỏ chạy song song với Transformer để đếm số thứ tự các từ.",
      "Chèn các thẻ HTML đặc biệt trước và sau mỗi từ để đánh dấu vị trí của chúng.",
      "Sắp xếp thứ tự các từ trong từ điển và luôn ưu tiên từ có số ID nhỏ hơn."
    ],
    answer: 0,
    explanation: "Transformer phân tích tất cả các từ cùng lúc (không tuần tự). Do đó, cần cộng thêm một vectơ 'Mã hóa vị trí' vào Embedding để mô hình biết từ nào đứng trước, từ nào đứng sau."
  },
  {
    type: "mcq",
    question: "Trong tính toán Điểm số Chú ý (Attention Score), tại sao phải chia cho căn bậc hai của kích thước chiều (square root of head_dim)?",
    options: [
      "Để giảm độ lớn của các giá trị tích số chấm, tránh làm hàm softmax bị bão hòa (gradient vanishing).",
      "Để chuẩn hóa chiều dài của các vectơ nhúng về đúng bằng 1 theo chuẩn Euclidean.",
      "Để đảm bảo rằng kích thước của ma trận kết quả luôn là một số chẵn có thể chia hết.",
      "Để làm cho tất cả các điểm số chú ý có giá trị bằng nhau trước khi tính tổng."
    ],
    answer: 0,
    explanation: "Tích số chấm (dot-product) của hai vectơ có số chiều lớn sẽ tạo ra các giá trị rất lớn. Khi đưa qua Softmax, các giá trị này sẽ đẩy phân phối xác suất về cực đoan (0 hoặc 1), làm mất gradient."
  },
  {
    type: "mcq",
    question: "Một cơ chế chú ý theo tích số chấm (dot-product attention) sử dụng phép toán nào để đo mức độ tương đồng giữa Query và Key?",
    options: [
      "Tích vô hướng (Tích số chấm) giữa vectơ Truy vấn (Query) và vectơ Khóa (Key).",
      "Tính khoảng cách Euclide tuyệt đối giữa hai điểm trong không gian ba chiều.",
      "Lấy giá trị trung bình của toàn bộ các phần tử trong ma trận đầu vào nguồn.",
      "So sánh độ dài chuỗi ký tự của hai từ để tìm ra sự tương đồng ngữ nghĩa."
    ],
    answer: 0,
    explanation: "Attention đo mức độ liên quan bằng cách lấy tích vô hướng (dot-product) giữa Query và các Key. Nếu tích vô hướng càng lớn, mức độ chú ý (sự tương quan) càng cao."
  },
  {
    type: "mcq",
    question: "Quá trình Teacher Forcing (Ép buộc giáo viên) khi huấn luyện mô hình Seq2Seq là gì?",
    options: [
      "Cung cấp cho bộ giải mã đầu vào là mã thông báo mục tiêu đúng thực sự, bất kể nó đã dự đoán sai trước đó.",
      "Sử dụng một mô hình lớn đã được huấn luyện sẵn để tạo ra nhãn cho một mô hình nhỏ học hỏi theo.",
      "Ép mạng RNN ngừng quá trình huấn luyện khi độ chính xác trên tập kiểm tra bắt đầu sụt giảm nhanh.",
      "Sử dụng thuật toán học củng cố để thưởng mô hình khi nó sinh ra một từ hoàn toàn chính xác."
    ],
    answer: 0,
    explanation: "Trong Teacher Forcing, tại mỗi bước huấn luyện, thay vì cho Decoder nhận đầu vào là dự đoán (có thể sai) của bước trước, ta cung cấp cho nó token mục tiêu đúng thực sự. Điều này giúp mô hình hội tụ nhanh và ổn định hơn."
  },
  {
    type: "fill",
    question: "Mô hình sinh từ kế tiếp dựa trên xác suất của các từ trước đó được gọi chung là mô hình _________.",
    options: ["ngôn ngữ", "language model", "language"],
    answer: "ngôn ngữ",
    explanation: "Mô hình ngôn ngữ (Language Model) phân bổ xác suất `p(token | past_tokens)`."
  },
  {
    type: "fill",
    question: "Kiến trúc thay thế hoàn toàn RNN bằng cơ chế Self-attention được gọi là _________.",
    options: ["Transformer", "Transformers"],
    answer: "Transformer",
    explanation: "Kiến trúc Transformer loại bỏ các lớp RNN, chỉ dựa vào Attention để xử lý các liên kết trong chuỗi."
  },
  {
    type: "fill",
    question: "Trong bài toán dịch máy, chuỗi nguồn sẽ đi qua Bộ mã hóa, gọi là _________, để tạo biểu diễn.",
    options: ["encoder", "Encoder"],
    answer: "Encoder",
    explanation: "Encoder (Bộ mã hóa) có nhiệm vụ đọc và nén/hiểu thông tin từ văn bản nguồn."
  },
  {
    type: "fill",
    question: "Trong Seq2Seq, bộ phận có nhiệm vụ nhận biểu diễn ngữ cảnh và sinh văn bản đích là _________.",
    options: ["decoder", "Decoder"],
    answer: "Decoder",
    explanation: "Decoder (Bộ giải mã) sử dụng ngữ cảnh từ Encoder và thông tin dịch ở quá khứ để sinh chuỗi ngôn ngữ đích."
  },
  {
    type: "fill",
    question: "Để mô hình Transformer nhận biết được thứ tự các từ trong câu, ta sử dụng mã hóa _________.",
    options: ["vị trí", "positional", "positional encoding"],
    answer: "vị trí",
    explanation: "Mã hóa vị trí (Positional Encoding) được cộng vào các Word Embedding để giữ lại thông tin về thứ tự từ."
  },
  {
    type: "matching",
    question: "Ghép nối thuật ngữ Attention với khái niệm tương đương (Ví dụ tìm kiếm):",
    options: [
      "Query (Truy vấn)",
      "Key (Khóa)",
      "Value (Giá trị)"
    ],
    answer: [
      "Từ khoá bạn nhập vào thanh tìm kiếm (Từ hiện tại đang xử lý)",
      "Các tiêu đề bài viết trong cơ sở dữ liệu (Các từ để so khớp)",
      "Nội dung bài viết thực sự được trả về (Thông tin ngữ cảnh gốc)"
    ],
    explanation: "Query đại diện cho thứ ta cần tìm, Key là nhãn đại diện cho thông tin có sẵn, Value là nội dung thực tế sẽ được lấy ra dựa trên độ khớp giữa Query và Key."
  },
  {
    type: "matching",
    question: "Ghép nối các loại Attention với tính năng của chúng:",
    options: [
      "Cross-attention (Chú ý chéo)",
      "Self-attention (Tự chú ý)",
      "Multi-head attention"
    ],
    answer: [
      "Query đến từ Decoder, Key và Value đến từ Encoder",
      "Query, Key, Value đều đến từ cùng một chuỗi (Encoder hoặc Decoder)",
      "Chia việc học chú ý thành nhiều không gian song song khác nhau"
    ],
    explanation: "Self-attention để mô hình tự hiểu câu. Cross-attention dùng trong Seq2Seq để đối chiếu câu đích với câu nguồn. Multi-head là một cách cải tiến biểu diễn."
  },
  {
    type: "matching",
    question: "Ghép nối các kiến trúc sau với đặc điểm hạn chế/ưu điểm nổi bật:",
    options: [
      "RNN (RNN truyền thống)",
      "Seq2Seq RNN cổ điển",
      "Transformer"
    ],
    answer: [
      "Xử lý tuần tự, gặp vấn đề quên quá khứ xa",
      "Nút thắt cổ chai thông tin ở vectơ trạng thái Encoder cuối cùng",
      "Tính toán song song tốt, khả năng kết nối toàn cục bất kể khoảng cách"
    ],
    explanation: "RNN truyền thống yếu khi chuỗi dài. Seq2Seq bị nút thắt trạng thái. Transformer giải quyết toàn bộ nhờ sự chú ý toàn cục (Global attention)."
  },
  {
    type: "matching",
    question: "Ghép nối các token đặc biệt với chức năng của chúng trong Seq2Seq:",
    options: [
      "[UNK] (Unknown)",
      "[PAD] (Padding)",
      "[start] / [BOS]",
      "[end] / [EOS]"
    ],
    answer: [
      "Đại diện cho từ không có trong từ vựng",
      "Thêm vào câu ngắn để cân bằng lô (batch)",
      "Cung cấp tín hiệu bắt đầu cho quá trình giải mã (Decoder)",
      "Tín hiệu cho mô hình biết đã dịch xong và dừng sinh từ"
    ],
    explanation: "Trong Seq2Seq, ta thường chèn [start] ở đầu câu đích và [end] ở cuối câu đích để điều khiển luồng giải mã."
  },
  {
    type: "matching",
    question: "Ghép nối các phép toán/hàm với chức năng trong Attention:",
    options: [
      "Tích vô hướng (Dot-product)",
      "Softmax",
      "Tổng có trọng số (Weighted sum)"
    ],
    answer: [
      "Tính điểm số tương đồng chưa chuẩn hóa giữa Query và Key",
      "Chuẩn hóa điểm số thành phân bố xác suất (tổng bằng 1)",
      "Tính toán vectơ ngữ cảnh cuối cùng kết hợp từ các Value"
    ],
    explanation: "Q x K tạo ra điểm thô. Softmax biến điểm thành trọng số từ 0 đến 1. Trọng số nhân với Value tạo ra ngữ cảnh đầu ra."
  },
  {
    type: "sorting",
    question: "Sắp xếp trình tự tạo từ (Autoregressive Generation) trong suy luận (inference) của mô hình ngôn ngữ:",
    options: [
      "Chuẩn bị một chuỗi 'prompt' (ví dụ: '[start]').",
      "Đưa chuỗi hiện tại vào mô hình để tính toán phân bố xác suất của từ tiếp theo.",
      "Lấy mẫu (argmax hoặc random) từ có xác suất cao nhất.",
      "Nối từ vừa được tạo vào cuối chuỗi hiện tại.",
      "Lặp lại quá trình cho đến khi gặp token '[end]'."
    ],
    answer: [0, 1, 2, 3, 4],
    explanation: "Quá trình suy luận sinh văn bản luôn yêu cầu mồi (prompt), dự đoán bước kế, thêm kết quả vào lịch sử rồi lặp lại (autoregressive loop)."
  },
  {
    type: "sorting",
    question: "Sắp xếp thứ tự các thao tác trong phương trình Attention (Self-attention):",
    options: [
      "Chiếu dữ liệu đầu vào thành các ma trận Q (Query), K (Key), V (Value).",
      "Tính tích số chấm giữa Q và ma trận chuyển vị của K.",
      "Chia kết quả cho căn bậc hai của kích thước chiều (Scale).",
      "Áp dụng hàm Softmax để nhận điểm số chú ý.",
      "Nhân điểm số chú ý với ma trận V (Value) để lấy ngữ cảnh."
    ],
    answer: [0, 1, 2, 3, 4],
    explanation: "Đây là các bước chính xác của Scaled Dot-Product Attention: Attention(Q, K, V) = softmax( (Q * K^T) / sqrt(d_k) ) * V."
  },
  {
    type: "sorting",
    question: "Sắp xếp thứ tự một bước truyền xuôi (Forward pass) qua Khối mã hóa Transformer (Transformer Encoder Block):",
    options: [
      "Cộng dữ liệu gốc với dữ liệu sau Attention (Residual Connection).",
      "Thực hiện chuẩn hóa (Layer Normalization).",
      "Đưa qua mạng chuyển tiếp (Feedforward Network).",
      "Cộng dữ liệu gốc với dữ liệu sau Feedforward (Residual Connection) & Chuẩn hóa."
    ],
    answer: [0, 1, 2, 3],
    explanation: "Thực tế, dòng dữ liệu đi vào lớp Multi-Head Attention, sau đó cộng phần dư (Add) và chuẩn hóa (Norm). Tiếp theo là lớp FeedForward, rồi lại Add & Norm."
  },
  {
    type: "sorting",
    question: "Sắp xếp độ phức tạp của việc học các phần phụ thuộc tầm xa (Long-range dependencies) theo thứ tự từ KÉM HIỆU QUẢ nhất đến TỐT NHẤT:",
    options: [
      "Mô hình Bag-of-words (Không có ngữ cảnh).",
      "Mạng lặp RNN cổ điển.",
      "Mạng lặp LSTM / GRU.",
      "Kiến trúc Transformer (Self-attention)."
    ],
    answer: [0, 1, 2, 3],
    explanation: "BoW không xét thứ tự. RNN nhớ rất ngắn. LSTM/GRU nhớ dài hơn nhờ cổng (gates). Transformer truy xuất trực tiếp bất kỳ token nào (tốt nhất cho chuỗi dài)."
  },
  {
    type: "sorting",
    question: "Sắp xếp các bước chuẩn bị dữ liệu huấn luyện cho bài toán Seq2Seq (Ví dụ Anh -> Tây Ban Nha):",
    options: [
      "Đọc cặp câu văn bản (source, target) từ tập tin.",
      "Chèn token '[start]' vào đầu và '[end]' vào cuối câu mục tiêu (target).",
      "Xây dựng từ vựng (Vocabulary) độc lập cho cả tiếng Anh và Tây Ban Nha.",
      "Chuyển đổi cặp câu thành cặp chuỗi số nguyên ID.",
      "Chuẩn bị nhãn (label) là chuỗi mục tiêu dịch chuyển một bước (offset by one) cho Teacher Forcing."
    ],
    answer: [0, 1, 2, 3, 4],
    explanation: "Đây là quy trình tiền xử lý chuẩn trong Keras cho các mô hình Seq2Seq với Tokenization riêng cho hai ngôn ngữ, và dịch chuyển nhãn 1 vị trí để đào tạo mô hình tự hồi quy."
  }
];

export default questions;
