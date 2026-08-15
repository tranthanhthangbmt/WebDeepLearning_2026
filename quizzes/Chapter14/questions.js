const questions = [
  {
    type: "mcq",
    question: "Xử lý ngôn ngữ tự nhiên (NLP) là lĩnh vực nghiên cứu về điều gì?",
    options: [
      "Sử dụng máy tính để diễn giải và thao tác với ngôn ngữ của con người.",
      "Thiết kế các ngôn ngữ lập trình mới cho máy tính như LISP hoặc XML.",
      "Mã hóa các quy tắc ngữ pháp một cách thủ công để máy tính dịch văn bản.",
      "Sử dụng mạng nơ-ron để nhận dạng các mẫu trong hình ảnh và video."
    ],
    answer: 0,
    explanation: "NLP (Natural Language Processing) là việc sử dụng máy tính để diễn giải, phân tích và thao tác với ngôn ngữ tự nhiên của con người (như tiếng Anh, tiếng Việt, v.v.)."
  },
  {
    type: "mcq",
    question: "Bước đầu tiên khi áp dụng học sâu vào văn bản tiếng Anh là gì?",
    options: [
      "Dịch văn bản viết thành các tensor chứa các con số.",
      "Huấn luyện trực tiếp một mạng nơ-ron trên các ký tự.",
      "Đếm số lượng từ vựng xuất hiện trong tập dữ liệu.",
      "Xây dựng một mạng lặp LSTM hai chiều (Bidirectional)."
    ],
    answer: 0,
    explanation: "Các mô hình học sâu không thể trực tiếp xử lý văn bản. Bước tiên quyết là phải chuyển đổi (vector hóa) văn bản viết thành các tensor số."
  },
  {
    type: "mcq",
    question: "Giai đoạn 'Tiêu chuẩn hóa' (Standardization) văn bản thường bao gồm những thao tác nào?",
    options: [
      "Chuyển đổi thành chữ thường và loại bỏ các ký tự dấu câu.",
      "Tách văn bản thành các từ và đánh chỉ mục cho từng từ đó.",
      "Ánh xạ các mã thông báo thành một không gian vector đa chiều.",
      "Đệm các chuỗi văn bản cho có cùng độ dài (Padding)."
    ],
    answer: 0,
    explanation: "Tiêu chuẩn hóa nhằm xóa bỏ những khác biệt không mong muốn (ví dụ: in hoa, in thường, dấu câu) để đơn giản hóa đầu vào, ví dụ: đưa tất cả về chữ thường và xóa dấu câu."
  },
  {
    type: "mcq",
    question: "Trong quy trình tiền xử lý, 'Tokenization' (Mã thông báo hóa) nghĩa là gì?",
    options: [
      "Chia văn bản thành các đơn vị nhỏ riêng lẻ như từ hoặc ký tự.",
      "Chuẩn hóa dữ liệu đầu vào bằng các phép biến đổi văn bản.",
      "Ánh xạ các mã thông báo tới các chỉ mục bằng từ vựng.",
      "Biến đổi một chuỗi văn bản thành một biểu diễn dense vector."
    ],
    answer: 0,
    explanation: "Tokenization là quá trình tách chuỗi văn bản thô thành các đơn vị có thể xử lý được gọi là token (mã thông báo). Token có thể là từ, ký tự, hoặc từ phụ (subword)."
  },
  {
    type: "mcq",
    question: "Mã thông báo đặc biệt `[UNK]` được sử dụng để làm gì?",
    options: [
      "Đại diện cho một mã thông báo chưa được biết đến (không có trong từ vựng).",
      "Đệm cho các chuỗi có độ dài ngắn để đảm bảo cùng kích thước lô.",
      "Đánh dấu sự kết thúc của một câu trong quá trình xử lý văn bản.",
      "Bỏ qua các từ phổ biến như 'the', 'a' để mô hình không bị nhiễu."
    ],
    answer: 0,
    explanation: "`[UNK]` (Unknown) được dùng làm chỉ số mặc định cho bất kỳ từ nào xuất hiện trong dữ liệu thực tế nhưng không có mặt trong bộ từ vựng đã học."
  },
  {
    type: "mcq",
    question: "Đâu là nhược điểm chính của việc sử dụng 'Trình mã thông báo cấp ký tự' (Character Tokenizer)?",
    options: [
      "Mã hóa từng đầu vào thành một chuỗi rất dài, khó cho mô hình xử lý.",
      "Cần một từ vựng cực kỳ lớn (hàng triệu thuật ngữ) để hoạt động.",
      "Không thể xử lý được các từ hiếm hoặc các từ viết sai chính tả.",
      "Nó làm mất đi toàn bộ thông tin về thứ tự của các ký tự."
    ],
    answer: 0,
    explanation: "Mặc dù từ vựng của mã thông báo ký tự rất nhỏ, nó khiến độ dài của chuỗi đầu vào trở nên cực kỳ lớn, gây khó khăn cho mô hình trong việc ghi nhớ các phụ thuộc xa."
  },
  {
    type: "mcq",
    question: "Mã hóa từ phụ (Subword Tokenization) có lợi ích gì so với mã hóa cấp độ từ (Word-level)?",
    options: [
      "Nén chuỗi đầu vào tốt đồng thời xử lý được các từ hiếm hiệu quả.",
      "Giữ nguyên 100% ngữ pháp tiếng Anh để dịch sang tiếng Nga.",
      "Tăng kích thước từ vựng lên hàng triệu để bắt mọi sắc thái từ.",
      "Tạo ra các vectơ one-hot trực giao cho mọi từ trong từ vựng."
    ],
    answer: 0,
    explanation: "Subword Tokenization kết hợp ưu điểm của cả mã hóa từ và ký tự: độ dài chuỗi trung bình ngắn gọn, trong khi vẫn sử dụng từ vựng nhỏ gọn để bao phủ mọi từ, kể cả từ hiếm."
  },
  {
    type: "mcq",
    question: "Mô hình 'Túi từ' (Bag-of-words) xử lý văn bản theo cách nào?",
    options: [
      "Loại bỏ thứ tự từ, chỉ quan tâm đến sự xuất hiện hay vắng mặt của các từ.",
      "Ghi nhớ hoàn toàn vị trí tuyệt đối của từng từ trong câu.",
      "Sử dụng mạng lặp (RNN) để xử lý từng từ một theo thứ tự thời gian.",
      "Ánh xạ mỗi từ vào một vectơ dày đặc 256 chiều để đo khoảng cách."
    ],
    answer: 0,
    explanation: "Mô hình Túi từ (Bag-of-words) bỏ qua thứ tự của các token, coi văn bản như một tập hợp (bag) chứa các từ xuất hiện, giúp tính toán nhanh nhưng mất thông tin ngữ cảnh."
  },
  {
    type: "mcq",
    question: "Trong ngữ cảnh mô hình túi từ, 'Bigram' là gì?",
    options: [
      "Một chuỗi hai mã thông báo xuất hiện liên tiếp trong văn bản.",
      "Một từ có độ dài từ hai âm tiết trở lên trong ngôn ngữ tự nhiên.",
      "Một ma trận nhị phân lưu trữ sự xuất hiện của hai loại từ khóa.",
      "Một thuật toán mã hóa cặp byte (Byte Pair Encoding) phức tạp."
    ],
    answer: 0,
    explanation: "Bigram (hoặc 2-gram) là cặp hai mã thông báo (thường là từ) đi liền nhau. Việc sử dụng bigram giúp mô hình giữ lại một số kiến thức về thứ tự từ cục bộ."
  },
  {
    type: "mcq",
    question: "Tại sao không gian của 4-gram (4 từ liên tiếp) thường khó sử dụng hiệu quả?",
    options: [
      "Không gian vô cùng lớn dẫn đến cần từ vựng khổng lồ, dễ bị quá khớp.",
      "Nó làm giảm tốc độ xử lý trên CPU do phải tính toán song song.",
      "Các mô hình RNN không thể xử lý các chuỗi lớn hơn 3 mã thông báo.",
      "Mã hóa đa điểm (multi-hot) không hỗ trợ bất kỳ chuỗi N-gram nào."
    ],
    answer: 0,
    explanation: "Khi tăng độ dài N-gram (ví dụ 4-gram), số lượng kết hợp có thể có tăng theo cấp số nhân, khiến từ vựng phình to và mô hình dễ dàng ghi nhớ toàn bộ đoạn văn (quá khớp)."
  },
  {
    type: "mcq",
    question: "Mô hình trình tự (Sequence models) khác với mô hình túi từ ở điểm nào?",
    options: [
      "Giữ nguyên thứ tự các mã thông báo và trực tiếp học các phụ thuộc vị trí.",
      "Chỉ mã hóa các đánh giá dưới dạng vectơ nhị phân (có hoặc không có từ).",
      "Luôn luôn sử dụng mã hóa một-điểm (one-hot encoding) để biểu diễn từ.",
      "Bỏ qua mọi từ phụ (subwords) và chỉ xử lý các ký tự trong bảng mã ASCII."
    ],
    answer: 0,
    explanation: "Các mô hình trình tự (như RNN, 1D ConvNet, Transformer) nhận đầu vào là chuỗi các mã thông báo theo đúng thứ tự của chúng, cho phép tìm hiểu cấu trúc tuần tự."
  },
  {
    type: "mcq",
    question: "Mục đích của việc sử dụng padding (đệm) là gì khi xử lý theo lô?",
    options: [
      "Đảm bảo tất cả các chuỗi đầu vào đều có cùng chiều dài để song song hóa tính toán.",
      "Loại bỏ các từ thường gặp (như 'the', 'is') ra khỏi các câu văn bản đầu vào.",
      "Thay thế các từ không có trong từ vựng bằng mã thông báo không xác định `[UNK]`.",
      "Tăng cường độ chính xác cho các bài đánh giá phim có độ dài quá ngắn."
    ],
    answer: 0,
    explanation: "Các tensor đầu vào bắt buộc phải có hình dạng chữ nhật cố định. Padding thêm các token `[PAD]` (thường là 0) để các câu ngắn có cùng độ dài với câu dài nhất trong lô."
  },
  {
    type: "mcq",
    question: "Vấn đề của biểu diễn One-hot encoding đối với văn bản là gì?",
    options: [
      "Các vectơ thưa thớt, kích thước lớn và không mã hóa bất kỳ mối quan hệ ngữ nghĩa nào.",
      "Chỉ có thể mã hóa các chữ cái in thường và không thể biểu diễn số hoặc ký hiệu.",
      "Vectơ được tạo ra quá đặc (dense) dẫn đến tình trạng quá tải bộ nhớ RAM.",
      "Phải sử dụng CPU để huấn luyện mạng thay vì tính toán nhanh trên GPU."
    ],
    answer: 0,
    explanation: "One-hot encoding giả định mọi từ độc lập với nhau (các vectơ trực giao). Các vectơ này có số chiều bằng kích thước từ vựng (rất lớn) và không thể hiện từ đồng nghĩa."
  },
  {
    type: "mcq",
    question: "Word Embedding (Nhúng từ) là gì?",
    options: [
      "Biểu diễn vectơ dày đặc, có chiều thấp, trong đó từ tương tự có vị trí gần nhau.",
      "Một mảng gồm các số 0 và 1, trong đó chỉ có một vị trí duy nhất chứa số 1.",
      "Quá trình biến đổi văn bản thô thành các chỉ số số nguyên tương ứng.",
      "Một mạng lặp LSTM hai chiều có thể hiểu được ý nghĩa tiếng Việt."
    ],
    answer: 0,
    explanation: "Word Embedding chuyển đổi từ thành các vectơ có số chiều thấp (dense vector) được học qua quá trình huấn luyện, sao cho các từ gần gũi về mặt ngữ nghĩa sẽ nằm gần nhau trong không gian vectơ."
  },
  {
    type: "mcq",
    question: "Lớp `Embedding` trong Keras hoạt động như thế nào?",
    options: [
      "Như một từ điển ánh xạ các chỉ số số nguyên thành các vectơ dấu phẩy động.",
      "Như một biểu thức chính quy để phân tách câu thành các mã thông báo con.",
      "Như một vòng lặp RNN giúp nhớ trạng thái của các từ xuất hiện phía trước.",
      "Như một bộ lọc Convolution để phát hiện các cụm từ (N-gram) cục bộ."
    ],
    answer: 0,
    explanation: "Lớp `Embedding` thực chất là một bảng tra cứu (lookup table) dùng chỉ số nguyên (token ID) làm khóa để lấy ra một vectơ nhúng tương ứng."
  },
  {
    type: "fill",
    question: "Quá trình biến đổi văn bản thành các đơn vị nhỏ (như từ hoặc cụm từ) được gọi là _________.",
    options: ["tokenization"],
    answer: "tokenization",
    explanation: "Tokenization (mã thông báo hóa) là bước cơ bản cắt văn bản thô thành các mảng đơn vị cơ sở để hệ thống xử lý."
  },
  {
    type: "fill",
    question: "Token đặc biệt được sử dụng để cân bằng chiều dài của các chuỗi trong quá trình xử lý theo lô là _________.",
    options: ["[PAD]", "PAD", "padding"],
    answer: "[PAD]",
    explanation: "Mã thông báo `[PAD]` (thường có index là 0) được thêm vào các chuỗi ngắn để tạo tensor dạng hình chữ nhật đồng nhất về kích thước."
  },
  {
    type: "fill",
    question: "Trong n-gram, n = 2 thì cụm 2 từ liền kề được gọi là _________.",
    options: ["bigram", "bigrams"],
    answer: "bigram",
    explanation: "Bigram là chuỗi 2 mã thông báo xuất hiện liên tiếp trong dữ liệu đầu vào. Tương tự, 3-gram gọi là trigram."
  },
  {
    type: "fill",
    question: "Biểu diễn vectơ thưa thớt, nhị phân, kích thước bằng kích thước từ vựng là mã hóa _________.",
    options: ["one-hot", "one hot", "one-hot encoding"],
    answer: "one-hot",
    explanation: "Mã hóa one-hot tạo ra một mảng toàn số 0, chỉ chứa duy nhất một số 1 tại vị trí chỉ mục của từ đó."
  },
  {
    type: "fill",
    question: "Phương pháp tạo vectơ dày đặc, có số chiều thấp và mang ý nghĩa ngữ nghĩa gọi là Word _________.",
    options: ["Embedding", "Embeddings"],
    answer: "Embedding",
    explanation: "Word Embedding giúp mô hình hiểu được sự liên quan về mặt ngữ nghĩa (semantic relationships) giữa các từ."
  },
  {
    type: "matching",
    question: "Ghép nối các kỹ thuật NLP với đặc điểm tương ứng:",
    options: [
      "Bag-of-words",
      "One-hot encoding",
      "Word Embedding",
      "Subword Tokenization"
    ],
    answer: [
      "Loại bỏ thứ tự từ, tính sự hiện diện",
      "Vectơ thưa thớt, nhị phân, trực giao",
      "Vectơ dày đặc học được ý nghĩa ngữ nghĩa",
      "Xử lý tốt các từ hiếm bằng các từ phụ"
    ],
    explanation: "Mỗi kỹ thuật có ưu nhược điểm riêng: Bag-of-words bỏ qua thứ tự, One-hot tốn bộ nhớ, Word Embedding tối ưu học đặc trưng, Subword tối ưu từ vựng."
  },
  {
    type: "matching",
    question: "Ghép nối các đối tượng/từ khóa với vai trò trong quy trình NLP:",
    options: [
      "Standardization",
      "Tokenization",
      "Indexing",
      "Padding"
    ],
    answer: [
      "Đưa về chữ thường, xóa dấu câu",
      "Tách văn bản thành từ hoặc ký tự",
      "Ánh xạ token sang số nguyên",
      "Cân bằng độ dài câu trong lô"
    ],
    explanation: "Tiêu chuẩn hóa làm sạch dữ liệu, mã thông báo hóa chia nhỏ, lập chỉ mục số hóa dữ liệu, và đệm để tạo tensor."
  },
  {
    type: "matching",
    question: "Ghép nối kiến trúc mô hình với đặc điểm phân loại văn bản:",
    options: [
      "Mô hình Unigram",
      "Mô hình Bigram",
      "Mô hình LSTM",
      "Lớp TextVectorization"
    ],
    answer: [
      "Chỉ xét từng từ riêng lẻ",
      "Có xét thứ tự cục bộ của hai từ",
      "Mạng lặp học sự phụ thuộc tuần tự xa",
      "Xử lý trước, tách và lập chỉ mục trong Keras"
    ],
    explanation: "Bigram tốt hơn Unigram vì có xét ngữ cảnh ngắn. LSTM tốt cho chuỗi dài nhưng tốc độ huấn luyện chậm hơn. TextVectorization hỗ trợ mọi khâu tiền xử lý văn bản ở Keras."
  },
  {
    type: "matching",
    question: "Ghép nối các khái niệm trong Word Embedding với ví dụ trực quan:",
    options: [
      "Vectơ vương miện",
      "Khoảng cách Cosine",
      "Số chiều nhúng",
      "Không gian hình học"
    ],
    answer: [
      "Vua + Nữ = Nữ hoàng",
      "Đo mức độ giống nhau của hai từ",
      "Thường là 256, 512, 1024",
      "Nơi các từ đồng nghĩa nằm gần nhau"
    ],
    explanation: "Không gian nhúng (Embedding space) mã hóa các tính chất ngôn ngữ học thành các phép toán đại số tuyến tính."
  },
  {
    type: "matching",
    question: "Ghép nối các loại mã thông báo hóa và ưu/nhược điểm:",
    options: [
      "Cấp độ từ (Word)",
      "Cấp độ ký tự (Character)",
      "Cấp độ từ phụ (Subword)",
      "Sử dụng biểu thức chính quy (Regex)"
    ],
    answer: [
      "Chuỗi ngắn, nhưng cần từ vựng rất lớn",
      "Từ vựng nhỏ, nhưng chuỗi quá dài",
      "Cân bằng giữa từ vựng nhỏ và chuỗi vừa",
      "Công cụ để thực hiện bước tách văn bản"
    ],
    explanation: "Mã hóa từ phụ (như Byte-Pair Encoding) ngày nay là chuẩn chung của hầu hết các mô hình ngôn ngữ lớn để xử lý hiệu quả từ vựng."
  },
  {
    type: "sorting",
    question: "Sắp xếp các bước tiêu chuẩn trong quy trình tiền xử lý văn bản:",
    options: [
      "Tiêu chuẩn hóa (Chữ thường, bỏ dấu câu)",
      "Mã thông báo hóa (Tách văn bản thành tokens)",
      "Xây dựng từ vựng (Vocabulary)",
      "Lập chỉ mục (Ánh xạ token thành ID nguyên)",
      "Đệm chuỗi (Padding / Truncating)"
    ],
    answer: [0, 1, 2, 3, 4],
    explanation: "Đầu tiên cần làm sạch văn bản (standardization), sau đó cắt thành các mã (tokenization), thống kê tạo từ vựng (vocabulary), gán số (indexing), cuối cùng là làm đều độ dài lô (padding)."
  },
  {
    type: "sorting",
    question: "Sắp xếp kích thước không gian vectơ cho các từ vựng 20,000 từ theo thứ tự tiết kiệm bộ nhớ dần (Từ tốn nhất đến tiết kiệm nhất):",
    options: [
      "Mã hóa one-hot (Vectơ 20,000 chiều)",
      "Nhúng từ 1024 chiều (Word Embedding)",
      "Nhúng từ 256 chiều (Word Embedding)"
    ],
    answer: [0, 1, 2],
    explanation: "One-hot lưu trữ số chiều ngang với số lượng từ vựng, cực kỳ lãng phí. Embedding biểu diễn ở không gian dense nên giảm chiều rất lớn, 256 chiều sẽ nhỏ hơn 1024 chiều."
  },
  {
    type: "sorting",
    question: "Sắp xếp thuật toán Mã hóa cặp byte (Byte-Pair Encoding) theo các bước thực hiện:",
    options: [
      "Bắt đầu với vốn từ vựng cơ bản gồm các ký tự đơn.",
      "Đếm tần suất xuất hiện của các cặp ký tự kề nhau.",
      "Hợp nhất cặp ký tự phổ biến nhất thành một ký hiệu mới.",
      "Thêm ký hiệu vừa hợp nhất vào vốn từ vựng (lặp lại)."
    ],
    answer: [0, 1, 2, 3],
    explanation: "BPE bắt đầu từ cấp độ ký tự, tiến hành thống kê cặp liền kề và hợp nhất dần để tạo ra các từ phụ phổ biến (subwords)."
  },
  {
    type: "sorting",
    question: "Sắp xếp khả năng học ngữ cảnh chuỗi của các mô hình (từ thấp nhất đến cao nhất):",
    options: [
      "Mô hình Unigram (Bag-of-words)",
      "Mô hình Bigram",
      "Mô hình lặp LSTM"
    ],
    answer: [0, 1, 2],
    explanation: "Unigram không có bất kỳ khái niệm thứ tự nào, Bigram chỉ có thứ tự cục bộ (2 từ kề nhau), LSTM có thể nhớ được trình tự của toàn bộ chuỗi dài."
  },
  {
    type: "sorting",
    question: "Sắp xếp các bước áp dụng Word Embedding vào một mô hình phân loại (Keras):",
    options: [
      "Sử dụng TextVectorization để chuyển văn bản thành số nguyên.",
      "Tạo lớp `Input` nhận tensor của các ID.",
      "Đưa qua lớp `Embedding` để chuyển ID thành vectơ dày đặc.",
      "Đưa qua LSTM và cuối cùng là Dense(1) để phân loại."
    ],
    answer: [0, 1, 2, 3],
    explanation: "Bước đầu là tiền xử lý ID, sau đó định nghĩa Input, gắn lớp Embedding ánh xạ ID ra Dense vector, sau đó là mô hình trình tự hoặc tuyến tính để ra kết quả cuối."
  }
];

export default questions;
