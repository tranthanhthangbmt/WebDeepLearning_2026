# Chapter 1: What is deep learning?

*(Bản dịch tự động từ PDF)*

11Học sâu là gì?

Chương này bao gồm ¡ Các định nghĩa cấp cao về các khái niệm cơ bản ¡ Giới thiệu sơ lược về các nguyên tắc đằng sau học máy ¡ Sự phổ biến ngày càng tăng của học sâu và tiềm năng trong tương lai Trong thập kỷ qua, trí tuệ nhân tạo (AI) đã trở thành chủ đề được truyền thông thổi phồng mạnh mẽ. Học máy, học sâu và AI xuất hiện trong vô số bài viết, thường nằm ngoài các ấn phẩm liên quan đến công nghệ. Chúng ta được hứa hẹn về một tương lai của các chatbot thông minh, ô tô tự lái và trợ lý ảo - một tương lai đôi khi được vẽ ra một cách u ám và đôi khi lại là điều không tưởng, nơi việc làm của con người sẽ khan hiếm và hầu hết hoạt động kinh tế sẽ được xử lý bởi robot hoặc tác nhân AI. Đối với người thực hành học máy, điều quan trọng là có thể nhận ra tín hiệu giữa tiếng ồn để bạn có thể nhận biết những diễn biến đang thay đổi thế giới từ các thông cáo báo chí được cường điệu hóa quá mức.

Tương lai của chúng ta đang bị đe dọa và đó là tương lai mà bạn đóng vai trò tích cực: sau khi đọc cuốn sách này, bạn sẽ là một trong những người có thể phát triển các hệ thống AI này. Vậy hãy

2 chương 1 học sâu là gì?

giải quyết những câu hỏi sau: Học sâu đã đạt được những gì cho đến nay? Nó quan trọng thế nào?

Tiếp theo chúng ta sẽ đi đâu? Bạn có nên tin vào sự cường điệu?

1.1 Trí tuệ nhân tạo, học máy và học sâu Trước tiên, chúng ta cần xác định rõ ràng những gì chúng ta đang nói đến khi đề cập đến AI. Trí tuệ nhân tạo, học máy và học sâu là gì (hình 1.1)? Chúng liên hệ với nhau như thế nào?

1.2 Trí tuệ nhân tạo Trí tuệ nhân tạo ra đời vào những năm 1950, khi một số nhà tiên phong trong lĩnh vực khoa học máy tính còn non trẻ bắt đầu hỏi liệu máy tính có thể được tạo ra để “suy nghĩ” hay không - một câu hỏi mà ngày nay chúng ta vẫn đang khám phá các phân nhánh của nó.

Trong khi nhiều ý tưởng cơ bản đã được hình thành trong nhiều năm và thậm chí nhiều thập kỷ trước, “trí tuệ nhân tạo” cuối cùng đã được kết tinh thành một lĩnh vực nghiên cứu vào năm 1956, khi John McCarthy, khi đó là Trợ lý Giáo sư Toán học trẻ tại Đại học Dartmouth, đã tổ chức một hội thảo mùa hè theo đề xuất sau:

Nghiên cứu này được tiến hành trên cơ sở phỏng đoán rằng mọi khía cạnh của việc học hoặc bất kỳ đặc điểm nào khác của trí thông minh về nguyên tắc đều có thể được mô tả chính xác đến mức có thể tạo ra một chiếc máy để mô phỏng nó. Một nỗ lực sẽ được thực hiện để tìm cách làm cho máy móc sử dụng ngôn ngữ, hình thành các khái niệm và khái niệm trừu tượng, giải quyết các loại vấn đề hiện chỉ dành cho con người và cải thiện bản thân. Chúng tôi nghĩ rằng có thể đạt được tiến bộ đáng kể trong một hoặc nhiều vấn đề này nếu một nhóm các nhà khoa học được lựa chọn cẩn thận cùng nhau giải quyết vấn đề đó trong một mùa hè.

Vào cuối mùa hè, hội thảo kết thúc mà vẫn chưa giải được hoàn toàn câu đố mà nó đặt ra để điều tra. Tuy nhiên, nó đã có sự tham gia của nhiều người, những người sẽ trở thành người tiên phong trong lĩnh vực này và nó đã khởi động một cuộc cách mạng trí tuệ vẫn đang tiếp diễn cho đến ngày nay.

Nói một cách ngắn gọn, AI có thể được mô tả là nỗ lực tự động hóa các nhiệm vụ trí tuệ thường do con người thực hiện. Như vậy, AI là một lĩnh vực chung bao gồm học máy và học sâu, nhưng cũng bao gồm nhiều cách tiếp cận khác có thể không liên quan đến bất kỳ việc học nào. Hãy xem xét điều đó cho đến những năm 1980, hầu hết các sách giáo khoa về AI đều không hề đề cập đến “việc học”! Ví dụ: các chương trình cờ vua ban đầu chỉ liên quan đến các quy tắc được mã hóa cứng do các lập trình viên tạo ra và không đủ tiêu chuẩn là học máy. Trên thực tế, trong một thời gian khá dài, hầu hết các chuyên gia đều tin rằng trí tuệ nhân tạo ở cấp độ con người có thể đạt được bằng cách yêu cầu các lập trình viên tạo ra một bộ quy tắc rõ ràng đủ lớn để thao tác Trí tuệ nhân tạo Học máy Học sâu Hình 1.1 Trí tuệ nhân tạo, học máy và học sâu

3 Kiến thức học máy được lưu trữ trong cơ sở dữ liệu rõ ràng. Cách tiếp cận này được gọi là AI tượng trưng. Đó là mô hình thống trị về AI từ những năm 1950 đến cuối những năm 1980 và đạt đến mức độ phổ biến cao nhất trong thời kỳ bùng nổ hệ thống chuyên gia những năm 1980.

Mặc dù AI biểu tượng tỏ ra phù hợp để giải quyết các vấn đề logic, được xác định rõ ràng, chẳng hạn như chơi cờ, nhưng hóa ra lại rất khó để tìm ra các quy tắc rõ ràng để giải quyết các vấn đề phức tạp, mờ hơn, chẳng hạn như phân loại hình ảnh, nhận dạng giọng nói hoặc dịch ngôn ngữ tự nhiên. Một cách tiếp cận mới đã nảy sinh để thay thế vị trí của AI mang tính biểu tượng: học máy.

1.3 Học máy Ở nước Anh thời Victoria, Lady Ada Lovelace là bạn và là cộng tác viên của Charles Bab bage, người phát minh ra Công cụ phân tích: chiếc máy tính cơ học đa năng đầu tiên được biết đến. Mặc dù có tầm nhìn xa và đi trước thời đại, Công cụ phân tích không được coi là một máy tính đa năng khi nó được thiết kế vào những năm 1830 và 1840, bởi vì khái niệm tính toán cho mục đích chung vẫn chưa được phát minh. Nó chỉ đơn thuần là một cách sử dụng các hoạt động cơ học để tự động hóa một số tính toán nhất định từ lĩnh vực phân tích toán học—do đó có tên là Công cụ phân tích. Như vậy, nó là hậu duệ trí tuệ của những nỗ lực trước đó trong việc mã hóa các phép toán ở dạng bánh răng, chẳng hạn như Pascaline, hay máy tính bước của Leibniz, một phiên bản cải tiến của Pascaline. Được thiết kế bởi Blaise Pascal vào năm 1642 (ở tuổi 19!), Pascaline là máy tính cơ học đầu tiên trên thế giới—nó có thể cộng, trừ, nhân hoặc thậm chí chia các chữ số.

Năm 1843, Ada Lovelace nhận xét về việc phát minh ra Máy phân tích:

Công cụ phân tích không có ý định tạo ra bất cứ thứ gì. Nó có thể làm bất cứ điều gì chúng ta biết, cách ra lệnh cho nó thực hiện… Nhiệm vụ của nó là hỗ trợ chúng ta cung cấp những gì chúng ta đã quen thuộc.

Ngay cả với quan điểm lịch sử kéo dài 182 năm, nhận xét của Quý bà Lovelace vẫn rất thu hút. Liệu một chiếc máy tính đa năng có thể “tạo ra” bất cứ thứ gì hay nó sẽ luôn bị ràng buộc phải thực hiện một cách chậm chạp các quy trình mà con người chúng ta hoàn toàn hiểu được? Nó có thể có khả năng của bất kỳ suy nghĩ ban đầu nào không? Nó có thể rút kinh nghiệm được không? Nó có thể thể hiện sự sáng tạo?

Nhận xét của cô sau đó được nhà tiên phong về AI Alan Turing trích dẫn là “sự phản đối của Lady Lovelace” trong bài báo mang tính bước ngoặt năm 1950 của ông “Máy tính và trí thông minh”,1 trong đó giới thiệu bài kiểm tra Turing2 cũng như các khái niệm chính sẽ hình thành nên AI. Turing có quan điểm—rất khiêu khích vào thời điểm đó—rằng về nguyên tắc, máy tính có thể được tạo ra để mô phỏng mọi khía cạnh của trí thông minh con người.

Cách thông thường để khiến máy tính thực hiện công việc hữu ích là nhờ một lập trình viên viết ra các quy tắc—một chương trình máy tính—để tuân theo nhằm biến dữ liệu đầu vào thành các câu trả lời thích hợp, giống như Lady Lovelace viết ra các hướng dẫn từng bước cho Bài phân tích lúc 1 giờ sáng Turing, “Máy tính và trí thông minh,” Mind 59, không. 236 (1950): 433-460.

2 Mặc dù bài kiểm tra Turing đôi khi được hiểu là một bài kiểm tra theo nghĩa đen - một mục tiêu mà lĩnh vực AI nên đặt ra - Turing chỉ có ý nghĩa nó như một công cụ khái niệm trong một cuộc thảo luận triết học về bản chất của nhận thức.

4 chương 1 học sâu là gì?

Động cơ để thực hiện. Học máy giải quyết vấn đề này: máy xem dữ liệu đầu vào và các câu trả lời tương ứng, đồng thời tìm ra các quy tắc nên là gì (hình 1.2).

Một hệ thống máy học được đào tạo thay vì được lập trình rõ ràng. Nó đưa ra nhiều ví dụ liên quan đến một nhiệm vụ và tìm thấy cấu trúc thống kê trong các ví dụ này mà cuối cùng cho phép hệ thống đưa ra các quy tắc để tự động hóa nhiệm vụ. Ví dụ: nếu bạn muốn tự động hóa tác vụ gắn thẻ cho các bức ảnh về kỳ nghỉ của mình, bạn có thể trình bày một hệ thống máy học với nhiều ví dụ về các bức ảnh đã được con người gắn thẻ và hệ thống sẽ tìm hiểu các quy tắc thống kê để liên kết các bức ảnh cụ thể với các thẻ cụ thể như “phong cảnh” hoặc “thực phẩm”. Mặc dù học máy chỉ bắt đầu phát triển vào những năm 1990 nhưng nó đã nhanh chóng trở thành trường con phổ biến nhất và thành công nhất của AI, một xu hướng được thúc đẩy bởi sự sẵn có của phần cứng nhanh hơn và bộ dữ liệu lớn hơn. Học máy có liên quan đến thống kê toán học, nhưng nó khác với thống kê ở một số điểm quan trọng—theo cùng một nghĩa là y học có liên quan đến hóa học nhưng không thể quy giản thành hóa học, vì y học xử lý các hệ thống riêng biệt với các đặc tính riêng biệt của chúng. Không giống như thống kê, học máy có xu hướng xử lý các tập dữ liệu lớn, phức tạp (chẳng hạn như tập dữ liệu gồm hàng triệu hình ảnh, mỗi hình ảnh bao gồm hàng chục nghìn pixel) mà phân tích thống kê cổ điển như phân tích Bayesian sẽ không thực tế. Kết quả là, học máy, và đặc biệt là học sâu, thể hiện tương đối ít lý thuyết toán học—có thể là quá ít—và về cơ bản là một môn kỹ thuật. Không giống như vật lý lý thuyết hay toán học, học máy là một lĩnh vực mang tính thực hành cao được thúc đẩy bởi những phát hiện thực nghiệm và phụ thuộc sâu sắc vào những tiến bộ trong phần mềm và phần cứng.

1.4 Các quy tắc học tập và biểu diễn từ dữ liệu Để xác định học sâu và hiểu sự khác biệt giữa học sâu và các phương pháp học máy khác, trước tiên chúng ta cần một số ý tưởng về thuật toán học máy làm gì. Chúng tôi vừa tuyên bố rằng học máy phát hiện ra các quy tắc để thực hiện tác vụ xử lý dữ liệu, đưa ra các ví dụ về những gì được mong đợi. Vì vậy, để thực hiện học máy, chúng ta cần ba điều:

¡ Điểm dữ liệu đầu vào —Ví dụ: nếu nhiệm vụ là nhận dạng giọng nói, những điểm dữ liệu này có thể là tệp âm thanh của người đang nói. Nếu nhiệm vụ là gắn thẻ hình ảnh, chúng có thể là hình ảnh.

¡ Ví dụ về kết quả đầu ra dự kiến ​​—Trong tác vụ nhận dạng giọng nói, đây có thể là bản ghi âm của các tệp âm thanh do con người tạo ra. Trong tác vụ hình ảnh, kết quả đầu ra dự kiến ​​có thể là các thẻ như “chó”, “mèo”, v.v.

¡ Một cách để đo xem thuật toán có hoạt động tốt hay không —Điều này là cần thiết để xác định khoảng cách giữa đầu ra hiện tại của thuật toán và Câu trả lời mong đợi của nó Quy tắc Dữ liệu Quy tắc lập trình cổ điển Câu trả lời dữ liệu Học máy Hình 1.2 Học máy: một mô hình lập trình mới

5 Quy tắc học tập và biểu diễn từ dữ liệu đầu ra. Phép đo được sử dụng làm tín hiệu phản hồi để điều chỉnh cách hoạt động của thuật toán. Bước điều chỉnh này được chúng tôi gọi là học tập.

Mô hình học máy chuyển đổi dữ liệu đầu vào của nó thành đầu ra có ý nghĩa, một quá trình được “học” từ việc tiếp xúc với các ví dụ đã biết về đầu vào và đầu ra. Do đó, vấn đề trọng tâm trong học máy và học sâu là biến đổi dữ liệu một cách có ý nghĩa: nói cách khác, tìm hiểu các cách biểu diễn hữu ích của dữ liệu đầu vào trong tầm tay—các cách biểu diễn giúp chúng ta tiến gần hơn đến kết quả đầu ra mong đợi.

Trước khi chúng ta đi xa hơn, đại diện là gì? Về cốt lõi, đó là một cách khác để xem dữ liệu để biểu diễn hoặc mã hóa dữ liệu. Ví dụ: một hình ảnh màu có thể được mã hóa ở định dạng RGB (đỏ-lục-xanh) hoặc ở định dạng HSV (giá trị độ bão hòa màu sắc): đây là hai cách biểu diễn khác nhau của cùng một dữ liệu. Một số nhiệm vụ có thể khó khăn với cách biểu diễn này có thể trở nên dễ dàng với cách biểu diễn khác. Ví dụ: tác vụ “Chọn tất cả các pixel màu đỏ trong hình ảnh” đơn giản hơn ở định dạng RGB, trong khi “Làm cho hình ảnh ít được xếp hạng satu” đơn giản hơn ở định dạng HSV. Các mô hình học máy đều tập trung vào việc tìm kiếm các cách trình bày thích hợp cho dữ liệu đầu vào của chúng—các phép biến đổi dữ liệu giúp dữ liệu trở nên phù hợp hơn với nhiệm vụ hiện tại.

Hãy làm cho điều này trở nên cụ thể. Xét trục x, trục y và một số điểm được biểu thị bằng tọa độ của chúng trong hệ (x, y), như trong hình 1.3.

Như bạn có thể thấy, chúng ta có một vài điểm trắng và một vài điểm đen. Giả sử chúng ta muốn phát triển một thuật toán có thể lấy tọa độ (x, y) của một điểm và đưa ra kết quả xem điểm đó có thể là đen hay trắng. Trong trường hợp này, ¡ Đầu vào là tọa độ các điểm của chúng tôi.

¡ Kết quả đầu ra mong đợi là màu sắc của các điểm của chúng tôi.

¡ Ví dụ: một cách để đo lường xem thuật toán của chúng tôi có hoạt động tốt hay không có thể là tỷ lệ phần trăm số điểm được phân loại chính xác.

Những gì chúng ta cần ở đây là một cách thể hiện mới cho dữ liệu của chúng ta để phân biệt rõ ràng các điểm trắng với các điểm đen. Một phép biến đổi mà chúng ta có thể sử dụng, trong số nhiều khả năng khác, sẽ là phép biến đổi tọa độ, được minh họa trong hình 1.4.

y2: Thay đổi tọa độ xy1: Dữ liệu thô xy3: Biểu diễn tốt hơn x Hình 1.4 Thay đổi tọa độ x Hình 1.3 Một số dữ liệu mẫu

6 chương 1 học sâu là gì?

Trong hệ tọa độ mới này, tọa độ các điểm của chúng tôi có thể được coi là cách thể hiện mới cho dữ liệu của chúng tôi. Và đó là một điều tốt! Với cách biểu diễn này, bài toán phân loại đen/trắng có thể được biểu diễn dưới dạng một quy tắc đơn giản: “Điểm đen sao cho x > 0” hoặc “Điểm trắng sao cho x < 0”. Cách biểu diễn mới này, kết hợp với quy tắc đơn giản này, sẽ giải quyết gọn gàng vấn đề phân loại.

Trong trường hợp này, chúng tôi đã xác định sự thay đổi tọa độ bằng tay: chúng tôi sử dụng trí thông minh của con người để đưa ra cách trình bày dữ liệu phù hợp của riêng mình. Điều này phù hợp với một vấn đề cực kỳ đơn giản như vậy, nhưng bạn có thể làm tương tự nếu nhiệm vụ là phân loại hình ảnh của các chữ số viết tay không? Bạn có thể viết các phép biến đổi hình ảnh rõ ràng, có thể thực hiện được trên máy tính để làm sáng tỏ sự khác biệt giữa số 6 và số 8, giữa số 1 và số 7, trên tất cả các loại chữ viết tay khác nhau không?

Điều này có thể xảy ra ở một mức độ nào đó. Các quy tắc dựa trên cách biểu diễn các chữ số, chẳng hạn như “đếm số vòng lặp khép kín” hoặc biểu đồ pixel dọc và ngang có thể thực hiện tốt công việc phân biệt các chữ số viết tay. Nhưng việc tìm kiếm các cách biểu diễn hữu ích như vậy bằng tay là một công việc khó khăn và như bạn có thể tưởng tượng, hệ thống dựa trên quy tắc thu được sẽ dễ vỡ và là một cơn ác mộng để duy trì. Mỗi khi bạn gặp một ví dụ mới về chữ viết tay có thể phá vỡ các quy tắc đã được suy nghĩ cẩn thận của bạn, bạn sẽ phải thêm các phép biến đổi dữ liệu mới và các quy tắc mới, đồng thời tính đến sự tương tác của chúng với mọi quy tắc trước đó.

Có lẽ bạn đang nghĩ, nếu quá trình này phức tạp đến vậy, liệu chúng ta có thể tự động hóa nó không? Điều gì sẽ xảy ra nếu chúng ta cố gắng tìm kiếm một cách có hệ thống các tập hợp dữ liệu và quy tắc được tạo tự động khác nhau dựa trên chúng, xác định những tập hợp tốt bằng cách sử dụng phần trăm tuổi của các chữ số được phân loại chính xác trong một số tập dữ liệu phát triển làm phản hồi? Sau đó chúng tôi sẽ thực hiện học máy. Học tập, trong bối cảnh học máy, mô tả một quy trình tìm kiếm tự động để chuyển đổi dữ liệu tạo ra các biểu diễn hữu ích của một số dữ liệu, được hướng dẫn bởi một số tín hiệu phản hồi—các biểu diễn tuân theo các quy tắc đơn giản hơn để giải quyết nhiệm vụ hiện tại.

Những phép biến đổi này có thể là các thay đổi tọa độ (như trong ví dụ phân loại tọa độ 2D của chúng tôi) hoặc biểu đồ pixel và vòng đếm (như trong ví dụ phân loại chữ số của chúng tôi), nhưng chúng cũng có thể là các phép chiếu tuyến tính, bản dịch và các phép toán phi tuyến (chẳng hạn như “Chọn tất cả các điểm sao cho x > 0”), v.v. Các thuật toán học máy thường không sáng tạo trong việc tìm ra những biến đổi này; họ chỉ đang tìm kiếm thông qua một tập hợp các phép toán được xác định trước, được gọi là không gian giả thuyết. Ví dụ: không gian của tất cả các thay đổi tọa độ có thể có sẽ là không gian giả thuyết của chúng tôi trong ví dụ phân loại tọa độ 2D.

Vì vậy, chính xác thì học máy là gì: tìm kiếm các biểu diễn và quy tắc hữu ích đối với một số dữ liệu đầu vào, trong không gian khả năng được xác định trước, sử dụng hướng dẫn từ tín hiệu phản hồi. Ý tưởng đơn giản này cho phép chúng tôi giải quyết một loạt các nhiệm vụ trí tuệ đáng chú ý, từ lái xe tự động đến trả lời câu hỏi bằng ngôn ngữ tự nhiên.

Bây giờ bạn đã hiểu ý nghĩa của việc học sâu, hãy cùng xem điều gì làm cho việc học sâu trở nên đặc biệt.

7 Cái “sâu” trong “học sâu” 1.5 Cái “sâu” trong “học sâu” Học sâu là một lĩnh vực con cụ thể của học máy; đó là một cách mới để học cách biểu diễn từ dữ liệu, trong đó nhấn mạnh đến việc học các lớp biểu diễn liên tiếp ngày càng có ý nghĩa. Từ “sâu” trong “học sâu” không ám chỉ đến bất kỳ loại hiểu biết sâu sắc nào đạt được bằng cách tiếp cận này; đúng hơn, nó đại diện cho ý tưởng về các lớp biểu diễn liên tiếp. Có bao nhiêu lớp đóng góp cho một mô hình dữ liệu được gọi là độ sâu của mô hình. Các tên thích hợp khác cho trường này có thể là học biểu diễn theo lớp hoặc học biểu diễn phân cấp. Học sâu hiện đại thường bao gồm hàng chục hoặc thậm chí hàng trăm lớp biểu diễn liên tiếp và tất cả chúng đều được học tự động khi tiếp xúc với dữ liệu huấn luyện. Trong khi đó, các cách tiếp cận khác đối với học máy có xu hướng chỉ tập trung vào việc học một hoặc hai lớp biểu diễn dữ liệu (ví dụ: lấy biểu đồ pixel và sau đó áp dụng quy tắc phân loại); do đó đôi khi chúng được gọi là học tập nông cạn.

Trong học sâu, các biểu diễn phân lớp này được học thông qua các mô hình gọi là mạng thần kinh, được cấu trúc theo các lớp chữ xếp chồng lên nhau. Thuật ngữ mạng lưới thần kinh là một tham chiếu đến sinh học thần kinh, nhưng mặc dù một số khái niệm trọng tâm trong học sâu được phát triển một phần bằng cách lấy cảm hứng từ sự hiểu biết của chúng ta về não bộ (đặc biệt là vỏ não thị giác), các mô hình học sâu không phải là mô hình của não. Không có bằng chứng nào cho thấy bộ não thực hiện bất cứ điều gì giống như cơ chế học tập được sử dụng trong các mô hình học sâu hiện đại. Bạn có thể bắt gặp các bài báo khoa học đại chúng tuyên bố rằng học sâu hoạt động giống như bộ não hoặc được mô phỏng theo bộ não, nhưng thực tế không phải vậy. Sẽ gây nhầm lẫn và phản tác dụng nếu những người mới tham gia vào lĩnh vực này nghĩ rằng học sâu có liên quan đến sinh học thần kinh theo bất kỳ cách nào; bạn không cần tấm màn bí ẩn và bí ẩn “giống như tâm trí của chúng ta”, và bạn cũng có thể quên bất cứ điều gì bạn có thể đã đọc về mối liên hệ giả thuyết giữa học sâu và sinh học. Đối với mục đích của chúng tôi, học sâu là một khung toán học để học các biểu diễn từ dữ liệu.

Các biểu diễn được học bằng thuật toán học sâu trông như thế nào? Chúng ta hãy xem xét cách một mạng sâu nhiều lớp (xem hình 1.5) biến đổi hình ảnh của một chữ số để nhận ra đó là chữ số nào.

Lớp 1 Đầu vào gốc Đầu ra cuối cùng Lớp 2 Lớp 3 Lớp 4 0 1 2 3 4 5 6 7 8 9 Hình 1.5 Mạng lưới thần kinh sâu để phân loại chữ số

8 chương 1 học sâu là gì?

Như bạn có thể thấy trong hình 1.6, mạng chuyển đổi hình ảnh số thành các biểu diễn ngày càng khác với hình ảnh gốc và ngày càng có nhiều thông tin về kết quả cuối cùng. Bạn có thể coi mạng sâu như một quá trình chắt lọc thông tin nhiều giai đoạn, trong đó thông tin đi qua các bộ lọc liên tiếp và ngày càng được tinh lọc hơn (nghĩa là hữu ích đối với một số nhiệm vụ).

Biểu diễn lớp 1 Đầu vào ban đầu Biểu diễn lớp 2 Biểu diễn lớp 3 Biểu diễn lớp 4 (đầu ra cuối cùng) Lớp 1 Lớp 2 Lớp 3 Lớp 40 1 2 3 4 5 6 7 8 9 Hình 1.6 Biểu diễn sâu được học bằng mô hình phân loại chữ số Vì vậy, về mặt kỹ thuật, học sâu là một cách học nhiều giai đoạn để học cách biểu diễn dữ liệu. Đó là một ý tưởng đơn giản, nhưng hóa ra, những cơ chế rất đơn giản, với quy mô vừa đủ, có thể trông giống như phép thuật.

1.6 Hiểu cách thức hoạt động của deep learning, trong ba hình. Tại thời điểm này, bạn đã biết rằng machine learning là ánh xạ đầu vào (chẳng hạn như hình ảnh) tới mục tiêu (chẳng hạn như nhãn “mèo”), được thực hiện bằng cách quan sát nhiều ví dụ về đầu vào và mục tiêu. Bạn cũng biết rằng mạng nơ-ron sâu thực hiện việc ánh xạ đầu vào tới mục tiêu này thông qua một chuỗi sâu các phép biến đổi dữ liệu đơn giản (lớp) và rằng các phép biến đổi dữ liệu này được học bằng cách tiếp xúc với các ví dụ. Bây giờ chúng ta hãy xem việc học này diễn ra như thế nào một cách cụ thể.

Thông số kỹ thuật về những gì một lớp thực hiện đối với dữ liệu đầu vào của nó được lưu trữ trong các trọng số của lớp, về bản chất là một loạt các con số. Về mặt kỹ thuật, chúng tôi cho rằng phép biến đổi được thực hiện bởi một lớp được tham số hóa bằng các trọng số của nó (xem hình 1.7). (Trọng số đôi khi còn được gọi là tham số của một lớp.) Trong ngữ cảnh này, việc học có nghĩa là tìm một tập hợp các giá trị cho trọng số của tất cả các lớp trong mạng, sao cho mạng sẽ ánh xạ chính xác các đầu vào mẫu tới các mục tiêu liên quan của chúng. Nhưng vấn đề là thế này: một mạng lưới thần kinh sâu có thể chứa hàng chục triệu tham số. Tìm đúng

9 Hiểu cách thức hoạt động của deep learning, giá trị ba con số đối với tất cả chúng có vẻ như là một nhiệm vụ khó khăn, đặc biệt khi việc sửa đổi giá trị của một tham số sẽ ảnh hưởng đến hành vi của tất cả các tham số khác! Mục tiêu: Tìm các giá trị phù hợp cho các trọng số này Lớp (chuyển đổi dữ liệu) Lớp trọng số đầu vào X (chuyển đổi dữ liệu) Dự đoán Y'Weights Hình 1.7 Mạng lưới thần kinh được tham số hóa theo các trọng số của nó.

Để kiểm soát một cái gì đó, trước tiên bạn cần có khả năng quan sát nó. Để kiểm soát đầu ra của mạng nơ-ron, bạn cần có khả năng đo lường đầu ra này khác với những gì bạn mong đợi bao xa. Đây là công việc của hàm mất mát của mạng, đôi khi còn được gọi là hàm mục tiêu hoặc hàm chi phí. Hàm mất mát lấy các dự đoán của mạng và mục tiêu thực sự (những gì bạn muốn mạng xuất ra) và tính điểm khoảng cách, ghi lại mức độ hoạt động của mạng trên ví dụ cụ thể này (xem hình 1.8).

Thủ thuật cơ bản trong deep learning là sử dụng điểm này làm tín hiệu phản hồi để điều chỉnh giá trị của các trọng số một chút, theo hướng làm giảm điểm mất mát cho ví dụ hiện tại (xem hình 1.9). Sự điều chỉnh này là công việc của trình tối ưu hóa, thực hiện thuật toán Backpropagation: thuật toán trung tâm trong học sâu. Chương tiếp theo sẽ giải thích chi tiết hơn về cách thức hoạt động của lan truyền ngược.

Ban đầu, các trọng số của mạng được gán các giá trị ngẫu nhiên, do đó mạng chỉ thực hiện một loạt các phép biến đổi ngẫu nhiên. Đương nhiên, sản lượng của nó khác xa so với mức lý tưởng và tỷ lệ tổn thất theo đó là rất cao. Nhưng với mỗi ví dụ mà mạng xử lý, các trọng số được điều chỉnh một chút theo đúng hướng và điểm mất mát sẽ giảm xuống. Đây là vòng lặp huấn luyện, lặp lại đủ Lớp (chuyển đổi dữ liệu)Đầu vào X Lớp trọng số (chuyển đổi dữ liệu) Dự đoán Y'True mục tiêu YWeights Chức năng mất Điểm mất Hình 1.8 Hàm mất đo lường chất lượng đầu ra của mạng.

10 chương 1 học sâu là gì?

Lớp (chuyển đổi dữ liệu)Đầu vào X Trọng số Lớp (chuyển đổi dữ liệu) Dự đoán Y'Cập nhật trọng lượng Mục tiêu đúng YTrọng lượng Chức năng mất Trình tối ưu hóa Điểm mất Hình 1.9 Điểm mất được sử dụng làm tín hiệu phản hồi để điều chỉnh trọng số.

số lần (thường là hàng chục lần vượt qua hàng nghìn ví dụ), mang lại các giá trị trọng số giúp giảm thiểu hàm mất mát. Mạng có tổn hao tối thiểu là mạng có đầu ra gần với mục tiêu nhất có thể: mạng đã được huấn luyện. Một lần nữa, đó là một cơ chế đơn giản, khi được thu nhỏ lại sẽ trông giống như một phép thuật.

1.7 Điều gì làm cho deep learning trở nên khác biệt Có điều gì đặc biệt về mạng lưới thần kinh sâu khiến chúng trở thành phương pháp tiếp cận “đúng đắn” để các công ty đầu tư và để các nhà nghiên cứu đổ xô vào không? Liệu chúng ta có còn sử dụng mạng lưới thần kinh sâu sau 20 năm nữa không?

Học sâu có một số đặc tính chứng minh vị thế của nó như một cuộc cách mạng AI và nó sẽ tiếp tục tồn tại. Chúng ta có thể không sử dụng mạng lưới thần kinh trong nhiều thập kỷ kể từ bây giờ, nhưng bất cứ điều gì chúng ta sử dụng sẽ kế thừa trực tiếp từ học sâu hiện đại và các khái niệm cốt lõi của nó.

Những thuộc tính quan trọng này có thể được chia thành ba loại:

¡ Đơn giản — Học sâu giúp việc giải quyết vấn đề dễ dàng hơn nhiều vì nó tự động kết hợp những gì từng là bước quan trọng nhất trong quy trình học máy:

kỹ thuật tính năng. Các kỹ thuật học máy trước đây—học nông—chỉ liên quan đến việc chuyển đổi dữ liệu đầu vào thành một hoặc hai không gian biểu diễn liên tiếp, không đủ biểu cảm cho hầu hết các vấn đề. Do đó, con người đã phải nỗ lực rất nhiều để làm cho dữ liệu đầu vào ban đầu dễ xử lý hơn bằng các phương pháp này: họ phải thiết kế các cách biểu diễn tốt cho dữ liệu của mình theo cách thủ công. Điều này được gọi là kỹ thuật tính năng. Mặt khác, học sâu hoàn toàn tự động hóa bước này: với học sâu, bạn học tất cả các tính năng trong một lần thay vì phải tự mình thiết kế chúng. Điều này đã đơn giản hóa đáng kể quy trình học máy, thường thay thế các đường ống nhiều tầng phức tạp bằng một mô hình học sâu duy nhất, đơn giản, từ đầu đến cuối.

¡ Khả năng mở rộng — Học sâu có khả năng song song hóa cao trên GPU hoặc phần cứng học máy chuyên dụng hơn, do đó, nó có thể tận dụng tối đa lợi thế của Moore

11 Học sâu đã đạt được những gì cho đến nay luật. Ngoài ra, các mô hình học sâu được đào tạo bằng cách lặp lại các lô dữ liệu nhỏ, cho phép chúng được đào tạo trên các tập dữ liệu có kích thước tùy ý. (Nút thắt cổ chai duy nhất là lượng sức mạnh tính toán song song sẵn có, theo định luật Moore, là một rào cản chuyển động nhanh.) ¡ Tính linh hoạt và khả năng sử dụng lại —Không giống như nhiều phương pháp học máy trước đây, các mô hình học sâu có thể được huấn luyện trên dữ liệu bổ sung mà không cần khởi động lại từ đầu, khiến chúng có khả năng học trực tuyến liên tục—một đặc tính quan trọng đối với các mô hình sản xuất rất lớn. Hơn nữa, các mô hình deep learning được đào tạo có thể tái sử dụng và do đó có thể tái sử dụng: đây là ý tưởng lớn đằng sau “mô hình nền tảng”—các mô hình lớn được đào tạo trên lượng dữ liệu khổng lồ, có thể được sử dụng cho nhiều nhiệm vụ mới mà không cần đào tạo lại hoặc thậm chí không cần đào tạo lại.

1.8 Thời đại của AI sáng tạo Có lẽ ví dụ nổi tiếng nhất về học sâu ngày nay là làn sóng ứng dụng AI sáng tạo gần đây—các trợ lý chatbot như ChatGPT, Gemini và Claude, cũng như các dịch vụ tạo hình ảnh như Midjourney. Những ứng dụng này đã thu hút trí tưởng tượng của công chúng nhờ khả năng tạo ra nội dung mang tính thông tin hoặc thậm chí sáng tạo để đáp ứng những lời nhắc đơn giản, làm mờ đi ranh giới giữa khả năng sáng tạo của con người và máy móc.

AI sáng tạo được hỗ trợ bởi các “mô hình nền tảng” rất lớn, học cách tái tạo lại nội dung văn bản và hình ảnh được đưa vào chúng—tái tạo lại một hình ảnh sắc nét từ một phiên bản ồn ào, dự đoán từ tiếp theo trong câu, v.v. Điều này có nghĩa là các mục tiêu trong hình 1.8 được lấy từ chính đầu vào. Điều này được gọi là học tự giám sát và nó cho phép các mô hình đó sử dụng lượng lớn dữ liệu chưa được gắn nhãn. Việc loại bỏ các chú thích dữ liệu thủ công từng gây tắc nghẽn cho các thương hiệu học máy trước đây đã mở ra một mức độ quy mô chưa từng thấy trước đây—một số mô hình nền tảng này có hàng trăm tỷ tham số và được đào tạo trên hơn 1 petabyte dữ liệu, với chi phí hàng chục triệu đô la.

Các mô hình nền tảng này hoạt động như một loại cơ sở dữ liệu mờ về kiến ​​thức của con người, khiến chúng có thể phù hợp với rất nhiều ứng dụng mà không cần lập trình hoặc đào tạo lại cho mục đích đặc biệt. Bởi vì họ đã ghi nhớ rất nhiều nên họ có thể giải quyết các vấn đề mới chỉ bằng cách nhắc nhở — truy vấn các biểu diễn kiến ​​thức mà họ đã học và trả về kết quả có nhiều khả năng liên quan đến lời nhắc của bạn nhất.

AI sáng tạo chỉ mới được phổ biến rộng rãi vào năm 2022, nhưng nó đã có lịch sử lâu đời— những thử nghiệm sớm nhất về việc tạo văn bản có từ những năm 1990. Ấn bản đầu tiên của cuốn sách này, được phát hành vào năm 2017, đã có một chương dày đặc có tựa đề “Generative AI” khám phá các kỹ thuật tạo văn bản và tạo hình ảnh vào thời điểm đó, đồng thời hứa hẹn một quan điểm kỳ quặc khi đó rằng, “sớm thôi”, phần lớn nội dung văn hóa mà chúng ta tiêu thụ sẽ được tạo ra với sự trợ giúp của AI.

1.9 Học sâu đã đạt được những gì cho đến nay Trong thập kỷ qua, học sâu đã đạt được một cuộc cách mạng công nghệ, bắt đầu với những kết quả đáng chú ý về các nhiệm vụ nhận thức từ năm 2013 đến năm 2017,

12 chương 1 Học sâu là gì?

sau đó đạt tiến bộ nhanh chóng trong các nhiệm vụ xử lý ngôn ngữ tự nhiên từ năm 2017 đến năm 2022 và đỉnh cao là làn sóng ứng dụng AI có khả năng biến đổi từ năm 2022 đến nay.

Học sâu đã tạo ra những đột phá lớn, tất cả đều giải quyết được những vấn đề cực kỳ khó khăn mà máy móc đã bỏ qua từ lâu:

¡ Các chatbot thông thạo và rất linh hoạt như ChatGPT và Gemini ¡ Trợ lý lập trình như GitHub Copilot ¡ Tạo hình ảnh thực tế ¡ Phân loại hình ảnh ở cấp độ con người ¡ Phiên âm giọng nói ở cấp độ con người ¡ Phiên âm chữ viết tay ở cấp độ con người và phiên âm văn bản in ¡ Dịch máy được cải thiện đáng kể ¡ Chuyển đổi văn bản thành giọng nói được cải thiện đáng kể ¡ Lái xe tự động ở cấp độ con người, đã được triển khai cho công chúng ở Phoenix, San Francisco, Los Angeles và Austin kể từ 2025 ¡ Hệ thống đề xuất được cải tiến, được YouTube, Netflix hoặc Spotify sử dụng ¡ Chơi cờ vây, cờ vua và poker siêu phàm Chúng tôi vẫn đang khám phá toàn bộ những gì học sâu có thể làm. Chúng tôi đã bắt đầu áp dụng nó rất thành công cho nhiều vấn đề được cho là không thể giải quyết chỉ cách đây vài năm—tự động sao chép hàng chục nghìn bản thảo cổ được lưu giữ trong Kho lưu trữ Bí mật Vatican, phát hiện và phân loại bệnh thực vật trên các cánh đồng bằng cách sử dụng điện thoại thông minh đơn giản, hỗ trợ các bác sĩ ung thư hoặc bác sĩ X quang diễn giải dữ liệu hình ảnh y tế, dự đoán các thảm họa thiên nhiên như lũ lụt, bão và thậm chí cả động đất. Với mỗi cột mốc quan trọng, chúng ta đang tiến gần hơn đến thời đại mà học sâu hỗ trợ chúng ta trong mọi hoạt động và mọi lĩnh vực nỗ lực của con người—khoa học, y học, sản xuất, năng lượng, giao thông vận tải, phát triển phần mềm, văn hóa nông nghiệp và thậm chí cả sáng tạo nghệ thuật.

1.10 Hãy cẩn thận với sự cường điệu ngắn hạn Chuỗi thành công dường như không thể ngăn cản này đã dẫn đến một làn sóng cường điệu mãnh liệt, một số trong đó có phần có căn cứ, nhưng hầu hết chỉ là những câu chuyện cổ tích hư cấu. Vào đầu năm 2023, ngay sau khi OpenAI phát hành GPT-4, nhiều chuyên gia đã tuyên bố rằng “không ai cần phải làm việc nữa” và tình trạng thất nghiệp hàng loạt sẽ xảy ra trong vòng một năm hoặc năng suất kinh tế sẽ sớm tăng từ 10 × đến 100 ×. Tất nhiên, hai năm sau, không điều nào trong số này thành hiện thực—tỷ lệ thất nghiệp ở Mỹ vẫn ở mức thấp, trong khi các chỉ số năng suất còn lâu mới đạt được mức bùng nổ như đã hứa. Đừng hiểu lầm: tác động của AI—đặc biệt là AI tạo sinh—đã rất đáng kể và nó đang phát triển nhanh chóng đáng kể. Tính đến giữa năm 2025, AI tạo ra đã tạo ra doanh thu hàng chục tỷ đô la mỗi năm, một con số cực kỳ ấn tượng đối với một ngành công nghiệp chưa từng tồn tại ba năm trước đó! Nhưng nó vẫn chưa tạo được nhiều ảnh hưởng trong

13 Hãy cẩn thận với sự cường điệu ngắn hạn của nền kinh tế tổng thể và những lời hứa hẹn hoàn toàn không thể kiểm soát mà chúng tôi đã tràn ngập khi mới bắt đầu.

Trong khi các cuộc thảo luận về tình trạng thất nghiệp và mức tăng năng suất gấp 100 lần nhờ AI đã gây ra lo lắng, thì sự cường điệu về AI thậm chí còn có một khía cạnh giật gân hơn. Bên này tuyên bố sắp xuất hiện trí thông minh tổng hợp cấp độ con người (AGI), hay thậm chí là “siêu trí tuệ” vượt xa khả năng của con người. Những tuyên bố này đang làm dấy lên những lo ngại ngoài sự gián đoạn kinh tế—bản thân loài người có thể có nguy cơ bị thay thế bởi những sáng tạo kỹ thuật số của chúng ta.

Những người mới tham gia lĩnh vực này có thể dễ dàng cho rằng chính thành công thực tế của AI tạo ra đã tạo nên niềm tin vào AGI trong thời gian ngắn, nhưng thực tế đó là điều ngược lại. Những tuyên bố về AGI ngắn hạn được đưa ra đầu tiên và chúng góp phần đáng kể vào sự phát triển của AI thế hệ. Ngay từ năm 2013, giới tinh hoa công nghệ đã lo ngại rằng AGI có thể xuất hiện trong vòng vài năm tới. Vào thời điểm đó, ý tưởng là DeepMind, một công ty khởi nghiệp nghiên cứu AI ở London được Google mua lại, đang trên đà đạt được điều đó. Niềm tin này là động lực đằng sau việc thành lập OpenAI vào năm 2015, ban đầu nhằm mục đích trở thành đối trọng nguồn mở với DeepMind. OpenAI đóng một vai trò quan trọng trong việc khởi đầu AI thế hệ, do đó, trong một bước ngoặt đặc biệt, chính niềm tin vào AGI trong thời gian ngắn đã thúc đẩy sự phát triển của AI thế hệ chứ không phải ngược lại. Năm 2016, mục tiêu tuyển dụng của OpenAI là nó sẽ đạt được AGI vào năm 2020! Tuy nhiên, công bằng mà nói, hồi đó chỉ có một số ít người trong ngành công nghệ tin vào mốc thời gian lạc quan như vậy. Tuy nhiên, đến đầu năm 2023, một bộ phận đáng kể các kỹ sư ở Khu vực Vịnh San Francisco dường như bị thuyết phục rằng AGI sẽ ra mắt trong vài năm tới.

Điều quan trọng là phải tiếp cận những tuyên bố như vậy với một thái độ hoài nghi lành mạnh. Bất chấp tên gọi của nó, “trí tuệ nhân tạo” ngày nay được mô tả chính xác hơn là “tự động hóa nhận thức” --- mã hóa và vận hành các kỹ năng và kiến ​​thức của con người. AI vượt trội trong việc giải quyết các vấn đề với các yêu cầu được xác định hẹp hoặc những vấn đề có sẵn nhiều ví dụ chính xác. Đó là việc nâng cao khả năng của máy tính chứ không phải việc tái tạo trí tuệ con người.

Nói rõ hơn, tự động hóa nhận thức cực kỳ hữu ích. Nhưng trí thông minh – khả năng tự chủ về nhận thức – lại là một sinh vật hoàn toàn khác. Hãy nghĩ theo cách này: AI giống như một nhân vật hoạt hình, trong khi trí thông minh giống như một sinh vật sống. Một bộ phim hoạt hình, dù thực tế đến đâu, cũng chỉ có thể diễn những cảnh mà nó được vẽ ra. Mặt khác, một sinh vật sống có thể thích nghi với những điều bất ngờ.

“Nếu phim hoạt hình được vẽ đủ chân thực và bao gồm đủ nhiều cảnh thì có gì khác biệt?” bạn có thể hỏi. Nếu một mô hình ngôn ngữ lớn có thể đưa ra một câu trả lời đủ giống con người khi được hỏi một câu hỏi, liệu nó có sở hữu khả năng tự chủ nhận thức thực sự hay không? Sự khác biệt chính là khả năng thích ứng. Trí thông minh là khả năng đối mặt với những điều chưa biết, thích ứng với nó và học hỏi từ nó. Tự động hóa, ngay cả ở mức tốt nhất, cũng chỉ có thể xử lý các tình huống mà nó đã được đào tạo hoặc lập trình. Đó là lý do tại sao việc tạo ra quy trình tự động hóa mạnh mẽ lại rất khó khăn—nó đòi hỏi phải tính toán mọi tình huống có thể xảy ra.

Vì vậy, đừng lo lắng về việc AI đột nhiên có khả năng tự nhận thức và tiếp quản nhân loại. Công nghệ ngày nay đơn giản là không đi theo hướng đó. Ngay cả với ý nghĩa đáng kể

14 chương 1 Học sâu là gì?

tiến bộ, AI sẽ vẫn là một công cụ tinh vi, không phải là một sinh vật có tri giác. Nó giống như việc mong đợi một chiếc đồng hồ tốt hơn sẽ dẫn đến việc du hành thời gian - chúng hoàn toàn là những thứ khác nhau.

1.11 Mùa hè có thể chuyển sang mùa đông Nguy cơ của những kỳ vọng ngắn hạn bị thổi phồng là khi công nghệ không thể tránh khỏi tình trạng thiếu hụt, đầu tư nghiên cứu có thể cạn kiệt, làm chậm tiến độ trong một thời gian dài. Điều này đã xảy ra trước đây. Hai lần trong quá khứ, AI đã trải qua một chu kỳ lạc quan mãnh liệt, sau đó là sự thất vọng và hoài nghi, dẫn đến tình trạng thiếu vốn. Nó bắt đầu với AI mang tính biểu tượng vào những năm 1960. Trong những ngày đầu đó, những dự đoán về AI rất cao. Một trong những người tiên phong và ủng hộ nổi tiếng nhất của cách tiếp cận AI mang tính biểu tượng là Marvin Minsky, người đã tuyên bố vào năm 1967, “Trong vòng một thế hệ... vấn đề tạo ra ‘trí tuệ nhân tạo’ về cơ bản sẽ được giải quyết.” Ba năm sau, vào năm 1970, ông đưa ra một dự đoán định lượng chính xác hơn: “Trong vòng ba đến tám năm nữa, chúng ta sẽ có một cỗ máy có trí thông minh chung của một con người bình thường”. Vào năm 2025, thành tựu như vậy dường như vẫn còn rất xa trong tương lai—vì chúng ta không có cách nào dự đoán được sẽ mất bao lâu—nhưng vào những năm 1960 và đầu những năm 1970, một số chuyên gia tin rằng nó sắp đến gần (nhiều người ngày nay cũng vậy). Vài năm sau, khi những kỳ vọng cao này không thành hiện thực, các nhà nghiên cứu và quỹ chính phủ đã quay lưng lại với lĩnh vực này, đánh dấu sự khởi đầu của mùa đông AI đầu tiên (ám chỉ mùa đông hạt nhân, vì thời điểm này diễn ra ngay sau đỉnh cao của Chiến tranh Lạnh).

Nó sẽ không phải là cái cuối cùng. Vào những năm 1980, một xu hướng mới về AI mang tính biểu tượng, các hệ thống chuyên gia, bắt đầu thu hút sự chú ý của các công ty lớn. Một số câu chuyện thành công ban đầu đã kích hoạt một làn sóng đầu tư, với việc các tập đoàn trên khắp thế giới thành lập bộ phận AI nội bộ của riêng họ để phát triển hệ thống chuyên gia. Khoảng năm 1985, các công ty chi hơn 1 tỷ USD mỗi năm cho công nghệ này; nhưng đến đầu những năm 1990, những hệ thống này đã tỏ ra tốn kém để bảo trì, khó mở rộng quy mô, phạm vi hạn chế và sự quan tâm cũng giảm dần.

Thế là bắt đầu mùa đông AI thứ hai. Hiện tại, chúng ta có thể đang chứng kiến ​​chu kỳ thứ ba của sự cường điệu và thất vọng về AI—và chúng ta vẫn đang trong giai đoạn lạc quan mãnh liệt.

Quan điểm hiện tại của tôi là chúng ta khó có thể chứng kiến ​​sự rút lui hoàn toàn khỏi nghiên cứu AI như chúng ta đã thấy vào những năm 1990. Nếu có mùa đông thì trời sẽ rất ôn hòa. AI đã chứng minh được giá trị thay đổi thế giới của mình. Tuy nhiên, có vẻ như không thể tránh khỏi việc phải thoát ra khỏi bong bóng AI 2023–2025. Hiện tại, đầu tư vào AI, chủ yếu vào các trung tâm dữ liệu và GPU, vượt qua 200 tỷ USD hàng năm, trong khi việc tạo ra doanh thu lại chậm lại đáng kể, gần 30 tỷ USD. AI hiện đang được các nhà điều hành và nhà đầu tư đánh giá không phải bởi những gì nó đã đạt được mà bởi những gì chúng ta được biết rằng nó có thể sớm làm được—phần lớn trong số đó sẽ vĩnh viễn nằm ngoài tầm với của các công nghệ hiện có. Một cái gì đó sẽ phải cho đi. Nhưng điều gì sẽ xảy ra chính xác khi bong bóng AI xì hơi vẫn còn chưa rõ ràng.

1.12 Lời hứa của AI Mặc dù chúng ta có thể có những kỳ vọng ngắn hạn không thực tế đối với AI, nhưng bức tranh dài hạn có vẻ tươi sáng. Chúng tôi chỉ mới bắt đầu áp dụng deep learning cho nhiều vấn đề quan trọng mà nó có thể mang lại sự biến đổi, từ chẩn đoán y tế đến trợ lý kỹ thuật số.

15 Lời hứa của AI Năm 2017, trong chính cuốn sách này, tôi đã viết:

Hiện tại, có vẻ khó tin rằng AI có thể có tác động lớn đến thế giới của chúng ta vì nó chưa được triển khai rộng rãi - giống như hồi năm 1995, thật khó để tin vào tác động trong tương lai của Internet. Hồi đó, hầu hết mọi người đều không hiểu Internet có liên quan đến họ như thế nào và nó sẽ thay đổi cuộc sống của họ như thế nào. Điều này cũng đúng với deep learning và AI ngày nay. Nhưng đừng nhầm lẫn: AI đang đến. Trong một tương lai không xa, AI sẽ là trợ lý, thậm chí là bạn của bạn; nó sẽ trả lời các câu hỏi của bạn, giúp giáo dục con bạn và theo dõi sức khỏe của bạn. Nó sẽ giao hàng tạp hóa đến tận nhà và đưa bạn từ điểm A đến điểm B. Nó sẽ là cầu nối của bạn với một thế giới ngày càng phức tạp và chứa nhiều thông tin. Và quan trọng hơn nữa, AI sẽ giúp nhân loại nói chung tiến lên bằng cách hỗ trợ các nhà khoa học con người thực hiện những khám phá mang tính đột phá mới trên tất cả các lĩnh vực khoa học, từ gen đến toán học.

Chuyển nhanh đến năm 2025, hầu hết những điều này đã trở thành hiện thực hoặc sắp trở thành hiện thực—và đây mới chỉ là khởi đầu:

¡ Hàng chục triệu người đang sử dụng các chatbot AI như ChatGPT, Gemini và Claude làm trợ lý hàng ngày. Trên thực tế, việc trả lời câu hỏi và “giáo dục con bạn” (hỗ trợ bài tập về nhà) hóa ra lại là những ứng dụng hàng đầu của những chatbot này! Đối với nhiều người, AI đã là giao diện tiếp cận thông tin của thế giới.

¡ Hàng trăm nghìn người tương tác với “những người bạn” AI trong các ứng dụng như Character.ai.

¡ Lái xe tự động hoàn toàn đã được triển khai trên quy mô lớn ở các thành phố như Phoenix, San Francisco, Los Angeles và Austin.

¡ AI đang có những bước tiến lớn trong việc giúp thúc đẩy khoa học. Mô hình AlphaFold của DeepMind đang giúp các nhà sinh học dự đoán cấu trúc protein với độ chính xác chưa từng có. Nhà toán học nổi tiếng Terence Tao tin rằng vào khoảng năm 2026, AI có thể trở thành đồng tác giả đáng tin cậy trong nghiên cứu toán học và các lĩnh vực khác khi được sử dụng phù hợp.

Cuộc cách mạng AI, từng là một tầm nhìn xa vời, giờ đây đang nhanh chóng mở ra trước mắt chúng ta. Trên đường đi, chúng ta có thể phải đối mặt với một số trở ngại – giống như cách ngành công nghiệp Internet đã bị thổi phồng quá mức vào những năm 1998–1999 và phải chịu một cuộc khủng hoảng khiến nguồn đầu tư cạn kiệt trong suốt đầu những năm 2000. Nhưng cuối cùng chúng ta sẽ đến đó. AI cuối cùng sẽ được áp dụng cho hầu hết mọi quá trình tạo nên xã hội và cuộc sống hàng ngày của chúng ta, giống như mạng Internet ngày nay.

Đừng tin vào sự cường điệu ngắn hạn mà hãy tin vào tầm nhìn dài hạn. Có thể phải mất một thời gian để AI phát huy hết tiềm năng thực sự của nó—một tiềm năng ở mức độ tối đa mà chưa ai dám mơ tới—nhưng AI đang xuất hiện và nó sẽ biến đổi thế giới của chúng ta theo một cách tuyệt vời.

