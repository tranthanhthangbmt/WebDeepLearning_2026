# Chapter 2: The mathematical building blocks of neural networks

*(Bản dịch tự động từ PDF sử dụng công nghệ nhận diện Code Blocks mới)*

16


2


Toán học


khối xây dựng của


mạng lưới thần kinh


Chương này bao gồm


- Ví dụ đầu tiên về mạng nơ-ron


- Tensor và các phép toán tensor


- Mạng lưới thần kinh học như thế nào thông qua lan truyền ngược


và giảm độ dốc


Để hiểu được deep learning đòi hỏi phải làm quen với nhiều khái niệm toán học đơn giản: tensor, tensor, vi phân, giảm gradient, v.v. Mục tiêu của chúng tôi trong chương này là xây dựng trực giác của bạn về những khái niệm này mà không cần quá chú trọng đến kỹ thuật. Đặc biệt, chúng tôi sẽ tránh xa các ký hiệu toán học, vốn có thể tạo ra những rào cản không cần thiết đối với những người không có nền tảng toán học và không cần thiết để giải thích rõ ràng mọi thứ. Mô tả chính xác, rõ ràng nhất của một phép toán là mã thực thi của nó.


Để cung cấp đủ bối cảnh cho việc giới thiệu các tensor và độ dốc giảm dần, chúng ta sẽ bắt đầu chương này với một ví dụ thực tế về mạng nơ-ron. Sau đó, chúng ta sẽ xem xét từng khái niệm mới được giới thiệu. Hãy nhớ rằng những khái niệm này sẽ rất cần thiết để bạn hiểu được các ví dụ thực tế sẽ có trong các chương sau!


17 Cái nhìn đầu tiên về mạng nơ-ron


Sau khi đọc chương này, bạn sẽ có hiểu biết trực quan về lý thuyết toán học đằng sau học sâu và bạn sẽ sẵn sàng bắt đầu đi sâu vào các khuôn khổ học sâu hiện đại, trong chương 3.


Chạy mã trong cuốn sách này


Cuốn sách này chứa đầy mã Python có thể chạy được. Mỗi chương được ghép nối với một sổ ghi chép Jupyter chứa tất cả mã từ chương đó. Sổ ghi chép Jupyter là một loại sổ ghi chép Python trực tiếp, nơi bạn có thể chạy mã, dữ liệu biểu đồ, xem hình ảnh một cách tương tác, v.v. Bạn sẽ thu được nhiều kiến ​​thức thực tế hơn từ cuốn sách này nếu bạn chạy và thử nghiệm mã khi đọc.


Cho đến nay, cách dễ nhất để thiết lập môi trường deep learning để chạy các sổ ghi chép này là Google Colaboratory (hay gọi tắt là Colab), một môi trường lưu trữ dành cho sổ ghi chép Jupyter đã trở thành tiêu chuẩn ngành cho những người thực hành ML. Với Colab, bạn có thể chạy mã cho cuốn sách này một cách tương tác trong trình duyệt, kết nối với thời gian chạy trên đám mây bằng phần cứng có thể định cấu hình. Theo mặc định, sổ ghi chép trong cuốn sách này sẽ chạy trên thời gian chạy GPU miễn phí của Colab.


Nếu muốn, bạn cũng có thể chạy các sổ ghi chép này cục bộ trên máy của chính mình. Nên sử dụng GPU, đặc biệt khi bạn tiếp cận các mô hình lớn hơn và có tính toán chuyên sâu hơn ở phần sau của cuốn sách này.


Bạn có thể tìm thấy hướng dẫn chạy cục bộ và trên Colab, cùng với mã tại https://github.com/fchollet/deep-learning-with-python-notebooks.


2.1 Cái nhìn đầu tiên về mạng nơ-ron


Hãy xem một ví dụ cụ thể về mạng nơ-ron sử dụng thư viện máy học Keras để học cách phân loại các chữ số viết tay. Chúng tôi sẽ sử dụng Keras rộng rãi trong suốt cuốn sách này. Đó là một thư viện cấp cao, đơn giản cho phép chúng tôi tập trung vào các khái niệm mà chúng tôi muốn đề cập.


Trừ khi bạn đã có kinh nghiệm với Keras hoặc các thư viện tương tự, bạn sẽ không hiểu mọi thứ về ví dụ đầu tiên này ngay lập tức. Điều đó ổn thôi. Trong một số phần, chúng ta sẽ xem xét từng phần tử trong ví dụ và giải thích chi tiết. Vì vậy, đừng lo lắng nếu một số bước có vẻ tùy tiện hoặc giống như phép thuật đối với bạn! Chúng ta phải bắt đầu từ đâu đó.


Vấn đề chúng tôi đang cố gắng giải quyết ở đây là phân loại hình ảnh thang độ xám của các chữ số viết tay (28 × 28 pixel) thành 10 loại (0 đến 9). Chúng ta sẽ sử dụng bộ dữ liệu MNIST, một bộ dữ liệu cổ điển trong cộng đồng học máy, đã tồn tại gần như lâu đời trong lĩnh vực này và đã được nghiên cứu chuyên sâu. Đó là một bộ gồm 60.000 hình ảnh huấn luyện, cộng với 10.000 hình ảnh thử nghiệm, được tập hợp bởi Viện Tiêu chuẩn và Công nghệ Quốc gia (NIST ở MNIST) vào những năm 1980. Bạn có thể coi việc “giải quyết” MNIST giống như “Xin chào thế giới” của học sâu—đó là những gì bạn làm để xác minh rằng thuật toán của bạn đang hoạt động như mong đợi. Khi trở thành người thực hành học máy, bạn sẽ thấy MNIST xuất hiện nhiều lần trong các bài báo khoa học, bài đăng trên blog, v.v. Bạn có thể xem một số mẫu MNIST trong hình 2.1.


18 Chương 2  Các khối xây dựng toán học của mạng lưới thần kinh


LƯU Ý  Trong học máy, một danh mục trong bài toán phân loại được gọi là lớp.  Điểm dữ liệu được gọi là mẫu. Lớp liên kết với một mẫu cụ thể được gọi là nhãn.


Bộ dữ liệu MNIST được tải sẵn trong Keras, dưới dạng một bộ bốn mảng NumPy.


Liệt kê 2.1  Đang tải tập dữ liệu MNIST trong Keras


```python
from keras.datasets import mnist
```

```python
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
```

train_images và train_labels tạo thành tập huấn luyện, dữ liệu mà mô hình sẽ học từ đó. Sau đó, mô hình sẽ được kiểm tra trên tập kiểm tra, test_images và test_labels. Hình ảnh được mã hóa dưới dạng mảng NumPy và nhãn là một mảng chữ số, nằm trong khoảng từ 0 đến 9. Hình ảnh và nhãn có sự tương ứng một-một.


LƯU Ý  NumPy là thư viện Python rất phổ biến để tính toán số.  Bạn sẽ thấy nó xuất hiện thường xuyên trong hành trình học máy của mình. Nó hiếm khi được sử dụng để triển khai các thuật toán học máy hiện đại do thiếu hỗ trợ GPU và khả năng tự phân biệt, nhưng mảng NumPy thường được sử dụng làm định dạng trao đổi dữ liệu số—như ở đây, cho các chữ số MNIST và nhãn của chúng.


Hãy xem dữ liệu đào tạo:


```python
>>> train_images.shape
(60000, 28, 28)
>>> len(train_labels)
60000
>>> train_labels
array([5, 0, 4, ..., 5, 6, 8], dtype=uint8)
```

Và đây là dữ liệu thử nghiệm:


```python
>>> test_images.shape
(10000, 28, 28)
>>> len(test_labels)
10000
>>> test_labels
array([7, 2, 1, ..., 4, 5, 6], dtype=uint8)
```

Quy trình làm việc sẽ như sau. Đầu tiên, chúng ta sẽ cung cấp dữ liệu huấn luyện cho mạng lưới thần kinh,


train_images và train_labels. Mạng sau đó sẽ học cách liên kết hình ảnh và


Hình 2.1  chữ số mẫu MNIST


19 Cái nhìn đầu tiên về mạng nơ-ron


nhãn. Cuối cùng, chúng tôi sẽ yêu cầu mạng tạo dự đoán cho test_images và chúng tôi sẽ xác minh xem những dự đoán này có khớp với nhãn từ test_labels hay không.


Hãy xây dựng mạng lưới—một lần nữa, hãy nhớ rằng bạn chưa cần phải hiểu mọi thứ về ví dụ này.


Liệt kê 2.2  Kiến trúc mạng


```python
import keras
from keras import layers
```

```python
model = keras.Sequential(
    [
        layers.Dense(512, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ]
)
```

```python
The core building block of neural networks is the layer. You can think of a layer as a 
filter for data: some data goes in, and it comes out in a more useful form. Specifically, 
layers extract representations out of the data fed into them—hopefully, representations 
that are more meaningful for the problem at hand. Most of deep learning consists of 
chaining together simple layers that will implement a form of progressive data distilla-
tion. A deep learning model is like a sieve for data processing, made of a succession of 
increasingly refined data filters—the layers.
```

```python
Here, our model consists of a sequence of two Dense layers, which are densely con-
nected (also called fully connected) neural layers. The second (and last) layer is a 10-way
```

```python
softmax classification layer, which means it will return an array of 10 probability scores 
(summing to 1). Each score will be the probability that the current digit image belongs 
to one of our 10 digit classes.
```

Để làm cho mô hình sẵn sàng cho việc đào tạo, chúng ta cần chọn thêm ba thứ nữa, như một phần của bước biên dịch:


- Hàm mất mát—Làm thế nào mô hình có thể đo lường hiệu suất của nó trên


dữ liệu huấn luyện và do đó làm thế nào nó có thể tự điều khiển đi đúng hướng.


- Trình tối ưu hóa—Cơ chế mà qua đó mô hình sẽ tự cập nhật dựa trên


trên dữ liệu huấn luyện mà nó nhìn thấy để cải thiện hiệu suất của nó.


- Các số liệu cần theo dõi trong quá trình đào tạo và kiểm tra—Ở đây, chúng tôi chỉ quan tâm đến độ chính xác


(tỷ lệ hình ảnh được phân loại chính xác).


Mục đích chính xác của hàm mất mát và trình tối ưu hóa sẽ được làm rõ trong hai chương tiếp theo.


Liệt kê 2.3  Bước biên dịch


```python
model.compile(
    optimizer="adam",
```

20 Chương 2  Các khối xây dựng toán học của mạng lưới thần kinh


loss="sparse_categorical_crossentropy", số liệu=["độ chính xác"], )


Trước khi đào tạo, chúng tôi sẽ xử lý trước dữ liệu bằng cách định hình lại dữ liệu thành hình dạng mà mô hình mong đợi và chia tỷ lệ sao cho tất cả các giá trị đều nằm trong khoảng [0, 1]. Trước đây, hình ảnh huấn luyện của chúng ta được lưu trữ trong một mảng có hình dạng (60000, 28, 28) thuộc loại uint8 với các giá trị trong khoảng [0, 255]. Chúng tôi chuyển đổi nó thành một mảng hình float32 (60000, 28 * 28) với các giá trị từ 0 đến 1.


Liệt kê 2.4  Chuẩn bị dữ liệu hình ảnh


```python
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype("float32") / 255
```

Bây giờ chúng ta đã sẵn sàng huấn luyện mô hình, việc này trong Keras được thực hiện thông qua lệnh gọi tới mô hình


fit()—chúng tôi điều chỉnh mô hình phù hợp với dữ liệu huấn luyện của nó.


Liệt kê 2.5  “Lắp” mô hình


```python
model.fit(train_images, train_labels, epochs=5, batch_size=128)
```

Hai đại lượng được hiển thị trong quá trình huấn luyện: sự mất mát của mô hình trên dữ liệu huấn luyện và độ chính xác của mô hình trên dữ liệu huấn luyện. Chúng tôi nhanh chóng đạt được độ chính xác 0,989 (98,9%) trên dữ liệu huấn luyện.


Bây giờ chúng ta đã có một mô hình được huấn luyện, chúng ta có thể sử dụng nó để dự đoán xác suất của lớp cho các chữ số mới—các hình ảnh không thuộc dữ liệu huấn luyện, giống như các hình ảnh trong tập kiểm tra.


Liệt kê 2.6  Sử dụng mô hình để đưa ra dự đoán


```python
>>> test_digits = test_images[0:10]
>>> predictions = model.predict(test_digits)
>>> predictions[0]
array([1.0726176e-10, 1.6918376e-10, 6.1314843e-08, 8.4106023e-06,
       2.9967067e-11, 3.0331331e-09, 8.3651971e-14, 9.9999106e-01,
       2.6657624e-08, 3.8127661e-07], dtype=float32)
```

Mỗi số chỉ số i trong mảng đó tương ứng với xác suất ảnh chữ số đó


test_digits[0] thuộc lớp i.


Chữ số kiểm tra đầu tiên này có điểm xác suất cao nhất (0,99999106, gần như 1) ở chỉ số 7, vì vậy theo mô hình của chúng tôi, nó phải là 7:


21 Biểu diễn dữ liệu cho mạng nơ-ron


```python
>>> predictions[0].argmax()
7
>>> predictions[0][7]
0.99999106
```

Chúng tôi có thể kiểm tra xem nhãn thử nghiệm có đồng ý không:


```python
>>> test_labels[0]
7
```

Trung bình, mô hình của chúng tôi phân loại các chữ số chưa từng thấy như vậy tốt đến mức nào? Hãy kiểm tra bằng cách tính độ chính xác trung bình trên toàn bộ tập kiểm tra.


Liệt kê 2.7  Đánh giá mô hình trên dữ liệu mới


```python
>>> test_loss, test_acc = model.evaluate(test_images, test_labels)
>>> print(f"test_acc: {test_acc}")
test_acc: 0.9785
```

Độ chính xác của tập kiểm tra hóa ra là 97,8%—gần gấp đôi tỷ lệ lỗi của tập huấn luyện (với độ chính xác 98,9%). Khoảng cách giữa độ chính xác của quá trình đào tạo và độ chính xác của bài kiểm tra là một ví dụ về việc trang bị quá mức: thực tế là các mô hình học máy có xu hướng hoạt động kém hơn trên dữ liệu mới so với dữ liệu đào tạo của chúng. Overfitting là chủ đề trọng tâm trong chương 5.


Điều này kết thúc ví dụ đầu tiên của chúng tôi. Bạn vừa thấy cách có thể xây dựng và huấn luyện mạng nơ-ron để phân loại các chữ số viết tay trong chưa đầy 15 dòng mã Python. Trong chương này và chương tiếp theo, chúng ta sẽ đi vào chi tiết về từng phần chuyển động mà chúng ta vừa xem trước và làm rõ những gì đang diễn ra ở hậu trường. Bạn sẽ tìm hiểu về tensor, các đối tượng lưu trữ dữ liệu đi vào mô hình; các phép toán tensor, các lớp được tạo thành từ đâu; và giảm độ dốc, cho phép mô hình của bạn học hỏi từ các ví dụ huấn luyện của nó.


2.2 Biểu diễn dữ liệu cho mạng nơ-ron


Trong ví dụ trước, chúng ta bắt đầu từ dữ liệu được lưu trữ trong mảng NumPy đa chiều, còn được gọi là tensor. Nhìn chung, tất cả các hệ thống máy học hiện tại đều sử dụng tensor làm cấu trúc dữ liệu cơ bản. Tensor là nền tảng của lĩnh vực này—cơ bản đến mức khung TensorFlow được đặt theo tên của chúng. Vậy tensor là gì?


Về cốt lõi, tensor là nơi chứa dữ liệu—thường là dữ liệu số. Vì vậy, nó là một nơi chứa các con số. Có thể bạn đã quen thuộc với ma trận, là các tensor hạng 2: tensor là sự tổng quát hóa của ma trận thành một số chiều tùy ý (lưu ý rằng trong ngữ cảnh của tensor, một chiều thường được gọi là trục).


Lúc đầu, việc xem xét chi tiết các tensor có vẻ hơi trừu tượng. Nhưng điều đó rất đáng giá—việc thao tác các tensor sẽ là cốt lõi của bất kỳ mã học máy nào bạn từng viết.


22 Chương 2  Các khối xây dựng toán học của mạng lưới thần kinh


2.2.1 Vô hướng (tensor hạng 0)


Một tensor chỉ chứa một số được gọi là tensor vô hướng (hoặc tensor vô hướng, tensor hạng 0, hoặc tensor 0D). Trong NumPy, số float32 hoặc float64 là một tensor vô hướng (hoặc mảng vô hướng). Bạn có thể hiển thị số trục của tensor NumPy thông qua thuộc tính ndim;  một tensor vô hướng có 0 trục (ndim == 0). Số trục của một tenxơ còn được gọi là hạng của nó. Đây là một vô hướng NumPy:


```python
>>> import numpy as np
>>> x = np.array(12)
>>> x
array(12)
>>> x.ndim
0
```

2.2.2 Vector (tensor hạng 1)


Một mảng số được gọi là vectơ (hoặc tensor hạng 1 hoặc tensor 1D). Tenxơ hạng 1 có đúng một trục. Sau đây là một vectơ NumPy:


```python
>>> x = np.array([12, 3, 6, 14, 7])
>>> x
array([12, 3, 6, 14, 7])
>>> x.ndim
1
```

Vectơ này có năm mục và được gọi là vectơ 5 chiều. Đừng nhầm lẫn vectơ 5D với tensor 5D! Vectơ 5D chỉ có một trục và có năm chiều dọc theo trục của nó, trong khi một tensor 5D có năm trục (và có thể có bất kỳ số chiều nào dọc theo mỗi trục). Thứ nguyên có thể biểu thị số lượng mục dọc theo một trục cụ thể (như trong trường hợp vectơ 5D của chúng ta) hoặc số trục trong một tenxơ (chẳng hạn như tenxơ 5D), đôi khi có thể gây nhầm lẫn. Trong trường hợp thứ hai, về mặt kỹ thuật sẽ đúng hơn khi nói về một tensor cấp 5 (cấp của tensor là số trục), nhưng ký hiệu mơ hồ 5D tensor vẫn phổ biến.


2.2.3 Ma trận (tensor bậc 2)


Mảng các vectơ là một ma trận (hoặc tensor hạng 2 hoặc tensor 2D). Một ma trận có hai trục (thường được gọi là hàng và cột). Bạn có thể diễn giải ma trận một cách trực quan dưới dạng một lưới số hình chữ nhật. Đây là ma trận NumPy:


```python
>>> x = np.array([[5, 78, 2, 34, 0],
...               [6, 79, 3, 35, 1],
...               [7, 80, 4, 36, 2]])
```

23 Biểu diễn dữ liệu cho mạng nơ-ron


```python
>>> x.ndim
2
```

Các mục từ trục đầu tiên được gọi là hàng và các mục từ trục thứ hai được gọi là cột. Trong ví dụ trước, [5, 78, 2, 34, 0] là hàng đầu tiên của x và [5, 6, 7] là cột đầu tiên.


2.2.4 Tenxơ hạng 3 và tensor hạng cao hơn


Nếu bạn đóng gói các ma trận như vậy vào một mảng mới, bạn sẽ thu được một tensor cấp 3 (hoặc tensor 3D), mà bạn có thể diễn giải một cách trực quan dưới dạng một khối số. Sau đây là tenxơ cấp 3 NumPy:


```python
>>> x = np.array([[[5, 78, 2, 34, 0],
...                [6, 79, 3, 35, 1],
...                [7, 80, 4, 36, 2]],
...               [[5, 78, 2, 34, 0],
...                [6, 79, 3, 35, 1],
...                [7, 80, 4, 36, 2]],
...               [[5, 78, 2, 34, 0],
...                [6, 79, 3, 35, 1],
...                [7, 80, 4, 36, 2]]])
>>> x.ndim
3
```

Bằng cách đóng gói các tensor hạng 3 trong một mảng, bạn có thể tạo ra một tensor hạng 4, v.v. Trong học sâu, bạn thường thao tác các tensor với cấp độ từ 0 đến 4, mặc dù bạn có thể tăng lên cấp độ 5 nếu xử lý dữ liệu video.


2.2.5 Thuộc tính chính


Một tensor được xác định bởi ba thuộc tính chính:


- Số trục (cấp)—Ví dụ: một tenxơ cấp 3 có ba trục và một ma trận


có hai trục. Đây còn được gọi là ndim của tensor trong các thư viện Python như NumPy, JAX, TensorFlow và PyTorch.


- Hình dạng—Đây là một bộ số nguyên mô tả tensor có bao nhiêu chiều


có dọc theo mỗi trục. Chẳng hạn, ví dụ ma trận trước có hình dạng (3, 5) và ví dụ tenxơ cấp 3 có hình dạng (3, 3, 5). Một vectơ có hình dạng với một phần tử duy nhất, chẳng hạn như (5,), trong khi một đại lượng vô hướng có hình dạng trống, ().


- Kiểu dữ liệu (thường được gọi là dtype trong thư viện Python)—Đây là kiểu dữ liệu


chứa trong tensor; Ví dụ: kiểu của tensor có thể là float16, float32,


float64, uint8, bool, v.v. Trong TensorFlow, bạn cũng có thể gặp


tensor dây.


Để làm cho điều này cụ thể hơn, hãy xem lại dữ liệu chúng tôi đã xử lý trong ví dụ MNIST. Đầu tiên, chúng tôi tải tập dữ liệu MNIST:


24 Chương 2  Các khối xây dựng toán học của mạng lưới thần kinh


```python
from keras.datasets import mnist
```

```python
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
```

Tiếp theo, chúng ta hiển thị số trục của tensor train_images, thuộc tính ndim:


```python
>>> train_images.ndim
3
```

Đây là hình dạng của nó:


```python
>>> train_images.shape
(60000, 28, 28)
```

Và đây là kiểu dữ liệu của nó, thuộc tính dtype:


```python
>>> train_images.dtype
uint8
```

Vì vậy, những gì chúng ta có ở đây là một tenxơ hạng 3 gồm các số nguyên 8 bit. Chính xác hơn, đó là một mảng gồm 60.000 ma trận số nguyên 28 × 28. Mỗi ma trận như vậy là một ảnh thang độ xám, có hệ số từ 0 đến 255.


Hãy hiển thị chữ số thứ tư trong tensor hạng 3 này, bằng cách sử dụng thư viện Matplotlib (một phần của bộ Python khoa học tiêu chuẩn); xem hình 2.2.


Liệt kê 2.8  Hiển thị chữ số thứ tư


```python
import matplotlib.pyplot as plt
```

```python
digit = train_images[4]
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()
```

Đương nhiên, nhãn tương ứng chỉ là số nguyên 9:


```python
>>> train_labels[4]
9
```

Hình 2.2  Mẫu thứ tư trong tập dữ liệu của chúng tôi


25 Biểu diễn dữ liệu cho mạng nơ-ron


2.2.6 Thao tác với tensor trong NumPy


Trong ví dụ trước, chúng ta đã chọn một chữ số cụ thể dọc theo trục đầu tiên bằng cú pháp train_images[i]. Việc chọn các phần tử cụ thể trong một tensor được gọi là cắt tensor.  Hãy xem các thao tác cắt tensor mà bạn có thể thực hiện trên mảng NumPy.


Ví dụ sau chọn các chữ số từ #10 đến #100 (không bao gồm #100) và đặt chúng thành một mảng có hình dạng (90, 28, 28):


```python
>>> my_slice = train_images[10:100]
>>> my_slice.shape
(90, 28, 28)
```

Nó tương đương với ký hiệu chi tiết hơn này, trong đó chỉ định chỉ mục bắt đầu và chỉ mục dừng cho lát cắt dọc theo mỗi trục tensor. Lưu ý rằng : tương đương với việc chọn toàn bộ trục:


```python
>>> my_slice = train_images[10:100, :, :]               
>>> my_slice.shape
(90, 28, 28)
>>> my_slice = train_images[10:100, 0:28, 0:28]        
>>> my_slice.shape
(90, 28, 28)
```

Nói chung, bạn có thể chọn các lát cắt giữa hai chỉ số bất kỳ dọc theo mỗi trục tensor. Ví dụ: để chọn 14 × 14 pixel ở góc dưới bên phải của tất cả hình ảnh, bạn sẽ làm như sau:


```python
my_slice = train_images[:, 14:, 14:]
```

Cũng có thể sử dụng các chỉ số tiêu cực. Giống như các chỉ số âm trong danh sách Python, chúng biểu thị vị trí tương đối so với điểm cuối của trục hiện tại. Để cắt hình ảnh thành các mảng có kích thước 14 × 14 pixel ở giữa, hãy thực hiện việc này:


```python
my_slice = train_images[:, 7:-7, 7:-7]
```

2.2.7 Khái niệm lô dữ liệu


Nói chung, trục đầu tiên (trục 0, vì quá trình lập chỉ mục bắt đầu từ 0) trong tất cả các tensor dữ liệu mà bạn gặp trong học sâu sẽ là trục mẫu. Trong ví dụ MNIST, “mẫu” là hình ảnh của các chữ số.


Ngoài ra, các mô hình học sâu không xử lý toàn bộ tập dữ liệu cùng một lúc; thay vào đó, họ chia dữ liệu thành các “đợt” nhỏ hoặc nhóm mẫu có kích thước cố định. Cụ thể, đây là một lô chữ số MNIST của chúng tôi, với kích thước lô là 128:


Tương đương với ví dụ trước


Cũng tương đương với ví dụ trước


26 Chương 2  Các khối xây dựng toán học của mạng lưới thần kinh


```python
batch = train_images[:128]
```

Và đây là đợt tiếp theo:


```python
batch = train_images[128:256]
```

Và đợt thứ n:


```python
n = 3
batch = train_images[128 * n : 128 * (n + 1)]
```

Khi xem xét một tensor lô như vậy, trục đầu tiên (trục 0) được gọi là trục lô (hoặc thứ nguyên lô). Bạn sẽ thường xuyên gặp thuật ngữ này khi sử dụng Keras và các thư viện deep learning khác.


2.2.8 Ví dụ thực tế về tensor dữ liệu


Hãy làm cho tensor dữ liệu trở nên cụ thể hơn bằng một vài ví dụ tương tự với những gì bạn sẽ gặp sau này. Dữ liệu bạn sẽ thao tác hầu như sẽ luôn thuộc một trong các loại sau:


- Dữ liệu vectơ—tensor cấp 2 về hình dạng (mẫu, đặc trưng), trong đó mỗi mẫu là một


vectơ thuộc tính số (“đặc điểm”)


- Dữ liệu chuỗi thời gian hoặc dữ liệu chuỗi—Các tensor cấp 3 có hình dạng (mẫu, dấu thời gian,


các tính năng), trong đó mỗi mẫu là một chuỗi (có độ dài dấu thời gian) của vectơ đặc trưng


- Hình ảnh—Các tenxơ cấp 4 về hình dạng (mẫu, chiều cao, chiều rộng, kênh), trong đó


mỗi mẫu là một lưới pixel 2D và mỗi pixel được biểu thị bằng một vectơ giá trị (“kênh”)


- Video—Các tenxơ cấp 5 về hình dạng (mẫu, khung, chiều cao, chiều rộng, kênh),


trong đó mỗi mẫu là một chuỗi (khung hình có độ dài) hình ảnh


Dữ liệu vectơ Dữ liệu vectơ là một trong những trường hợp phổ biến nhất. Trong tập dữ liệu như vậy, mỗi điểm dữ liệu có thể được mã hóa dưới dạng vectơ và do đó, một loạt dữ liệu sẽ được mã hóa dưới dạng tenxơ cấp 2 (nghĩa là một mảng vectơ), trong đó trục đầu tiên là trục mẫu và trục thứ hai là trục tính năng.


Chúng ta hãy xem hai ví dụ:


- Một tập dữ liệu thống kê về con người, trong đó chúng tôi xem xét độ tuổi, giới tính và


thu nhập. Mỗi người có thể được mô tả như một vectơ có ba giá trị và do đó, toàn bộ tập dữ liệu gồm 100.000 người có thể được lưu trữ trong một tenxơ cấp 2 có hình dạng


```python
(100000, 3).
```

27 Biểu diễn dữ liệu cho mạng nơ-ron


- Một tập dữ liệu về các tài liệu văn bản, trong đó chúng tôi biểu thị từng tài liệu bằng số lượng


mỗi từ xuất hiện trong đó bao nhiêu lần (trong từ điển 20.000 từ thông dụng). Mỗi tài liệu có thể được mã hóa dưới dạng vectơ gồm 20.000 giá trị (một giá trị cho mỗi từ trong từ điển) và do đó toàn bộ tập dữ liệu gồm 500 tài liệu có thể được lưu trữ dưới dạng tensor có hình dạng (500, 20000).


Dữ liệu chuỗi thời gian hoặc dữ liệu chuỗi Bất cứ khi nào thời gian quan trọng trong dữ liệu của bạn (hoặc khái niệm về thứ tự chuỗi), bạn nên lưu trữ dữ liệu đó trong một tensor hạng 3 với trục thời gian rõ ràng. Mỗi mẫu có thể được mã hóa dưới dạng một chuỗi vectơ (tenxơ hạng 2) và do đó, một loạt dữ liệu sẽ được mã hóa dưới dạng tenxơ hạng 3 (xem hình 2.3).


Trục thời gian luôn là trục thứ hai (trục chỉ số 1), theo quy ước. Hãy xem xét một vài ví dụ:


- Tập dữ liệu về giá cổ phiếu—Mỗi phút, chúng tôi lưu trữ giá hiện tại của cổ phiếu, giá trị


giá cao nhất trong phút vừa qua và giá thấp nhất trong phút vừa qua. Do đó, mỗi phút được mã hóa dưới dạng vectơ 3D, toàn bộ ngày giao dịch được mã hóa dưới dạng ma trận hình dạng (390, 3) (có 390 phút trong một ngày giao dịch) và dữ liệu có giá trị trong 250 ngày có thể được lưu trữ ở dạng tenxơ cấp 3 (250, 390, 3). Ở đây, mỗi mẫu sẽ có giá trị dữ liệu trong một ngày.


- Một tập hợp các tweet, trong đó chúng tôi mã hóa mỗi tweet dưới dạng một chuỗi gồm 280 ký tự trong một


bảng chữ cái gồm 128 ký tự duy nhất—Trong cài đặt này, mỗi ký tự có thể được mã hóa dưới dạng vectơ nhị phân có kích thước 128 (vectơ toàn số 0 ngoại trừ mục nhập 1 ở chỉ mục tương ứng với ký tự). Sau đó, mỗi tweet có thể được mã hóa dưới dạng tenxơ cấp 2 có hình dạng (280, 128) và tập dữ liệu gồm 1 triệu tweet có thể được lưu trữ dưới dạng tenxơ có hình dạng (1000000, 280, 128).


Dữ liệu hình ảnh Hình ảnh thường có ba chiều: chiều cao, chiều rộng và độ sâu màu. Mặc dù hình ảnh thang độ xám (như chữ số MNIST của chúng tôi) chỉ có một kênh màu duy nhất và do đó có thể được lưu trữ ở các tensor cấp 2, nhưng theo quy ước, tenxơ hình ảnh luôn ở cấp 3, với kênh màu một chiều cho hình ảnh thang độ xám. Do đó, một lô 128 ảnh thang độ xám có kích thước 256 × 256 có thể được lưu trữ dưới dạng tenxơ có hình dạng (128, 256, 256, 1) và một lô gồm 128 ảnh màu có thể được lưu trữ dưới dạng tenxơ có hình dạng (128, 256, 256, 3) (xem hình 2.4).


Có hai quy ước về hình dạng của tensor hình ảnh: quy ước kênh cuối cùng (là tiêu chuẩn trong JAX và TensorFlow, cũng như hầu hết các công cụ học sâu khác hiện có) và quy ước kênh đầu tiên (là tiêu chuẩn trong PyTorch).


Quy ước cuối cùng của kênh đặt trục độ sâu màu ở cuối: (mẫu,


chiều cao, chiều rộng, màu sắc_độ sâu). Trong khi đó, quy ước ưu tiên kênh đặt trục độ sâu màu ngay sau trục lô: (mẫu, color_deep, chiều cao, chiều rộng).


## Đặc trưng


## Dấu thời gian


## Mẫu


Hình 2.3  Tenxơ dữ liệu chuỗi thời gian cấp 3


28 Chương 2  Các khối xây dựng toán học của mạng lưới thần kinh


Với quy ước kênh đầu tiên, các ví dụ trước đó sẽ trở thành (128, 1, 256, 256) và


(128, 3, 256, 256). API Keras cung cấp hỗ trợ cho cả hai định dạng.


Dữ liệu video Dữ liệu video là một trong số ít loại dữ liệu trong thế giới thực mà bạn cần có tensor cấp 5. Một video có thể được hiểu là một chuỗi các khung hình, mỗi khung hình là một hình ảnh màu.  Bởi vì mỗi khung có thể được lưu trữ trong một tensor hạng 3 (chiều cao, chiều rộng,


color_deep), một chuỗi các khung hình có thể được lưu trữ trong một tensor cấp 4 (khung hình, chiều cao, chiều rộng, color_deep) và do đó, một loạt video khác nhau có thể được lưu trữ trong một tenxơ cấp 5 có hình dạng (mẫu, khung,


chiều cao, chiều rộng, màu sắc_độ sâu).


Ví dụ: một video clip YouTube dài 60 giây, 144 × 256 được lấy mẫu ở 4 khung hình mỗi giây sẽ có 240 khung hình. Một loạt bốn video clip như vậy sẽ được lưu trữ dưới dạng tensor có hình dạng (4, 240, 144, 256, 3). Đó là tổng cộng 106.168.320 giá trị! Nếu dtype của tensor là float32 thì mỗi giá trị sẽ được lưu trữ trong 32 bit, do đó tensor sẽ đại diện cho 425 MB. Nặng! Các video bạn gặp trong đời thực nhẹ hơn nhiều vì chúng không được lưu trữ trong float32 và chúng thường được nén bởi một hệ số lớn (chẳng hạn như định dạng MPEG).


2.3 Các bánh răng của mạng nơ-ron: Các phép toán tensor


Giống như bất kỳ chương trình máy tính nào cuối cùng cũng có thể được rút gọn thành một tập hợp nhỏ các phép toán nhị phân trên các đầu vào nhị phân (AND, OR, NOR, v.v.), tất cả các phép biến đổi được học bởi mạng lưới thần kinh sâu có thể được rút gọn thành một số phép toán tensor (hoặc hàm tensor) áp dụng cho tensor của dữ liệu số. Ví dụ: có thể cộng các tensor, nhân tensor, v.v.


Trong ví dụ ban đầu, chúng tôi đã xây dựng mô hình của mình bằng cách xếp chồng các lớp Dày đặc lên nhau. Một phiên bản lớp Keras trông như thế này:


```python
keras.layers.Dense(512, activation="relu")
```

Lớp này có thể được hiểu là một hàm, lấy đầu vào là một ma trận và trả về một ma trận khác—một cách biểu diễn mới cho tensor đầu vào. Cụ thể, hàm như sau (trong đó W là ma trận và b là vectơ, cả hai thuộc tính của lớp):


```python
output = relu(matmul(input, W) + b)
```

Kênh màu


## Chiều cao


## Chiều rộng


## Mẫu


Hình 2.4  Tenxor dữ liệu hình ảnh cấp 4


29 Các bánh răng của mạng nơ-ron: Các phép toán tensor


Hãy giải nén cái này. Chúng ta có ba phép toán tensor ở đây:


- Tích tensor (matmul) giữa tensor đầu vào và tensor có tên W.


- Phép cộng (+) giữa ma trận thu được và vectơ b.


- Phép toán relu: relu(x) bằng max(x, 0). "relu" là viết tắt của “Đơn vị tuyến tính được điều chỉnh lại”.


LƯU Ý  Mặc dù phần này đề cập hoàn toàn đến các biểu thức đại số tuyến tính nhưng bạn sẽ không tìm thấy bất kỳ ký hiệu toán học nào trong cuốn sách này. Tôi nhận thấy rằng các khái niệm toán học có thể dễ dàng được nắm vững hơn bởi các lập trình viên không có nền tảng toán học nếu chúng được biểu diễn dưới dạng các đoạn mã Python ngắn thay vì các phương trình toán học. Vì vậy, chúng tôi sẽ sử dụng mã NumPy xuyên suốt.


2.3.1 Hoạt động theo phần tử


Phép toán và phép cộng relu là các phép toán theo phần tử: các phép toán được áp dụng độc lập cho từng phần tử trong tensor đang được xem xét. Điều này có nghĩa là các hoạt động này rất dễ tuân thủ các triển khai song song ồ ạt (các triển khai được vector hóa, một thuật ngữ xuất phát từ kiến ​​trúc siêu máy tính bộ xử lý vector từ giai đoạn 1970–1990). Nếu bạn muốn viết một bản triển khai Python đơn giản của một thao tác theo phần tử, bạn sử dụng vòng lặp for, như trong cách triển khai đơn giản này của một thao tác relu theo phần tử:


```python
def naive_relu(x):
    assert len(x.shape) == 2                                     
    x = x.copy()                                                 
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] = max(x[i, j], 0)
    return x
```

Bạn có thể làm tương tự để bổ sung:


```python
def naive_add(x, y):
    assert len(x.shape) == 2                                   
    assert x.shape == y.shape
    x = x.copy()                                               
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] += y[i, j]
    return x
```

Theo nguyên tắc tương tự, bạn có thể thực hiện phép nhân, phép trừ theo từng phần tử, v.v.


Trong thực tế, khi xử lý mảng NumPy, các thao tác này có sẵn dưới dạng các hàm NumPy tích hợp được tối ưu hóa tốt, chính các hàm này giao phó công việc nặng nhọc cho việc triển khai Chương trình con đại số tuyến tính cơ bản (BLAS). BLAS ở mức độ thấp,


x là một tensor NumPy hạng 2.


Tránh ghi đè tensor đầu vào


x và y là các tenxơ NumPy hạng 2.


Tránh ghi đè tensor đầu vào


30 Chương 2  Các khối xây dựng toán học của mạng lưới thần kinh


các quy trình thao tác tensor hiệu quả, song song cao thường được triển khai trong Fortran hoặc C.


Vì vậy, trong NumPy, bạn có thể thực hiện thao tác theo từng phần tử sau đây và nó sẽ hoạt động rất nhanh:


```python
import numpy as np
```

```python
z = x + y                                                              
z = np.maximum(z, 0.0)
```

Hãy thực sự tính toán thời gian chênh lệch:


```python
import time
```

```python
x = np.random.random((20, 100))
y = np.random.random((20, 100))
```

```python
t0 = time.time()
for _ in range(1000):
    z = x + y
    z = np.maximum(z, 0.0)
print("Took: {0:.2f} s".format(time.time() - t0))
```

Quá trình này mất 0,02 giây. Trong khi đó, phiên bản ngây thơ mất 2,45 giây tuyệt đẹp:


```python
t0 = time.time()
for _ in range(1000):
    z = naive_add(x, y)
    z = naive_relu(z)
print("Took: {0:.2f} s".format(time.time() - t0))
```

Tương tự như vậy, khi chạy mã JAX/TensorFlow/PyTorch trên GPU, các hoạt động theo từng phần tử được thực thi thông qua triển khai CUDA được vector hóa hoàn toàn để có thể sử dụng tốt nhất kiến ​​trúc chip GPU song song cao.


2.3.2 Phát sóng


Việc triển khai ngây thơ trước đây của chúng tôi về ngây thơ_add chỉ hỗ trợ việc bổ sung các tensor hạng 2 có hình dạng giống hệt nhau. Nhưng trong lớp Mật độ được giới thiệu trước đó, chúng tôi đã thêm một tenxơ cấp 2 kèm theo một vectơ. Điều gì xảy ra với phép cộng khi hình dạng của hai tensor được cộng khác nhau?


Khi có thể và nếu không có sự mơ hồ, tensor nhỏ hơn sẽ được phát sóng để phù hợp với hình dạng của tensor lớn hơn. Phát sóng bao gồm hai bước:


- Các trục (gọi là trục phát sóng) được thêm vào tensor nhỏ hơn để phù hợp với ndim của


tensor càng lớn.


Phép cộng theo phần tử


Relu theo yếu tố


31 Các bánh răng của mạng nơ-ron: Các phép toán tensor


- Tenxor nhỏ hơn được lặp lại dọc theo các trục mới này để khớp với hình dạng đầy đủ


của tensor lớn hơn.


Hãy xem một ví dụ cụ thể. Xét X có hình dạng (32, 10) và y có hình dạng


```python
(10,):
```

```python
import numpy as np
```

```python
X = np.random.random((32, 10))                          
y = np.random.random((10,))
```

Đầu tiên, chúng ta thêm một trục trống đầu tiên vào y, có hình dạng trở thành (1, 10):


```python
y = np.expand_dims(y, axis=0)
```

Sau đó, chúng ta lặp lại y 32 lần dọc theo trục mới này để có được một tenxơ Y có hình dạng (32, 10), trong đó Y[i, :] == y cho i trong phạm vi (0, 32):


```python
Y = np.tile(y, (32, 1))
```

Tại thời điểm này, chúng ta có thể thêm X và Y vì chúng có hình dạng giống nhau.


```python
In terms of implementation, no new rank-2 tensor is created because that would be 
terribly inefficient. The repetition operation is entirely virtual: it happens at the algo-
rithmic level rather than at the memory level. But thinking of the vector being repeated 
32 times alongside a new axis is a helpful mental model. Here’s what a naive implemen-
tation would look like:
```

```python
def naive_add_matrix_and_vector(x, y):
    assert len(x.shape) == 2                                
    assert len(y.shape) == 1                                
    assert x.shape[1] == y.shape[0]
    x = x.copy()                                            
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] += y[j]
    return x
```

Với việc phát sóng, bạn thường có thể áp dụng các phép toán theo phần tử hai tensor nếu một tensor có hình dạng (a, b, … n, n + 1, … m) và cái còn lại có hình dạng (n, n + 1, … m).  Việc phát sóng sau đó sẽ tự động diễn ra cho các trục từ a đến n - 1.


X là ma trận ngẫu nhiên có dạng (32, 10).


y là một vectơ ngẫu nhiên có hình dạng (10,).


Hình dạng của y bây giờ là (1, 10).


Lặp lại y 32 lần dọc theo trục 0 để thu được Y có hình dạng (32, 10).


x là một tensor NumPy hạng 2.


y là một vectơ NumPy.


Tránh ghi đè tensor đầu vào


32 Chương 2  Các khối xây dựng toán học của mạng lưới thần kinh


Ví dụ sau áp dụng phép toán tối đa theo phần tử cho hai tensor có hình dạng khác nhau thông qua việc phát sóng:


```python
import numpy as np
```

```python
x = np.random.random((64, 3, 32, 10))                     
y = np.random.random((32, 10))                            
z = np.maximum(x, y)
```

2.3.3 Tích tensor


Tích tensor, còn được gọi là tích số chấm hoặc matmul (viết tắt của “phép nhân ma trận”) là một trong những phép toán tensor phổ biến nhất, hữu ích nhất.


```python
In NumPy, a tensor product is done using the np.matmul function, and in Keras, with 
the keras.ops.matmul function. Its shorthand is the @ operator in Python:
```

```python
x = np.random.random((32,))
y = np.random.random((32,))
```

```python
z = np.matmul(x, y)                                    
z = x @ y
```

Trong ký hiệu toán học, bạn sẽ ghi chú phép toán bằng dấu chấm (^) (do đó có tên là “tích số chấm”):


```python
z = x • y
```

Về mặt toán học, phép toán matmul làm gì? Hãy bắt đầu với tích của hai vectơ x và y. Nó được tính như sau:


```python
def naive_vector_product(x, y):
    assert len(x.shape) == 1                           
    assert len(y.shape) == 1                           
    assert x.shape[0] == y.shape[0]
    z = 0.0
    for i in range(x.shape[0]):
        z += x[i] * y[i]
    return z
```

Bạn sẽ nhận thấy rằng tích giữa hai vectơ là vô hướng và chỉ các vectơ có cùng số phần tử mới tương thích với phép toán này.


x là một tensor ngẫu nhiên có hình dạng (64, 3, 32, 10).


y là một tensor ngẫu nhiên có hình dạng (32, 10).


Đầu ra z có hình dạng (64, 3, 32, 10) giống x.


Lấy tích giữa x và y


Điều này là tương đương.


x và y là các vectơ NumPy.


33 Các bánh răng của mạng nơ-ron: Các phép toán tensor


Bạn cũng có thể lấy tích giữa ma trận x và vectơ y, nó trả về một vectơ trong đó các hệ số là tích giữa y và các hàng của x. Bạn thực hiện nó như sau:


```python
def naive_matrix_vector_product(x, y):
    assert len(x.shape) == 2                                  
    assert len(y.shape) == 1                                  
    assert x.shape[1] == y.shape[0]                           
    z = np.zeros(x.shape[0])                                  
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            z[i] += x[i, j] * y[j]
    return z
```

Bạn cũng có thể sử dụng lại mã mà chúng tôi đã viết trước đó, trong đó nêu bật mối quan hệ giữa tích vectơ ma trận và tích vectơ:


```python
def naive_matrix_vector_product(x, y):
    z = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        z[i] = naive_vector_product(x[i, :], y)
    return z
```

Lưu ý rằng ngay khi một trong hai tensor có ndim lớn hơn 1, matmul không còn đối xứng nữa, nghĩa là matmul(x, y) không giống với matmul(y, x).


Tất nhiên, tích tensor tổng quát hóa thành tensor với số trục tùy ý.  Các ứng dụng phổ biến nhất có thể là tích giữa hai ma trận. Bạn có thể lấy tích của hai ma trận x và y (matmul(x, y)) khi và chỉ khi x.shape[1] ==


y.shape[0]. Kết quả là một ma trận có hình dạng (x.shape[0], y.shape[1]), trong đó các hệ số là tích vectơ giữa các hàng của x và các cột của y. Đây là cách thực hiện ngây thơ:


```python
def naive_matrix_product(x, y):
    assert len(x.shape) == 2                                  
    assert len(y.shape) == 2                                  
    assert x.shape[1] == y.shape[0]                           
    z = np.zeros((x.shape[0], y.shape[1]))                   
    for i in range(x.shape[0]):                               
        for j in range(y.shape[1]):                           
            row_x = x[i, :]
            column_y = y[:, j]
```

x là ma trận NumPy.


y là một vectơ NumPy.


Chiều thứ nhất của x phải bằng chiều thứ 0 của y!


Thao tác này trả về một vectơ 0 với số hàng bằng x.


x và y là ma trận NumPy.


Chiều thứ nhất của x phải bằng chiều thứ 0 của y!


Thao tác này trả về ma trận 0 với hình dạng cụ thể.


Lặp lại các hàng của x…


… và trên các cột của y.


34 Chương 2  Các khối xây dựng toán học của mạng lưới thần kinh


```python
z[i, j] = naive_vector_product(row_x, column_y)
    return z
```

Để hiểu khả năng tương thích của hình dạng sản phẩm vector, nó giúp hình dung các tensor đầu vào và đầu ra bằng cách căn chỉnh chúng như trong hình 2.5.


x, y và z được mô tả dưới dạng hình chữ nhật (hộp các hệ số theo nghĩa đen). Vì các hàng của x và các cột của y phải có cùng kích thước nên chiều rộng của x phải bằng chiều cao của y. Nếu bạn tiếp tục phát triển các thuật toán học máy mới, bạn có thể sẽ thường xuyên vẽ những sơ đồ như vậy.


Tổng quát hơn, bạn có thể lấy tích giữa các tensor có chiều cao hơn, tuân theo các quy tắc tương tự về tính tương thích hình dạng như đã nêu trước đó đối với trường hợp 2D:


```python
(a, b, c, d) • (d,) -> (a, b, c)
(a, b, c, d) • (d, e) -> (a, b, c, e)
```

Và vân vân.


Một


b


```python
x • y = z
```

b


x.shape:


```python
(a, b)
```

y.hình dạng:


```python
(b, c)
```

z.hình dạng:


```python
(a, c)
```

Hàng x


Cột của y


```python
z [ i,  j ]
```

c


Hình 2.5  Sơ đồ hộp sản phẩm ma trận


2.3.4 Định hình lại tensor


Loại hoạt động tensor thứ ba cần phải hiểu là định hình lại tensor.  Mặc dù nó không được sử dụng trong các Lớp dày đặc trong ví dụ về mạng thần kinh đầu tiên của chúng tôi, nhưng chúng tôi đã sử dụng nó khi xử lý trước dữ liệu chữ số trước khi đưa nó vào mô hình của mình:


```python
train_images = train_images.reshape((60000, 28 * 28))
```

35 Các bánh răng của mạng nơ-ron: Các phép toán tensor


Định hình lại một tensor có nghĩa là sắp xếp lại các hàng và cột của nó để phù hợp với hình dạng mục tiêu.  Đương nhiên, tensor được định hình lại có tổng số hệ số giống như tensor ban đầu. Việc định hình lại được hiểu rõ nhất qua các ví dụ đơn giản:


```python
>>> x = np.array([[0., 1.],
...               [2., 3.],
...               [4., 5.]])
>>> x.shape
(3, 2)
>>> x = x.reshape((6, 1))
>>> x
array([[ 0.],
       [ 1.],
       [ 2.],
       [ 3.],
       [ 4.],
       [ 5.]])
>>> x = x.reshape((2, 3))
>>> x
array([[ 0.,  1.,  2.],
       [ 3.,  4.,  5.]])
```

Một trường hợp đặc biệt của việc định hình lại thường gặp là chuyển vị. Chuyển vị một ma trận có nghĩa là hoán đổi các hàng và cột của nó, sao cho x[i, :] trở thành x[:, i]:


```python
>>> x = np.zeros((300, 20))        
>>> x = np.transpose(x)
>>> x.shape
(20, 300)
```

2.3.5 Giải thích hình học của các phép toán tensor


Bởi vì nội dung của các tensor được điều khiển bởi các phép toán tensor có thể được hiểu như là tọa độ của các điểm trong một không gian hình học nào đó, nên tất cả các phép toán tensor đều có cách diễn giải hình học. Ví dụ, hãy xem xét phép cộng. Chúng ta sẽ bắt đầu với vectơ sau:


## ```python
A = [0.5, 1]
```

Đó là một điểm trong không gian 2D (xem hình 2.6). Người ta thường hình dung một vectơ như một mũi tên nối điểm gốc với điểm, như trong hình 2.7.


```python
Let’s consider a new point, B = [1, 0.25], which we’ll add to the previous one. This is 
done geometrically by chaining together the vector arrows, with the resulting location 
being the vector representing the sum of the previous two vectors (see figure 2.8). As 
you can see, adding a vector B to a vector A represents the action of copying point A in
```

Tạo một ma trận hình dạng toàn số 0 (300, 20)


36 Chương 2  Các khối xây dựng toán học của mạng lưới thần kinh


1


1


```python
A
[0.5, 1]
```

Hình 2.6  Một điểm trong không gian 2D


vị trí mới, có khoảng cách và hướng từ điểm A ban đầu được xác định bởi vectơ B. Nếu bạn áp dụng cùng một phép cộng vectơ cho một nhóm điểm trong mặt phẳng (“đối tượng”), bạn sẽ tạo một bản sao của toàn bộ đối tượng ở một vị trí mới (xem hình 2.9). Do đó, phép cộng tensor thể hiện hành động dịch một đối tượng (di chuyển đối tượng mà không làm biến dạng nó) theo một lượng nhất định theo một hướng nhất định.


Nói chung, các phép toán hình học cơ bản, chẳng hạn như dịch chuyển, xoay, chia tỷ lệ, nghiêng, v.v., có thể được biểu diễn dưới dạng các phép toán tenxơ.  Dưới đây là một vài ví dụ:


- Dịch—Như bạn vừa thấy, việc thêm một vectơ vào một điểm sẽ di chuyển điểm này một khoảng


lượng cố định theo một hướng cố định. Áp dụng cho một tập hợp các điểm (chẳng hạn như đối tượng 2D), điều này được gọi là “bản dịch” (xem hình 2.9).


x y


K K


hệ số dọc


hệ số ngang


hệ số ngang


dọc_factor +


Hình 2.9  Phép dịch 2D dưới dạng phép cộng vectơ


1


1


```python
A
[0.5, 1]
```

Hình 2.7  Một điểm trong không gian 2D được biểu diễn dưới dạng mũi tên


1


1


## MỘT


## B


## ```python
A + B
```

Hình 2.8  Giải thích hình học của tổng hai vectơ


37 Các bánh răng của mạng nơ-ron: Các phép toán tensor


- Xoay—Một phép quay ngược chiều kim đồng hồ của vectơ 2D một góc theta (xem hình-


```python
ure 2.10) can be achieved via a product with a 2 × 2 matrix R = [[cos(theta),
```

```python
-sin(theta)], [sin(theta), cos(theta)]].
```

```python
cos(theta) –sin(theta)
sin(theta)
cos(theta)
```

x y


Theta K


## K


Hình 2.10  Xoay 2D (ngược chiều kim đồng hồ) dưới dạng tích ma trận


- Chia tỷ lệ—Có thể chia tỷ lệ theo chiều dọc và chiều ngang của hình ảnh (xem hình 2.11)


```python
achieved via a product with a 2 × 2 matrix S = (note that such a matrix is called 
a “diagonal matrix” because it only has non-zero coefficients in its “diagonal,” 
going from the top left to the bottom right).
```

1 0 0 –0,5 x y K K


Hình 2.11  Chia tỷ lệ 2D dưới dạng sản phẩm ma trận


- Biến đổi tuyến tính—Một sản phẩm có ma trận tùy ý thực hiện một phép biến đổi tuyến tính


hình thức. Lưu ý rằng việc chia tỷ lệ và xoay, đã thấy trước đây, theo định nghĩa, là các phép biến đổi tuyến tính.


- Biến đổi affine—Biến đổi affine (xem hình 2.12) là sự kết hợp của một


```python
linear transform (achieved via a matrix product) and a translation (achieved via 
a vector addition). As you have probably recognized, that’s exactly the y = W @ x
```

+ b tính toán được thực hiện bởi lớp dày đặc! Lớp dày đặc không có chức năng kích hoạt là lớp affine.


38 Chương 2  Các khối xây dựng toán học của mạng lưới thần kinh


## K


W • x + b


Hình 2.12  Biến đổi affine trong mặt phẳng


- Lớp dày đặc có kích hoạt relu—Một quan sát quan trọng về các phép biến đổi affine


```python
is that if you apply many of them repeatedly, you still end up with an affine trans-
form (so you could just have applied that one affine transform in the first place). 
Let’s try it with two: affine2(affine1(x)) = W2 @ (W1 @ x + b1) + b2 = (W2 @ W1)
```

@x+(W2@b1+b2). Đó là một phép biến đổi affine trong đó phần tuyến tính là ma trận W2@W1 và phần dịch là vectơ W2@b1 + b2. Do đó, một mạng lưới thần kinh nhiều lớp được tạo hoàn toàn bằng các lớp Mật độ dày đặc mà không có sự kích hoạt sẽ tương đương với một lớp Mật độ cao duy nhất. Mạng lưới thần kinh “sâu” này sẽ chỉ là một mô hình tuyến tính trá hình! Đây là lý do tại sao chúng ta cần các hàm kích hoạt, như relu (xem trong hình 2.13). Nhờ các hàm kích hoạt, một chuỗi các Lớp dày đặc có thể được tạo ra để thực hiện phép biến đổi hình học phi tuyến rất phức tạp, mang lại không gian giả thuyết rất phong phú cho mạng lưới thần kinh sâu của bạn. Chúng tôi đề cập đến ý tưởng này chi tiết hơn trong chương tiếp theo.


## K


```python
relu(W • x + b)
```

Hình 2.13  Biến đổi affine sau đó kích hoạt relu


2.3.6 Giải thích hình học của học sâu


Bạn vừa biết rằng mạng lưới thần kinh bao gồm toàn bộ chuỗi các phép toán tensor và tất cả các phép toán tensor này chỉ là các phép biến đổi hình học đơn giản của dữ liệu đầu vào. Theo đó, bạn có thể hiểu mạng lưới thần kinh là một phép biến đổi hình học rất phức tạp trong không gian nhiều chiều, được thực hiện thông qua một loạt các bước đơn giản.


Trong không gian 3D, hình ảnh tinh thần sau đây có thể hữu ích. Hãy tưởng tượng hai tờ giấy màu: một màu đỏ và một màu xanh. Đặt cái này lên trên cái kia. Bây giờ vò chúng lại với nhau thành một quả bóng nhỏ. Quả bóng giấy nhàu nát đó chính là dữ liệu đầu vào của bạn, và mỗi tờ giấy là một lớp dữ liệu trong bài toán phân loại. Mục đích của mạng lưới thần kinh là hình dung


39 Động cơ của mạng nơ-ron: Tối ưu hóa dựa trên độ dốc


đưa ra một phép biến đổi của quả bóng giấy để làm cho nó không bị nhàu nát để làm cho hai lớp trở lại có thể tách biệt rõ ràng (xem hình 2.14). Với học sâu, điều này sẽ được thực hiện dưới dạng một loạt các phép biến đổi đơn giản của không gian 3D, chẳng hạn như những phép bạn có thể áp dụng trên quả bóng giấy bằng ngón tay của mình, mỗi lần một chuyển động.


Hình 2.14  Tách dữ liệu đa dạng phức tạp


Những quả bóng giấy không bị nhàu nát là nội dung của máy học: tìm ra các biểu diễn gọn gàng cho các đa tạp dữ liệu phức tạp, có độ gấp cao trong không gian nhiều chiều (đa tạp là một bề mặt liên tục, giống như tờ giấy nhàu nát của chúng ta). Tại thời điểm này, bạn nên có một trực giác khá tốt về lý do tại sao deep learning lại vượt trội ở lĩnh vực này: nó sử dụng phương pháp phân hủy dần dần một phép biến đổi hình học phức tạp thành một chuỗi dài các biến đổi cơ bản, gần như là chiến lược mà con người sẽ tuân theo để làm phẳng một quả bóng giấy. Mỗi lớp trong mạng sâu áp dụng một phép biến đổi để tách dữ liệu ra một chút—và việc xếp chồng các lớp sâu khiến cho quá trình gỡ rối cực kỳ phức tạp trở nên dễ dàng thực hiện.


2.4 Động cơ của mạng nơ-ron: Tối ưu hóa dựa trên độ dốc


Như bạn đã thấy trong phần trước, mỗi lớp thần kinh từ ví dụ mô hình đầu tiên của chúng tôi sẽ biến đổi dữ liệu đầu vào của nó như sau:


```python
output = relu(matmul(input, W) + b)
```

Trong biểu thức này, W và b là các tensor là thuộc tính của lớp. Chúng được gọi là trọng số hoặc tham số có thể huấn luyện của lớp (tương ứng là thuộc tính hạt nhân và độ lệch).  Các trọng số này chứa thông tin mà mô hình đã học được từ việc tiếp xúc với dữ liệu huấn luyện.


Ban đầu, các ma trận trọng số này chứa đầy các giá trị ngẫu nhiên nhỏ (bước này gọi là khởi tạo ngẫu nhiên). Tất nhiên, không có lý do gì để mong đợi điều đó relu(matmul(input, W)


+ b), khi W và b là ngẫu nhiên, sẽ mang lại bất kỳ biểu diễn hữu ích nào. Các biểu diễn thu được là vô nghĩa—nhưng chúng là điểm khởi đầu. Điều tiếp theo là điều chỉnh dần dần các trọng số này dựa trên tín hiệu phản hồi. Sự điều chỉnh dần dần này, còn được gọi là đào tạo, về cơ bản là quá trình học tập mà học máy hướng tới.


Điều này xảy ra trong vòng lặp huấn luyện, hoạt động như sau. Lặp lại các bước này trong một vòng lặp cho đến khi tổn thất có vẻ đủ thấp:


40 Chương 2  Các khối xây dựng toán học của mạng lưới thần kinh


1 Vẽ một loạt mẫu huấn luyện x và mục tiêu tương ứng y_true.


2 Chạy mô hình trên x (bước được gọi là chuyển tiếp) để thu được dự đoán y_pred.


3 Tính toán độ hao hụt của mô hình trên lô, thước đo độ không phù hợp giữa


y_pred và y_true.


4 Cập nhật tất cả các trọng số của mô hình theo cách giảm nhẹ tổn thất trong lô này.


Cuối cùng, bạn sẽ thu được một mô hình có mức tổn thất dữ liệu huấn luyện rất thấp: mức độ không khớp thấp giữa dự đoán y_pred và mục tiêu dự kiến ​​y_true. Mô hình đã “học” cách ánh xạ đầu vào của nó tới các mục tiêu chính xác. Nhìn xa thì có vẻ như ảo thuật nhưng khi rút gọn lại thành những bước cơ bản thì lại trở nên đơn giản.


Bước 1 nghe có vẻ dễ dàng—đó chỉ là mã I/O. Bước 2 và 3 chỉ đơn thuần là việc áp dụng một số phép toán tensor, vì vậy bạn có thể thực hiện các bước này hoàn toàn từ những gì bạn đã học ở phần trước. Phần khó khăn nhất là bước 4: cập nhật trọng số của mô hình. Với một hệ số trọng số riêng lẻ trong mô hình, làm thế nào bạn có thể tính toán xem hệ số này nên tăng hay giảm và tăng bao nhiêu?


```python
One naive solution would be to freeze all weights in the model except the one scalar 
coefficient being considered and try different values for this coefficient. Let’s say the 
initial value of the coefficient is 0.3. After the forward pass on a batch of data, the loss of 
the model on the batch is 0.5. If you change the coefficient’s value to 0.35 and rerun the 
forward pass, the loss increases to 0.6. But if you lower the coefficient to 0.25, the loss 
falls to 0.4. In this case, it seems that updating the coefficient by –0.05 would contribute 
to minimizing the loss. This would have to be repeated for all coefficients in the model.
```

Nhưng cách tiếp cận như vậy sẽ cực kỳ kém hiệu quả vì bạn cần tính hai lần chuyển tiếp (rất tốn kém) cho mỗi hệ số riêng lẻ (trong đó có rất nhiều, thường ít nhất là vài nghìn và có khả năng lên tới hàng tỷ). Rất may, có một cách tiếp cận tốt hơn nhiều: giảm độ dốc.


Giảm dần độ dốc là kỹ thuật tối ưu hóa hỗ trợ các mạng lưới thần kinh hiện đại. Đây là ý chính của nó. Tất cả các hàm được sử dụng trong mô hình của chúng tôi (chẳng hạn như matmul hoặc


```python
+) transform their input in a smooth and continuous way: if you look at z = x + y, for 
instance, a small change in y only results in a small change in z, and if you know the 
direction of the change in y, you can infer the direction of the change in z. Mathe-
matically, you’d say these functions are differentiable. If you chain together such func-
tions, the bigger function you obtain is still differentiable. In particular, this applies to 
the function that maps the model’s coefficients to the loss of the model on a batch of 
data: a small change of the model’s coefficients results in a small, predictable change 
of the loss value. This enables you to use a mathematical operator called the gradient to 
describe how the loss varies as you move the model’s coefficients in different directions. 
If you compute this gradient, you can use it to move the coefficients (all at once in a sin-
gle update, rather than one at a time) in a direction that decreases the loss.
```

Nếu bạn đã biết khả vi nghĩa là gì và độ dốc là gì, bạn có thể bỏ qua hai phần tiếp theo. Nếu không, phần sau đây sẽ giúp bạn hiểu những khái niệm này.


41 Động cơ của mạng nơ-ron: Tối ưu hóa dựa trên độ dốc


2.4.1 Công cụ phái sinh là gì?


```python
Consider a continuous, smooth function f(x) = y, mapping a number x to a new num-
ber y. We can use the function in figure 2.15 as an example.
```

Vì hàm số này là liên tục nên một thay đổi nhỏ trong x chỉ có thể dẫn đến một thay đổi nhỏ trong y—đó là trực giác đằng sau tính liên tục. Giả sử bạn tăng x lên một hệ số nhỏ epsilon_x: điều này dẫn đến một thay đổi nhỏ epsilon_y thành y, như minh họa trong hình 2.16.


```python
y = f(x)
```

x


y


Hình 2.15  Hàm số trơn tru, liên tục


Ngoài ra, do hàm này trơn tru (đường cong của nó không có góc đột ngột) nên khi epsilon_x đủ nhỏ, xung quanh một điểm p nhất định, có thể xấp xỉ


f là hàm tuyến tính của độ dốc a, do đó epsilon_y trở thành * epsilon_x:


```python
f(x + epsilon_x) = y + a * epsilon_x
```

Rõ ràng, phép tính gần đúng tuyến tính này chỉ đúng khi x đủ gần với p.


Hệ số góc a được gọi là đạo hàm của f trong p. Nếu a âm, điều đó có nghĩa là x tăng nhẹ xung quanh


p sẽ làm giảm f(x), như thể hiện trong hình 2.17, và nếu a dương, x tăng nhẹ sẽ dẫn đến f(x tăng). Hơn nữa, giá trị tuyệt đối của a (độ lớn của đạo hàm) cho bạn biết sự tăng hoặc giảm này sẽ diễn ra nhanh như thế nào.


```python
For every differentiable function f(x) (dif-
ferentiable means “can be derived”: for example, 
smooth, continuous functions can be derived), there exists a derivative function f'(x) 
that maps values of x to the slope of the local linear approximation of f in those points. 
For instance, the derivative of cos(x) is -sin(x), the derivative of f(x) = a * x is f'(x)
```

= a, v.v.


```python
y = f(x)
```

epsilon_x


epsilon_y


x


y


Hình 2.16  Với hàm liên tục, một thay đổi nhỏ của x sẽ dẫn đến một thay đổi nhỏ của y.


```python
y = f(x)
```

x


y


Xấp xỉ tuyến tính cục bộ của , với độ dốc a


f


Hình 2.17  Đạo hàm của f theo p


42 Chương 2  Các khối xây dựng toán học của mạng lưới thần kinh


Khả năng suy ra các hàm là một công cụ rất mạnh khi nói đến tối ưu hóa, nhiệm vụ tìm các giá trị của x sao cho giá trị của f(x) cực tiểu hóa. Nếu bạn đang cố gắng cập nhật


x theo hệ số epsilon_x để cực tiểu hoá f(x) và bạn biết đạo hàm của f, thì công việc của bạn đã xong: đạo hàm mô tả đầy đủ f(x) diễn biến như thế nào khi bạn thay đổi x. Nếu muốn giảm giá trị của f(x), bạn chỉ cần di chuyển x một chút theo hướng ngược lại với đạo hàm.


2.4.2 Đạo hàm của phép toán tensor: Độ dốc


Hàm chúng ta vừa xem xét đã biến một giá trị vô hướng x thành một giá trị vô hướng y khác: bạn có thể vẽ nó dưới dạng một đường cong trong mặt phẳng 2D. Bây giờ, hãy tưởng tượng một hàm biến một bộ giá trị vô hướng (x, y) thành giá trị vô hướng z: đó sẽ là một phép toán vectơ. Bạn có thể vẽ nó dưới dạng bề mặt 2D trong không gian 3D (được lập chỉ mục theo tọa độ x, y, z). Tương tự như vậy, bạn có thể tưởng tượng các hàm lấy làm ma trận đầu vào, các hàm lấy làm tensor cấp 3 đầu vào, v.v.


Khái niệm đạo hàm có thể được áp dụng cho bất kỳ hàm nào như vậy, miễn là các bề mặt mà chúng mô tả là liên tục và nhẵn. Đạo hàm của phép toán tensor (hoặc hàm tensor) được gọi là gradient. Độ dốc chỉ là sự khái quát hóa khái niệm đạo hàm cho các hàm lấy tensor làm đầu vào. Hãy nhớ làm thế nào, đối với hàm vô hướng, đạo hàm biểu thị độ dốc cục bộ của đường cong của hàm? Theo cách tương tự, gradient của hàm tensor biểu thị độ cong của bề mặt đa chiều được mô tả bởi hàm. Nó mô tả cách đầu ra của hàm thay đổi khi các tham số đầu vào của nó thay đổi.


Hãy xem một ví dụ dựa trên học máy. Coi như


- Một vectơ đầu vào x (một mẫu trong tập dữ liệu)


- Ma trận W (trọng số của mô hình)


- Mục tiêu y_true (những gì mô hình nên học để liên kết với x)


- Mất chức năng mất mát (có nghĩa là đo khoảng cách giữa dòng điện của mô hình


dự đoán và y_true).


Bạn có thể sử dụng W để tính toán ứng viên mục tiêu y_pred và sau đó tính toán sự mất mát hoặc không khớp giữa ứng viên mục tiêu y_pred và y_true đích:


```python
y_pred = matmul(x, W)                                               
loss_value = loss(y_pred, y_true)
```

Bây giờ, chúng tôi muốn sử dụng độ dốc để tìm ra cách cập nhật W để làm cho loss_value nhỏ hơn. Chúng ta làm điều đó bằng cách nào?


Với đầu vào cố định x và y_true, các thao tác trước đó có thể được hiểu là hàm ánh xạ các giá trị của W (trọng số của mô hình) thành các giá trị tổn thất:


Chúng tôi sử dụng trọng số mô hình W để đưa ra dự đoán cho x.


Chúng tôi ước tính mức độ sai lệch của dự đoán.


43 Động cơ của mạng nơ-ron: Tối ưu hóa dựa trên độ dốc


```python
loss_value = f(W)
```

Giả sử giá trị hiện tại của W là W0. Khi đó đạo hàm của f tại điểm W0 là một tenxơ


grad(loss_value, W0), có hình dạng giống như W, trong đó mỗi hệ số grad(loss_


```python
value, W0)[i, j] indicates the direction and magnitude of the change in loss_value 
you observe when modifying W0[i, j]. That tensor grad(loss_value, W0) is the gradi-
ent of the function f(W) = loss_value in W0, also called “gradient of loss_value with 
respect to W around W0.”
```

```python
NOTE  The tensor operation grad(f(W), W) (which takes as input a matrix W) 
can be expressed as a combination of scalar functions grad_ij(f(W), w_ij), 
each of which would return the derivative of loss_value = f(W) with respect 
to the coefficient W[i, j] of W, assuming all other coefficients are constant.
```

grad_ij được gọi là đạo hàm riêng của f đối với W[i, j].


```python
Concretely, what does grad(loss_value, W0) represent? You saw earlier that the deriv-
ative of a function f(x) of a single coefficient can be interpreted as the slope of the 
curve of f. Likewise, grad(loss_value, W0) can be interpreted as the tensor describing 
the curvature of loss_value = f(W) around W0. Each partial derivative describes the cur-
vature of f in a specific direction.
```

```python
We just saw how for a function f(x), you can reduce the value of f(x) by moving x a 
little in the opposite direction from the derivative. In much the same way, with a func-
tion f(W) of a tensor, you can reduce loss_value = f(W) by moving W in the opposite 
direction from the gradient, such as an update of W1 = W0 - step * grad(f(W0), W0) 
where step is a small scaling factor. That means going against the curvature, which intu-
itively should put you lower on the curve. Note that the scaling factor step is needed 
because grad(loss_value, W0) only approximates the curvature when you’re close to
```

W0, vì vậy bạn không muốn đi quá xa W0.


2.4.3 Giảm độ dốc ngẫu nhiên


Cho một hàm khả vi, về mặt lý thuyết có thể tìm cực tiểu của nó theo phương pháp phân tích: người ta biết rằng cực tiểu của hàm là điểm có đạo hàm bằng 0, vì vậy tất cả những gì bạn phải làm là tìm tất cả các điểm mà đạo hàm tiến tới 0 và kiểm tra xem điểm nào trong số các điểm này hàm số có giá trị thấp nhất.


```python
Applied to a neural network, that means finding analytically the combination of 
weight values that yields the smallest possible loss function. This can be done by solving 
the equation grad(f(W), W) = 0 for W. This is a polynomial equation of N variables, where
```

```python
N is the number of coefficients in the model. Although it would be possible to solve such 
an equation for N = 2 or N = 3, doing so is intractable for real neural networks, where 
the number of parameters is never less than a few thousand and can sometimes be in 
the billions.
```

f mô tả đường cong (hoặc bề mặt nhiều chiều) được hình thành bởi các giá trị tổn thất khi W thay đổi.


44 Chương 2  Các khối xây dựng toán học của mạng lưới thần kinh


Thay vào đó, bạn có thể sử dụng thuật toán bốn bước được nêu ở đầu phần này: sửa đổi từng chút một các tham số dựa trên giá trị tổn thất hiện tại trên một lô dữ liệu ngẫu nhiên. Bởi vì bạn đang xử lý một hàm khả vi, bạn có thể tính toán độ dốc của nó, điều này mang lại cho bạn một cách hiệu quả để thực hiện bước 4. Nếu bạn cập nhật các trọng số theo hướng ngược lại với độ dốc, thì tổn thất sẽ ít hơn một chút mỗi lần:


1 Vẽ một loạt mẫu huấn luyện x và mục tiêu tương ứng y_true.


2 Chạy mô hình trên x để thu được dự đoán y_pred (đây được gọi là chuyển tiếp).


3 Tính toán độ hao hụt của mô hình trên lô, thước đo độ không phù hợp giữa


y_pred và y_true.


4 Tính toán độ dốc của tổn thất theo các tham số của mô hình (điều này được gọi là truyền ngược).


5 Di chuyển các tham số một chút theo hướng ngược lại với gradient—ví dụ: W -= learning_rate * gradient—do đó giảm tổn thất trên lô một chút. Tốc độ học tập (learning_rate ở đây) sẽ là một hệ số vô hướng điều chỉnh “tốc độ” của quá trình giảm độ dốc.


Đủ dễ dàng! Những gì chúng tôi vừa mô tả được gọi là giảm độ dốc ngẫu nhiên theo lô nhỏ (SGD lô nhỏ). Thuật ngữ ngẫu nhiên đề cập đến thực tế là mỗi lô dữ liệu được rút ngẫu nhiên (ngẫu nhiên là từ đồng nghĩa khoa học của ngẫu nhiên). Hình 2.18 minh họa điều gì xảy ra trong 1D, khi mô hình chỉ có một tham số và bạn chỉ có một mẫu huấn luyện.


Bằng trực giác, chúng ta có thể thấy rằng điều quan trọng là chọn một giá trị hợp lý cho hệ số learning_rate.  Nếu nó quá nhỏ, việc đi xuống đường cong sẽ mất nhiều lần lặp lại và nó có thể bị kẹt ở mức tối thiểu cục bộ. Nếu learning_rate quá lớn, các cập nhật của bạn có thể đưa bạn đến các vị trí hoàn toàn ngẫu nhiên trên đường cong.


Lưu ý rằng một biến thể của thuật toán SGD lô nhỏ sẽ là vẽ một mẫu và mục tiêu duy nhất ở mỗi lần lặp, thay vì vẽ một loạt dữ liệu. Đây sẽ là SGD thực sự (ngược lại với SGD lô nhỏ). Ngoài ra, đi theo hướng ngược lại, bạn có thể chạy từng bước trên tất cả dữ liệu có sẵn, được gọi là giảm độ dốc hàng loạt. Mỗi bản cập nhật sau đó sẽ chính xác hơn nhưng đắt hơn nhiều. Sự thỏa hiệp hiệu quả giữa hai thái cực này là sử dụng các lô nhỏ có quy mô hợp lý.


Mặc dù hình 2.18 minh họa việc giảm độ dốc trong không gian tham số 1D, nhưng trong thực tế, bạn sẽ sử dụng việc giảm độ dốc trong không gian có nhiều chiều: mọi hệ số trọng số trong mạng nơ-ron là một chiều tự do trong không gian và có thể có hàng chục nghìn hoặc thậm chí hàng triệu chiều đó. Để giúp bạn xây dựng trực quan về các bề mặt mất mát, bạn cũng có thể trực quan hóa độ dốc giảm dần dọc theo bề mặt mất mát 2D, như trong hình 2.19. Nhưng


Giá trị tổn thất Điểm bắt đầu (t=0)


Tỷ lệ học tập


```python
t=1
```

```python
t=2
```

```python
t=3
```

## tham số


giá trị


Hình 2.18  SGD xuống đường cong tổn thất 1D (một tham số có thể học được)


45 Động cơ của mạng nơ-ron: Tối ưu hóa dựa trên độ dốc


bạn không thể hình dung được quá trình đào tạo mạng lưới thần kinh thực sự trông như thế nào—bạn không thể biểu diễn một không gian 1.000.000 chiều theo cách có ý nghĩa đối với con người. Vì vậy, bạn nên nhớ rằng trực giác mà bạn phát triển thông qua các biểu diễn chiều thấp này có thể không phải lúc nào cũng chính xác trong thực tế. Điều này trước đây từng là nguồn gốc của các vấn đề trong thế giới nghiên cứu học sâu.


Điểm xuất phát


Điểm cuối cùng


45 40 35 30 25 20 15 10 5 Hình 2.19  Độ dốc giảm xuống bề mặt mất 2D (hai tham số có thể học được)


Ngoài ra, tồn tại nhiều biến thể của SGD khác nhau bằng cách tính đến các cập nhật trọng lượng trước đó khi tính toán cập nhật trọng số tiếp theo, thay vì chỉ nhìn vào giá trị hiện tại của độ dốc. Ví dụ: có SGD có động lượng, cũng như Adagrad, RMSprop và một số loại khác. Các biến thể như vậy được gọi là phương pháp tối ưu hóa hoặc trình tối ưu hóa. Đặc biệt, khái niệm động lượng, được sử dụng trong nhiều biến thể này, đáng để bạn chú ý. Động lượng giải quyết hai vấn đề với SGD: tốc độ hội tụ và cực tiểu cục bộ. Hãy xem hình 2.20, biểu thị đường cong tổn thất là hàm của tham số mô hình.


Như bạn có thể thấy, xung quanh một giá trị tham số nhất định sẽ có mức tối thiểu cục bộ: xung quanh điểm đó, di chuyển sang trái sẽ dẫn đến tổn thất tăng lên, nhưng di chuyển sang phải cũng vậy. Nếu tham số đang được xem xét đang được tối ưu hóa thông qua SGD với tốc độ học nhỏ thì quá trình tối ưu hóa sẽ bị kẹt ở mức tối thiểu cục bộ thay vì đạt đến mức tối thiểu toàn cầu.


Bạn có thể tránh những vấn đề như vậy bằng cách sử dụng động lượng, lấy cảm hứng từ vật lý. Một hình ảnh hữu ích trong đầu ở đây là hãy nghĩ về quá trình tối ưu hóa như một quả bóng nhỏ lăn xuống đường cong tổn thất. Nếu có đủ


Giá trị tổn thất


## tham số


giá trị


Tối thiểu địa phương


Tối thiểu toàn cầu


Hình 2.20  Mức tối thiểu cục bộ và mức tối thiểu toàn cầu


46 Chương 2  Các khối xây dựng toán học của mạng lưới thần kinh


theo đà, quả bóng sẽ không bị mắc kẹt trong khe núi và sẽ dừng lại ở mức tối thiểu toàn cầu.  Động lượng được thực hiện bằng cách di chuyển quả bóng ở mỗi bước không chỉ dựa trên giá trị độ dốc hiện tại (gia tốc hiện tại) mà còn dựa trên vận tốc hiện tại (do gia tốc trong quá khứ). Trong thực tế, điều này có nghĩa là cập nhật tham số w không chỉ dựa trên giá trị gradient hiện tại mà còn dựa trên cập nhật tham số trước đó, chẳng hạn như trong cách triển khai đơn giản này:


```python
past_velocity = 0.0
momentum = 0.1                                                       
while loss > 0.01:                                                   
    w, loss, gradient = get_current_parameters()
    velocity = past_velocity * momentum - learning_rate * gradient
    w = w + momentum * velocity - learning_rate * gradient
    past_velocity = velocity
    update_parameter(w)
```

2.4.4 Đạo hàm chuỗi: Thuật toán lan truyền ngược


Trong thuật toán đã thảo luận trước đó, chúng ta đã giả định một cách ngẫu nhiên rằng vì một hàm khả vi nên chúng ta có thể dễ dàng tính toán độ dốc của nó. Nhưng điều đó có đúng không? Làm thế nào chúng ta có thể tính toán gradient của các biểu thức phức tạp trong thực tế? Trong ví dụ về mạng hai lớp của chúng ta, làm thế nào chúng ta có thể nhận được gradient của tổn thất theo trọng số? Đó là lúc thuật toán Backpropagation xuất hiện.


Quy tắc dây chuyền Lan truyền ngược là một cách sử dụng đạo hàm của các phép toán đơn giản (chẳng hạn như phép cộng,


relu, hoặc tích tensor) để dễ dàng tính toán gradient của các tổ hợp phức tạp tùy ý của các phép toán nguyên tử này. Điều quan trọng là mạng lưới thần kinh bao gồm nhiều phép toán tensor được xâu chuỗi lại với nhau, mỗi phép toán có một đạo hàm đơn giản đã biết. Ví dụ, mô hình từ ví dụ đầu tiên của chúng ta có thể được biểu diễn dưới dạng hàm được tham số hóa bởi các biến W1, b1, W2 và b2 (thuộc lớp dày đặc thứ nhất và thứ hai), liên quan đến các phép toán nguyên tử matmul, relu, softmax và +, cũng như hàm mất mát, mất mát của chúng ta, tất cả đều dễ dàng phân biệt được:


```python
loss_value = loss(
    y_true,
    softmax(matmul(relu(matmul(inputs, W1) + b1), W2) + b2),
)
```

```python
Calculus tells us that such a chain of functions can be derived using the following iden-
tity, called the chain rule. Consider two functions f and g, as well as the composed func-
tion fg such that y = fg(x) == f(g(x)):
```

Hệ số động lượng không đổi


Vòng lặp tối ưu hóa


47 Động cơ của mạng nơ-ron: Tối ưu hóa dựa trên độ dốc


```python
def fg(x):
    x1 = g(x)
    y = f(x1)
    return y
```

Khi đó, quy tắc dây chuyền nêu rõ rằng grad(y, x) == grad(y, x1) * grad(x1, x). Điều này cho phép bạn tính đạo hàm của fg miễn là bạn biết đạo hàm của f và g. Quy tắc chuỗi được đặt tên như thế này vì khi bạn thêm nhiều hàm trung gian hơn, nó sẽ bắt đầu trông giống như một chuỗi:


```python
def fghj(x):
    x1 = j(x)
    x2 = h(x1)
    x3 = g(x2)
    y = f(x3)
    return y
```

```python
grad(y, x) == grad(y, x3) * grad(x3, x2) * grad(x2, x1) * grad(x1, x)
```

Áp dụng quy tắc chuỗi để tính toán các giá trị gradient của mạng nơ-ron sẽ tạo ra một thuật toán gọi là lan truyền ngược. Hãy xem nó hoạt động như thế nào, cụ thể.


```python
Automatic differentiation with computation graphs
A useful way to think about backpropagation is in terms of computation graphs. A com-
putation graph is the data structure at the heart of the deep learning revolution. It’s 
a directed acyclic graph of operations—in our case, tensor operations. For instance, 
figure 2.21 is the graph representation of our first model.
```

Đồ thị tính toán là một sự trừu tượng cực kỳ thành công trong khoa học máy tính vì chúng cho phép chúng ta coi tính toán là dữ liệu: một biểu thức tính toán được mã hóa dưới dạng cấu trúc dữ liệu máy đọc được và có thể được sử dụng làm đầu vào hoặc đầu ra của một chương trình khác. Ví dụ: bạn có thể tưởng tượng một chương trình nhận biểu đồ tính toán và trả về biểu đồ tính toán mới triển khai phiên bản phân tán quy mô lớn của cùng một phép tính—điều này có nghĩa là bạn có thể phân phối bất kỳ phép tính nào mà không cần phải tự viết logic phân phối. Hoặc hãy tưởng tượng… một chương trình nhận một biểu đồ tính toán và có thể tự động tạo ra đạo hàm của biểu thức mà nó biểu diễn. Sẽ dễ dàng hơn nhiều để thực hiện những điều này nếu tính toán của bạn được thể hiện dưới dạng cấu trúc dữ liệu biểu đồ rõ ràng thay vì các dòng ký tự ASCII trong tệp .py.


Để giải thích rõ ràng về lan truyền ngược, chúng ta hãy xem một ví dụ thực sự cơ bản về biểu đồ tính toán. Chúng ta sẽ xem xét một phiên bản đơn giản của biểu đồ trong hình 2.21, trong đó chúng ta chỉ có một lớp tuyến tính và tất cả các biến đều là vô hướng, được hiển thị trong hình 2.22. Chúng ta sẽ lấy hai biến vô hướng w, b, đầu vào vô hướng x và áp dụng một số thao tác với chúng để kết hợp thành đầu ra y. Cuối cùng, chúng ta sẽ áp dụng hàm mất lỗi giá trị tuyệt đối: loss_val =


abs(y_true - y). Vì chúng ta muốn cập nhật w và b theo cách giảm thiểu loss_val, nên chúng ta quan tâm đến việc tính grad(loss_val, b) và grad(loss_val, w).


48 Chương 2  Các khối xây dựng toán học của mạng lưới thần kinh


dấu chấm


```python
+
```

x


## W1


b1


mất_val


relu


dấu chấm


```python
+
```

## W2


b2


softmax


mất y_true


Hình 2.21  Biểu diễn đồ thị tính toán của mô hình hai lớp của chúng tôi


Hãy đặt các giá trị cụ thể cho “nút đầu vào” trong biểu đồ—nghĩa là đầu vào x, y_true đích,


w và b (hình 2.23). Chúng tôi truyền các giá trị này đến tất cả các nút trong biểu đồ, từ trên xuống dưới, cho đến khi đạt loss_val. Đây là đường chuyền về phía trước.


Bây giờ chúng ta hãy “đảo ngược” đồ thị: với mỗi cạnh trong đồ thị đi từ A đến B, chúng ta sẽ tạo một cạnh đối diện từ B đến A và hỏi: “B thay đổi bao nhiêu khi A thay đổi?” Đó là, cái gì là


```python
*
```

```python
+
```

sự mất mát


mất_val


x1


x2


x


b


w


y_true


Hình 2.22  Một ví dụ cơ bản về đồ thị tính toán


```python
*
```

```python
+
```

sự mất mát


```python
loss_val = 3
```

2


```python
x1 = 6
```

```python
x2 = 7
```

x


b


1


thứ 3


4


y_true


Hình 2.23  Chạy tiến


49 Động cơ của mạng nơ-ron: Tối ưu hóa dựa trên độ dốc


tốt nghiệp (B, A)? Chúng ta sẽ chú thích từng cạnh đảo ngược bằng giá trị này (hình 2.24). Biểu đồ lùi này biểu thị đường chuyền lùi.


```python
*
```

```python
+
```

sự mất mát


mất_val


2


x1


```python
x2
grad(loss_val, x2) = 1
```

```python
grad(x2, x1) = 1
```

```python
grad(x1, w) = 2
```

```python
grad(x2, b) = 1
```

x


b


1


thứ 3


4


y_true


Hình 2.24  Chạy lùi


chúng tôi có


```python
- grad(loss_val, x2) = 1 because as x2 varies by an amount epsilon, loss_val =
```

abs(4 - x2) thay đổi theo cùng một lượng.


```python
- grad(x2, x1) = 1 because as x1 varies by an amount epsilon, x2 = x1 + b = x1 + 1
```

thay đổi một lượng như nhau.


```python
- grad(x2, b) = 1 because as b varies by an amount epsilon, x2 = x1 + b = 6 + b
```

thay đổi một lượng như nhau.


```python
- grad(x1, w) = 2 because as w varies by an amount epsilon, x1 = x * w = 2 * w
```

thay đổi 2 * epsilon.


```python
What the chain rule says about this backward graph is that you can obtain the deriv-
ative of a node with respect to another node by multiplying the derivatives for each edge 
along the path linking the two nodes. For instance, grad(loss_val, w) = grad(loss_val,
```

```python
x2) * grad(x2, x1) * grad(x1, w) (see figure 2.25).
```

Bằng cách áp dụng quy tắc dây chuyền vào biểu đồ của mình, chúng tôi có được những gì mình đang tìm kiếm:


```python
- grad(loss_val, w) = 1 * 1 * 2 = 2
```

```python
- grad(loss_val, b) = 1 * 1 = 1
```

50 Chương 2  Các khối xây dựng toán học của mạng lưới thần kinh


```python
*
```

```python
+
```

abs_diff


mất_val


2


x1


```python
x2
grad(loss_val, x2) = 1
```

```python
grad(x2, x1) = 1
```

```python
grad(x1, w) = 2
```

```python
grad(x2, b) = 1
```

x


b


1


thứ 3


4


y_true


Hình 2.25  Đường dẫn từ loss_val đến w trong đồ thị lùi


LƯU Ý  Nếu có nhiều đường dẫn liên kết hai nút quan tâm a, b trong biểu đồ ngược, chúng ta sẽ thu được grad(b, a) bằng cách tính tổng đóng góp của tất cả các đường dẫn.


Và với điều đó, bạn vừa thấy hoạt động lan truyền ngược! Lan truyền ngược đơn giản là việc áp dụng quy tắc dây chuyền vào biểu đồ tính toán. Không có gì hơn với nó.  Lan truyền ngược bắt đầu với giá trị tổn thất cuối cùng và hoạt động ngược từ lớp trên cùng xuống lớp dưới cùng, tính toán mức đóng góp của mỗi tham số trong giá trị tổn thất. Đó là nguồn gốc của cái tên “lan truyền ngược”: chúng tôi “truyền ngược” phần đóng góp tổn thất của các nút khác nhau trong biểu đồ tính toán.


Ngày nay, mọi người triển khai mạng lưới thần kinh trong các khuôn khổ hiện đại có khả năng phân biệt tự động, chẳng hạn như JAX, TensorFlow và PyTorch. Việc vi phân tự động được thực hiện bằng loại biểu đồ tính toán đã trình bày trước đó.  Phép vi phân tự động giúp có thể truy xuất độ dốc của các thành phần tùy ý của các phép toán tensor khả vi mà không cần thực hiện bất kỳ công việc bổ sung nào ngoài việc ghi lại đường chuyển tiếp. Khi tôi viết mạng nơ-ron đầu tiên của mình bằng C vào những năm 2000, tôi phải viết gradient bằng tay. Giờ đây, nhờ các công cụ phân biệt tự động hiện đại, bạn sẽ không bao giờ phải tự mình thực hiện lan truyền ngược. Hãy coi mình là người may mắn!


51 Nhìn lại ví dụ đầu tiên của chúng ta


2.5 Nhìn lại ví dụ đầu tiên của chúng ta


Bạn sắp kết thúc chương này và bây giờ bạn đã có hiểu biết chung về những gì đang diễn ra đằng sau hậu trường trong mạng lưới thần kinh. Chiếc hộp đen huyền diệu ở đầu chương đã trở thành một bức tranh rõ ràng hơn, như được minh họa trong hình 2.26: mô hình, bao gồm các lớp được xâu chuỗi lại với nhau, ánh xạ dữ liệu đầu vào tới các dự đoán. Sau đó, hàm mất mát sẽ so sánh các dự đoán này với các mục tiêu, tạo ra giá trị tổn thất: thước đo mức độ phù hợp của các dự đoán của mô hình với những gì được mong đợi. Trình tối ưu hóa sử dụng giá trị tổn thất này để cập nhật trọng số của mô hình.


Lớp (chuyển đổi dữ liệu)


## Đầu vào X


## Trọng lượng


Lớp (chuyển đổi dữ liệu)


## Dự đoán


Y' Cập nhật cân nặng


Mục tiêu thực sự


## Y


## Trọng lượng


Trình tối ưu hóa chức năng mất


Điểm thua


Hình 2.26  Mối quan hệ giữa mạng, các lớp, hàm mất mát và trình tối ưu hóa


Hãy quay lại ví dụ đầu tiên và xem lại từng phần của nó dựa trên những gì bạn đã học được trong các phần trước.


Đây là dữ liệu đầu vào:


```python
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype("float32") / 255
```

Bây giờ bạn hiểu rằng hình ảnh đầu vào được lưu trữ trong các tenxơ NumPy, ở đây được định dạng dưới dạng các tenxơ float32 có hình dạng (60000, 784) (dữ liệu huấn luyện) và (10000,


784) (dữ liệu thử nghiệm), tương ứng. Đây là mô hình của chúng tôi:


52 Chương 2  Các khối xây dựng toán học của mạng lưới thần kinh


```python
model = keras.Sequential(
    [
        layers.Dense(512, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ]
)
```

Bây giờ bạn hiểu rằng mô hình này bao gồm một chuỗi gồm hai lớp Mật độ cao, mỗi lớp áp dụng một vài phép toán tensor đơn giản cho dữ liệu đầu vào và các hoạt động này liên quan đến các tensor trọng số. Các tensor trọng số, là thuộc tính của các lớp, là nơi duy trì kiến ​​thức về mô hình.


Đây là bước biên dịch mô hình:


```python
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
```

Bây giờ bạn đã hiểu rằng "sparse_categorical_crossentropy" là hàm mất mát được sử dụng làm tín hiệu phản hồi để tìm hiểu các tensor trọng số mà giai đoạn huấn luyện sẽ cố gắng giảm thiểu. Bạn cũng biết rằng việc giảm tổn thất này xảy ra thông qua việc giảm độ dốc ngẫu nhiên theo lô nhỏ. Các quy tắc chính xác chi phối việc sử dụng cụ thể việc giảm độ dốc được xác định bởi trình tối ưu hóa "adam" được truyền làm đối số đầu tiên.


Cuối cùng, đây là vòng đào tạo:


```python
model.fit(
    train_images,
    train_labels,
    epochs=5,
    batch_size=128,
)
```

Bây giờ bạn đã hiểu điều gì xảy ra khi bạn gọi là phù hợp: mô hình sẽ bắt đầu lặp lại dữ liệu huấn luyện theo lô nhỏ gồm 128 mẫu, lặp lại 5 lần (mỗi lần lặp trên tất cả dữ liệu huấn luyện được gọi là một kỷ nguyên). Đối với mỗi lô, mô hình sẽ tính toán độ dốc của tổn thất đối với các trọng số (sử dụng thuật toán Lan truyền ngược, xuất phát từ quy tắc dây chuyền trong phép tính) và di chuyển các trọng số theo hướng sẽ làm giảm giá trị tổn thất cho lô này.


Sau 5 kỷ nguyên này, mô hình sẽ thực hiện 2.345 cập nhật độ dốc (469 mỗi kỷ nguyên) và độ mất mát của mô hình sẽ đủ thấp để mô hình có khả năng phân loại các chữ số viết tay với độ chính xác cao.


53 Nhìn lại ví dụ đầu tiên của chúng ta


Tại thời điểm này, bạn đã biết hầu hết những điều cần biết về mạng lưới thần kinh.  Hãy chứng minh điều đó bằng cách từng bước triển khai lại phiên bản đơn giản của ví dụ đầu tiên đó, chỉ sử dụng các thao tác cấp thấp.


2.5.1 Thực hiện lại ví dụ đầu tiên của chúng tôi từ đầu


Còn gì tốt hơn để thể hiện sự hiểu biết đầy đủ, rõ ràng hơn là thực hiện mọi thứ từ đầu? Tất nhiên, ý nghĩa “từ đầu” ở đây chỉ mang tính tương đối: chúng tôi sẽ không triển khai lại các phép toán tensor cơ bản và chúng tôi sẽ không triển khai lan truyền ngược. Nhưng chúng ta sẽ đi tới mức thấp đến mức mỗi bước tính toán sẽ được trình bày rõ ràng.


Đừng lo lắng nếu bạn chưa hiểu từng chi tiết nhỏ trong ví dụ này. Chương tiếp theo sẽ đi sâu vào chi tiết hơn về API Keras. Hiện tại, bạn chỉ cần cố gắng theo dõi ý chính của những gì đang diễn ra—mục đích của ví dụ này là giúp bạn đúc kết hiểu biết của bạn về toán học của deep learning bằng cách triển khai cụ thể. Đi thôi!


Một lớp Dense đơn giản Bạn đã biết trước đó rằng lớp Dense thực hiện phép biến đổi đầu vào sau, trong đó W và b là các tham số mô hình và kích hoạt là một hàm theo phần tử (thường là relu):


```python
output = activation(matmul(input, W) + b)
```

Hãy triển khai một lớp Python đơn giản NaiveDense để tạo hai biến Keras W và b, đồng thời hiển thị phương thức __call__() áp dụng phép biến đổi trước đó:


```python
import keras                                           
from keras import ops
```

```python
class NaiveDense:
    def __init__(self, input_size, output_size, activation=None):
        self.activation = activation
        self.W = keras.Variable(
            shape=(input_size, output_size), initializer="uniform"   
        )                                                             
        self.b = keras.Variable(shape=(output_size,), initializer="zeros")
```

```python
def __call__(self, inputs):                             
        x = ops.matmul(inputs, self.W)
        x = x + self.b
        if self.activation is not None:
            x = self.activation(x)
        return x
```

```python
keras.ops is where you will find all 
the tensor operations you need.
```

Tạo ma trận W có hình dạng


(input_size, out_size), được khởi tạo với các giá trị ngẫu nhiên được rút ra từ phân bố đồng đều


Tạo một vectơ b có hình dạng (output_size,), được khởi tạo bằng số 0


Áp dụng chuyển tiếp


54 Chương 2  Các khối xây dựng toán học của mạng lưới thần kinh


```python
@property
    def weights(self):                                    
        return [self.W, self.b]
```

```python
A simple Sequential class
Now, let’s create a NaiveSequential class to chain these layers. It wraps a list of layers and 
exposes a __call__() method that simply calls the underlying layers on the inputs, in 
order. It also features a weights property to easily keep track of the layers’ parameters:
```

```python
class NaiveSequential:
    def __init__(self, layers):
        self.layers = layers
```

```python
def __call__(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x
```

```python
@property
    def weights(self):
        weights = []
        for layer in self.layers:
            weights += layer.weights
        return weights
```

Sử dụng lớp NaiveDense này và lớp NaiveSequential này, chúng ta có thể tạo mô hình Keras giả:


```python
model = NaiveSequential(
    [
        NaiveDense(input_size=28 * 28, output_size=512, activation=ops.relu),
        NaiveDense(input_size=512, output_size=10, activation=ops.softmax),
    ]
)
assert len(model.weights) == 4
```

Trình tạo lô Tiếp theo, chúng ta cần một cách để lặp lại dữ liệu MNIST theo lô nhỏ. Điều này thật dễ dàng:


```python
import math
```

```python
class BatchGenerator:
    def __init__(self, images, labels, batch_size=128):
```

Phương pháp thuận tiện để lấy trọng số của lớp


55 Nhìn lại ví dụ đầu tiên của chúng ta


```python
assert len(images) == len(labels)
        self.index = 0
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.num_batches = math.ceil(len(images) / batch_size)
```

```python
def next(self):
        images = self.images[self.index : self.index + self.batch_size]
        labels = self.labels[self.index : self.index + self.batch_size]
        self.index += self.batch_size
        return images, labels
```

2.5.2 Chạy một bước huấn luyện


Phần khó khăn nhất của quy trình là “bước huấn luyện”: cập nhật trọng số của mô hình sau khi chạy nó trên một loạt dữ liệu. Chúng ta cần phải


- Tính toán các dự đoán của mô hình cho các ảnh trong batch


- Tính giá trị tổn thất cho những dự đoán này dựa trên nhãn thực tế


- Tính gradient của tổn thất theo trọng số của mô hình


- Di chuyển các trọng số một lượng nhỏ theo hướng ngược lại với gradient


Liệt kê 2.9  Một bước huấn luyện


```python
def one_training_step(model, images_batch, labels_batch):
    predictions = model(images_batch)                                  
    loss = ops.sparse_categorical_crossentropy(labels_batch, predictions)  
    average_loss = ops.mean(loss)
    gradients = get_gradients_of_loss_wrt_weights(loss, model.weights)   
    update_weights(gradients, model.weights)                             
    return loss
```

Bước cập nhật trọng lượng Như bạn đã biết, mục đích của bước “cập nhật trọng lượng” (thể hiện bằng


update_weights()) là di chuyển các trọng số “một chút” theo hướng sẽ giảm tổn thất trong lô này. Độ lớn của chuyển động được xác định bởi


Chạy “chuyển tiếp”


Tính toán độ dốc của tổn thất liên quan đến


trọng lượng. Đầu ra, gradient, là một danh sách trong đó mỗi mục tương ứng với một


```python
weight from the model.
weights list. We haven’t 
defined this function yet!
```

Cập nhật trọng số bằng cách sử dụng gradient.


Chúng tôi chưa xác định chức năng này!


56 Chương 2  Các khối xây dựng toán học của mạng lưới thần kinh


“tỷ lệ học tập”, thường là một số lượng nhỏ. Cách đơn giản nhất để thực hiện bản cập nhật này_


Hàm Weights() là trừ gradient * learning_rate khỏi mỗi trọng số:


```python
learning_rate = 1e-3
```

```python
def update_weights(gradients, weights):
    for g, w in zip(gradients, weights):
        w.assign(w - g * learning_rate)
```

Trong thực tế, bạn hầu như sẽ không bao giờ thực hiện bước cập nhật trọng lượng như thế này bằng tay.  Thay vào đó, bạn sẽ sử dụng phiên bản Trình tối ưu hóa từ Keras—như thế này:


```python
from keras import optimizers
```

```python
optimizer = optimizers.SGD(learning_rate=1e-3)
```

```python
def update_weights(gradients, weights):
    optimizer.apply_gradients(zip(gradients, weights))
```

Tính toán độ dốc Bây giờ, chúng ta vẫn còn thiếu một thứ: tính toán độ dốc (được biểu thị bằng hàm get_gradients_of_loss_wrt_weights() trong danh sách 2.9). Trong phần trước, chúng ta đã phác thảo cách chúng ta có thể sử dụng quy tắc dây chuyền để thu được gradient của một chuỗi các hàm dựa trên đạo hàm riêng của chúng, một quá trình được gọi là lan truyền ngược.  Chúng tôi có thể triển khai lại quá trình lan truyền ngược từ đầu, nhưng điều đó sẽ khá cồng kềnh, đặc biệt là vì chúng tôi đang sử dụng phép toán softmax và mất mát entropy chéo, có đạo hàm khá dài dòng.


Thay vào đó, chúng ta có thể dựa vào cơ chế phân biệt tự động được tích hợp trong một trong các khung cấp thấp được Keras hỗ trợ, chẳng hạn như TensorFlow, JAX hoặc PyTorch.  Để làm ví dụ, chúng ta hãy sử dụng TensorFlow tại đây. Bạn sẽ tìm hiểu thêm về TensorFlow, JAX và PyTorch trong chương tiếp theo.


```python
The API through which you can use TensorFlow’s automatic differentiation capabili-
ties is the tf.GradientTape object. It’s a Python scope that will “record” the tensor oper-
ations that run inside it, in the form of a computation graph (sometimes called a tape). 
This graph can then be used to retrieve the gradient of any scalar value with respect to 
any set of input values:
```

```python
import tensorflow as tf
```

```python
x = tf.zeros(shape=())
```

Gán một giá trị mới cho biến, tại chỗ


Khởi tạo một tenxơ vô hướng có giá trị 0


57 Nhìn lại ví dụ đầu tiên của chúng ta


```python
with tf.GradientTape() as tape:                     
    y = 2 * x + 3                                   
grad_of_y_wrt_x = tape.gradient(y, x)
```

Hãy viết lại hàm one_training_step() của chúng ta bằng cách sử dụng TensorFlow gradientTape (bỏ qua nhu cầu về hàm get_gradients_of_loss_wrt_weights() riêng biệt):


```python
def one_training_step(model, images_batch, labels_batch):
    with tf.GradientTape() as tape:
        predictions = model(images_batch)
        loss = ops.sparse_categorical_crossentropy(labels_batch, predictions)
        average_loss = ops.mean(loss)
    gradients = tape.gradient(average_loss, model.weights)
    update_weights(gradients, model.weights)
    return average_loss
```

Bây giờ bước đào tạo theo đợt của chúng tôi đã sẵn sàng, chúng tôi có thể chuyển sang triển khai toàn bộ giai đoạn đào tạo.


2.5.3 Vòng đào tạo đầy đủ


Một kỷ nguyên huấn luyện chỉ đơn giản bao gồm việc lặp lại bước huấn luyện cho từng đợt trong dữ liệu huấn luyện và vòng lặp huấn luyện đầy đủ chỉ đơn giản là sự lặp lại của một kỷ nguyên:


```python
def fit(model, images, labels, epochs, batch_size=128):
    for epoch_counter in range(epochs):
        print(f"Epoch {epoch_counter}")
        batch_generator = BatchGenerator(images, labels)
        for batch_counter in range(batch_generator.num_batches):
            images_batch, labels_batch = batch_generator.next()
            loss = one_training_step(model, images_batch, labels_batch)
            if batch_counter % 100 == 0:
                print(f"loss at batch {batch_counter}: {loss:.2f}")
```

Hãy lái thử nó:


```python
from keras.datasets import mnist
```

```python
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
```

```python
train_images = train_images.reshape((60000, 28 * 28))
```

Mở phạm vi gradientTape


Bên trong phạm vi, áp dụng một số phép toán tensor cho biến của chúng ta


Sử dụng băng để truy xuất độ dốc của đầu ra y đối với biến x của chúng tôi


58 Chương 2  Các khối xây dựng toán học của mạng lưới thần kinh


```python
train_images = train_images.astype("float32") / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype("float32") / 255
```

fit(model, train_images, train_labels, epochs=10, batch_size=128)


2.5.4 Đánh giá mô hình


Chúng ta có thể đánh giá mô hình bằng cách lấy argmax của các dự đoán của nó trên các hình ảnh thử nghiệm và so sánh nó với các nhãn dự kiến:


```python
>>> predictions = model(test_images)
>>> predicted_labels = ops.argmax(predictions, axis=1)
>>> matches = predicted_labels == test_labels
>>> f"accuracy: {ops.mean(matches):.2f}"
accuracy: 0.83
```

Tất cả đã xong! Như bạn có thể thấy, việc thực hiện “bằng tay” những gì bạn có thể làm trong một vài dòng mã Keras là khá tốn công. Nhưng vì bạn đã trải qua các bước này nên giờ đây bạn sẽ hiểu rõ ràng về những gì diễn ra bên trong mạng lưới thần kinh khi bạn gọi fit(). Việc có mô hình tinh thần cấp thấp này về những gì mã của bạn đang thực hiện ở hậu trường sẽ giúp bạn có thể tận dụng tốt hơn các tính năng cấp cao của API Keras.


2.6 Tóm tắt


- Tensors tạo thành nền tảng của hệ thống máy học hiện đại. Họ bước vào


nhiều hương vị khác nhau của dtype, cấp bậc và hình dạng.


- Bạn có thể thao tác các tensor số thông qua các phép toán tensor (chẳng hạn như phép cộng,


tích tensor hoặc phép nhân theo phần tử), có thể được hiểu là mã hóa các phép biến đổi hình học. Nói chung, mọi thứ trong deep learning đều có thể tuân theo cách giải thích hình học.


- Các mô hình deep learning bao gồm các chuỗi phép toán tensor đơn giản, tham số-


được xác định bằng các trọng số, bản thân chúng cũng là tensor. Trọng số của một mô hình là nơi lưu trữ “kiến thức” của nó.


- Học có nghĩa là tìm một tập giá trị cho các trọng số của mô hình để giảm thiểu


hàm mất mát cho một tập hợp mẫu dữ liệu huấn luyện nhất định và các mục tiêu tương ứng của chúng.


- Việc học diễn ra bằng cách vẽ các lô mẫu dữ liệu ngẫu nhiên và mục tiêu của chúng


và tính toán độ dốc của các tham số mô hình liên quan đến tổn thất trên lô. Các tham số mô hình sau đó được di chuyển một chút (độ lớn của


59 Tóm tắt


di chuyển được xác định bởi tốc độ học) theo hướng ngược lại với gradient.  Điều này được gọi là giảm độ dốc hàng loạt nhỏ.


- Toàn bộ quá trình học tập được thực hiện nhờ thực tế là tất cả các phép toán tensor


trong các mạng thần kinh có thể phân biệt được và do đó, có thể áp dụng quy tắc đạo hàm chuỗi để tìm hàm gradient ánh xạ các tham số hiện tại và lô dữ liệu hiện tại thành một giá trị gradient. Điều này được gọi là lan truyền ngược.


- Hai khái niệm chính mà bạn sẽ gặp thường xuyên trong các chương sau là mất mát và tối ưu hóa.


Đây là hai điều bạn cần xác định trước khi bắt đầu cung cấp dữ liệu vào mô hình: – Sự mất mát là số lượng bạn sẽ cố gắng giảm thiểu trong quá trình đào tạo, do đó, nó sẽ


đại diện cho thước đo thành công cho nhiệm vụ mà bạn đang cố gắng giải quyết. – Trình tối ưu hóa chỉ định cách chính xác độ dốc của tổn thất sẽ được thực hiện


được sử dụng để cập nhật các tham số: chẳng hạn, nó có thể là trình tối ưu hóa RMSProp, SGD có động lượng, v.v.


