# Vì sao CUDA không hỗ trợ đệ quy hiệu quả ? #
***
Thứ nhất, hãy hiểu về GPU core khác thế nào với CPU core
* Một CPU thường có 4-16 cores hiệu năng cao 
 - CPU core thì mạnh mẽ hơn, có thể điều khiển hệ thống, xử lý nhiều kiểu lệnh phức tạp 
* Trong khi đó 1 GPU hiện đại (ví dụ NVIDIA) có thể có: 
 - Hàng trăm đến hàng nghìn "CUDA cores" (mini core)
 - CUDA cores là đơn vị rất nhỏ, đơn giản, chuyên làm tác vụ tính toán song song nhẹ
 - Ví dụ: NVIDIA RTX 3050 (đang dùng) có 2560 cores 
 - GPU gôm nhiều SM (Stream Multiprocessor):
  * Mỗi SM chứa hàng chục đến hàng trăm CUDA cores
 - Khi bạn chạy CUDA kernel:
  * Hàng nghìn luồng (threads) được chia ra chạy trên các CUDA cores theo nhóm một
  * Các threads được tổ chức thành wrap (32 threads), phải chạy đồng bộ cùng lúc và cùng thực hiện 1 thao tác trên nhiều dữ liệu nên NẾU một thread rẽ nhánh để đệ quy còn thread khác thì không -> Sẽ gây ra hiện tượng phân kỳ (divergence) -> Giảm hiệu suất GPU 
* Ngoài ra GPU phù hợp hơn với các bài toán tách sẵn được dữ liệu, chia đều -> Không có phụ thuộc 
***
➡️ Đây cũng chính là lý do mà GPU không tối ưu cho cho đệ quy 
### Giải thích ###
***1. GPU không được thiết kế cho đệ quy sâu***
  * Đệ quy cần stack call cho từng mức gọi hàm 
  * Trên CPU, stack được cấp phát thoải mái 
  * Trên GPU, mỗi luồng chỉ có stack nhỏ (thường 1-2KB mặc định), không phù hợp để gọi đệ quy sâu (gọi lồng nhau nhiều lần)
  * Việc gọi đệ quy nhiều cấp sẽ nhanh chóng hết stack, gây lỗi hoặc bị chặn bởi compiler <br>
***2. Luồng GPU không hiệu quả khi rẽ nhánh (branching)***
  * Trong thuật toán đệ quy, mỗi nhãnh đi theo hướng khác nhau: `left`, `right`, `merge`,...
  * Nếu nhiều luồng CUDA chạy `mergeSort(left)` trong khi các luồng khác chạy `mergeSort(right)`, ta có divergence (phân kỳ)- GPU phải chạy tuần tự từng nhánh, mất hiệu suất
  * GPU chỉ chạy hiệu quả nhất khi nhiều luồng cùng làm một việc tại một thời điểm (*SIMT model: Single Instruction, Multiple Threads*) <br>
***3. Không phải kiến trúc GPU nào cũng hỗ trợ đệ quy***
  * Một số GPU cũ (Compute Capability < 2.0) không hỗ trợ recursion
  * Các GPU mới (>=2.0) hỗ trợ device-side recursion nhưng: 
    - Phải bật `-rdc=true` (relocatable device code)
    - Gây tăng thời gian compile và giảm hiệu suất
    - Vẫn bị giới hạn bởi stack nếu gọi đệ quy sâu

# Vậy ta dùng giải pháp gì ? #
👉 ***Dùng thuật toán Bottom-Up Merge Sort***
* Không cần đệ quy 
* Mỗi bước chia mảng thành nhiều đoạn nhỏ cố định (width), rồi merge song song
* Tối ưu cho GPU vì: 
 - Không cần stack
 - Không có rẽ nhánh phức tạp
 - Dễ ánh xạ lên mỗi thread xử lý 1 đoạn con.

***Ý tưởng chính***
* Không chia đệ quy nữa
* Thay vào đó ta: 
 1. Chia mảng thành các đoạn nhỏ cố định (ví dụ mỗi đoạn 2 phần tử)
 2. Sử dụng mỗi luồng (thread) để merge một cặp đoạn con đó 
 3. Sau đó tăng độ rộng mỗi đoạn con (width *= 2) và lặp lại
|                   Input                   |
|-------------------------------------------|
|      8 3 1 9 1 2 7 5 9 3 6 4 2 0 2 5      |
|                                           |
|  Thread1 | Thread2  | Thread3  | Thread4  |
|----------|----------|----------|----------|
| 8 3 1 9  | 1 2 7 5  | 9 3 6 4  | 2 0 2 5  |
|  38 19   |  12 57   |  39 46   |  02 25   |
|---------------------|---------------------|
|       Thread1       |        Thread2      |
|   1398       1257   |   3469       0225   |
|       11235789      |       02234569      |
|-------------------------------------------|
|                  Thread1                  |
|      0 1 1 2 2 2 3 3 4 5 5 6 7 8 9 9      |
|                                           |

***🧠 Ví dụ minh họa***
Giả sử mảng có 8 phần tử: <br>
```text
arr = [7, 3, 5, 2, 9, 1, 6, 8]
``` 
* Vòng 1 (width = 1):
 - Luồng 0 merge [7] và [3] -> [3,7]
 - Luồng 1 merge [5] và [2] -> [2,5]
 - ...
* Vòng 2 (width = 2):
 - Luồng 0 merge [3,7] và [2,5] -> [2,3,5,7]
 - ...
* Vòng 3 (width = 4):
 - Luồng 0 merge [2,3,5,7] và [1,6,8,9] -> Kết quả cuối 
