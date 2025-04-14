Merge Sort Song Song Sử Dụng MPI
Triển khai Merge Sort bằng MPI
Tóm tắt
Mặc dù Merge Sort đã được nghiên cứu kỹ trong lý thuyết thuật toán song song, nhưng việc triển khai thực tế với các nền tảng lập trình song song phổ biến như MPI vẫn chưa được nghiên cứu đầy đủ. Việc hiểu rõ hơn về cách song song hóa thuật toán Merge Sort sẽ góp phần cải thiện hiểu biết tổng thể về phương pháp "chia để trị" (Divide-and-Conquer) trong môi trường song song. Trong tài liệu này, tôi khảo sát cả hai phiên bản Merge Sort: tuần tự (serial) và song song (parallel), đặc biệt là với giao tiếp bằng cách gửi/nhận thông điệp trong MPI. Qua các thí nghiệm, so sánh thời gian thực thi giữa hai phiên bản cho thấy phiên bản song song đạt được hiệu suất cao hơn rõ rệt.

1. Thuật Toán Merge Sort
Merge Sort tuân theo nguyên lý “chia để trị”. Tuy nhiên, nó không chỉ chia mảng thành hai phần mà chia thành N mảng con, mỗi mảng chỉ chứa một phần tử (vì một phần tử đã được xem là đã sắp xếp). Sau đó, các mảng con này được gộp dần lại theo từng bước để tạo thành mảng được sắp xếp cuối cùng. Merge Sort có thời gian thực thi là O(n log n) và là một thuật toán ổn định (các phần tử bằng nhau vẫn giữ nguyên thứ tự ban đầu sau khi sắp xếp).

Hàm Merge Sort sẽ đệ quy sắp xếp dãy con từ array[p..r]. Khi kích thước của dãy con là 0 hoặc 1 thì không cần làm gì vì đã được sắp xếp. Nếu không, ta chia tiếp và gọi merge(array, p, q, r) để gộp hai dãy con đã sắp xếp lại.

Việc trộn cần tạo ra các mảng tạm thời để lưu hai dãy con trước khi gộp vào array[p..r] vì không thể ghi đè khi chưa lưu lại giá trị cũ.

2. Cài Đặt Merge Sort Tuần Tự Trong Môi Trường MPI
2.1. Hàm sort()
Hàm này nhận một mảng, chỉ số đầu và cuối. Tính chỉ số giữa để chia mảng thành hai phần và tiếp tục gọi đệ quy cho đến khi chỉ còn một phần tử. Khi hai phần nhỏ đã được sắp xếp, ta gọi hàm merge() để gộp chúng lại.

2.2. Hàm merge()
Hàm này nhận hai mảng đã sắp xếp cùng kích thước và trả về một mảng đã sắp xếp. Ba biến chỉ mục (fi, si, mi) dùng để duyệt từng mảng. So sánh phần tử đầu mỗi mảng, phần tử nhỏ hơn sẽ được đưa vào mảng M. Khi một trong hai mảng hết phần tử, các phần tử còn lại của mảng kia sẽ được đưa vào mảng M giữ nguyên thứ tự, vì chúng chắc chắn lớn hơn những phần đã được đưa vào. Cuối cùng, chia mảng M thành hai phần mới thay thế mảng đầu vào ban đầu.

2.3. Hàm main()
Hàm chính khởi tạo dữ liệu và dùng hàm clock() để đo thời gian chạy.

3. Cài Đặt Merge Sort Song Song
3.1. Chia Dữ Liệu Đều
Ban đầu chia mảng đều cho các processor, tuy nhiên có thể không chia đều được nên phải xử lý phần dư. Lấy thương + 1 để đảm bảo mỗi processor có cùng số phần tử và gán thêm các phần tử 0 cho đủ. Sau đó, sinh dữ liệu ngẫu nhiên có 3 chữ số bằng cách lấy rand() % 1000.

3.2. Sắp Xếp Dữ Liệu
Dùng MPI_Bcast để gửi kích thước của mảng con đến các processor. Dùng MPI_Scatter để gửi mảng con cho từng processor. Sau đó gọi hàm sort() trên mỗi processor.

3.3. Gộp Dữ Liệu
Sử dụng chiến lược gộp theo cấu trúc cây nhị phân. Processor có rank chia hết cho step sẽ nhận dữ liệu từ processor có rank không chia hết và gộp với mảng của mình. Step tăng theo lũy thừa của 2 để lần lượt gộp các mảng lại.

4. Đo Thời Gian
Dùng hàm clock() để đo thời gian thực thi giữa hai phiên bản Merge Sort: tuần tự và song song.