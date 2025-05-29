# **Để phân tích hiệu suất của chương trình CUDA, sử dụng `NSIGHT COMPUTE` và `NSIGHT SYSTEMS`** #
### NSIGHT COMPUTE và NSIGHT SYSTEMS là gì ? ###
1. `NSIGHT COMPUTE`(`nsys`)
- Là công cụ giúp bạn phân tích hiệu suất toàn bộ pipeline chương trình CUDA, nó theo dõi: 
  * Thời gian thực thi của kernel
  * Hoạt động sao chép bộ nhớ giữa *host* và *device*
  * Sự đồng bộ giữa CPU và GPU 
  * Thời gian thực thi trên CPU
 
**👉 Lệnh sử dụng:** <br>
Tổng quan hiệu suất kernel:
```bash
nsys profile --stats=true ./your_program.exe
```
![image](https://github.com/user-attachments/assets/01e9ce81-db05-47e9-bff3-bb22f8adc79c)

2. `NSIGHT COMPUTE` (`ncu`)
- Là công cụ để phân tích chi tiết kernel CUDA. Nó cung cấp thông tin về: 
 * Truy cập bộ nhớ (memory accesses)
 * Occupancy (mức độ tận dụng tài nguyên GPU). Nó đánh giá khả năng ẩn latency và khai thác phần cứng 
 * Cache miss, register usage và thời gian thực thi từng dòng lệnh.
> #### **Occupancy là gì ?** ####
> * Occupancy là tỷ lệ phần trăm số warp đang họat động (active warp) trên mỗi SM so với tổng số warp tối đa mà SM đó có thể chứa `Occupancy = (Số warp đang hoạt đông / Số warp trên mỗi SM) x 100%
> * Occupancy quan trọng vì khi nó cao, sẽ giúp che giấu độ trễ (latency) của: 
>  - Truy cập bộ nhớ (global memory, DRAM)
>  - Thao tác tính toán bị phụ thuộc 
> * SM không idle -> Hiệu năng tổng thể cao hơn 
> * Giả sử: SM chứa tối đa 64 warp, nhưng kernel của bạn chỉ cho phép 32 warp chạy đồng thời. Dẫn đến Occupancy = 32 / 64 = 50%
> *Nhưng Occupancy không phải cứ càng cao càng tốt !* <br>
> 50-80% là mức tốt, tùy theo loại kernel
#### **Yếu tố ảnh hưởng đến Occupancy** ####
|Yếu tố  |     Ảnh hưởng |
|--------|---------------|
|Số register mỗi thread | Dùng nhiều -> ít thread fit vào SM |
|Shared memory mỗi block | Dùng nhiều -> ít block chạy cùng lúc | 
| Threads/block | Cấu hình quá thấp hoặc quá cao đều ảnh hưởng | 
| Kernel đồng bộ nhiều (`__syncthread()`) | Làm SM chờ đợi -> giảm hiệu quả thực tế | 

**👉 Lệnh sử dụng:**
```bash
ncu ./your_program.exe 
```
Ngoài ra còn một số lệnh để lọc phần phân tích:
| Mục tiêu                 | Lệnh sử dụng               | 
|--------------------------|----------------------------|
| Đo hiệu suất truy cập bộ nhớ | `ncu --section MemoryWorkloadAnalysis  .\your_program.exe` |
| Kiểm tra occupancy kernel | `ncu --section SpeedOfLight .\your_program.exe` | 
> Lưu ý, lệnh này chỉ dùng được khi mở quyền Admin cho command prompt hay PowerShell
![image](https://github.com/user-attachments/assets/f2ea7d79-32f1-45b6-89fe-e95fcad501ff)


