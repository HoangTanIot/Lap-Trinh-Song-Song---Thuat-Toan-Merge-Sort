import pandas as pd
import matplotlib.pyplot as plt
import os

# === Danh sách file và số lượng phần tử mảng ===
file_info = {
            r"D:\C-C++_project\Project_2024-2\Merge_sort_cuda\main\Time_test\100k\e_100k_128mk_16ms.csv": 100_000,
            r"D:\C-C++_project\Project_2024-2\Merge_sort_cuda\main\Time_test\1mil\e_1mil_128mk_16ms.csv": 1_000_000,
            r"D:\C-C++_project\Project_2024-2\Merge_sort_cuda\main\Time_test\10mil\e_10mil_128mk_16ms.csv": 10_000_000,
            r"D:\C-C++_project\Project_2024-2\Merge_sort_cuda\main\Time_test\60mil\e_60mil_128mk_16ms.csv":60_000_000,
            r"D:\C-C++_project\Project_2024-2\Merge_sort_cuda\main\Time_test\100mil\128mk_16ms\e_100mil_128mk_16ms(1).csv": 100_000_000
}

x_values = []
y_means = []

# === Đọc giá trị từ dòng Average ===
for file_path, array_size in file_info.items():
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            last_line = lines[-1]

            if last_line.startswith('Average'):
                avg_time = float(last_line.strip().split(',')[1])
                x_values.append(array_size)
                y_means.append(avg_time)

                print(f"[+] {os.path.basename(file_path)}: Thời gian trung bình {avg_time:.4f} giây")
            else:
                print(f"[!] File {file_path} không có dòng Average hợp lệ!")

    except Exception as e:
        print(f"[Lỗi] {file_path}: {e}")

# === Vẽ biểu đồ ===
plt.figure(figsize=(10, 6))
plt.plot(x_values, y_means, marker='o', linestyle='-', color='b', label='Thời gian trung bình')

plt.xlabel("Số lượng phần tử mảng", fontsize=12)
plt.ylabel("Thời gian trung bình (s)", fontsize=12)
plt.title("Thời gian thực thi CUDA theo kích thước mảng", fontsize=14)
plt.grid(True)
plt.legend()
plt.xscale('log')
plt.xticks(x_values, [f"{int(x):,}" for x in x_values])

plt.tight_layout()
plt.savefig("D:/C-C++_project/Project_2024-2/Compare_png/AverTime_128mk_16ms.png", dpi=300)
plt.show()