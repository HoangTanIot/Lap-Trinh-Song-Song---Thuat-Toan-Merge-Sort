import pandas as pd
import matplotlib.pyplot as plt
import os

# === Danh sách file và số lượng phần tử mảng ===
file_info = {
            r"D:\C-C++_project\Project_2024-2\Merge_sort_sequential\Time_test\e_100k.csv": 100_000,
            r"D:\C-C++_project\Project_2024-2\Merge_sort_sequential\Time_test\e_1mil.csv": 1_000_000,
            r"D:\C-C++_project\Project_2024-2\Merge_sort_sequential\Time_test\e_10mil.csv": 10_000_000,
            r"D:\C-C++_project\Project_2024-2\Merge_sort_sequential\Time_test\e_100mil.csv": 100_000_000,
            r"D:\C-C++_project\Project_2024-2\Merge_sort_sequential\Time_test\e_500mil.csv": 500_000_000
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
                avg_time = float(last_line.split(',')[1])
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
plt.title("Thời gian thực thi Merge Sort (OMP) theo kích thước mảng", fontsize=14)
plt.grid(True)
plt.legend()
plt.xscale('log')
plt.xticks(x_values, [f"{int(x):,}" for x in x_values])

plt.tight_layout()
plt.savefig("Project_2024-2/Compare_png/Average_time_sequential.png", dpi=300)
plt.show()