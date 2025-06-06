import pandas as pd
import matplotlib.pyplot as plt
import os

# === Các file tương ứng của từng phương pháp ===
sequential_files = [
    r"D:\C-C++_project\Project_2024-2\Merge_sort_sequential\Time_test\e_100k.csv",
    r"D:\C-C++_project\Project_2024-2\Merge_sort_sequential\Time_test\e_1mil.csv",
    r"D:\C-C++_project\Project_2024-2\Merge_sort_sequential\Time_test\e_10mil.csv",
    r"D:\C-C++_project\Project_2024-2\Merge_sort_sequential\Time_test\e_60mil.csv",
    r"D:\C-C++_project\Project_2024-2\Merge_sort_sequential\Time_test\e_100mil.csv"

]

cuda_files = [
    r"D:\C-C++_project\Project_2024-2\Merge_sort_cuda\main\Time_test\100k\e_100k_1024mk_32ms.csv",
    r"D:\C-C++_project\Project_2024-2\Merge_sort_cuda\main\Time_test\1mil\e_1mil_1024mk_32ms.csv",
    r"D:\C-C++_project\Project_2024-2\Merge_sort_cuda\main\Time_test\10mil\e_10mil_1024mk_32ms(1).csv",
    r"D:\C-C++_project\Project_2024-2\Merge_sort_cuda\main\Time_test\60mil\e_60mil_1024mk_32ms.csv",
    r"D:\C-C++_project\Project_2024-2\Merge_sort_cuda\main\Time_test\100mil\1024mk_32ms\e_100mil_1024mk_32ms(1).csv"
]

omp_task_files = [
    r"D:\C-C++_project\Project_2024-2\Merge_sort_omp\merge_omp_task\Time_test\e_100k.csv",
    r"D:\C-C++_project\Project_2024-2\Merge_sort_omp\merge_omp_task\Time_test\e_1mil.csv",
    r"D:\C-C++_project\Project_2024-2\Merge_sort_omp\merge_omp_task\Time_test\e_10mil.csv",
    r"D:\C-C++_project\Project_2024-2\Merge_sort_omp\merge_omp_task\Time_test\e_60mil.csv",
    r"D:\C-C++_project\Project_2024-2\Merge_sort_omp\merge_omp_task\Time_test\e_100mil.csv"

]

# Tên mốc số lượng phần tử
array_sizes = ["100,000", "1,000,000", "10,000,000", "60,000,000", "100,000,000"]

def extract_average(file_list):
    times = []
    for file in file_list:
        try:
            with open(file, 'r') as f:
                lines = f.readlines()
                last_line = lines[-1]
                if 'Average' in last_line:
                    avg_value = float(last_line.strip().split(',')[1])
                    times.append(avg_value)
                else:
                    print(f"[!] Không tìm thấy 'Average' trong file: {file}")
                    times.append(None)
        except Exception as e:
            print(f"Lỗi khi đọc file {file}: {e}")
            times.append(None)
    return times

# Lấy giá trị trung bình
sequential_avg = extract_average(sequential_files)
sections_avg = extract_average(cuda_files)
omp_task_avg = extract_average(omp_task_files)

# === Vẽ biểu đồ ===
plt.figure(figsize=(10,6))
plt.plot(array_sizes, sequential_avg, marker='o', label='Sequential', color="green")
plt.plot(array_sizes, sections_avg, marker='s', label='Cuda', color="blue")
plt.plot(array_sizes, omp_task_avg, marker='^', label='Omp Task', color="red")

plt.xlabel("Số lượng phần tử")
plt.ylabel("Thời gian trung bình (s)")
plt.title("So sánh thời gian thực thi Merge Sort")
plt.legend(title="Phương pháp:")
plt.grid(True)
plt.tight_layout()

# Lưu và hiển thị
plt.savefig("D:/C-C++_project/Project_2024-2/Compare_png/Compare_AverTime_1024mk_32ms.png", dpi=300)
plt.show()
