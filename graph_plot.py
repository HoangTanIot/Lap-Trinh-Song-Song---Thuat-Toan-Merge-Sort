import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

# === Tìm tất cả các file CSV trong thư mục hiện tại ===
file_paths = [r"D:\C-C++_project\Project_2024-2\Merge_sort_omp\merge_omp_task\Task_omp_merge_sort.csv",
              r"D:\C-C++_project\Project_2024-2\Merge_sort_omp\merge_sections\Sections_merge_sort.csv",
              r"D:\C-C++_project\Project_2024-2\Merge_sort_sequential\Sequential_merge_sort.csv"]  
# nếu muốn lọc riêng, sửa tên ví dụ: "Task_omp_merge_sort_*.csv"

plt.figure(figsize=(12, 6))

# === Vẽ từng file một ===
for file_path in file_paths:
    try:
        df = pd.read_csv(file_path)  # Đọc file CSV

        # Kiểm tra cột dữ liệu có đủ không
        if 'Lan' not in df.columns or 'thoi gian(s)' not in df.columns:
            print(f"[!] Bỏ qua {file_path} - Không có cột 'Lan' hoặc 'thoi gian(s)'!")
            continue

        # Lấy tên file (bỏ .csv) làm chú thích
        label_name = os.path.splitext(os.path.basename(file_path))[0]

        # Vẽ đường biểu diễn
        plt.plot(df['Lan'], df['thoi gian(s)'], marker='x', label=label_name)

    except Exception as e:
        print(f"Lỗi khi đọc {file_path}: {e}")

# === Tùy chỉnh biểu đồ ===
plt.xlabel("Lần chạy (Run)", fontsize=12)
plt.ylabel("Thời gian thực thi (s)", fontsize=12)
plt.title("So sánh thời gian thực thi", fontsize=14)
plt.legend(title="Tên file:", fontsize=10)
plt.grid(True)
plt.tight_layout()

# === Hiển thị hoặc lưu ảnh ===
plt.savefig("Project_2024-2/compare_execution_time.png", dpi=300)  # Lưu file ảnh
plt.show()
