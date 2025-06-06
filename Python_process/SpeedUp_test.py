import matplotlib.pyplot as plt

# Dữ liệu
threads = [50_000, 500_000, 5_000_000, 15_000_000, 30_000_000, 50_000_000]
speedups = [0.906, 3.546, 11.274, 12.013, 12.174, 11.033]

# Ve duong y = x
ideal_speedup = [threads[i] / threads[0] * speedups[0] for i in range(len(threads))]

# Vẽ biểu đồ
plt.figure(figsize=(10, 6))
plt.plot(threads, speedups, marker='o', linestyle='-', color='blue', linewidth=2, markersize=8)

# plt.plot(threads, ideal_speedup, linestyle='--', color='gray', linewidth=2, label='y = x')

plt.title('Biểu đồ Speedup của Merge Sort song song (GPU) so với tuần tự (CPU)', fontsize=14)

plt.xlabel('Số phần tử', fontsize=12)
plt.ylabel('Speedup (Lần)', fontsize=12)
plt.grid(True)

plt.xticks(threads, [f'{x:,}' for x in threads])  # Hiển thị ngăn cách hàng nghìn
plt.yticks(range(0, int(max(speedups)) + 5, 5))
plt.tight_layout()
plt.legend()

plt.savefig("D:/C-C++_project/Project_2024-2/Compare_png/SpeedUp_1024mk_32ms.png", dpi=300)
plt.show()
