/**
 * @brief Bottom-Up merge sort on GPU 
 * @author Luong Huu Phuc
 * @copyright Joey Ohman
 */
#include <iostream>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "pseudo_num_gen.h"
#include "neces_lib.cuh"
#include "merge_sort_cudalib.cuh"

//Version nay chay duoc 10 trieu phan tu -> 100tr bi crash
double benchmarkGPU(long *arr, int n){
  startTimer();
  mergeSortGPU(arr, n);
  return (elaspedTimer() / 1000.0f); //Tra ve giay
}

//Version nay chi chay duoc 1 trieu phan tu la max 
double benchmarkGPU_ver2(long *arr, long n, dim3 threads, dim3 blocks){
  startTimer();
  mergeSortGPU_ver2(arr, n, threads, blocks);
  return (elaspedTimer() / 1000.0f);
}

int main(void){
  long num;
  const int numSorts = 15;
  double num_gen_time, total_time = 0.0f;

  // dim3 threadsPerBlock, blocksPerGrid;

  // threadsPerBlock.x = 32;
  // threadsPerBlock.y = 1;
  // threadsPerBlock.z = 1;

  // blocksPerGrid.x = 8;
  // blocksPerGrid.y = 1;
  // blocksPerGrid.z = 1;

  std::cout << "Nhap kich thuoc mang: ";
  std::cin >> num;
  std::cin.ignore();

  long *arr = new long[num];
  if(arr == NULL){
    std::cerr << "Mang cap phat khong thanh cong !" << std::endl;
    return 1;
  }

  //Tao folder Time_test neu chua ton tai
  std::filesystem::create_directories("D:/C-C++_project/Project_2024-2/Merge_sort_cuda/main/Time_test");

  std::ofstream outFile("D:/C-C++_project/Project_2024-2/Merge_sort_cuda/main/Time_test/result.csv");
  if(!outFile.is_open()){
    std::cerr << "Khong the mo file de ghi !" << std::endl;
    delete[] arr;
    return 1;
  }
  outFile << "Lan,thoi_gian(s),thoi_gian_sinh_so(s)" << "\n";

  for(int i = 0; i < numSorts; i++){
    num_gen_time = pseudo_number_generate(arr, num);

    double duration = benchmarkGPU(arr, num);
    total_time += duration;

    bool check = isSorted(arr, num);
    if(!check){
      std::cerr << "Lan " << i + 1 << ": check Failed !" << std::endl;
      std::cout << "Error array: " << std::endl;
      printArray(arr, num);
      outFile << "Lan " << i + 1 << ": check Failed !" << "\n";
      delete[] arr;
      outFile.close();
      return 1;
    }

    //In ket qua ra man hinh + luu vao file (tgian thuc thi + tgian tao mang)
    std::cout << "Lan "<< i + 1 << ": " << duration << " ms, check: OK !" << std::endl;
    std::cout << "Number generate time: " << num_gen_time << " seconds" << std::endl;
    outFile << i + 1 << "," << std::fixed << std::setprecision(10) << duration << ","
                            << std::fixed << std::setprecision(10) << num_gen_time << "\n";
  }

  double average_time = total_time / numSorts;
  std::cout << "Thoi gian trung binh: " << average_time << std::endl;
  outFile << "Average," << std::fixed << std::setprecision(10) << average_time << "\n";

  delete[] arr;
  outFile.close();
  return 0;
}

/**
 * @note Loi khi chay 100 trieu phan tu la do so luong lon khien so lan goi kernel qua nhieu gay tran bo nho
 */