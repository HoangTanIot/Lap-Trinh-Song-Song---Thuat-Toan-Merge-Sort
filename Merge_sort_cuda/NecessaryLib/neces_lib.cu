/**
 * @author Luong Huu Phuc
 * @file neces_lib.cpp
 */
#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "neces_lib.cuh"

cudaEvent_t start_time = nullptr;
cudaEvent_t end_time = nullptr;
float elasped_time = 0.0f;
int count = 0;
std::chrono::high_resolution_clock::time_point tStart;

__host__ void startTimer(void){
  cudaEventCreate(&start_time);
  cudaEventCreate(&end_time);
  cudaEventRecord(start_time, 0);
}

__host__ double elaspedTimer(void){
  cudaEventRecord(end_time, 0);
  cudaEventSynchronize(end_time); //Dam bao kernel da chay xong
  
  cudaEventElapsedTime(&elasped_time, start_time, end_time); //thoi gian tinh bang milliseconds
  cudaEventDestroy(start_time);
  cudaEventDestroy(end_time);

  return elasped_time;
}

__host__ void swapPointers(long **a, long **b){
  long *temp = *a; 
  *a = *b; 
  *b = temp;
}

void printArray(long *arr, int n){
  for(int i = 0; i < n; i++){
     std::cout << arr[i] << " ";
     count++;
     if(count == 20){
      std::cout << std::endl;
      count = 0;
     }
  }
}

int tm(void){
  static bool initialized = false;
  if(!initialized){
    tStart = std::chrono::high_resolution_clock::now();
    initialized = true;
    return 0;
  }

  auto tEnd = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(tEnd - tStart).count();
  tStart = tEnd;

  return static_cast<int>(duration); //Tra ve microseconds
}