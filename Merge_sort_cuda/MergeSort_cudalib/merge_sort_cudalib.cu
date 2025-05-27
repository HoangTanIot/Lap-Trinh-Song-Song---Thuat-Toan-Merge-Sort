/**
 * @file merge_sort_cudalib.cu
 * @date 2025/04/05
 * @author Luong Huu Phuc 
 * @copyright JoeyOhman
 * \anchor https://github.com/JoeyOhman/GPUMergeSort
 */
#include <iostream>
#include <algorithm>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "neces_lib.cuh"
#include "merge_sort_cudalib.cuh"
#include "Cuda_check_err.cu"

//Ham kiem tra xem thu tu phan tu da chuan chua
__host__ bool isSorted(long *arr, long n){
  for(int i = 1; i < n; i++){
    // if(arr[i] != i){
    //   return 0;
    // }
    if(arr[i - 1] > arr[i]){
      return 0;
    }
  }
  return 1;
}

/************************** VERSION 1 ***************************/

__device__ int binarySearch(long *arr, int val, int left, int right){
  if(right <= left){ //Khi index right == index left (con 1 phan tu)
    return (val > arr[left] ? (left + 1) : left); //Neu gia tri can tim (val) lon hon gia tri left
  }
  int mid = (left + right) / 2;
  if(val > arr[mid]){ //Neu gia tri can tim lon hon gia tri o giua 
    return binarySearch(arr, val, mid + 1, right); //Bo het mang ben trai (be hon val), de quy tiep mang ben phai
  }
  return binarySearch(arr, val, left, mid); //Khong thi bo het mang ben phai (lon hon val), de quy tiep mang ben trai
}

__device__ int getIndex(long *subAux, int ownIndex, int nLeft, int nTot){
  int scanIndex; //Bien luu gioi han duoi cho binary search
  int upperBound; //Bien luu gioi han tren cho binary search 
  bool isLeft = ownIndex < nLeft; //Xem xem phan tu hien tai thuoc mang ben nao (xet index cua subAux)

  if(isLeft){ //Dung => Phan tu thuoc mang trai => tim xem bao nhieu phan tu ben PHAI nho hon no
    scanIndex = nLeft; //la chi so dau tien cua mang phai 
    upperBound = nTot; //Gioi han tren (diem cuoi cua mang phai)
  }else{ //Neu phan tu thuoc mang phai => tim xem bao nhieu phan tu ben TRAI nho hon no
    scanIndex = 0; //la chi so dau tien cua mang trai
    upperBound = nLeft; //sau cuoi mang trai
  }

  //Tim kiem bang Binary Search cho 2 nua TRAI hoac PHAI do 
  scanIndex = binarySearch(subAux, subAux[ownIndex], scanIndex, upperBound - 1); //Tra ve so luong phan tu nho hon tu phia con lai
  return ownIndex + scanIndex - nLeft; //Tru di nLeft vi mang subAux dang duoc danh chi so tu [0]
} 

__global__ void mergeKernel(long *arr, long *aux, int left, int mid, int right){
  /**
   * @brief Tinh so thread trong gird cua CUDA, moi thread phu trach 1 phan tu tu aux[left..]
   * @param blockIdx Chi so block moi grid (0 -> gridDim.x - 1)
   * @param blockDim So thread moi block 
   * @param threadIdx Chi so thread hien tai trong block do (0 -> blockDim.x - 1)
   */
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  int nLeft = mid - left + 1; //Xac dinh vi tri ket thuc cua mang ben trai (= So luong phan tu mang trai)
  int nRight = right - mid; //Xac dinh vi tri bat dau cua mang ben phai (= So luong phan tu mang phai)
  int nTot = nLeft + nRight; //Tong so phan tu cua ca 2 mang 

  //Neu thread co chi so vuot ngoai tong so phan tu thi dung lai - khong lam gi ca
  if(idx >= nTot){
    return;
  }

  //Xac dinh phan tu aux[left + idx] dang thuoc mang con trai hay phai
  //Sau do se thuc hien tim kiem nhi phan trong mang con lai de dem xem co bao nhieu phan tu nho hon no
  //Nham xac dinh vi tri chinh xac sau khi merge roi push vao trong mang arr cuoi cung
  int arrIndex = getIndex(&aux[left], idx, nLeft, nTot);
  arr[left + arrIndex] = aux[left + idx]; //Ghi phan tu vao dung vi tri trong mang arr
}

__host__ __device__ void merge(long *arr, long *aux, int left, int mid, int right){
  int i = 0;
  int j = 0;
  int mergeIndex = left;
  int nLeft = mid - left + 1; //So phan tu mang ben trai
  int nRight = right - mid; //So phan tu mang ben phai

  while(i < nLeft && j < nRight){
    if(aux[left + i] < aux[mid + 1 + j]){ //Neu phan tu nao mang ben trai < be hon ben phai
      arr[mergeIndex] = aux[left + i]; //Day vao mang arr 
      i++;
    }else{
      arr[mergeIndex] = aux[mid + 1 + j]; //Khong thi day phan tu ben phai vao mang
      j++;
    }
    mergeIndex++;
  }

  while(i < nLeft){
    arr[mergeIndex] = aux[left + i];
    i++;
    mergeIndex++;
  }

  while(j < nRight){
    arr[mergeIndex] = aux[mid + 1 + j];
    j++;
    mergeIndex++;
  }
}

__global__ void mergeSort_coordinate(long *arr, long *aux, int currentSize, int n, int afterSize){
  int idx = blockIdx.x * blockDim.x + threadIdx.x; 

  int left = idx * afterSize; //Chi so bat dau cua doan dang xet
  /**
   * Thread thu idx se thuc hien gop 2 mang con bat dau tu phan tu `left`
   * Moi mang con dai currentSize, nen thread nay se merge doan tu `left` den `left + afterSize - 1`
   * Gia su: currentSize = 2 -> afterSize = currentSize * 2 = 4 (sau khi gop mang 2 phan tu thi se duoc mang co do dai 4 phan tu), idx = 3
   * => left = idx * afterSize = 3 * 4 = 12;
   * => mid = left + currentSize - 1 = 12 + 2 - 1 = 13
   * => right = left + afterSize - 1 = 12 + 4 - 1 = 15
   * -> Thread SO 3 se merge 2 doan con:
   *  * Mang con trai: arr[12], arr[13]
   *  * Mang con phai: arr[14], arr[15]
   */

  //Neu left nam ngoai pham vi mang
  if(left >= n - currentSize || left < 0){
    return;
  }

  int mid = left + currentSize - 1; //Ket thuc cua mang con ben trai
  int right = min_local(left + afterSize - 1, n - 1); //Ket thuc cua mang con ben phai

  int nTot = right - left + 1; //So threads duoc sinh ra (Tong so phan tu trong doan [left, right])

  if(nTot > MERGE_PARALLEL_THRESHOLD){ //Neu so luong phan tu can merge lon hon nguong de co the song song kernel ma khong gay overhead
    int numThreadsPerBlock = 1024; //1024 thread moi block
    int numBlocks = (nTot + numThreadsPerBlock - 1) / numThreadsPerBlock; //So blocks duoc sinh ra theo so phan tu mang

    mergeKernel <<< numBlocks, numThreadsPerBlock >>>(arr, aux, left, mid, right);
    cudaCheckErrorDev(); //Check loi sau moi lan goi kernel
  }else{ //Neu phan tu nho hon nguong quy dinh (de tranh overhead)
    merge(arr, aux, left, mid, right);
  }
}

void mergeSortGPU(long *arr, int n){
  //Hai mang trong device (GPU)
  long *deviceArr; //Mang chua phan tu sau khi da sap xep xong (cuoi cung)
  long *auxArr; //Mang trung gian luu ket qua cua tung buoc hop nhat

  //Uu tien su dung cache L1 hon shared memory -> Toi uu hieu suat cho kernel nho
  cudaSafeCall(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
  cudaSafeCall(cudaMalloc((void**)&deviceArr, n * sizeof(long)));
  cudaSafeCall(cudaMalloc((void**)&auxArr, n * sizeof(long)));
  cudaSafeCall(cudaMemcpy(deviceArr, arr, n * sizeof(long), cudaMemcpyHostToDevice)); //Sao chep du lieu tu mang arr(host - CPU) vao deviceArr(GPU)

  //currentSize: kich thuoc hien tai cua mang con dang merge
  //Duyet qua cac mang con co kich thuoc currentSize = 1, 2, 4, 8,...n
  //Moi lan lap se merge cac doan co do dai currentSize, sau moi vong lap so phan tu trong mang se tang len 2 lan
  for(int currentSize = 1; currentSize < n; currentSize *= 2){
    //Tinh toan tham so kernel
    int afterSize = currentSize * 2; //Bien luu tong do dai cua 2 mang con sau khi duoc merge (2 -> 4 -> 8,...)
    int numSorts = (n + afterSize - 1) / afterSize; //So luong sorting thread sinh ra (so merge can thuc hien)
    int numThreadsPerBlock = 32; 
    int numBlocks = (numSorts + numThreadsPerBlock - 1) / numThreadsPerBlock;

    cudaSafeCall(cudaMemcpy(auxArr, deviceArr, n * sizeof(long), cudaMemcpyDeviceToDevice)); //Truoc khi hop nhat copy du lieu tu deviceArr sang auxArr
    //Trong moi vong lap currentSize, no goi kernel mergeSort de xu ly nhieu doan nho song song 
    //Trong kernel mergeSort, neu doan mang du lon thi ta tiep tuc goi 1 kernel con mergeKernel() de song song hoa qua trinh merge tung phan tu trong doan do
    mergeSort_coordinate <<< numBlocks, numThreadsPerBlock >>> (deviceArr, auxArr, currentSize, n, afterSize);
    cudaDeviceSynchronize(); //__host__ function
    cudaCheckError();
  }

  //Sau khi sap xep xong thi tra ve arr tren CPU 
  cudaSafeCall(cudaMemcpy(arr, deviceArr, n * sizeof(long), cudaMemcpyDeviceToHost));

  cudaSafeCall(cudaFree(deviceArr));
  cudaSafeCall(cudaFree(auxArr));
}

/************************** VERSION 2 ***************************/

__device__ unsigned int getIndex_kernel(dim3 *threads, dim3 *blocks){
  int x;
  return (threadIdx.x + 
          threadIdx.y * (x = threads->x) + 
          threadIdx.z * (x *= threads->y) + 
          blockIdx.x * (x *= threads->z) + 
          blockIdx.y * (x *= blocks->z) +
          blockIdx.z * (x *= blocks->y));
}

__device__ void gpu_bottomUpMerge_ver2(long *arr, long *aux, long left, long mid, long right){
  long i = left;
  long j =  mid;
  for(long k = left; k < right; k++){
    if(i < mid && (j >= right || arr[i] < arr[j])){
      aux[k] = arr[i];
      i++;
    }else{
      aux[k] = arr[j];
      j++;
    }
  }
}

__global__ void gpu_mergeSort_ver2(long *arr, long *aux, long n, long afterSize, long slices, dim3 *threads, dim3 *blocks){
  unsigned int idx = getIndex_kernel(threads, blocks);
  long left = afterSize * idx * slices, mid, right;

  for(long slice = 0; slice < slices; slice++){
    if(left >= n){
      break;
    }
    mid = min_local(left + (afterSize >> 1), n);
    right = min_local(left + afterSize, n);
    gpu_bottomUpMerge_ver2(arr, aux, left, mid, right);
    left += afterSize;
  }
}

void mergeSortGPU_ver2(long *arr, long n, dim3 threadsPerBlock, dim3 blocksPerGrid){
  long *deviceArr;
  long *auxArr;
  dim3 *deviceThreads;
  dim3 *deviceBlocks;

  // tm();

  cudaSafeCall(cudaMalloc((void**)&deviceArr, n * sizeof(long)));
  cudaSafeCall(cudaMalloc((void**)&auxArr, n * sizeof(long)));
  cudaSafeCall(cudaMemcpy(deviceArr, arr, n * sizeof(long), cudaMemcpyHostToDevice));

  cudaSafeCall(cudaMalloc((void**)&deviceThreads, sizeof(dim3)));
  cudaSafeCall(cudaMalloc((void**)&deviceBlocks, sizeof(dim3)));
  cudaSafeCall(cudaMemcpy(deviceThreads, &threadsPerBlock, sizeof(dim3), cudaMemcpyHostToDevice));
  cudaSafeCall(cudaMemcpy(deviceBlocks, &blocksPerGrid, sizeof(dim3), cudaMemcpyHostToDevice));

  long *A = deviceArr;
  long *B = auxArr;

  long nThreads = threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z * 
                 blocksPerGrid.x * blocksPerGrid.y * blocksPerGrid.z;

  for(int afterSize = 2; afterSize < (n << 1); afterSize <<= 1){
    long slices = n / ((nThreads) * afterSize) + 1;

    gpu_mergeSort_ver2 <<< blocksPerGrid, threadsPerBlock >>>(A, B, n, afterSize, slices, deviceThreads, deviceBlocks);

    A = A == deviceArr ? auxArr : deviceArr;
    B = B == deviceArr ? auxArr : deviceArr;
  }

  // tm();
  cudaSafeCall(cudaMemcpy(arr, A, n * sizeof(int), cudaMemcpyDeviceToHost));

  cudaSafeCall(cudaFree(deviceArr));
  cudaSafeCall(cudaFree(auxArr));
}