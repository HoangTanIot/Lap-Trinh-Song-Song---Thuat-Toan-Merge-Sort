#include <stdio.h>

#define CUDA_ERROR_CHECK

#define cudaCheckErrorDev()  __cudaCheckErrorDev(__FILE__, __LINE__)
#define cudaSafeCall(err)    __cudaSafeCall(err, __FILE__, __LINE__)
#define cudaCheckError()     __cudaCheckError(__FILE__, __LINE__)

#ifdef __cplusplus
extern "C"{
#endif 

__device__ inline void __cudaCheckErrorDev(const char *file, const int line){
  #ifdef CUDA_ERROR_CHECK
    cudaError err = cudaGetLastError();
    if(cudaSuccess != err){
      printf("%s %s %d\n", cudaGetErrorString(err), file, line);
    }

    // err = cudaDeviceSynchronize(); //Ham nay chi chay duoc tren CPU (host)
    // if(cudaSuccess != err){
    //   printf("%s %s %d\n", cudaGetErrorString(err), file, line);
    // }
  #endif 
}

__host__ inline void __cudaSafeCall(cudaError err, const char *file, const int line){
  #ifdef CUDA_ERROR_CHECK
    if(cudaSuccess != err){
      fprintf(stderr, "cudaSafeCall() failed at %s:%i :%s\n", file, line, cudaGetErrorString(err));
      exit(-1); 
    } 
  #endif
}               

__host__ inline void __cudaCheckError(const char *file, const int line){
  #ifdef CUDA_ERROR_CHECK
    cudaError err = cudaGetLastError();
    if(cudaSuccess != err){
      fprintf(stderr, "cudaCheckError() failed at: %s:%i: %s\n", file, line, cudaGetErrorString(err));
      exit(-1);
    }
    err = cudaDeviceSynchronize();
    if(cudaSuccess != err){
      fprintf(stderr, "cudaCheckError() with sync failed at: %s:%i : %s\n", file, line, cudaGetErrorString(err));
      exit(-1);
    }
  #endif
}

#ifdef __cplusplus
}
#endif
