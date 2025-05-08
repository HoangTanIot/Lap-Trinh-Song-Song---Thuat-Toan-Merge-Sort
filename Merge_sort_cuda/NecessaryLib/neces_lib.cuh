#ifndef NECESS_LIB_HPP
#define NECESS_LIB_HPP

#include <iostream>
#include <cuda_runtime.h>

extern cudaEvent_t start_time, end_time;
extern float elasped_time;
extern int count;

#ifdef __cplusplus
extern "C"{
#endif

__host__ void startTimer(void);

/**
 * @return Tra ve thoi gian milliseconds
 */
__host__ double elaspedTimer(void);

/**
 * @brief Ham nay nhan 2 con tro bac 2 
 * @param a la con tro tro den con tro kieu int 
 * @param b la con tro tro den con tro kieu int 
 */
void swapPointers(int **a, int **b);

void printArray(long *arr, int n);

int tm(void);

#ifdef __cplusplus
}
#endif 

#endif //NECESS_LIB_HPP