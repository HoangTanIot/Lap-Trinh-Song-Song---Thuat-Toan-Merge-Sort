/**
 * @date 2025/05/02
 * @author Luong Huu Phuc
 */
#include <stdio.h>
#include "pseudo_num_gen.h"
#include <omp.h>
#include <stdlib.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif 

double pseudo_number_generate(long *arr, long n){
  double start_time = omp_get_wtime();

  //Ham tuyen tinh dong du (linear congruential function)
  #pragma omp parallel for 
  for(int i = 0; i < n; i++){
    arr[i] = (i * 37 + 19) % n; //Sinh ra so tu 0 -> n -1
  }

  double end_time = omp_get_wtime();
  return (end_time - start_time); //Tra ve thoi gian gen ra so 
}

double fisher_yates_shuffle(int *arr, int n){
  double start_time = omp_get_wtime();

  #pragma omp parallel for 
  for(int i = 0; i < n; i++){
    arr[i] = i;
  }

  srand((unsigned int)time(NULL));

  //Fisher-Yates shuffle chay tuan tu de tranh race condition 
  for(int i = n - 1; i > 0; i--){
    int j = rand() % (i + 1);
    //Hoan doi arr[i] va arr[j]
    int temp = arr[i];
    arr[i] = arr[j];
    arr[j] = temp;
  }

  double end_time = omp_get_wtime();
  return (end_time - start_time);
}

#ifdef __cplusplus
}
#endif 
