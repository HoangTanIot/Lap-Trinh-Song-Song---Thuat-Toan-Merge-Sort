#ifndef PSEUDO_NUM_GEN_H
#define PSEUDO_NUM_GEN_H

#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif 

//Chay tren CPU nen khong can __host__ va day la file.h 
double pseudo_number_generate(long *arr, long n);

//Fisher-Yates Shuffle dam bao cac so tu 0 -> n - 1 duy nhat va xao tron ngau nhien hoan toan
double fisher_yates_shuffle(int *arr, int);

#ifdef __cplusplus
}
#endif 

#endif //PSEUDO_NUM_GEN_H