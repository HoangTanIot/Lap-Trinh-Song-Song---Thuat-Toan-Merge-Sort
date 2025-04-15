#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "omp.h"

void shuffle(int *arr, int n){
  if(n > 1){
    for(int i = 0; i < n; i++){

    }  
  }
}

void random_number_generate(int a[], int n, int min, int max){
  srand(time(NULL));

  for(int i = 0; i < n; i++){
    a[i] = rand() % (max - min + 1) + min;
  }
}

void merge(int arr[], int left, int mid, int right) {
  int i, j, k;
  int n1 = mid - left + 1;
  int n2 = right - mid;

  int *L = (int*)malloc(n1 * sizeof(int));
  int *R = (int*)malloc(n2 * sizeof(int));

  for (i = 0; i < n1; i++){
    L[i] = arr[left + i];
  }
   
  for (j = 0; j < n2; j++){
    R[j] = arr[mid + 1 + j];
  }

  i = 0; j = 0; k = left;

  while (i < n1 && j < n2){
    if (L[i] <= R[j]) {
      arr[k++] = L[i++];
    } else {
      arr[k++] = R[j++];
    }
  }

  while (i < n1){
    arr[k++] = L[i++];
  }

  while (j < n2){
    arr[k++] = R[j++];
  }

  free(L);
  free(R);
}

void mergeSort(int arr[], int left, int right) {
  if (left < right) {
    int mid = left + (right - left) / 2;
    mergeSort(arr, left, mid);
    mergeSort(arr, mid + 1, right);
    merge(arr, left, mid, right);
  }
}

void printArray(int arr[], int size) {
  for (int i = 0; i < size; i++)
    printf("%d ", arr[i]);
  printf("\n");
}

int main() {
  int num, min, max;
  int count = 0;
  printf("Nhap so luong phan tu cua mang: ");
  scanf("%d", &num);

  // printf("Nhap gia tri min, max: ");
  // scanf("%d %d", &min, &max);

  // Cấp phát mảng động
  int *arr = (int*)malloc(num * sizeof(int));
  if (arr == NULL) {
    printf("Khong the cap phat bo nho!\n");
    return 1; 
  }

  for(int i = 0; i < num; i++){
    arr[i] = i;
  }
  srand(time(NULL));


  printf("Mang truoc khi sap xep: \n");
  for(int i = 0; i < num; i++){
    printf("%d ", arr[i]);
    count++;
    if(count == 10){
      printf("\n");
      count = 0;
    }
  }

  double start_time = omp_get_wtime();

  mergeSort(arr, 0, num - 1);

  double end_time = omp_get_wtime();

  printf("\nSorted Array: \n");
  for(int i = 0; i < num; i++){
    printf("%d ", arr[i]);
    count++;
    if(count == 10){
      printf("\n");
      count = 0;
    }
  }

  printf("\nThoi gian sau khi sap xep: %.10f", end_time - start_time);

  free(arr);  // Giải phóng bộ nhớ
  return 0;
}
