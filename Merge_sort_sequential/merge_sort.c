#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "omp.h"

void print_arr(int *arr, int n){
  int count = 0;
  for(int i = 0; i < n; i++){
    printf("%d ", arr[i]);
    count++;
    if(count == 10){
      printf("\n");
      count = 0;
    }
  }
}

void random_number_generate(int a[], int n, int min, int max){
  srand(time(NULL));
  for(int i = 0; i < n; i++){
    a[i] = rand() % (max - min + 1) + min;
  }
}

//Ham tao gia tri "gia ngau nhien" song song, nhanh va an toan hon
void pseudo_number_generator(int *a, int n){
  #pragma omp parallel for
  for(int i = 0; i < n; i++){
    a[i] = (i * 37 + 19) % n;  
  }
}

void merge(int arr[], int left, int mid, int right) {
  int i, j, k;
  int size1 = mid - left + 1;
  int size2 = right - mid;

  int *L = (int*)malloc(size1 * sizeof(int));
  int *R = (int*)malloc(size2 * sizeof(int));

  for (i = 0; i < size1; i++){
    L[i] = arr[left + i];
  }
   
  for (j = 0; j < size2; j++){
    R[j] = arr[mid + 1 + j];
  }

  i = 0; j = 0; k = left; 

  while (i < size1 && j < size2){
    if (L[i] <= R[j]) {
      arr[k++] = L[i++];
    } else {
      arr[k++] = R[j++];
    }
  }

  while (i < size1){
    arr[k++] = L[i++];
  }

  while (j < size2){
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

int main() {
  int num;
  int count = 0;
  printf("Nhap so luong phan tu cua mang: ");
  scanf("%d", &num);

  // Cấp phát mảng động
  int *arr = (int*)malloc(num * sizeof(int));
  if (arr == NULL) {
    printf("Khong the cap phat bo nho!\n");
    return 1; 
  }

  FILE *output_file = fopen("Sequential_merge_sort.csv", "w");
  if(output_file == NULL){
    printf("Khong mo duoc file de ghi !\n");
    free(arr);
    return 1;
  }
  fprintf(output_file, "Lan,thoi gian(s)\n");
  double tong = 0.0f;
  
  for(int run = 1; run <= 15; run++){
    pseudo_number_generator(arr, num); //Reset mang

    double start_time = omp_get_wtime();
    
    mergeSort(arr, 0, num - 1);

    double end_time = omp_get_wtime();

    double elapsed = end_time - start_time;
    tong += elapsed;

    printf("Lan %2d: %.10f\n", run, elapsed);
    fprintf(output_file, "%d,%.10f\n", run , elapsed);
  }

  double aver = tong / 15.0f;
  printf("Thoi gian trung binh: %.10f", aver);
  fprintf(output_file, "Average,%.10f\n", aver);

  fclose(output_file);
  free(arr); 
  return 0;
}
