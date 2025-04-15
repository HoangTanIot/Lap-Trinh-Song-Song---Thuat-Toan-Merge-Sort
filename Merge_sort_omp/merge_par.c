#include <stdio.h>
#include <stdlib.h>
#include "omp.h"
#include "time.h"

void merge_sort(int a[], int i, int j);
void merge(int a[], int i1, int j1, int i2, int j2);
void random_number_generate(int a[], int n, int min, int max);

void shuffle(int *array, int n) {
  if (n > 1) {
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int t = array[j];
        array[j] = array[i];
        array[i] = t;
    }
  }
}

void random_number_generate(int a[], int n, int min, int max){
  srand(time(NULL));

  for(int i = 0; i < n; i++){
    a[i] = rand() % (max - min + 1) + min;
  }
}

void merge_sort(int a[], int i, int j){
  int mid;

  if(i < j){
    mid = (i + j) / 2;

    #pragma omp parallel sections //Tao ra nhieu thread the thuc thi khoi lenh dong thoi
    {
      #pragma omp section //Moi section gan cho 1 thread
      {
        //Thread 1
        merge_sort(a, i, mid); //Left recursion
      }

      #pragma omp section
      {
        //Thread 2
        merge_sort(a, mid + 1, j); //right recursion
      }
    }
    merge(a, i, mid, mid + 1, j);
  }
}

void merge(int a[], int i1, int j1, int i2, int j2){
  int temp[1000];    //array used for merging
  int i = i1;    //beginning of the first list
  int j = i2;    //beginning of the second list
  int k=0;
  
  while(i <= j1 && j <= j2){
    if(a[i] < a[j]){
      temp[k++] = a[i++];
    }
    else{
      temp[k++] = a[j++];
    }
  }
  
  while(i <= j1){ //copy remaining elements of the first list
    temp[k++] = a[i++];
  }    
      
  while(j <= j2){  //copy remaining elements of the second list
    temp[k++] = a[j++];
  } 

  //Transfer elements from temp[] back to a[]
  for(i = i1, j = 0;i <= j2; i++, j++)
    a[i] = temp[j];
}

int main(void){
  int num, min, max;
  int count = 0;
  printf("Nhap so phan tu can sap xep: ");
  scanf("%d", &num);

  // printf("Nhap gia tri min, max: ");
  // scanf("%d %d", &min, &max);

  int *a = (int*)malloc(sizeof(int) * num);
  for(int i = 0; i < num; i++){
    a[i] = i;
  }
  srand(time(NULL));
  shuffle(a, num);
  
  // printf("Mang truoc khi sap xep: \n");
  // for(int i = 0; i < num; i++){
  //   printf("%d ", a[i]);
  //   count++;
  //   if(count == 10){
  //     printf("\n");
  //     count = 0;
  //   }
  // }

  double start_time = omp_get_wtime();

  merge_sort(a, 0, num - 1);

  double end_time = omp_get_wtime();

  // printf("\nSorted array: \n");
  // for(int i = 0; i < num; i++){
  //   printf("%d ", a[i]);
  //   count++;
  //   if(count == 10){
  //     printf("\n");
  //     count = 0;
  //   }
  // }

  printf("\nThoi gian thuc hien: %.10f", end_time - start_time);

  free(a);
  return 0;
}