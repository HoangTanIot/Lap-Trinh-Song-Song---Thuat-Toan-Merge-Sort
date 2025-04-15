/**
 * @author Luong Huu Phuc
 * @date 2025/04/15
 */
#include <stdio.h>
#include <stdlib.h>
#include "omp.h"
#include "time.h"

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

//Ham tron mang tuan tu, song song kho (race condition)
void shuffle_number_generator(int *arr, int n){
  #pragma omp parallel for 
  for(int i = 0; i < n; i++){
    arr[i] = i;
  }
  srand(time(NULL));
  if (n > 1) {
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int t = arr[j];
        arr[j] = arr[i];
        arr[i] = t;
    }
  }
}

//Ham nay tao so ngau nhien tuan tu, song song se rat kho 
void random_number_generator(int a[], int n, int min, int max){
  srand(time(NULL));
  for(int i = 0; i < n; i++){
    a[i] = rand() % (max - min + 1) + min;  
  }
}

//Ham tao gia tri "gia ngau nhien" song song, nhanh va an toan hon
void pseudo_number_generator(int *a, int n){
  #pragma omp parallel for //Song song hoa duoc vi khong phu thuoc du lieu  
  for(int i = 0; i < n; i++){
    a[i] = (i * 37 + 19) % n; //Cong thuc tinh nhanh va phan phoi deu
  }
}

void merge(int arr[], int left, int mid, int right) {
  int i, j, k;
  int size1 = mid - left + 1;
  int size2 = right - mid;

  int *L = (int*)malloc(size1 * sizeof(int));
  int *R = (int*)malloc(size2 * sizeof(int));

  //Ham for nay khong song song duoc (?)
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

/**
 * \note Han che dung "#pragma omp parallel sections" vi voi so luong n lon, de quy se bi long song song vo tan 
 * \note Voi n = 1tr phan tu, de quy sau hang chuc ngan cap, moi cap lai tao 1 thread moi, may tinh se het ram va cpu
 * \note Moi lan merge_sort() goi, lai sinh ra thread, moi thread lai tiep tuc sinh ra thread de chay de quy
 * \note Thread no tung: CPU het RAM, he dieu hanh kill process ngay lap tuc -> Crash nhung khong bao loi
 * @return "#pragma omp parallel sections" dung trong de quy de crash neu khong co kiem soat neu size nho !
 */
void merge_sort(int a[], int left, int right){
  if(left < right){
    int mid = (left + right) / 2;

    //Neu mang du nho thi sort truc tiep (tranh tao thread lung tung)
    //Song song den 10000 phan tu thi dung, neu khong co if thi 100% bi crash 
    if(right - left <= 10000){
      merge_sort(a, left, mid);
      merge_sort(a, mid + 1, right);
    }else{
      #pragma omp parallel sections //Tao ra nhieu thread the thuc thi khoi lenh dong thoi
      {
        #pragma omp section //Thread 1
        merge_sort(a, left, mid); //De quy cho nua ben trai
  
        #pragma omp section //Thread 2
        merge_sort(a, mid + 1, right); //De quy cho nua ben phai
      }
    }
    merge(a, left, mid, right); //Hop nhat 2 nua lai
  }
} 

int main(void){
  int num;
  double tong = 0.0f;
  printf("Nhap so phan tu can sap xep: ");
  scanf("%d", &num);
  
  int *a = (int*)malloc(sizeof(int) * num);
  if(a == NULL){
    printf("Khong the cap phat bo nho!\n");
    return 1;
  } 

  FILE *output_file = fopen("Time_test/e_10mil.csv", "w");
  if(output_file == NULL){
    printf("Khong the mo file de ghi !\n");
    free(a);
    return 1;
  }
  fprintf(output_file, "Lan,thoi gian(s)\n");

  //Thay doi so core xu ly
  omp_set_num_threads(8); 

  for(int run = 1; run <= 15; run++){
    pseudo_number_generator(a, num); //Cu mot lan run tang thi mang lai reset

    double start_time = omp_get_wtime();

    merge_sort(a, 0, num - 1);

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
  free(a);
  return 0;
}