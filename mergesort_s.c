/* Serial Merge Sort */
#include <stdio.h>
#include <time.h>

#define N 1000000


int * merge(int *First, int Fsize, int *Second, int Ssize);
void sort(int *Arr, int start, int end);

int * merge(int *First, int Fsize, int *Second, int Ssize) {
	int fi = 0, si = 0, mi = 0, i;
	int *merged;
	int Msize = Fsize+Ssize;

	merged = (int *)malloc(Msize*sizeof(int));
	while ((fi < Fsize) && (si < Ssize)) {
		if (First[fi] <= Second[si]) {
			merged[mi] = First[fi];
			mi++; fi++;
		} else {
			merged[mi] = Second[si];
			mi++; si++;
		}
	}

	if (fi >= Fsize)
		for (i = mi; i < Msize; i++, si++)
			merged[i] = Second[si];
	else if (si >= Ssize)
		for (i = mi; i < Msize; i++, fi++)
			merged[i] = First[fi];

	for (i = 0; i < Fsize; i++)
		First[i] = merged[i];
	for (i = 0; i < Ssize; i++)
		Second[i] = merged[Fsize+i];

	return merged;
}

void sort(int *Arr, int start, int end)
{
	int *sortedArr;
	int mid = (start+end)/2;
	int leftCount = mid - start + 1;
	int rightCount = end - mid;

	/* If the range consists of a single element, it's already sorted */
	if (end == start) {
		return;
	} else {
		/* sort the left half */
		sort(Arr, start, mid);
		/* sort the right half */
		sort(Arr, mid+1, end);
		/* merge the two halves */
		sortedArr = merge(Arr + start, leftCount, Arr + mid + 1, rightCount);
	}
}

main(int argc, char **argv)
{
	int * data;
	int n=N;
	int i;
	double seconds;
	double start,stop;

	start = clock();

		data = (int *)malloc(n*sizeof(int));
		for(i=0;i<n;i++)
			data[i] = random() % 100;

		//Print data before sort
		/*
		printf("\n******* Unsorted *******\n");
        for(i=0;i<n;i++)
            printf("%d ",data[i]);
        putchar('\n');
        */

		sort(data, 0, n-1);

	stop = clock();

    seconds = (stop-start)/CLOCKS_PER_SEC;
    printf("\nArray with %d; %d processors, took %f seconds\n",n,1,seconds);

    //Print data after sort

    /*
    printf("\n******* Sorted *******\n");
    for(i=0;i<n;i++)
        printf("%d ",data[i]);
    putchar('\n');
    */

}
