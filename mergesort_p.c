/* Parallel merge sort */
#include <stdio.h>
#include <mpi.h>
#include <time.h>

#define N 1000000

int * merge(int *First, int Fsize, int *Second, int Ssize);
void sort(int *A, int begin, int end);
void printElements(int rank, int *v, int n);

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

void printElements(int rank, int *t, int n)
{
	int i;
	printf("%d: ",rank);
	for(i=0;i<n;i++)
        printf("%d ",t[i]);
    printf("\n\n");
}

main(int argc, char **argv)
{
	int * data;
	int * local_data;
	int * otherArr;
	int q,n=N;
	int my_rank,comm_sz;
	int local_n = 0;
	int i;
	int step;
	double start,stop;
	MPI_Status status;

	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
	MPI_Comm_size(MPI_COMM_WORLD,&comm_sz);

	start = clock();
	if(my_rank == 0)
	{
		int r;
		local_n = n/comm_sz;
		r = n%comm_sz;
		int size = n+local_n-r;
		data = (int *)malloc((size)*sizeof(int));
		for(i=0;i<n;i++)
			data[i] = random() % 1000;
		if(r!=0)
		{
			for(i=n;i<size;i++)
				data[i]=0;
			local_n=local_n+1;
		}

		//Print unsorted data
		MPI_Bcast(&local_n,1,MPI_INT,0,MPI_COMM_WORLD);
		local_data = (int *)malloc(local_n*sizeof(int));
		MPI_Scatter(data,local_n,MPI_INT,local_data,local_n,MPI_INT,0,MPI_COMM_WORLD);
		sort(local_data, 0, local_n-1);
	}
	else
	{
		MPI_Bcast(&local_n,1,MPI_INT,0,MPI_COMM_WORLD);
		local_data = (int *)malloc(local_n*sizeof(int));
		MPI_Scatter(data,local_n,MPI_INT,local_data,local_n,MPI_INT,0,MPI_COMM_WORLD);
		sort(local_data, 0, local_n-1);

	}

	//Print sorted data per processor
	step = 1;
	while(step < comm_sz)
	{
		if(my_rank % (2*step) == 0)
		{
			if(my_rank+step < comm_sz)
			{
				MPI_Recv(&q, 1, MPI_INT, my_rank+step, 0, MPI_COMM_WORLD, &status);
				otherArr = (int *)malloc(q*sizeof(int));
				MPI_Recv(otherArr, q, MPI_INT, my_rank+step, 0, MPI_COMM_WORLD, &status);
				local_data = merge(local_data, local_n, otherArr, q);
				local_n = local_n + q;
			}
		}
		else
		{
			int dest = my_rank-step;
			MPI_Send(&local_n, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
			MPI_Send(local_data, local_n, MPI_INT, dest, 0, MPI_COMM_WORLD);
			break;
		}
		step = step * 2;
	}
	stop = clock();

	if(my_rank == 0)
	{
	    //Print data after sort
        printf("\nArray with %d; %d processors, took %f second\n\n",local_n, comm_sz,(stop-start)/CLOCKS_PER_SEC);
	}

    MPI_Finalize();
}
