#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <time.h>

struct timespec diff(struct timespec start, struct timespec end)
{
struct timespec temp;

if(end.tv_sec - start.tv_sec == 0)
{
    temp.tv_nsec = end.tv_nsec - start.tv_nsec;
}
else
{
    temp.tv_nsec = ((end.tv_sec - start.tv_sec)*1000000000) + end.tv_nsec - start.tv_nsec;
}

return temp;
}

int main()
{
unsigned int N;
struct timespec t_start, t_end;
clock_t start, end;

srand(time(NULL));

FILE *f = fopen("out.txt", "w");
if(f == NULL)
{
    printf("Could not open output\n");
    return -1;
}

for(N = 1000000; N < 10000000; N += 1000000)
{
    fprintf(f, "%d\t", N);
    int* array = (int*)malloc(sizeof(int)*N);
    if(array == NULL)
    {
        printf("Not enough space\n");
        return -1;
    }
    for(unsigned int i = 0; i<N; i++) array[i] = rand();

    int max_val = 0.0;

    // clock_gettime(CLOCK_MONOTONIC, &t_start);

    // #pragma omp parallel for reduction(max:max_val)
    // for(unsigned int i=0; i<N; i++)
    // {
        // if(array[i] > max_val) max_val = array[i];
    // }

    // clock_gettime(CLOCK_MONOTONIC, &t_end);

    // fprintf(f, "%lf\t", (double)(diff(t_start, t_end).tv_nsec / 1000000000.0));

    max_val = 0.0;

    clock_gettime(CLOCK_MONOTONIC, &t_start);
    for(unsigned int i = 0; i<N; i++)
    {
        if(array[i] > max_val) max_val = array[i];
    }
    clock_gettime(CLOCK_MONOTONIC, &t_end);

    fprintf(f, "%lf\n", (double)(diff(t_start, t_end).tv_nsec / 1000000000.0));

    free(array);
}

fclose(f);

return 0;
}
