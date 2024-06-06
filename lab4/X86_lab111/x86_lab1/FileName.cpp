#include<iostream>
#include<stdlib.h>
#include<iomanip>
#include<mpi.h>
#include<omp.h>
#include <xmmintrin.h> //SSE
#include <emmintrin.h> //SSE2
#include <pmmintrin.h> //SSE3
#include <tmmintrin.h> //SSSE3
#include <smmintrin.h> //SSE4.1
#include <nmmintrin.h> //SSSE4.2
#include <immintrin.h> //AVX、AVX2
using namespace std;
int num_proc = 6;
const int n = 3000;
int NUM_THREADS = 8;
float A[n][n];
void init(float A[n][n])//n*n矩阵赋值
{
    srand((unsigned)time(NULL));
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < i; j++)
        {
            A[i][j] = 0;
        }
        A[i][i] = 1.0;
        for (int j = i + 1; j < n; j++)
        {
            A[i][j] = rand() % 100;//模取一万保证矩阵数据不会过大
        }
    }
    for (int k = 0; k < n; k++)
    {
        for (int i = k + 1; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                A[i][j] += A[k][j];
                A[i][j] = float(int(A[i][j]) % 100);//同理
            }
        }
    }
}
void display(float A[n][n])
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            cout << setw(11) << A[i][j] << " ";
        }
        cout << endl;
    }
}
void Gauss_normal(int rank)//普通高斯消元算法
{
    init(A);
    double start, end;
    start = MPI_Wtime();
    for (int k = 0; k < n; k++)
    {
        for (int j = k + 1; j < n; j++)
        {
            A[k][j] = A[k][j] / A[k][k];
        }
        A[k][k] = 1.0;
        for (int i = k + 1; i < n; i++)
        {
            for (int j = k + 1; j < n; j++)
            {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
    end = MPI_Wtime();
    if (rank == 0)
    {
        cout << "高斯消元普通时间" << (end - start) * 1000 << "ms" << endl;
    }
}
void clone(float A[n][n], float** B)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            B[i][j] = A[i][j];
        }
    }
}
void Gauss_mpi(int rank)
{
    int j;
    double start, end;
    int len = (n - n % num_proc) / num_proc;
    int r1 = rank * len;
    int r2 = (rank + 1) * len;
    if (rank == num_proc - 1)
    {
        r2 = n;
    }
    if (rank == 0)
    {
        init(A);
        for (j = 1; j < num_proc; j++)
        {
            int t1 = j * len;
            int t2 = (j + 1) * len;
            if (j == num_proc - 1)
            {
                t2 = n;
            }
            MPI_Send(&A[t1][0], n * (t2 - t1), MPI_FLOAT, j, 0, MPI_COMM_WORLD);
        }
    }
    else
    {
        MPI_Recv(&A[r1][0], n * (r2 - r1), MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
    for (int k = 0; k < n; k++)
    {
        if (k >= r1 && k < r2)
        {
            for (j = k + 1; j < n; j++)
            {
                A[k][j] = A[k][j] / A[k][k];
            }
            A[k][k] = 1.0;
            for (j = rank + 1; j < num_proc; j++)
            {
                MPI_Send(&A[k][0], n, MPI_FLOAT, j, 2, MPI_COMM_WORLD);
            }
        }
        else
        {
            int tmp = k / len;
            if (tmp < rank)
            {
                MPI_Recv(&A[k][0], n, MPI_FLOAT, tmp, 2, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
            }
        }
        for (int i = max(r1, k + 1); i < r2; i++)
        {
            for (j = k + 1; j < n; j++)
            {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
    end = MPI_Wtime();
    if (rank == num_proc - 1)
    {
        cout << "高斯消元mpi块划分优化" << (end - start) * 1000 << "ms" << endl;
    }
}
void Gauss_mpi_loop(int rank)
{
    int j;
    double start, end;
    if (rank == 0)
    {
        init(A);
        for (int k = 0; k < n; k++)
        {
            int tmp = k % num_proc;
            if (tmp != 0)
            {
                MPI_Send(&A[k][0], n, MPI_FLOAT, tmp, 0, MPI_COMM_WORLD);
            }
        }
    }
    else
    {
        for (int k = 0; k < n; k++)
        {
            int tmp = k % num_proc;
            if (tmp == rank)
            {
                MPI_Recv(&A[k][0], n, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
    for (int k = 0; k < n; k++)
    {
        if (k % num_proc == rank)
        {
            for (j = k + 1; j < n; j++)
            {
                A[k][j] = A[k][j] / A[k][k];
            }
            A[k][k] = 1.0;
            for (j = rank + 1; j < num_proc; j++)
            {
                MPI_Send(&A[k][0], n, MPI_FLOAT, j, 2, MPI_COMM_WORLD);
            }
        }
        else
        {
            int tmp = k % num_proc;
            if (tmp < rank)
            {
                MPI_Recv(&A[k][0], n, MPI_FLOAT, tmp, 2, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
            }
        }
        for (int i = 0; i < n; i++)
        {
            if (i % num_proc == rank && i > k)
            {
                for (j = k + 1; j < n; j++)
                {
                    A[i][j] = A[i][j] - A[i][k] * A[k][j];
                }
                A[i][k] = 0;
            }
        }
    }
    end = MPI_Wtime();
    if (rank == num_proc - 1)
    {
        cout << "高斯消元mpi循环划分优化" << (end - start) * 1000 << "ms" << endl;
    }
}
void Gauss_mpi_sse(int rank)
{
    int j;
    double start, end;
    int len = (n - n % num_proc) / num_proc;
    int r1 = rank * len;
    int r2 = (rank + 1) * len;
    if (rank == num_proc - 1)
    {
        r2 = n;
    }
    if (rank == 0)
    {
        init(A);
        for (j = 1; j < num_proc; j++)
        {
            int t1 = j * len;
            int t2 = (j + 1) * len;
            if (j == num_proc - 1)
            {
                t2 = n;
            }
            MPI_Send(&A[t1][0], n * (t2 - t1), MPI_FLOAT, j, 0, MPI_COMM_WORLD);
        }
    }
    else
    {
        MPI_Recv(&A[r1][0], n * (r2 - r1), MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
    for (int k = 0; k < n; k++)
    {
        if (k >= r1 && k < r2)
        {
            __m128 vt = _mm_set1_ps(A[k][k]);
            for (j = k + 1; j + 4 <= n; j += 4)
            {
                __m128 va = _mm_loadu_ps(&A[k][j]);
                va = _mm_div_ps(va, vt);
                _mm_storeu_ps(&A[k][j], va);
            }
            for (; j < n; j++)
            {
                A[k][j] = A[k][j] / A[k][k];
            }
            A[k][k] = 1.0;
            for (j = rank + 1; j < num_proc; j++)
            {
                MPI_Send(&A[k][0], n, MPI_FLOAT, j, 2, MPI_COMM_WORLD);
            }
        }
        else
        {
            int tmp = k / len;
            if (tmp < rank)
            {
                MPI_Recv(&A[k][0], n, MPI_FLOAT, tmp, 2, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
            }
        }
        for (int i = max(r1, k + 1); i < r2; i++)
        {
            __m128 vaik = _mm_set1_ps(A[i][k]);
            for (j = k + 1; j + 4 <= n; j += 4)
            {
                __m128 vakj = _mm_loadu_ps(&A[k][j]);
                __m128 vaij = _mm_loadu_ps(&A[i][j]);
                __m128 vx = _mm_mul_ps(vakj, vaik);
                vaij = _mm_sub_ps(vaij, vx);
                _mm_storeu_ps(&A[i][j], vaij);
            }
            for (; j < n; j++)
            {
                A[i][j] = A[i][j] - A[k][j] * A[i][k];
            }
            A[i][k] = 0;
        }
    }
    end = MPI_Wtime();
    if (rank == num_proc - 1)
    {
        cout << "高斯消元mpi块划分sse优化" << (end - start) * 1000 << "ms" << endl;
    }
}
void Gauss_mpi_loop_sse(int rank)
{
    int j;
    double start, end;
    if (rank == 0)
    {
        init(A);
        for (int k = 0; k < n; k++)
        {
            int tmp = k % num_proc;
            if (tmp != 0)
            {
                MPI_Send(&A[k][0], n, MPI_FLOAT, tmp, 0, MPI_COMM_WORLD);
            }
        }
    }
    else
    {
        for (int k = 0; k < n; k++)
        {
            int tmp = k % num_proc;
            if (tmp == rank)
            {
                MPI_Recv(&A[k][0], n, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
    for (int k = 0; k < n; k++)
    {
        if (k % num_proc == rank)
        {
            __m128 vt = _mm_set1_ps(A[k][k]);
            for (j = k + 1; j + 4 <= n; j += 4)
            {
                __m128 va = _mm_loadu_ps(&A[k][j]);
                va = _mm_div_ps(va, vt);
                _mm_storeu_ps(&A[k][j], va);
            }
            for (; j < n; j++)
            {
                A[k][j] = A[k][j] / A[k][k];
            }
            A[k][k] = 1.0;
            for (j = rank + 1; j < num_proc; j++)
            {
                MPI_Send(&A[k][0], n, MPI_FLOAT, j, 2, MPI_COMM_WORLD);
            }
        }
        else
        {
            int tmp = k % num_proc;
            if (tmp < rank)
            {
                MPI_Recv(&A[k][0], n, MPI_FLOAT, tmp, 2, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
            }
        }
        for (int i = 0; i < n; i++)
        {
            if (i % num_proc == rank && i > k)
            {
                __m128 vaik = _mm_set1_ps(A[i][k]);
                for (j = k + 1; j + 4 <= n; j += 4)
                {
                    __m128 vakj = _mm_loadu_ps(&A[k][j]);
                    __m128 vaij = _mm_loadu_ps(&A[i][j]);
                    __m128 vx = _mm_mul_ps(vakj, vaik);
                    vaij = _mm_sub_ps(vaij, vx);
                    _mm_storeu_ps(&A[i][j], vaij);
                }
                for (; j < n; j++)
                {
                    A[i][j] = A[i][j] - A[k][j] * A[i][k];
                }
                A[i][k] = 0;
            }
        }
    }
    end = MPI_Wtime();
    if (rank == num_proc - 1)
    {
        cout << "高斯消元mpi循环划分sse优化" << (end - start) * 1000 << "ms" << endl;
    }
}
void Gauss_mpi_omp(int rank)
{
    int i = 0;
    int j = 0;
    int k = 0;
    float temp = 0;
    int len = (n - n % num_proc) / num_proc;
    int r1 = rank * len;
    int r2 = (rank + 1) * len;
    double start, end;
    if (rank == num_proc - 1)
    {
        r2 = n;
    }
    if (rank == 0)
    {
        init(A);
        for (j = 1; j < num_proc; j++)
        {
            int t1 = j * len;
            int t2 = (j + 1) * len;
            if (j == num_proc - 1)
            {
                t2 = n;
            }
            MPI_Send(&A[t1][0], n * (t2 - t1), MPI_FLOAT, j, 0, MPI_COMM_WORLD);
        }
    }
    else
    {
        MPI_Recv(&A[r1][0], n * (r2 - r1), MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
#pragma omp parallel num_threads(NUM_THREADS), private(i, j, k, temp)
    for (k = 0; k < n; k++)
    {
#pragma omp single
        if (k >= r1 && k < r2)
        {
            temp = A[k][k];
            for (j = k + 1; j < n; j++)
            {
                A[k][j] = A[k][j] / temp;
            }
            A[k][k] = 1.0;
            for (j = rank + 1; j < num_proc; j++)
            {
                MPI_Send(&A[k][0], n, MPI_FLOAT, j, 2, MPI_COMM_WORLD);
            }
        }
        else
        {
            int tmp = k / len;
            if (tmp < rank)
            {
                MPI_Recv(&A[k][0], n, MPI_FLOAT, tmp, 2, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
            }
        }
#pragma omp for schedule(guided)
        for (i = max(r1, k + 1); i < r2; i++)
        {
            temp = A[i][k];
            for (j = k + 1; j < n; j++)
            {
                A[i][j] = A[i][j] - temp * A[k][j];
            }
            A[i][k] = 0;
        }
    }
    end = MPI_Wtime();
    if (rank == num_proc - 1)
    {
        cout << "高斯消元mpi块划分openmp优化" << (end - start) * 1000 << "ms" << endl;
    }
}
void Gauss_mpi_loop_omp(int rank)
{
    int i = 0;
    int j = 0;
    int k = 0;
    float temp = 0;
    double start, end;
    if (rank == 0)
    {
        init(A);
        for (k = 0; k < n; k++)
        {
            int tmp = k % num_proc;
            if (tmp != 0)
            {
                MPI_Send(&A[k][0], n, MPI_FLOAT, tmp, 0, MPI_COMM_WORLD);
            }
        }
    }
    else
    {
        for (k = 0; k < n; k++)
        {
            int tmp = k % num_proc;
            if (tmp == rank)
            {
                MPI_Recv(&A[k][0], n, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
#pragma omp parallel num_threads(NUM_THREADS), private(i, j, k, temp)
    for (k = 0; k < n; k++)
    {
#pragma omp single
        if (k % num_proc == rank)
        {
            temp = A[k][k];
            for (j = k + 1; j < n; j++)
            {
                A[k][j] = A[k][j] / temp;
            }
            A[k][k] = 1.0;
            for (j = rank + 1; j < num_proc; j++)
            {
                MPI_Send(&A[k][0], n, MPI_FLOAT, j, 2, MPI_COMM_WORLD);
            }
        }
        else
        {
            int tmp = k % num_proc;
            if (tmp < rank)
            {
                MPI_Recv(&A[k][0], n, MPI_FLOAT, tmp, 2, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
            }
        }
#pragma omp for schedule(guided)
        for (i = 0; i < n; i++)
        {
            if (i % num_proc == rank && i > k)
            {
                temp = A[i][k];
                for (j = k + 1; j < n; j++)
                {
                    A[i][j] = A[i][j] - temp * A[k][j];
                }
                A[i][k] = 0;
            }
        }
    }
    end = MPI_Wtime();
    if (rank == num_proc - 1)
    {
        cout << "高斯消元mpi循环划分openmp优化" << (end - start) * 1000 << "ms" << endl;
    }
}
void Gauss_mpi_omp_sse(int rank)
{
    int i = 0;
    int j = 0;
    int k = 0;
    float temp = 0;
    double start, end;
    int len = (n - n % num_proc) / num_proc;
    int r1 = rank * len;
    int r2 = (rank + 1) * len;
    if (rank == num_proc - 1)
    {
        r2 = n;
    }
    if (rank == 0)
    {
        init(A);
        for (j = 1; j < num_proc; j++)
        {
            int t1 = j * len;
            int t2 = (j + 1) * len;
            if (j == num_proc - 1)
            {
                t2 = n;
            }
            MPI_Send(&A[t1][0], n * (t2 - t1), MPI_FLOAT, j, 0, MPI_COMM_WORLD);
        }
    }
    else
    {
        MPI_Recv(&A[r1][0], n * (r2 - r1), MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
#pragma omp parallel num_threads(NUM_THREADS), private(i, j, k, temp)
    for (k = 0; k < n; k++)
    {
#pragma omp single
        if (k >= r1 && k < r2)
        {
            temp = A[k][k];
            __m128 vt = _mm_set1_ps(temp);
            for (j = k + 1; j + 4 <= n; j += 4)
            {
                __m128 va = _mm_loadu_ps(&A[k][j]);
                va = _mm_div_ps(va, vt);
                _mm_storeu_ps(&A[k][j], va);
            }
            for (; j < n; j++)
            {
                A[k][j] = A[k][j] / A[k][k];
            }
            A[k][k] = 1.0;
            for (j = rank + 1; j < num_proc; j++)
            {
                MPI_Send(&A[k][0], n, MPI_FLOAT, j, 2, MPI_COMM_WORLD);
            }
        }
        else
        {
            int tmp = k / len;
            if (tmp < rank)
            {
                MPI_Recv(&A[k][0], n, MPI_FLOAT, tmp, 2, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
            }
        }
#pragma omp for schedule(guided)
        for (i = max(r1, k + 1); i < r2; i++)
        {
            temp = A[i][k];
            __m128 vaik = _mm_set1_ps(temp);
            for (j = k + 1; j + 4 <= n; j += 4)
            {
                __m128 vakj = _mm_loadu_ps(&A[k][j]);
                __m128 vaij = _mm_loadu_ps(&A[i][j]);
                __m128 vx = _mm_mul_ps(vakj, vaik);
                vaij = _mm_sub_ps(vaij, vx);
                _mm_storeu_ps(&A[i][j], vaij);
            }
            for (; j < n; j++)
            {
                A[i][j] = A[i][j] - A[k][j] * A[i][k];
            }
            A[i][k] = 0;
        }
    }
    end = MPI_Wtime();
    if (rank == num_proc - 1)
    {
        cout << "高斯消元mpi块划分openmp+sse优化" << (end - start) * 1000 << "ms" << endl;
    }
}
void Gauss_mpi_loop_omp_sse(int rank)
{
    int i = 0;
    int j = 0;
    int k = 0;
    float temp = 0;
    double start, end;
    if (rank == 0)
    {
        init(A);
        for (k = 0; k < n; k++)
        {
            int tmp = k % num_proc;
            if (tmp != 0)
            {
                MPI_Send(&A[k][0], n, MPI_FLOAT, tmp, 0, MPI_COMM_WORLD);
            }
        }
    }
    else
    {
        for (k = 0; k < n; k++)
        {
            int tmp = k % num_proc;
            if (tmp == rank)
            {
                MPI_Recv(&A[k][0], n, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
#pragma omp parallel num_threads(NUM_THREADS), private(i, j, k, temp)
    for (k = 0; k < n; k++)
    {
#pragma omp single
        if (k % num_proc == rank)
        {
            temp = A[k][k];
            __m128 vt = _mm_set1_ps(temp);
            for (j = k + 1; j + 4 <= n; j += 4)
            {
                __m128 va = _mm_loadu_ps(&A[k][j]);
                va = _mm_div_ps(va, vt);
                _mm_storeu_ps(&A[k][j], va);
            }
            for (; j < n; j++)
            {
                A[k][j] = A[k][j] / A[k][k];
            }
            A[k][k] = 1.0;
            for (j = rank + 1; j < num_proc; j++)
            {
                MPI_Send(&A[k][0], n, MPI_FLOAT, j, 2, MPI_COMM_WORLD);
            }
        }
        else
        {
            int tmp = k % num_proc;
            if (tmp < rank)
            {
                MPI_Recv(&A[k][0], n, MPI_FLOAT, tmp, 2, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
            }
        }
#pragma omp for schedule(guided)
        for (i = 0; i < n; i++)
        {
            if (i % num_proc == rank && i > k)
            {
                temp = A[i][k];
                __m128 vaik = _mm_set1_ps(temp);
                for (j = k + 1; j + 4 <= n; j += 4)
                {
                    __m128 vakj = _mm_loadu_ps(&A[k][j]);
                    __m128 vaij = _mm_loadu_ps(&A[i][j]);
                    __m128 vx = _mm_mul_ps(vakj, vaik);
                    vaij = _mm_sub_ps(vaij, vx);
                    _mm_storeu_ps(&A[i][j], vaij);
                }
                for (; j < n; j++)
                {
                    A[i][j] = A[i][j] - A[k][j] * A[i][k];
                }
                A[i][k] = 0;
            }
        }
    }
    end = MPI_Wtime();
    if (rank == num_proc - 1)
    {
        cout << "高斯消元mpi循环划分openmp+sse优化" << (end - start) * 1000 << "ms" << endl;
    }
}
double calculate_MPI_pipeline(int rank) {
    double start_time, end_time;

    
    int size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    // 只有是0号进程，才进行初始化工作
    if (rank == 0) {
        init(A);
    }
    start_time = MPI_Wtime();
    int task_num = rank < n % size ? n / size + 1 : n / size;
    // 0号进程负责任务的初始分发工作
    auto* buff = new float[task_num * n];
    if (rank == 0) {
        for (int p = 1; p < size; p++) {
            for (int i = p; i < n; i += size) {
                for (int j = 0; j < n; j++) {
                    buff[i / size * n + j] = A[i][j];
                }
            }
            int count = p < n % size ? n / size + 1 : n / size;
            MPI_Send(buff, count * n, MPI_FLOAT, p, 0, MPI_COMM_WORLD);
        }
    }
    // 非0号进程负责任务的接收工作
    else {
        MPI_Recv(&A[rank][0], task_num * n, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (int i = 0; i < task_num; i++) {
            for (int j = 0; j < n; j++) {
                A[rank + i * size][j] = A[rank + i][j];
            }
        }
    }
    // 做消元运算
    int pre_proc = (rank + (size - 1)) % size;
    int next_proc = (rank + 1) % size;
    for (int k = 0; k < n; k++) {
        // 如果除法操作是本进程负责的任务，并将除法结果广播
        if (k % size == rank) {
            for (int j = k + 1; j < n; j++) {
                A[k][j] /= A[k][k];
            }
            A[k][k] = 1;
            MPI_Send(&A[k][0], n, MPI_FLOAT, next_proc, 1, MPI_COMM_WORLD);
        }
        // 其余进程接收除法行的结果
        else {
            MPI_Recv(&A[k][0], n, MPI_FLOAT, pre_proc, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (next_proc != k % size) {
                MPI_Send(&A[k][0], n, MPI_FLOAT, next_proc, 1, MPI_COMM_WORLD);
            }
        }
        // 进行消元操作
        int begin = n / size * size + rank < n ? n / size * size + rank : n / size * size + rank - size;
        for (int i = begin; i > k; i -= size) {
            for (int j = k + 1; j < n; j++) {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
    end_time = MPI_Wtime();
    if (rank == num_proc - 1)
    {
        cout << "高斯消元mpi块划分pipeline优化" << (end_time - start_time) * 1000 << "ms" << endl;
    }
    return (end_time - start_time) * 1000;
}
int main(int argc, char* argv[])
{
   /* int step = 500;
    for (; n <= 6000; n += step)
    {*/
   
        MPI_Init(NULL, NULL);
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      //  Gauss_normal(rank);
        Gauss_mpi(rank);
        calculate_MPI_pipeline(rank);
       // Gauss_mpi_loop(rank);
      /*  Gauss_mpi_sse(rank);
        Gauss_mpi_loop_sse(rank);
        Gauss_mpi_omp(rank);
        Gauss_mpi_loop_omp(rank);
        Gauss_mpi_omp_sse(rank);
        Gauss_mpi_loop_omp_sse(rank);*/
        MPI_Finalize();

  //      if (n >= 1000) step = 1000;
  //      if (n >= 2000) step = 2000;
  ////  }
    return 0;
}


