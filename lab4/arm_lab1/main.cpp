#include <iostream>
#include <mpi.h>
#include <sys/time.h>
#include <omp.h>
#include <arm_neon.h>

using namespace std;

const int max_threads = 4;

void init_A(float** arr, int N)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            arr[i][j] = 0;
        }
        arr[i][i] = 1.0;
        for (int j = i + 1; j < N; j++)
            arr[i][j] = rand() % 100;
    }

    for (int i = 0; i < N; i++)
    {
        int k1 = rand() % N;
        int k2 = rand() % N;
        for (int j = 0; j < N; j++)
        {
            arr[i][j] += arr[0][j];
            arr[k1][j] += arr[k2][j];
        }
    }
}

void reset_A(float** A, float** arr, int N)
{
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            A[i][j] = arr[i][j];
}

void f_ordinary(float** A, float** arr, int N)
{
    reset_A(A, arr, N);
    timeval t_start;
    timeval t_end;
    gettimeofday(&t_start, NULL);

    for (int k = 0; k < N; k++)
    {
        for (int j = k + 1; j < N; j++)
        {
            A[k][j] = A[k][j] * 1.0 / A[k][k];
        }
        A[k][k] = 1.0;

        for (int i = k + 1; i < N; i++)
        {
            for (int j = k + 1; j < N; j++)
            {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
    gettimeofday(&t_end, NULL);
    cout << "ordinary time cost: "
        << 1000 * (t_end.tv_sec - t_start.tv_sec) +
        0.001 * (t_end.tv_usec - t_start.tv_usec) << "ms" << endl;
}

void LU(float** A, int N, int rank, int num_proc)
{
    int block = N / num_proc;
    int remain = N % num_proc;

    int begin = rank * block;
    int end = rank != num_proc - 1 ? begin + block : begin + block + remain;

    for (int k = 0; k < N; k++)
    {
        if (k >= begin && k < end)
        {
            for (int j = k + 1; j < N; j++)
                A[k][j] = A[k][j] / A[k][k];
            A[k][k] = 1.0;
            for (int p = 0; p < num_proc; p++)
                if (p != rank)
                    MPI_Send(&A[k][0], N, MPI_FLOAT, p, 2, MPI_COMM_WORLD);
        }
        else
        {
            int cur_p = k / block;
            MPI_Recv(&A[k][0], N, MPI_FLOAT, cur_p, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        for (int i = begin; i < end && i < N; i++)
        {
            if (i >= k + 1)
            {
                for (int j = k + 1; j < N; j++)
                    A[i][j] = A[i][j] - A[i][k] * A[k][j];
                A[i][k] = 0.0;
            }
        }
    }
}

void f_mpi(float** A, float** arr, int N)
{
    timeval t_start;
    timeval t_end;

    int num_proc;
    int rank;

    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int block = N / num_proc;
    int remain = N % num_proc;

    if (rank == 0)
    {
        reset_A(A, arr, N);
        gettimeofday(&t_start, NULL);
        for (int i = 1; i < num_proc; i++)
        {
            if (i != num_proc - 1)
            {
                for (int j = 0; j < block; j++)
                    MPI_Send(&A[i * block + j][0], N, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
            }
            else
            {
                for (int j = 0; j < block + remain; j++)
                    MPI_Send(&A[i * block + j][0], N, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
            }
        }
        LU(A, N, rank, num_proc);
        for (int i = 1; i < num_proc; i++)
        {
            if (i != num_proc - 1)
            {
                for (int j = 0; j < block; j++)
                    MPI_Recv(&A[i * block + j][0], N, MPI_FLOAT, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            else
            {
                for (int j = 0; j < block + remain; j++)
                    MPI_Recv(&A[i * block + j][0], N, MPI_FLOAT, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
        gettimeofday(&t_end, NULL);
        cout << "Block MPI LU time cost: "
            << 1000 * (t_end.tv_sec - t_start.tv_sec) +
            0.001 * (t_end.tv_usec - t_start.tv_usec) << "ms" << endl;
    }
    else
    {
        if (rank != num_proc - 1)
        {
            for (int j = 0; j < block; j++)
                MPI_Recv(&A[rank * block + j][0], N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        else
        {
            for (int j = 0; j < block + remain; j++)
                MPI_Recv(&A[rank * block + j][0], N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        LU(A, N, rank, num_proc);
        if (rank != num_proc - 1)
        {
            for (int j = 0; j < block; j++)
                MPI_Send(&A[rank * block + j][0], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
        }
        else
        {
            for (int j = 0; j < block + remain; j++)
                MPI_Send(&A[rank * block + j][0], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
        }
    }
}

void LU_opt(float** A, int N, int rank, int num_proc)
{
    float32x4_t t1, t2, t3;
    int block = N / num_proc;
    int remain = N % num_proc;
    int begin = rank * block;
    int end = rank != num_proc - 1 ? begin + block : begin + block + remain;
#pragma omp parallel num_threads(max_threads) private(t1, t2, t3)
    for (int k = 0; k < N; k++)
    {
        if (k >= begin && k < end)
        {
            float temp1[4] = { A[k][k], A[k][k], A[k][k], A[k][k] };
            t1 = vld1q_f32(temp1);
#pragma omp for schedule(static)
            for (int j = k + 1; j < N - 3; j += 4)
            {
                t2 = vld1q_f32(A[k] + j);
                t3 = vdivq_f32(t2, t1);
                vst1q_f32(A[k] + j, t3);
            }
            for (int j = N - N % 4; j < N; j++)
            {
                A[k][j] = A[k][j] / A[k][k];
            }
            A[k][k] = 1.0;
            for (int p = rank + 1; p < num_proc; p++)
                MPI_Send(&A[k][0], N, MPI_FLOAT, p, 2, MPI_COMM_WORLD);
        }
        else
        {
            int cur_p = k / block;
            if (cur_p < rank)
                MPI_Recv(&A[k][0], N, MPI_FLOAT, cur_p, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        for (int i = begin; i < end && i < N; i++)
        {
            if (i >= k + 1)
            {
                float temp2[4] = { A[i][k], A[i][k], A[i][k], A[i][k] };
                t1 = vld1q_f32(temp2);
#pragma omp for schedule(static)
                for (int j = k + 1; j <= N - 3; j += 4)
                {
                    t2 = vld1q_f32(A[i] + j);
                    t3 = vld1q_f32(A[k] + j);
                    t3 = vmulq_f32(t1, t3);
                    t2 = vsubq_f32(t2, t3);
                    vst1q_f32(A[i] + j, t2);
                }
                for (int j = N - N % 4; j < N; j++)
                    A[i][j] = A[i][j] - A[i][k] * A[k][j];
                A[i][k] = 0;
            }
        }
    }
}

void f_mpi_opt(float** A, float** arr, int N)
{
    timeval t_start;
    timeval t_end;

    int num_proc;
    int rank;

    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int block = N / num_proc;
    int remain = N % num_proc;

    if (rank == 0)
    {
        reset_A(A, arr, N);
        gettimeofday(&t_start, NULL);
        for (int i = 1; i < num_proc; i++)
        {
            if (i != num_proc - 1)
            {
                for (int j = 0; j < block; j++)
                    MPI_Send(&A[i * block + j][0], N, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
            }
            else
            {
                for (int j = 0; j < block + remain; j++)
                    MPI_Send(&A[i * block + j][0], N, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
            }
        }
        LU_opt(A, N, rank, num_proc);
        for (int i = 1; i < num_proc; i++)
        {
            if (i != num_proc - 1)
            {
                for (int j = 0; j < block; j++)
                    MPI_Recv(&A[i * block + j][0], N, MPI_FLOAT, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            else
            {
                for (int j = 0; j < block + remain; j++)
                    MPI_Recv(&A[i * block + j][0], N, MPI_FLOAT, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
        gettimeofday(&t_end, NULL);
        cout << "Block MPI LU with NEON and OpenMP time cost: "
            << 1000 * (t_end.tv_sec - t_start.tv_sec) +
            0.001 * (t_end.tv_usec - t_start.tv_usec) << "ms" << endl;
    }
    else
    {
        if (rank != num_proc - 1)
        {
            for (int j = 0; j < block; j++)
                MPI_Recv(&A[rank * block + j][0], N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        else
        {
            for (int j = 0; j < block + remain; j++)
                MPI_Recv(&A[rank * block + j][0], N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        LU_opt(A, N, rank, num_proc);
        if (rank != num_proc - 1)
        {
            for (int j = 0; j < block; j++)
                MPI_Send(&A[rank * block + j][0], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
        }
        else
        {
            for (int j = 0; j < block + remain; j++)
                MPI_Send(&A[rank * block + j][0], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
        }
    }
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    for (int proc_count = 1; proc_count <= 4; proc_count *= 2) // Loop over different process counts (e.g., 1, 2, 4)

    {


       int step=500;
        for (int N = 1000; N <= 4000; N += step) // Loop over different matrix sizes (e.g., 1000, 2000)
        {
            float** arr = new float* [N];
            float** A = new float* [N];
            for (int i = 0; i < N; i++)
            {
                arr[i] = new float[N];
                A[i] = new float[N];
            }

            if (MPI_Comm_rank(MPI_COMM_WORLD,&rank)== 0) {
                cout << "Running with " << proc_count << " processes and matrix size " << N << "x" << N << endl;
            }

            init_A(arr, N);
            f_ordinary(A, arr, N);

            if (proc_count > 1) { // Skip MPI tests if proc_count is 1
                f_mpi(A, arr, N);
                f_mpi_opt(A, arr, N);
            }

            for (int i = 0; i < N; i++)
            {
                delete[] arr[i];
                delete[] A[i];
            }
            delete[] arr;
            delete[] A;

            if(N>=1000) step=1000;
            if(N>=2000) step=2000;
        }
    }

    MPI_Finalize();
    return 0;
}
