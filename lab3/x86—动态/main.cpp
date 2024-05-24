#include <iostream>
#include <cstdlib>
#include <windows.h>
#include <immintrin.h>

using namespace std;

const int n = 500;
float A[n][n];
int worker_count = 4; //工作线程数量

void init()
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            A[i][j] = 0;
        }
        A[i][i] = 1.0;
        for (int j = i + 1; j < n; j++)
            A[i][j] = rand() % 100;
    }

    for (int i = 0; i < n; i++)
    {
        int k1 = rand() % n;
        int k2 = rand() % n;
        for (int j = 0; j < n; j++)
        {
            A[i][j] += A[0][j];
            A[k1][j] += A[k2][j];
        }
    }
}

void f_ordinary()
{
    for (int k = 0; k < n; k++)
    {
        for (int j = k + 1; j < n; j++)
        {
            A[k][j] = A[k][j] * 1.0 / A[k][k];
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
}

struct threadParam_t
{
    int k;     //消去的轮次
    int t_id;  // 线程 id
};

void* threadFunc(void* param)
{
    __m256 vaik, vakj;

    threadParam_t* p = (threadParam_t*)param;
    int k = p->k;    //消去的轮次
    int t_id = p->t_id;  //线程编号
    int i = k + t_id + 1; //获取自己的计算任务
    for (int m = k + 1 + t_id; m < n; m += worker_count)
    {
        vaik = _mm256_set1_ps(A[m][k]);
        int j;
        for (j = k + 1; j + 8 <= n; j += 8)
        {
            vakj = _mm256_loadu_ps(&(A[k][j]));
            __m256 vaij = _mm256_loadu_ps(&(A[m][j]));
            __m256 vx = _mm256_mul_ps(vakj, vaik);
            vaij = _mm256_sub_ps(vaij, vx);
            _mm256_storeu_ps(&A[i][j], vaij);
        }
        for (; j < n; j++)
            A[m][j] = A[m][j] - A[m][k] * A[k][j];

        A[m][k] = 0;
    }

    return NULL;
}

int main()
{
    init();
    LARGE_INTEGER frequency, start, end;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&start); //开始计时
for(int q=0;q<20;q++)
{


    for (int k = 0; k < n; k++)
    {
        __m256 vt = _mm256_set1_ps(A[k][k]);
        int j;
        for (j = k + 1; j + 8 <= n; j += 8)
        {
            __m256 va = _mm256_loadu_ps(&(A[k][j]));
            va = _mm256_div_ps(va, vt);
            _mm256_storeu_ps(&(A[k][j]), va);
        }

        for (; j < n; j++)
        {
            A[k][j] = A[k][j] * 1.0 / A[k][k];
        }
        A[k][k] = 1.0;

        //创建工作线程，进行消去操作
        HANDLE* handles = new HANDLE[worker_count]; // 创建对应的 Handle
        threadParam_t* param = new threadParam_t[worker_count]; // 创建对应的线程数据结构

        //分配任务
        for (int t_id = 0; t_id < worker_count; t_id++)
        {
            param[t_id].k = k;
            param[t_id].t_id = t_id;
        }
        //创建线程
        for (int t_id = 0; t_id < worker_count; t_id++)
            handles[t_id] = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)threadFunc, (LPVOID)&param[t_id], 0, NULL);

        //主线程挂起等待所有的工作线程完成此轮消去工作
        WaitForMultipleObjects(worker_count, handles, TRUE, INFINITE);

        //关闭线程句柄
        for (int t_id = 0; t_id < worker_count; t_id++)
            CloseHandle(handles[t_id]);

        delete[] handles;
        delete[] param;
    }
}
    QueryPerformanceCounter(&end); //结束计时
    double seconds = (double)(end.QuadPart - start.QuadPart) * 1000.0 /( frequency.QuadPart*20); //单位 ms

    cout << "pthread_dong: " << seconds << " ms" << endl;

    return 0;
}
