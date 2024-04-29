#include <stdio.h>
#include <iostream>
#include <windows.h>
#include <stdlib.h>
#include <emmintrin.h> // 引入 SSE 头文件
#define ALIGNMENT 16
using namespace std;

long long head, tail, freq; // timers
const int n = 6072;
int t = 13;
int T = 3; // 循环五次，取时间均值
int nm[13] = { 500, 600, 700, 800, 1000, 1200, 1400, 1700, 2000, 2200, 2500, 2800, 3000 };
float m[n][n];
float result[n]{};
float** aligned_m;

void m_reset(int N)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < i; j++)
            m[i][j] = 0;
        m[i][i] = 1.0;
        for (int j = i + 1; j < N; j++)
            m[i][j] = rand() % 1000;
    }
    for (int k = 0; k < N; k++)
        for (int i = k + 1; i < N; i++)
            for (int j = 0; j < N; j++)
                m[i][j] += m[k][j];
}
void m_reset_align(int N)
{
    aligned_m = (float**)malloc(N * sizeof(float*)); // 分配内存给 aligned_m
    for (int i = 0; i < N; i++) {
        aligned_m[i] = (float*)_aligned_malloc(N * sizeof(float), ALIGNMENT);
    }

    // 填充临时矩阵，使其按列主元顺序存储
    for (int j = 0; j < N; j++)
    {
        for (int i = 0; i < N; i++)
        {
            if (i == j)
                aligned_m[i][j] = 1.0; // 对角线元素为1
            else
                aligned_m[i][j] = rand() % 1000; // 非对角线元素随机填充
        }
    }

    for (int k = 0; k < N; k++)
        for (int i = k + 1; i < N; i++)
            for (int j = 0; j < N; j++)
                m[i][j] += m[k][j];

}
// SSE 优化的高斯消元函数
void guass_elimination_sse(int N)
{
    for (int k = 0; k < N; k++)
    {
        __m128 inv_kk = _mm_set1_ps(1.0f / m[k][k]); // 1 / m[k][k]
        for (int j = k + 1; j < N; j += 4) // 使用 SSE 加速
        {
            __m128 m_kj = _mm_loadu_ps(&m[k][j]); // 加载 m[k][j] 到 xmm 寄存器
            __m128 m_kj_div_kk = _mm_mul_ps(m_kj, inv_kk); // m[k][j] / m[k][k]
            _mm_storeu_ps(&m[k][j], m_kj_div_kk); // 存储结果回 m[k][j]
        }

        for (int i = k + 1; i < N; i++)
        {
            __m128 vik = _mm_set1_ps(m[i][k]);
            for (int j = k + 1; j < N; j += 4)
            {
                __m128 vkj = _mm_loadu_ps(&m[k][j]);
                __m128 vij = _mm_loadu_ps(&m[i][j]);
                __m128 vx = _mm_mul_ps(vik, vkj);
                vij = _mm_sub_ps(vij, vx);
                _mm_storeu_ps(&m[i][j], vij);
            }
        }
    }

    for (int i = N - 1; i >= 0; i--)
    {
        result[i] = m[i][N] / m[i][i];

        for (int j = i - 1; j >= 0; j--)
        {
            m[j][N] -= m[j][i] * result[i];
        }
    }
}

void guass_elimination_sse_align(int N)
{
    for (int k = 0; k < N; k++)
    {
        __m128 inv_kk = _mm_set1_ps(1.0f / aligned_m[k][k]); // 1 / aligned_m[k][k]

        int j = (k + 1 + 3) & ~3;
        for (; j + 4 <= N; j += 4) // 使用 SSE 加速
        {
            __m128 m_kj = _mm_load_ps(&aligned_m[k][j]); // 加载 aligned_m[k][j] 到 xmm 寄存器
            __m128 m_kj_div_kk = _mm_mul_ps(m_kj, inv_kk); // aligned_m[k][j] / aligned_m[k][k]
            _mm_store_ps(&aligned_m[k][j], m_kj_div_kk); // 存储结果回 aligned_m[k][j]
        }
        //处理对齐操作下剩余的元素
        for (; j < N; j++)
        {
            aligned_m[k][j] = aligned_m[k][j] * 1.0 / aligned_m[k][k];
        }

        for (int i = k + 1; i < N; i++)
        {

            int j = k + 1;
            j = (k + 1 + 3) & ~3;
            __m128 vik = _mm_set1_ps(aligned_m[i][k]);
            for (; j + 4 <= N; j += 4)
            {
                __m128 vkj = _mm_load_ps(&aligned_m[k][j]);
                __m128 vij = _mm_load_ps(&aligned_m[i][j]);
                __m128 vx = _mm_mul_ps(vik, vkj);
                vij = _mm_sub_ps(vij, vx);
                _mm_store_ps(&aligned_m[i][j], vij);
            }
            for (; j < N; j++)
            {
                aligned_m[i][j] -= aligned_m[k][j] * aligned_m[i][k];
            }
            aligned_m[i][k] = 0.0;

        }
    }

    for (int i = N - 1; i >= 0; i--)
    {
        result[i] = aligned_m[i][N] / aligned_m[i][i];

        for (int j = i - 1; j >= 0; j--)
        {
            aligned_m[j][N] -= aligned_m[j][i] * result[i];
        }
    }
}




int main()
{
    /*
    for (int i = 0; i < t; i++)
    {
        double total_time = 0.0;
        for (int j = 0; j < T; j++)
        {
            m_reset(nm[i]);
            QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
            QueryPerformanceCounter((LARGE_INTEGER*)&head);

            guass_elimination_sse(nm[i]); // 调用 SSE 优化的高斯消元函数

            QueryPerformanceCounter((LARGE_INTEGER*)&tail); // end time
            total_time += (tail - head) * 1000.0 / freq;
        }
        // 输出平均时间
        cout << "数据规模为：" << nm[i] << " sse平均消耗时间为：";
        cout << total_time / T << "ms" << endl;
    }*/

    for (int i = 0; i < t; i++)
    {
        double total_time = 0.0;
        for (int j = 0; j < T; j++)
        {
            m_reset_align(nm[i]);
            QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
            QueryPerformanceCounter((LARGE_INTEGER*)&head);

            guass_elimination_sse_align(nm[i]); // 调用 SSE 优化的高斯消元函数

            QueryPerformanceCounter((LARGE_INTEGER*)&tail); // end time
            total_time += (tail - head) * 1000.0 / freq;
        }
        // 输出平均时间
        cout << "数据规模为：" << nm[i] << " sse对齐优化平均消耗时间为：";
        cout << total_time / T << "ms" << endl;
    }
    return 0;
}
