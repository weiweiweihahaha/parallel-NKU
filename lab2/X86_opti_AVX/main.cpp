#include <stdio.h>
#include <iostream>
#include <windows.h>
#include <stdlib.h>
#include <immintrin.h>

using namespace std;

long long head, tail, freq; // timers
const int n = 6072;
int t = 13;
int T = 2; // 循环五次，取时间均值
int nm[13] = {500, 600, 700, 800, 1000, 1200, 1400, 1700, 2000, 2200, 2500, 2800, 3000};
float m[n][n + 1]; // 扩展矩阵
float result[n] = {};

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

void guass_elimination_avx(int N)
{
    for (int k = 0; k < N; k++)
    {
        __m256 t1 = _mm256_set1_ps(m[k][k]);
        int j = 0;
        for (j = k + 1; j + 8 <= N; j += 8)
        {
            __m256 t2 = _mm256_loadu_ps(&m[k][j]);
            t2 = _mm256_div_ps(t2, t1);
            _mm256_storeu_ps(&m[k][j], t2);
        }
        for (; j < N; j++)
        {
            m[k][j] = m[k][j] / m[k][k];
        }
        m[k][k] = 1.0;
        for (int i = k + 1; i < N; i++)
        {
            __m256 vik = _mm256_set1_ps(m[i][k]);
            for (j = k + 1; j + 8 <= N; j += 8)
            {
                __m256 vkj = _mm256_loadu_ps(&m[k][j]);
                __m256 vij = _mm256_loadu_ps(&m[i][j]);
                __m256 vx = _mm256_mul_ps(vik, vkj);
                vij = _mm256_sub_ps(vij, vx);
                _mm256_storeu_ps(&m[i][j], vij);
            }
            for (; j < N; j++)
            {
                m[i][j] -= m[i][k] * m[k][j];
            }
            m[i][k] = 0;
        }
    }

    // 解方程
    for (int i = N - 1; i >= 0; i--)
    {
        __m256 res_i = _mm256_broadcast_ss(&m[i][N]);
        __m256 diag_i = _mm256_broadcast_ss(&m[i][i]);
        for (int j = i + 1; j < N; j += 8)
        {
            __m256 res_j = _mm256_loadu_ps(&result[j]);
            __m256 m_ij = _mm256_loadu_ps(&m[i][j]);
            __m256 res_j_updated = _mm256_sub_ps(res_j, _mm256_mul_ps(m_ij, res_i));
            _mm256_store_ps(&result[j], res_j_updated);
        }
        __m256 res_i_div_diag_i = _mm256_div_ps(res_i, diag_i);
        _mm256_store_ps(&result[i], res_i_div_diag_i);
    }
}

int main()
{
    for (int i = 0; i < t; i++)
    {
        double total_time = 0.0;
        for (int j = 0; j < T; j++)
        {
            m_reset(nm[i]);
            QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
            QueryPerformanceCounter((LARGE_INTEGER *)&head);

            guass_elimination_avx(nm[i]);

            QueryPerformanceCounter((LARGE_INTEGER *)&tail); // 结束时间
            total_time += (tail - head) * 1000.0 / freq;
        }
        // 输出平均时间
        cout << "数据规模为：" << nm[i] << " 平均消耗时间为：";
        cout << total_time / T << "ms" << endl;
        cout<<endl;
    }
    return 0;
}
