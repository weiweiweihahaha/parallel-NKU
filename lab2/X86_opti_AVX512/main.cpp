#include <stdio.h>
#include <iostream>
#include <windows.h>
#include <stdlib.h>
#include <immintrin.h>

using namespace std;

long long head, tail, freq; // 计时器
const int n = 6072;
int t = 18;
int T = 2; // 循环五次，取时间均值
int nm[18] = {500, 600, 700, 800, 1000, 1200, 1400, 1700, 2000, 2200, 2500, 2800, 3000, 3500, 4000, 4500, 5000, 6000};
float m[n][n];
float result[n]{};

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

void guass_elimination_avx512_unalign(int N)
{
    __m512 inv_k, m_kj, m_ik, m_ij;
    for (int k = 0; k < N; k++)
    {
        inv_k = _mm512_rcp14_ps(_mm512_broadcastss_ps(_mm_load_ss(&m[k][k])));
        for (int j = k + 1; j < N; j++)
        {
            m[k][j] *= inv_k[0];
        }
        m[k][k] = 1.0f;
        for (int i = k + 1; i < N; i++)
        {
            m_ik = _mm512_broadcastss_ps(_mm_load_ss(&m[i][k]));
            for (int j = k + 1; j < N; j += 16)
            {
                m_ij = _mm512_loadu_ps(&m[i][j]);
                m_kj = _mm512_loadu_ps(&m[k][j]);
                m_ij = _mm512_sub_ps(m_ij, _mm512_mul_ps(m_ik, m_kj));
                _mm512_storeu_ps(&m[i][j], m_ij);
            }
            m[i][k] = 0.0f;
        }
    }

    for (int i = N - 1; i >= 0; i--)
    {
        __m512 res_i = _mm512_broadcastss_ps(_mm_load_ss(&m[i][N]));
        __m512 diag_i = _mm512_broadcastss_ps(_mm_load_ss(&m[i][i]));
        for (int j = i + 1; j < N; j += 16)
        {
            __m512 res_j = _mm512_loadu_ps(&result[j]);
            __m512 m_ij = _mm512_loadu_ps(&m[i][j]);
            __m512 res_j_updated = _mm512_sub_ps(res_j, _mm512_mul_ps(m_ij, res_i));
            _mm512_storeu_ps(&result[j], res_j_updated);
        }
        __m512 res_i_div_diag_i = _mm512_div_ps(res_i, diag_i);
        _mm512_storeu_ps(&result[i], res_i_div_diag_i);
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

            guass_elimination_avx512_unalign(nm[i]);

            QueryPerformanceCounter((LARGE_INTEGER *)&tail); // 结束时间
            total_time += (tail - head) * 1000.0 / freq;
        }
        // 输出平均时间
        cout << "unalign_数据规模为：" << nm[i] << " 平均消耗时间为：";
        cout << total_time / T << "ms" << endl;
    }
    return 0;
}
