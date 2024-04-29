#include <stdio.h>
#include <iostream>
#include <windows.h>
#include <stdlib.h>
#include <emmintrin.h>

using namespace std;

long long head, tail, freq; // timers
const int n = 6072;
int t = 18;
int T = 2; // 循环五次，取时间均值
int nm[18] = { 500, 600, 700, 800, 1000, 1200, 1400, 1700, 2000, 2200, 2500, 2800, 3000, 3500, 4000, 4500, 5000, 6000 };
float m[n][n];
float result[n];

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
        __m128 res_i = _mm_set1_ps(m[i][N] / m[i][i]); // 计算结果向量中的第 i 个元素，用 SSE 寄存器保存
        for (int j = i - 1; j >= 0; j--)
        {
            __m128 res_j = _mm_loadu_ps(&m[j][N]); // 加载第 j 行的结果向量
            __m128 m_ji = _mm_set1_ps(m[j][i]); // 加载第 j 行的第 i 列元素
            res_j = _mm_sub_ps(res_j, _mm_mul_ps(res_i, m_ji)); // 更新结果向量
            _mm_storeu_ps(&m[j][N], res_j); // 存储更新后的结果向量
        }
        _mm_store_ss(&result[i], res_i); // 存储结果
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
            QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
            QueryPerformanceCounter((LARGE_INTEGER*)&head);

            guass_elimination_sse(nm[i]);

            QueryPerformanceCounter((LARGE_INTEGER*)&tail); // end time
            total_time += (tail - head) * 1000.0 / freq;
        }
        // 输出平均时间
        cout << "数据规模为：" << nm[i] << " 平均消耗时间为：";
        cout << total_time / T << "ms" << endl;
    }
    return 0;
}
