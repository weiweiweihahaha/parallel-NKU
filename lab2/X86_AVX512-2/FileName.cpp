#include <stdio.h>
#include <iostream>
#include <windows.h>
#include <stdlib.h>
#include <immintrin.h>
#define ALIGNMENT 64
using namespace std;

long long head, tail, freq; // timers
const int n = 6072;
int t = 13;
int T = 1; // ѭ����Σ�ȡʱ���ֵ
int nm[13] = { 500, 600, 700, 800, 1000, 1200, 1400, 1700, 2000, 2200, 2500, 2800, 3000 };
float m[n][n + 1]; // ��չ����
float result[n] = {};
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
        aligned_m = (float**)malloc(N * sizeof(float*)); // �����ڴ�� aligned_m
        for (int i = 0; i < N; i++) {
            aligned_m[i] = (float*)_aligned_malloc(N * sizeof(float), ALIGNMENT);
        }

        // �����ʱ����ʹ�䰴����Ԫ˳��洢
        for (int j = 0; j < N; j++)
        {
            for (int i = 0; i < N; i++)
            {
                if (i == j)
                    aligned_m[i][j] = 1.0; // �Խ���Ԫ��Ϊ1
                else
                    aligned_m[i][j] = rand() % 1000; // �ǶԽ���Ԫ��������
            }
        }
    
    for (int k = 0; k < N; k++)
        for (int i = k + 1; i < N; i++)
            for (int j = 0; j < N; j++)
                m[i][j] += m[k][j];

}

void free_align(int N)
{
    if (aligned_m) { // ȷ�� aligned_m ��Ϊ��
        for (int i = 0; i < N; i++) {
            _aligned_free(aligned_m[i]); // �ͷ��ڴ�
        }
        free(aligned_m); // �ͷ� aligned_m ���鱾��
    }
}

void guass_elimination_avx512(int N)
{
    for (int k = 0; k < N; k++)
    {
        __m512 t1 = _mm512_set1_ps(m[k][k]);
        int j = 0;
        for (j = k + 1; j + 16 <= N; j += 16)
        {
            __m512 t2 = _mm512_loadu_ps(&m[k][j]);
            t2 = _mm512_div_ps(t2, t1);
            _mm512_storeu_ps(&m[k][j], t2);
        }
        for (; j < N; j++)
        {
            m[k][j] = m[k][j] / m[k][k];
        }
        m[k][k] = 1.0;
        for (int i = k + 1; i < N; i++)
        {
            __m512 vik = _mm512_set1_ps(m[i][k]);
            for (j = k + 1; j + 16 <= N; j += 16)
            {
                __m512 vkj = _mm512_loadu_ps(&m[k][j]);
                __m512 vij = _mm512_loadu_ps(&m[i][j]);
                __m512 vx = _mm512_mul_ps(vik, vkj);
                vij = _mm512_sub_ps(vij, vx);
                _mm512_storeu_ps(&m[i][j], vij);
            }
            for (; j < N; j++)
            {
                m[i][j] -= m[i][k] * m[k][j];
            }
            m[i][k] = 0;
        }
    }

    
    // �ⷽ��
    for (int i = N - 1; i >= 0; i--)
    {
        result[i] = m[i][N] / m[i][i];

        for (int j = i - 1; j >= 0; j--)
        {
            m[j][N] -= m[j][i] * result[i];
        }
    }
}
void guass_elimination_avx512_align(int N)
{
    for (int k = 0; k < N; k++)
    {
        __m512 t1 = _mm512_set1_ps(aligned_m[k][k]);
        int j = (k + 1 + 15) & ~15; // ���뵽 16 �ı���

        for (; j + 16 <= N; j += 16)
        {
            __m512 t2 = _mm512_load_ps(&aligned_m[k][j]); // ���ض��������
            t2 = _mm512_div_ps(t2, t1);
            _mm512_store_ps(&aligned_m[k][j], t2); // �洢���������
        }

        for (; j < N; j++)
        {
            aligned_m[k][j] = aligned_m[k][j] / aligned_m[k][k];
        }
        aligned_m[k][k] = 1.0;

        for (int i = k + 1; i < N; i++)
        {
            __m512 vik = _mm512_set1_ps(aligned_m[i][k]);
            j = (k + 1 + 15) & ~15;

            for (; j + 16 <= N; j += 16)
            {
                __m512 vkj = _mm512_load_ps(&aligned_m[k][j]);
                __m512 vij = _mm512_load_ps(&aligned_m[i][j]);
                __m512 vx = _mm512_mul_ps(vik, vkj);
                vij = _mm512_sub_ps(vij, vx);
                _mm512_store_ps(&aligned_m[i][j], vij);
            }

            for (; j < N; j++)
            {
                aligned_m[i][j] -= aligned_m[i][k] * aligned_m[k][j];
            }
            aligned_m[i][k] = 0;
        }
    }

    // �ⷽ��
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

            guass_elimination_avx512(nm[i]);

            QueryPerformanceCounter((LARGE_INTEGER*)&tail); // ����ʱ��
            total_time += (tail - head) * 1000.0 / freq;
        }
        // ���ƽ��ʱ��
        cout << "���ݹ�ģΪ��" << nm[i] << " AVX512ƽ������ʱ��Ϊ��";
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

            guass_elimination_avx512_align(nm[i]);

            QueryPerformanceCounter((LARGE_INTEGER*)&tail); // ����ʱ��
            total_time += (tail - head) * 1000.0 / freq;
            free_align(nm[i]);
        }
        // ���ƽ��ʱ��
        cout << "���ݹ�ģΪ��" << nm[i] << " AVX512����ƽ������ʱ��Ϊ��";
        cout << total_time / T << "ms" << endl;

    }
    return 0;
}
