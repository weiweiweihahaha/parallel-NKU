#include <iostream>
#include <cstdlib>
#include <sys/time.h>
#include <arm_neon.h>
//neon�Ż����������������������ش���ȫ���Ż�
//δ�����
using namespace std;

const int n = 6072;
/*
int t = 13;
int T = 1; // ѭ����Σ�ȡʱ���ֵ
int nm[13] = {500, 600, 700, 800, 1000, 1200, 1400, 1700, 2000, 2200, 2500, 2800, 3000};
*/
int t = 5;
int T = 1; // ѭ����Σ�ȡʱ���ֵ
int nm[5] = {3500,4000,4500,5000,6000};
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

void guass_elimination_neon1(int N)
{
    for (int k = 0; k < N; k++)
    {
        // ʹ�� NEON ���й�һ��
        float32x4_t div_factor = vdupq_n_f32(1.0f / m[k][k]);

        for (int j = k + 1; j < N; j += 4)
        {

            float32x4_t m_k_j = vld1q_f32(&m[k][j]);
            m_k_j = vmulq_f32(m_k_j, div_factor);

            vst1q_f32(&m[k][j], m_k_j);
        }

        // ���Խ���Ԫ������Ϊ1
        m[k][k] = 1.0;

        for (int i = k + 1; i < N; i++)
        {
            for (int j = k + 1; j < N; j++)
            {
                m[i][j] -= m[i][k] * m[k][j];
            }
            m[i][k] = 0;
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

void guass_elimination_neon2(int N)
{
    for (int k = 0; k < N; k++)
    {
        for (int j = k + 1; j < N; j++)
        {
            m[k][j] = m[k][j] / m[k][k];
        }
        m[k][k] = 1.0;

        for (int i = k + 1; i < N; i++)
        {
            float32x4_t m_i_k = vdupq_n_f32(m[i][k]); // ����m[i][k]��NEON�Ĵ�����
            for (int j = k + 1; j < N; j += 4)
            {
                float32x4_t m_k_j = vld1q_f32(&m[k][j]); // ����m[k][j]��NEON�Ĵ�����
                float32x4_t m_i_j = vld1q_f32(&m[i][j]); // ����m[i][j]��NEON�Ĵ�����
                float32x4_t product = vmulq_f32(m_i_k, m_k_j); // �˷�����
                m_i_j = vsubq_f32(m_i_j, product); // ��������
                vst1q_f32(&m[i][j], m_i_j); // �洢������ڴ�
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

void guass_elimination_neon3(int N)
{
    for (int k = 0; k < N; k++)
    {
        for (int j = k + 1; j < N; j++)
        {
            m[k][j] = m[k][j] / m[k][k];
        }
        m[k][k] = 1.0;
        for (int i = k + 1; i < N; i++)
        {
            for (int j = k + 1; j < N; j++)
            {
                m[i][j] -= m[i][k] * m[k][j];
            }
            m[i][k] = 0;
        }
    }

    for (int i = N - 1; i >= 0; i--)
    {
        result[i] = m[i][N] / m[i][i];

        for (int j = i - 1; j >= 0; j--)
        {
            float32x4_t result_i = vdupq_n_f32(result[i]);
            // ���� result[i] �� NEON �Ĵ�����
            float32x4_t m_j_i = vld1q_f32(&m[j][i]);
            // ���� m[j][i] �� NEON �Ĵ�����
            float32x4_t product = vmulq_f32(result_i, m_j_i);
            // �˷�����
            float32x4_t m_j_N = vld1q_f32(&m[j][N]);
            // ���� m[j][N] �� NEON �Ĵ�����
            m_j_N = vsubq_f32(m_j_N, product);
            // ��������
            vst1q_f32(&m[j][N], m_j_N);
             // �洢������ڴ�
        }
    }
}

void guass_elimination_neon_all(int N)
{
   for (int k = 0; k < N; k++)
    {
        // ʹ�� NEON ���й�һ��
        float32x4_t div_factor = vdupq_n_f32(1.0f / m[k][k]);
        for (int j = k + 1; j < N; j += 4)
        {
            float32x4_t m_k_k = vdupq_n_f32(m[k][k]);
            float32x4_t m_k_j = vld1q_f32(&m[k][j]);
            float32x4_t normalized = vdivq_f32(m_k_j, m_k_k);
            vst1q_f32(&m[k][j], normalized);
        }

        // ���Խ���Ԫ������Ϊ1
        m[k][k] = 1.0;

        // ʹ�� NEON �Ż��˷��ͼ���
        for (int i = k + 1; i < N; i++)
        {
            float32x4_t m_i_k = vdupq_n_f32(m[i][k]); // ���� m[i][k] �� NEON �Ĵ�����
            for (int j = k + 1; j < N; j += 4)
            {
                float32x4_t m_k_j = vld1q_f32(&m[k][j]); // ���� m[k][j] �� NEON �Ĵ�����
                float32x4_t m_i_j = vld1q_f32(&m[i][j]); // ���� m[i][j] �� NEON �Ĵ�����
                float32x4_t product = vmulq_f32(m_i_k, m_k_j); // �˷�����
                m_i_j = vsubq_f32(m_i_j, product); // ��������
                vst1q_f32(&m[i][j], m_i_j); // �洢������ڴ�
            }
        }
    }

    for (int i = N - 1; i >= 0; i--)
    {
        float32x4_t result_i = vdupq_n_f32(result[i]); // ���� result[i] �� NEON �Ĵ�����
        for (int j = i - 1; j >= 0; j--)
        {
            float32x4_t m_j_i = vld1q_f32(&m[j][i]); // ���� m[j][i] �� NEON �Ĵ�����
            float32x4_t product = vmulq_f32(result_i, m_j_i); // �˷�����
            float32x4_t m_j_N = vld1q_f32(&m[j][N]); // ���� m[j][N] �� NEON �Ĵ�����
            m_j_N = vsubq_f32(m_j_N, product); // ��������
            vst1q_f32(&m[j][N], m_j_N); // �洢������ڴ�
        }
    }
}
int main()
{
    timeval start_time, end_time;

   /* for (int i = 0; i < t; i++)
    {
        double total_time = 0.0;
        for (int j = 0; j < T; j++)
        {
            m_reset(nm[i]);
            gettimeofday(&start_time, NULL);

            guass_elimination_neon1(nm[i]);

            gettimeofday(&end_time, NULL);
            double elapsed_time = (end_time.tv_sec - start_time.tv_sec) * 1000.0 + (end_time.tv_usec - start_time.tv_usec) / 1000.0;
            total_time += elapsed_time;
        }
        // ���ƽ��ʱ��
        cout << "scale��" << nm[i] << " neon1time��";
        cout << total_time / T << "ms" << endl;
    }

    for (int i = 0; i < t; i++)
    {
        double total_time = 0.0;
        for (int j = 0; j < T; j++)
        {
            m_reset(nm[i]);
            gettimeofday(&start_time, NULL);

            guass_elimination_neon2(nm[i]);

            gettimeofday(&end_time, NULL);
            double elapsed_time = (end_time.tv_sec - start_time.tv_sec) * 1000.0 + (end_time.tv_usec - start_time.tv_usec) / 1000.0;
            total_time += elapsed_time;
        }
        // ���ƽ��ʱ��
        cout << "scale��" << nm[i] << " neon2time��";
        cout << total_time / T << "ms" << endl;
    }

   for (int i = 0; i < t; i++)
    {
        double total_time = 0.0;
        for (int j = 0; j < T; j++)
        {
            m_reset(nm[i]);
            gettimeofday(&start_time, NULL);

            guass_elimination_neon3(nm[i]);

            gettimeofday(&end_time, NULL);
            double elapsed_time = (end_time.tv_sec - start_time.tv_sec) * 1000.0 + (end_time.tv_usec - start_time.tv_usec) / 1000.0;
            total_time += elapsed_time;
        }
        // ���ƽ��ʱ��
        cout << "scale��" << nm[i] << " neon3time��";
        cout << total_time / T << "ms" << endl;
    }*/

    for (int i = 0; i < t; i++)
    {
        double total_time = 0.0;
        for (int j = 0; j < T; j++)
        {
            m_reset(nm[i]);
            gettimeofday(&start_time, NULL);

            guass_elimination_neon_all(nm[i]);

            gettimeofday(&end_time, NULL);
            double elapsed_time = (end_time.tv_sec - start_time.tv_sec) * 1000.0 + (end_time.tv_usec - start_time.tv_usec) / 1000.0;
            total_time += elapsed_time;
        }
        // ���ƽ��ʱ��
        cout << "scale��" << nm[i] << " neonalltime��";
        cout << total_time / T << "ms" << endl;
    }
    return 0;
}
