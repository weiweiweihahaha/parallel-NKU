#include <iostream>
#include <cstdlib>
#include <sys/time.h>

using namespace std;

const int n = 6072;
/*int t = 13;
int T = 5; // 循环五次，取时间均值
int nm[13] = {500, 600, 700, 800, 1000, 1200, 1400, 1700, 2000, 2200, 2500, 2800, 3000};*/
int t = 5;
int T = 1; // 循环五次，取时间均值
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

void guass_elimination_ordinary(int N)
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
            m[j][N] -= m[j][i] * result[i];
        }
    }
}

int main()
{
    timeval start_time, end_time;

    for (int i = 0; i < t; i++)
    {
        double total_time = 0.0;
        for (int j = 0; j < T; j++)
        {
            m_reset(nm[i]);
            gettimeofday(&start_time, NULL);

            guass_elimination_ordinary(nm[i]);

            gettimeofday(&end_time, NULL);
            double elapsed_time = (end_time.tv_sec - start_time.tv_sec) * 1000.0 + (end_time.tv_usec - start_time.tv_usec) / 1000.0;
            total_time += elapsed_time;
        }
        // 输出平均时间
        cout << "数据规模为：" << nm[i] << " 平均消耗时间为：";
        cout << total_time / T << "ms" << endl;
    }
    return 0;
}
