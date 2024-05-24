//���ļ����е���̽���˲�ȡ��ͬ��̲��ԣ���̬��̡���̬+�ź�������̬+barrier��ʱ��
// ��ͨ��˹��ȥ�㷨�ڲ�ͬ��ģ��500��1000��3000��6000���Ͳ�ͬ�߳���Ŀ��4��6��8���µ����ܱ���
//����ˮƽ���ֵķ�ʽ����avx����

#include <iostream>
#include <windows.h>
#include <stdlib.h>
#include <pthread.h>
#include <semaphore.h>
#include <nmmintrin.h>

using namespace std;

//�����ģ
const int N = 5000;

//ϵ������
//float **m;
float m[N][N];

void m_reset(int);
void m_gauss(int);
void m_gauss_d(int);
void m_gauss_h(int, int);
void m_gauss_v(int, int);
void print(int);

int main()
{
    long long head, tail, freq; // timers
    int step = 50; int a;

    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);

    for (int n = 50; n <= N; n += step)
    {
        cout << "�����ģn: " << n << endl;

        m_reset(n);
        QueryPerformanceCounter((LARGE_INTEGER*)&head);
        m_gauss(n);
        QueryPerformanceCounter((LARGE_INTEGER*)&tail);
        cout << "�����㷨ʱ�䣺" << (tail - head) * 1000.0 / freq << "ms" << endl;

        if (n <= 300) {
            m_reset(n);
            QueryPerformanceCounter((LARGE_INTEGER*)&head);
            m_gauss_d(n);
            QueryPerformanceCounter((LARGE_INTEGER*)&tail);
            cout << "��̬�̰߳汾ʱ�䣺" << (tail - head) * 1000.0 / freq << "ms" << endl;
        }

        int s = 1;
        //��ͬ�߳���
        for (int i = 2; i <= 21; i += s) {
            cout << "�߳���: " << i << endl;
            m_reset(n);
            QueryPerformanceCounter((LARGE_INTEGER*)&head);
            m_gauss_h(n, i);
            QueryPerformanceCounter((LARGE_INTEGER*)&tail);
            cout << "��̬�̡߳�ˮƽ���ְ汾ʱ�䣺" << (tail - head) * 1000.0 / freq << "ms" << endl;

            m_reset(n);
            QueryPerformanceCounter((LARGE_INTEGER*)&head);
            m_gauss_v(n, i);
            QueryPerformanceCounter((LARGE_INTEGER*)&tail);
            cout << "��̬�̡߳���ֱ���ְ汾ʱ�䣺" << (tail - head) * 1000.0 / freq << "ms" << endl;

            //if(i == 5) s = 2;
            if (i == 9) s = 4;
        }

        if (n == 100) step = 300;
        if (n == 1000) step = 1000;
    }

    return 0;
}

//��ʼ������Ԫ��
void m_reset(int n)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < i; j++) {
            m[i][j] = 0;
        }
        m[i][i] = 1.0;
        for (int j = i + 1; j < n; j++) {
            //  m[i][j]=rand()%10000+1;
            m[i][j] = rand();
        }

    }
    for (int k = 0; k < n; k++) {
        for (int i = k + 1; i < n; i++) {
            for (int j = 0; j < n; j++) {
                m[i][j] += m[k][j];
            }
        }
    }
}

//������ͨ��˹��ȥ�㷨
void m_gauss(int n)
{
    for (int k = 0; k < n; k++)
    {
        for (int j = k + 1; j < n; j++)
        {
            m[k][j] = m[k][j] / m[k][k];
        }
        m[k][k] = 1.0;
        for (int i = k + 1; i < n; i++)
        {
            for (int j = k + 1; j < n; j++)
            {
                m[i][j] = m[i][j] - m[i][k] * m[k][j];
            }
            m[i][k] = 0;

        }
    }
}

//��̬�̰߳汾���ݽṹ
typedef struct {
    int k; //��ȥ���ִ�
    int t_id; // �߳� id
    int n; //�����ģ
}threadParam_t_d;

//��̬�̰߳汾�̺߳���
void* threadFunc_d(void* param) {
    threadParam_t_d* p = (threadParam_t_d*)param;
    int k = p->k; //��ȥ���ִ�
    int n = p->n; //�����ģ
    int t_id = p->t_id; //�̱߳��
    int i = k + t_id + 1; //��ȡ�Լ��ļ�������

    __m128 vaik, vakj, vaij, vx;
    vaik = _mm_set_ps1(m[i][k]);
    int j;
    int start = k - k % 4 + 4;
    for (j = k + 1; j < start && j < n; j++) {
        m[i][j] = m[i][j] - m[k][j] * m[i][k];
    }
    if (j != n) {
        for (j = start; j + 4 <= n; j += 4) {
            vakj = _mm_load_ps(&m[k][j]);
            vaij = _mm_load_ps(&m[i][j]);
            vx = _mm_mul_ps(vakj, vaik);
            vaij = _mm_sub_ps(vaij, vx);
            _mm_store_ps(&m[i][j], vaij);
        }
        if (j < n) {
            for (; j < n; j++) {
                m[i][j] = m[i][j] - m[k][j] * m[i][k];
            }
        }
    }
    m[i][k] = 0;
    pthread_exit(NULL);
}

//��̬�̰߳汾
void m_gauss_d(int n) {
    for (int k = 0; k < n; ++k) {
        //���߳�����������
        __m128 vt, va;
        vt = _mm_set_ps1(m[k][k]);
        int j;
        int start = k - k % 4 + 4;
        for (j = k + 1; j < start && j < n; j++) {
            m[k][j] = m[k][j] / m[k][k];
        }
        if (j != n) {
            for (j = start; j + 4 <= n; j += 4) {
                va = _mm_load_ps(&m[k][j]);
                va = _mm_div_ps(va, vt);
                _mm_store_ps(&m[k][j], va);
            }
            if (j < n) {
                for (; j < n; j++) {
                    m[k][j] = m[k][j] / m[k][k];
                }
            }
        }
        m[k][k] = 1.0;

        //���������̣߳�������ȥ����
        int worker_count = n - 1 - k; //�����߳�����
        pthread_t* handles = (pthread_t*)malloc(worker_count * sizeof(pthread_t));// ������Ӧ�� Handle
        threadParam_t_d* param = (threadParam_t_d*)malloc(worker_count * sizeof(threadParam_t_d));// ������Ӧ���߳����ݽṹ
        //��������
        for (int t_id = 0; t_id < worker_count; t_id++) {
            param[t_id].k = k;
            param[t_id].n = n;
            param[t_id].t_id = t_id;
        }

        //�����߳�
        for (int t_id = 0; t_id < worker_count; t_id++) {
            pthread_create(&handles[t_id], NULL, threadFunc_d, (void*)&param[t_id]);
        }

        //���̹߳���ȴ����еĹ����߳���ɴ�����ȥ����
        for (int t_id = 0; t_id < worker_count; t_id++) {
            pthread_join(handles[t_id], NULL);
        }

        free(handles);
        free(param);
    }
}

//��̬�̰߳汾�߳����ݽṹ����
typedef struct {
    int t_id; //�߳� id
    int num_threads; //�߳���
    int n; //�����ģ
}threadParam_t;

//�ź�������
sem_t sem_Divsion;
sem_t sem_Elimination;

//��̬�߳�+�ź���ͬ����ȫ�������̣߳�ˮƽ���ְ汾�̺߳�������
void* threadFunc_h(void* param) {
    threadParam_t* p = (threadParam_t*)param;
    int t_id = p->t_id;
    int num_threads = p->num_threads;
    int n = p->n;
    __m128 vt, va, vaik, vakj, vaij, vx;

    for (int k = 0; k < n; ++k) {
        // t_id Ϊ 0 ���߳����������������������߳��ȵȴ�
        if (t_id == 0) {
            vt = _mm_set_ps1(m[k][k]);
            int j;
            int start = k - k % 4 + 4;
            for (j = k + 1; j < start && j < n; j++) {
                m[k][j] = m[k][j] / m[k][k];
            }
            if (j != n) {
                for (j = start; j + 4 <= n; j += 4) {
                    va = _mm_load_ps(&m[k][j]);
                    va = _mm_div_ps(va, vt);
                    _mm_store_ps(&m[k][j], va);
                }
                if (j < n) {
                    for (; j < n; j++) {
                        m[k][j] = m[k][j] / m[k][k];
                    }
                }
            }
            m[k][k] = 1.0;
        }
        else {
            sem_wait(&sem_Divsion); // �������ȴ���ɳ�������
        }

        // t_id Ϊ 0 ���̻߳������������̣߳�������ȥ����
        if (t_id == 0) {
            for (int t_id = 0; t_id < num_threads - 1; ++t_id) {
                sem_post(&sem_Divsion);
            }
        }

        //ѭ����������
        for (int i = k + 1 + t_id; i < n; i += num_threads) {
            //��ȥ
            vaik = _mm_set_ps1(m[i][k]);
            int j;
            int start = k - k % 4 + 4;
            for (j = k + 1; j < start && j < n; j++) {
                m[i][j] = m[i][j] - m[k][j] * m[i][k];
            }
            if (j != n) {
                for (j = start; j + 4 <= n; j += 4) {
                    vakj = _mm_load_ps(&m[k][j]);
                    vaij = _mm_load_ps(&m[i][j]);
                    vx = _mm_mul_ps(vakj, vaik);
                    vaij = _mm_sub_ps(vaij, vx);
                    _mm_store_ps(&m[i][j], vaij);
                }
                if (j < n) {
                    for (; j < n; j++) {
                        m[i][j] = m[i][j] - m[k][j] * m[i][k];
                    }
                }
            }
            m[i][k] = 0.0;
        }

        // �����߳�һ�������һ��
        if (t_id == 0) {
            for (int t_id = 0; t_id < num_threads - 1; ++t_id) {
                sem_post(&sem_Elimination);
            }
        }
        else {
            sem_wait(&sem_Elimination);
        }
    }

    pthread_exit(NULL);
}

//��̬�߳�+�ź���ͬ����ȫ�������̣߳�ˮƽ���ְ汾
void m_gauss_h(int n, int num_threads) {
    //��ʼ���ź���
    sem_init(&sem_Divsion, 0, 0);
    sem_init(&sem_Elimination, 0, 0);


    //�����߳�
    pthread_t handles[num_threads];// ������Ӧ�� Handle
    threadParam_t param[num_threads];// ������Ӧ���߳����ݽṹ
    for (int t_id = 0; t_id < num_threads; t_id++) {
        param[t_id].t_id = t_id;
        param[t_id].n = n;
        param[t_id].num_threads = num_threads;
        pthread_create(&handles[t_id], NULL, threadFunc_h, (void*)&param[t_id]);
    }

    for (int t_id = 0; t_id < num_threads; t_id++) {
        pthread_join(handles[t_id], NULL);
    }

    sem_destroy(&sem_Divsion);
    sem_destroy(&sem_Elimination);
}

//barrier����
pthread_barrier_t barrier_Divsion;
pthread_barrier_t barrier_Elimination;
int num_blocks;//�黮�ִ�С

//��̬�߳�+�ź���ͬ����ȫ�������̣߳���ֱ���ְ汾�̺߳�������
void* threadFunc_v(void* param) {
    threadParam_t* p = (threadParam_t*)param;
    int t_id = p->t_id;
    int num_threads = p->num_threads;
    int n = p->n;
    __m128 vt, va, vaik, vakj, vaij, vx;

    for (int k = 0; k < n; ++k) {
        // t_id Ϊ 0 ���߳����������������������߳��ȵȴ�
        if (t_id == 0) {
            vt = _mm_set_ps1(m[k][k]);
            int j;
            int start = k - k % 4 + 4;
            for (j = k + 1; j < start && j < n; j++) {
                m[k][j] = m[k][j] / m[k][k];
            }
            if (j != n) {
                for (j = start; j + 4 <= n; j += 4) {
                    va = _mm_load_ps(&m[k][j]);
                    va = _mm_div_ps(va, vt);
                    _mm_store_ps(&m[k][j], va);
                }
                if (j < n) {
                    for (; j < n; j++) {
                        m[k][j] = m[k][j] / m[k][k];
                    }
                }
            }
            m[k][k] = 1.0;
            num_blocks = (n - k - 1) / num_threads;

            if (num_blocks < 8) {
                num_blocks = 8;
            }
        }

        // �������ȴ���ɳ�������
        pthread_barrier_wait(&barrier_Divsion);

        //�黮������
        int j = k + 1 + t_id * num_blocks;
        int o = j;
        if (j < n) {
            int my_end = j + num_blocks;
            if (my_end > n) {
                my_end = n;
            }

            if (t_id == num_threads - 1) {
                my_end = n;
            }

            int my_start = j - j % 4 + 4;
            for (int i = k + 1; i < n; i++) {
                j = o;
                for (; j < my_start && j < my_end; j++) {
                    m[i][j] = m[i][j] - m[k][j] * m[i][k];
                }

                if (j != my_end) {
                    //��ȥ
                    vaik = _mm_set_ps1(m[i][k]);
                    for (j = my_start; j + 4 <= my_end; j += 4) {
                        vakj = _mm_load_ps(&m[k][j]);
                        vaij = _mm_load_ps(&m[i][j]);
                        vx = _mm_mul_ps(vakj, vaik);
                        vaij = _mm_sub_ps(vaij, vx);
                        _mm_store_ps(&m[i][j], vaij);
                    }
                    if (j < my_end) {
                        for (; j < my_end; j++) {
                            m[i][j] = m[i][j] - m[k][j] * m[i][k];
                        }
                    }
                }
            }
        }


        // �����߳�һ�������һ��
        pthread_barrier_wait(&barrier_Elimination);

        if (t_id == 1) {
            for (int i = k + 1; i < n; i++) {
                m[i][k] = 0.0;
            }
        }

    }

    pthread_exit(NULL);
}

//��̬�߳�+�ź���ͬ����ȫ�������̣߳���ֱ���ְ汾
void m_gauss_v(int n, int num_threads) {
    //��ʼ��barrier
    pthread_barrier_init(&barrier_Divsion, NULL, num_threads);
    pthread_barrier_init(&barrier_Elimination, NULL, num_threads);

    //�����߳�
    pthread_t handles[num_threads];// ������Ӧ�� Handle
    threadParam_t param[num_threads];// ������Ӧ���߳����ݽṹ
    for (int t_id = 0; t_id < num_threads; t_id++) {
        param[t_id].t_id = t_id;
        param[t_id].n = n;
        param[t_id].num_threads = num_threads;
        pthread_create(&handles[t_id], NULL, threadFunc_v, (void*)&param[t_id]);
    }

    for (int t_id = 0; t_id < num_threads; t_id++) {
        pthread_join(handles[t_id], NULL);
    }

    pthread_barrier_destroy(&barrier_Divsion);
    pthread_barrier_destroy(&barrier_Elimination);

}

void print(int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            cout << m[i][j] << " ";
        }
        cout << endl;
    }
}
