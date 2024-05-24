//该文件进行的是探讨了采取不同编程策略（动态编程、静态+信号量、静态+barrier）时，
// 普通高斯消去算法在不同规模（500，1000，3000，6000）和不同线程数目（4、6、8）下的性能表现
//采用水平划分的方式，与avx相结合

#include <iostream>
#include <windows.h>
#include <stdlib.h>
#include <pthread.h>
#include <semaphore.h>
#include <nmmintrin.h>

using namespace std;

//矩阵规模
const int N = 5000;

//系数矩阵
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
        cout << "问题规模n: " << n << endl;

        m_reset(n);
        QueryPerformanceCounter((LARGE_INTEGER*)&head);
        m_gauss(n);
        QueryPerformanceCounter((LARGE_INTEGER*)&tail);
        cout << "串行算法时间：" << (tail - head) * 1000.0 / freq << "ms" << endl;

        if (n <= 300) {
            m_reset(n);
            QueryPerformanceCounter((LARGE_INTEGER*)&head);
            m_gauss_d(n);
            QueryPerformanceCounter((LARGE_INTEGER*)&tail);
            cout << "动态线程版本时间：" << (tail - head) * 1000.0 / freq << "ms" << endl;
        }

        int s = 1;
        //不同线程数
        for (int i = 2; i <= 21; i += s) {
            cout << "线程数: " << i << endl;
            m_reset(n);
            QueryPerformanceCounter((LARGE_INTEGER*)&head);
            m_gauss_h(n, i);
            QueryPerformanceCounter((LARGE_INTEGER*)&tail);
            cout << "静态线程、水平划分版本时间：" << (tail - head) * 1000.0 / freq << "ms" << endl;

            m_reset(n);
            QueryPerformanceCounter((LARGE_INTEGER*)&head);
            m_gauss_v(n, i);
            QueryPerformanceCounter((LARGE_INTEGER*)&tail);
            cout << "静态线程、垂直划分版本时间：" << (tail - head) * 1000.0 / freq << "ms" << endl;

            //if(i == 5) s = 2;
            if (i == 9) s = 4;
        }

        if (n == 100) step = 300;
        if (n == 1000) step = 1000;
    }

    return 0;
}

//初始化矩阵元素
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

//串行普通高斯消去算法
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

//动态线程版本数据结构
typedef struct {
    int k; //消去的轮次
    int t_id; // 线程 id
    int n; //问题规模
}threadParam_t_d;

//动态线程版本线程函数
void* threadFunc_d(void* param) {
    threadParam_t_d* p = (threadParam_t_d*)param;
    int k = p->k; //消去的轮次
    int n = p->n; //问题规模
    int t_id = p->t_id; //线程编号
    int i = k + t_id + 1; //获取自己的计算任务

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

//动态线程版本
void m_gauss_d(int n) {
    for (int k = 0; k < n; ++k) {
        //主线程做除法操作
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

        //创建工作线程，进行消去操作
        int worker_count = n - 1 - k; //工作线程数量
        pthread_t* handles = (pthread_t*)malloc(worker_count * sizeof(pthread_t));// 创建对应的 Handle
        threadParam_t_d* param = (threadParam_t_d*)malloc(worker_count * sizeof(threadParam_t_d));// 创建对应的线程数据结构
        //分配任务
        for (int t_id = 0; t_id < worker_count; t_id++) {
            param[t_id].k = k;
            param[t_id].n = n;
            param[t_id].t_id = t_id;
        }

        //创建线程
        for (int t_id = 0; t_id < worker_count; t_id++) {
            pthread_create(&handles[t_id], NULL, threadFunc_d, (void*)&param[t_id]);
        }

        //主线程挂起等待所有的工作线程完成此轮消去工作
        for (int t_id = 0; t_id < worker_count; t_id++) {
            pthread_join(handles[t_id], NULL);
        }

        free(handles);
        free(param);
    }
}

//静态线程版本线程数据结构定义
typedef struct {
    int t_id; //线程 id
    int num_threads; //线程数
    int n; //问题规模
}threadParam_t;

//信号量定义
sem_t sem_Divsion;
sem_t sem_Elimination;

//静态线程+信号量同步，全部工作线程，水平划分版本线程函数定义
void* threadFunc_h(void* param) {
    threadParam_t* p = (threadParam_t*)param;
    int t_id = p->t_id;
    int num_threads = p->num_threads;
    int n = p->n;
    __m128 vt, va, vaik, vakj, vaij, vx;

    for (int k = 0; k < n; ++k) {
        // t_id 为 0 的线程做除法操作，其它工作线程先等待
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
            sem_wait(&sem_Divsion); // 阻塞，等待完成除法操作
        }

        // t_id 为 0 的线程唤醒其它工作线程，进行消去操作
        if (t_id == 0) {
            for (int t_id = 0; t_id < num_threads - 1; ++t_id) {
                sem_post(&sem_Divsion);
            }
        }

        //循环划分任务
        for (int i = k + 1 + t_id; i < n; i += num_threads) {
            //消去
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

        // 所有线程一起进入下一轮
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

//静态线程+信号量同步，全部工作线程，水平划分版本
void m_gauss_h(int n, int num_threads) {
    //初始化信号量
    sem_init(&sem_Divsion, 0, 0);
    sem_init(&sem_Elimination, 0, 0);


    //创建线程
    pthread_t handles[num_threads];// 创建对应的 Handle
    threadParam_t param[num_threads];// 创建对应的线程数据结构
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

//barrier定义
pthread_barrier_t barrier_Divsion;
pthread_barrier_t barrier_Elimination;
int num_blocks;//块划分大小

//静态线程+信号量同步，全部工作线程，垂直划分版本线程函数定义
void* threadFunc_v(void* param) {
    threadParam_t* p = (threadParam_t*)param;
    int t_id = p->t_id;
    int num_threads = p->num_threads;
    int n = p->n;
    __m128 vt, va, vaik, vakj, vaij, vx;

    for (int k = 0; k < n; ++k) {
        // t_id 为 0 的线程做除法操作，其它工作线程先等待
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

        // 阻塞，等待完成除法操作
        pthread_barrier_wait(&barrier_Divsion);

        //块划分任务
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
                    //消去
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


        // 所有线程一起进入下一轮
        pthread_barrier_wait(&barrier_Elimination);

        if (t_id == 1) {
            for (int i = k + 1; i < n; i++) {
                m[i][k] = 0.0;
            }
        }

    }

    pthread_exit(NULL);
}

//静态线程+信号量同步，全部工作线程，垂直划分版本
void m_gauss_v(int n, int num_threads) {
    //初始化barrier
    pthread_barrier_init(&barrier_Divsion, NULL, num_threads);
    pthread_barrier_init(&barrier_Elimination, NULL, num_threads);

    //创建线程
    pthread_t handles[num_threads];// 创建对应的 Handle
    threadParam_t param[num_threads];// 创建对应的线程数据结构
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
