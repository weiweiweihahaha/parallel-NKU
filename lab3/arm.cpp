#include <iostream>
#include <stdlib.h>
#include <pthread.h>
#include <semaphore.h>
#include <arm_neon.h>
#include <sys/time.h>

using namespace std;

// 矩阵规模
const int N = 6000;

// 系数矩阵
float m[N][N];

void m_reset(int);
void m_gauss(int);
void m_gauss_h(int, int);
long long get_current_time();

int main() {
    long long head, tail;
    int step = 500;

    for(int n = 500; n <= 2000; n += step) {
        cout << "问题规模n: " << n << endl;

        m_reset(n);
        head = get_current_time();
        m_gauss(n);
        tail = get_current_time();
        cout << "串行算法时间：" << (tail - head) << "ms" << endl;

        int s = 4;
        for(int i = 4; i <= 21; i += s) {
            cout << "线程数: " << i << endl;
            m_reset(n);
            head = get_current_time();
            m_gauss_h(n, i);
            tail = get_current_time();
            cout << "静态线程、水平划分、信号量同步版本时间：" << (tail - head) << "ms" << endl;

            if(i == 9) s = 4;
        }
    }

    return 0;
}

// 获取当前时间，单位ms
long long get_current_time() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec * 1000 + tv.tv_usec / 1000;
}

// 初始化矩阵元素
void m_reset(int n) {
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < i; j++) {
            m[i][j] = 0;
        }
        m[i][i] = 1.0;
        for(int j = i + 1; j < n; j++) {
            m[i][j] = rand();
        }
    }
    for(int k = 0; k < n; k++) {
        for(int i = k + 1; i < n; i++) {
            for(int j = 0; j < n; j++) {
                m[i][j] += m[k][j];
            }
        }
    }
}

// 串行普通高斯消去算法
void m_gauss(int n) {
    for(int k = 0; k < n; k++) {
        for(int j = k + 1; j < n; j++) {
            m[k][j] = m[k][j] / m[k][k];
        }
        m[k][k] = 1.0;
        for(int i = k + 1; i < n; i++) {
            for(int j = k + 1; j < n; j++) {
                m[i][j] = m[i][j] - m[i][k] * m[k][j];
            }
            m[i][k] = 0;
        }
    }
}

// 静态线程版本线程数据结构定义
typedef struct {
    int t_id; // 线程 id
    int num_threads; // 线程数
    int n; // 问题规模
} threadParam_t;

// 信号量定义
sem_t sem_Divsion;
sem_t sem_Elimination;

// 静态线程+信号量同步，全部工作线程，水平划分版本线程函数定义
void *threadFunc_h(void *param) {
    threadParam_t *p = (threadParam_t*)param;
    int t_id = p->t_id;
    int num_threads = p->num_threads;
    int n = p->n;
    float32x4_t vt, va, vaik, vakj, vaij, vx;

    for(int k = 0; k < n; ++k) {
        if(t_id == 0) {
            vt = vdupq_n_f32(m[k][k]);
            int j;
            int start = k - k % 4 + 4;
            for(j = k + 1; j < start && j < n; j++) {
                m[k][j] = m[k][j] / m[k][k];
            }
            if(j != n) {
                for(j = start; j + 4 <= n; j += 4) {
                    va = vld1q_f32(&m[k][j]);
                    va = vdivq_f32(va, vt);
                    vst1q_f32(&m[k][j], va);
                }
                if(j < n) {
                    for(; j < n; j++) {
                        m[k][j] = m[k][j] / m[k][k];
                    }
                }
            }
            m[k][k] = 1.0;
        }
        sem_post(&sem_Divsion);
        sem_wait(&sem_Elimination);

        for(int i = k + 1 + t_id; i < n; i += num_threads) {
            vaik = vdupq_n_f32(m[i][k]);
            int j;
            int start = k - k % 4 + 4;
            for(j = k + 1; j < start && j < n; j++) {
                m[i][j] = m[i][j] - m[k][j] * m[i][k];
            }
            if(j != n) {
                for(j = start; j + 4 <= n; j += 4) {
                    vakj = vld1q_f32(&m[k][j]);
                    vaij = vld1q_f32(&m[i][j]);
                    vx = vmulq_f32(vakj, vaik);
                    vaij = vsubq_f32(vaij, vx);
                    vst1q_f32(&m[i][j], vaij);
                }
                if(j < n) {
                    for(; j < n; j++) {
                        m[i][j] = m[i][j] - m[k][j] * m[i][k];
                    }
                }
            }
            m[i][k] = 0;
        }
        sem_post(&sem_Elimination);
    }
    pthread_exit(nullptr);
}

// 静态线程+信号量同步，水平划分版本高斯消去
void m_gauss_h(int n, int num_threads) {
    pthread_t *handles = (pthread_t*)malloc(num_threads * sizeof(pthread_t));
    threadParam_t *param = (threadParam_t*)malloc(num_threads * sizeof(threadParam_t));

    sem_init(&sem_Divsion, 0, 0);
    sem_init(&sem_Elimination, 0, 0);

    for(int t_id = 0; t_id < num_threads; t_id++) {
        param[t_id].t_id = t_id;
        param[t_id].num_threads = num_threads;
        param[t_id].n = n;
        pthread_create(&handles[t_id], nullptr, threadFunc_h, (void*)&param[t_id]);
    }

    for(int k = 0; k < n; k++) {
        sem_wait(&sem_Divsion);
        sem_post(&sem_Elimination);
    }

    for(int t_id = 0; t_id < num_threads; t_id++) {
        pthread_join(handles[t_id], nullptr);
    }

    sem_destroy(&sem_Divsion);
    sem_destroy(&sem_Elimination);
    free(handles);
    free(param);
}

