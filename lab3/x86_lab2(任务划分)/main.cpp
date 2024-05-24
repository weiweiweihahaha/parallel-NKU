#include <iostream>
#include <windows.h>
#include <stdlib.h>
#include <pthread.h>
#include <semaphore.h>
#include <nmmintrin.h>
#include <immintrin.h>
using namespace std;

//矩阵规模
const int N = 2000;

//系数矩阵
//float **m;
float m[N][N];

void m_reset(int);

void m_gauss_h1(int, int);
void m_gauss_h2(int, int);
void m_gauss_v(int, int);
void m_gauss_v1(int, int);
void print(int);

using namespace std;

int main()
{
    long long head, tail , freq ; // timers
    int step = 500;int a;

    QueryPerformanceFrequency((LARGE_INTEGER *)&freq );

    for(int n = 2000; n <= N; n += step)
    {
        cout << "问题规模n: " << n << endl;

         int s = 4;
         int i=8;
       // for(int i = 4; i <= 21; i += s){
       // if(n <= 300){
            cout << "线程数: " << i << endl;


             m_reset(n);
            QueryPerformanceCounter((LARGE_INTEGER *)&head);
            for(int w=0;w<5;w++)
                m_gauss_h2(n, i);
            QueryPerformanceCounter((LARGE_INTEGER *)&tail );
            cout << "静态线程、水平（穿插）划分、barrier同步版本时间：" << ( tail - head) * 1000.0 /(freq *5) << "ms" << endl;

//              m_reset(n);
//            QueryPerformanceCounter((LARGE_INTEGER *)&head);
//            for(int w=0;w<5;w++)
//                m_gauss_h1(n, i);
//            QueryPerformanceCounter((LARGE_INTEGER *)&tail );
//            cout << "静态线程、水平（块）划分、barrier同步版本时间：" << ( tail - head) * 1000.0 /(freq*5 ) << "ms" << endl;

//
//            m_reset(n);
//            QueryPerformanceCounter((LARGE_INTEGER *)&head);
//            for(int w=0;w<5;w++)
//                m_gauss_v1(n, i);
//            QueryPerformanceCounter((LARGE_INTEGER *)&tail );
//            cout << "静态线程、垂直（块）划分、barrier同步版本时间：" << ( tail - head) * 1000.0 /(freq*5 )<< "ms" << endl;

//             m_reset(n);
//            QueryPerformanceCounter((LARGE_INTEGER *)&head);
//            for(int w=0;w<5;w++)
//                m_gauss_v(n, i);
//            QueryPerformanceCounter((LARGE_INTEGER *)&tail );
//            cout << "静态线程、垂直（穿插）划分、barrier同步版本时间：" << ( tail - head) * 1000.0 /(freq*5 )<< "ms" << endl;

            //if(i == 5) s = 2;
            if(i == 9) s = 4;
     //   }

       // if(n == 1000) step = 1000;
        //if(n == 2000) step = 2000;
    }

    return 0;
}

//初始化矩阵元素
void m_reset(int n)
{
    for(int i=0;i<n;i++)
    {
        for(int j=0;j<i;j++){
            m[i][j]=0;
        }
        m[i][i]=1.0;
        for(int j=i+1;j<n;j++){
          //  m[i][j]=rand()%10000+1;
            m[i][j]=rand();
        }

    }
    for(int k=0;k<n;k++){
        for(int i=k+1;i<n;i++){
             for(int j=0;j<n;j++){
                m[i][j]+=m[k][j];
             }
        }
    }
}

//串行普通高斯消去算法
void m_gauss(int n)
{
    for(int k = 0 ; k < n ; k++)
    {
        for(int j = k+1 ; j < n ; j++)
        {
            m[k][j] = m[k][j]/m[k][k];
        }
        m[k][k] = 1.0;
        for(int i = k+1 ; i < n ; i++)
        {
            for(int j = k+1 ; j < n ; j++)
            {
                m[i][j] = m[i][j] - m[i][k] * m[k][j];
            }
            m[i][k] = 0;

        }
    }
}

//静态线程版本线程数据结构定义
typedef struct {
    int t_id; //线程 id
    int num_threads; //线程数
    int n; //问题规模
}threadParam_t;

//barrier定义
pthread_barrier_t barrier_Divsion2;
pthread_barrier_t barrier_Elimination2;


//静态线程+barrier同步，全部工作线程，水平划分（穿插划分）版本线程函数定义
void *threadFunc_h2(void *param) {
    threadParam_t *p = (threadParam_t*)param;
    int t_id = p -> t_id;
    int num_threads = p -> num_threads;
    int n = p -> n;
    __m128 vt, va, vaik, vakj, vaij, vx;

//    for(int k = 0; k < n; ++k){
//        // t_id 为 0 的线程做除法操作，其它工作线程先等待
//        if (t_id == 0){
//            vt = _mm_set_ps1(m[k][k]);
//            int j;
//            int start = k-k%4+4;
//            for(j = k+1; j < start && j < n; j++){
//                m[k][j] = m[k][j]/m[k][k];
//            }
//            if(j != n){
//                for(j = start; j+4 <= n; j+=4){
//                    va = _mm_load_ps(&m[k][j]);
//                    va = _mm_div_ps(va, vt);
//                    _mm_store_ps(&m[k][j], va);
//                }
//                if(j < n){
//                    for(;j < n; j++){
//                        m[k][j] = m[k][j]/m[k][k];
//                    }
//                }
//            }
//            m[k][k] = 1.0;
//            num_blocks = (n-k-1)/num_threads;
//
//            if(num_blocks < 8){
//                num_blocks = 8;
//            }
//        }
    for(int k = 0; k < n; ++k){
        // t_id 为 0 的线程做除法操作，其它工作线程先等待
        if (t_id == 0){
            vt = _mm_set_ps1(m[k][k]);
            int j;
            int start = k-k%4+4;
            for(j = k+1; j < start && j < n; j++){
                m[k][j] = m[k][j]/m[k][k];
            }
            if(j != n){
                for(j = start; j+4 <= n; j+=4){
                    va = _mm_load_ps(&m[k][j]);
                    va = _mm_div_ps(va, vt);
                    _mm_store_ps(&m[k][j], va);
                }
                if(j < n){
                    for(;j < n; j++){
                        m[k][j] = m[k][j]/m[k][k];
                    }
                }
            }
            m[k][k] = 1.0;
        }
        // 阻塞，等待完成除法操作
        pthread_barrier_wait(&barrier_Divsion2);
//
//        //块划分任务
//        int j = k+1+t_id*num_blocks;
//        int o = j;
//        if(j < n){
//            int my_end = j+num_blocks;
//            if(my_end > n){
//                my_end = n;
//            }
//
//            if(t_id == num_threads-1){
//                my_end = n;
//            }
//
//            int my_start = j-j%4+4;
//            for(int i = k+1; i < n; i++){
//                j = o;
//                for(; j < my_start && j < my_end; j++){
//                    m[i][j] = m[i][j] - m[k][j]*m[i][k];
//                }
//
//                if(j != my_end){
//                    //消去
//                    vaik = _mm_set_ps1(m[i][k]);
//                    for(j = my_start; j+4 <= my_end; j+=4){
//                        vakj = _mm_load_ps(&m[k][j]);
//                        vaij = _mm_load_ps(&m[i][j]);
//                        vx = _mm_mul_ps(vakj, vaik);
//                        vaij = _mm_sub_ps(vaij, vx);
//                        _mm_store_ps(&m[i][j], vaij);
//                    }
//                    if(j < my_end){
//                        for(;j < my_end; j++){
//                            m[i][j] = m[i][j] - m[k][j]*m[i][k];
//                        }
//                    }
//                }
//            }
//        }

        for(int i=k+1+t_id; i < n; i += num_threads){
            //消去
            vaik = _mm_set_ps1(m[i][k]);
            int j;
            int start = k-k%4+4;
            for(j = k+1; j < start && j < n; j++){
                m[i][j] = m[i][j] - m[k][j]*m[i][k];
            }
            if(j != n){
                for(j = start; j+4 <= n; j+=4){
                    vakj = _mm_load_ps(&m[k][j]);
                    vaij = _mm_load_ps(&m[i][j]);
                    vx = _mm_mul_ps(vakj, vaik);
                    vaij = _mm_sub_ps(vaij, vx);
                    _mm_store_ps(&m[i][j], vaij);
                }
                if(j < n){
                    for(;j < n; j++){
                        m[i][j] = m[i][j] - m[k][j]*m[i][k];
                    }
                }
            }
            m[i][k] = 0.0;
        }


        // 所有线程一起进入下一轮
        pthread_barrier_wait(&barrier_Elimination2);

        if (t_id == 1){
            for(int i = k+1; i < n; i++){
                m[i][k] = 0.0;
            }
        }
    }
    pthread_exit(NULL);
}

//静态线程+barrier同步，全部工作线程，水平划分（穿插划分）版本
void m_gauss_h2(int n, int num_threads){
    //初始化barrier
    pthread_barrier_init(&barrier_Divsion2, NULL, num_threads);
    pthread_barrier_init(&barrier_Elimination2, NULL, num_threads);

    //创建线程
    pthread_t handles[num_threads];// 创建对应的 Handle
    threadParam_t param[num_threads];// 创建对应的线程数据结构
    for(int t_id = 0; t_id < num_threads; t_id++){
        param[t_id].t_id = t_id;
        param[t_id].n = n;
        param[t_id].num_threads = num_threads;
        pthread_create(&handles[t_id], NULL, threadFunc_h2, (void*)&param[t_id]);
    }

    for(int t_id = 0; t_id < num_threads; t_id++){
        pthread_join(handles[t_id], NULL);
    }

    pthread_barrier_destroy(&barrier_Divsion2);
    pthread_barrier_destroy(&barrier_Elimination2);

}



// barrier 定义
pthread_barrier_t barrier_Divsion1;
pthread_barrier_t barrier_Elimination1;



// 静态线程+barrier同步，水平划分（块划分）版本线程函数定义
void *threadFunc_h1(void *param) {
    threadParam_t *p = (threadParam_t*)param;
    int t_id = p->t_id;
    int num_threads = p->num_threads;
    int n = p->n;

    for(int k = 0; k < n; ++k) {
        // t_id 为 0 的线程做除法操作，其它工作线程先等待
        if (t_id == 0) {
            __m256 vt = _mm256_set1_ps(m[k][k]);
            int j;
            for(j = k + 1; j <= n - 8; j += 8) { // 使用 256 位寄存器处理 8 个单精度浮点数
                __m256 va = _mm256_loadu_ps(&m[k][j]);
                va = _mm256_div_ps(va, vt);
                _mm256_storeu_ps(&m[k][j], va);
            }
            for(; j < n; j++) {
                m[k][j] = m[k][j] / m[k][k];
            }
            m[k][k] = 1.0f;
        }
        // 阻塞，等待完成除法操作
        pthread_barrier_wait(&barrier_Divsion1);

        // 块划分任务
        int chunk_size = (n - k - 1) / num_threads;
        int start_row = k + 1 + t_id * chunk_size;
        int end_row = (t_id == num_threads - 1) ? n : start_row + chunk_size;

        for(int i = start_row; i < end_row; i++) {
            __m256 vaik = _mm256_set1_ps(m[i][k]);
            int j;
            for(j = k + 1; j <= n - 8; j += 8) { // 使用 256 位寄存器处理 8 个单精度浮点数
                __m256 vakj = _mm256_loadu_ps(&m[k][j]);
                __m256 vaij = _mm256_loadu_ps(&m[i][j]);
                vaij = _mm256_sub_ps(vaij, _mm256_mul_ps(vakj, vaik));
                _mm256_storeu_ps(&m[i][j], vaij);
            }
            for(; j < n; j++) {
                m[i][j] = m[i][j] - m[k][j] * m[i][k];
            }
            m[i][k] = 0.0f;
        }
        // 所有线程一起进入下一轮
        pthread_barrier_wait(&barrier_Elimination1);
    }
    pthread_exit(NULL);
}


// 静态线程+barrier同步，水平划分（块划分）版本
void m_gauss_h1(int n, int num_threads) {
    // 初始化 barrier
    pthread_barrier_init(&barrier_Divsion1, NULL, num_threads);
    pthread_barrier_init(&barrier_Elimination1, NULL, num_threads);

    // 创建线程
    pthread_t handles[num_threads]; // 创建对应的 Handle
    threadParam_t param[num_threads]; // 创建对应的线程数据结构
    for(int t_id = 0; t_id < num_threads; t_id++) {
        param[t_id].t_id = t_id;
        param[t_id].n = n;
        param[t_id].num_threads = num_threads;
        pthread_create(&handles[t_id], NULL, threadFunc_h1, (void*)&param[t_id]);
    }

    for(int t_id = 0; t_id < num_threads; t_id++) {
        pthread_join(handles[t_id], NULL);
    }

    pthread_barrier_destroy(&barrier_Divsion1);
    pthread_barrier_destroy(&barrier_Elimination1);
}


//barrier定义
pthread_barrier_t barrier_Divsion;
pthread_barrier_t barrier_Elimination;
int num_blocks;//块划分大小

//静态线程+barrier同步，全部工作线程，垂直划分（块划分）版本线程函数定义
void *threadFunc_v1(void *param) {
    threadParam_t *p = (threadParam_t*)param;
    int t_id = p -> t_id;
    int num_threads = p -> num_threads;
    int n = p -> n;
    __m128 vt, va, vaik, vakj, vaij, vx;

    for(int k = 0; k < n; ++k){
        // t_id 为 0 的线程做除法操作，其它工作线程先等待
        if (t_id == 0){
            vt = _mm_set_ps1(m[k][k]);
            int j;
            int start = k-k%4+4;
            for(j = k+1; j < start && j < n; j++){
                m[k][j] = m[k][j]/m[k][k];
            }
            if(j != n){
                for(j = start; j+4 <= n; j+=4){
                    va = _mm_load_ps(&m[k][j]);
                    va = _mm_div_ps(va, vt);
                    _mm_store_ps(&m[k][j], va);
                }
                if(j < n){
                    for(;j < n; j++){
                        m[k][j] = m[k][j]/m[k][k];
                    }
                }
            }
            m[k][k] = 1.0;
            num_blocks = (n-k-1)/num_threads;

            if(num_blocks < 8){
                num_blocks = 8;
            }
        }

        // 阻塞，等待完成除法操作
        pthread_barrier_wait(&barrier_Divsion);

        //块划分任务
        int j = k+1+t_id*num_blocks;
        int o = j;
        if(j < n){
            int my_end = j+num_blocks;
            if(my_end > n){
                my_end = n;
            }

            if(t_id == num_threads-1){
                my_end = n;
            }

            int my_start = j-j%4+4;
            for(int i = k+1; i < n; i++){
                j = o;
                for(; j < my_start && j < my_end; j++){
                    m[i][j] = m[i][j] - m[k][j]*m[i][k];
                }

                if(j != my_end){
                    //消去
                    vaik = _mm_set_ps1(m[i][k]);
                    for(j = my_start; j+4 <= my_end; j+=4){
                        vakj = _mm_load_ps(&m[k][j]);
                        vaij = _mm_load_ps(&m[i][j]);
                        vx = _mm_mul_ps(vakj, vaik);
                        vaij = _mm_sub_ps(vaij, vx);
                        _mm_store_ps(&m[i][j], vaij);
                    }
                    if(j < my_end){
                        for(;j < my_end; j++){
                            m[i][j] = m[i][j] - m[k][j]*m[i][k];
                        }
                    }
                }
            }
        }


        // 所有线程一起进入下一轮
        pthread_barrier_wait(&barrier_Elimination);

        if (t_id == 1){
            for(int i = k+1; i < n; i++){
                m[i][k] = 0.0;
            }
        }

    }

    pthread_exit(NULL);
}

//静态线程+barrier同步，全部工作线程，垂直划分版本
void m_gauss_v1(int n, int num_threads){
    //初始化barrier
    pthread_barrier_init(&barrier_Divsion, NULL, num_threads);
    pthread_barrier_init(&barrier_Elimination, NULL, num_threads);

    //创建线程
    pthread_t handles[num_threads];// 创建对应的 Handle
    threadParam_t param[num_threads];// 创建对应的线程数据结构
    for(int t_id = 0; t_id < num_threads; t_id++){
        param[t_id].t_id = t_id;
        param[t_id].n = n;
        param[t_id].num_threads = num_threads;
        pthread_create(&handles[t_id], NULL, threadFunc_v1, (void*)&param[t_id]);
    }

    for(int t_id = 0; t_id < num_threads; t_id++){
        pthread_join(handles[t_id], NULL);
    }

    pthread_barrier_destroy(&barrier_Divsion);
    pthread_barrier_destroy(&barrier_Elimination);

}



pthread_barrier_t barrier_DivsionV;
pthread_barrier_t barrier_EliminationV;


// 静态线程+barrier同步，垂直划分（穿插划分）版本线程函数定义
void *threadFunc_v(void *param){

    threadParam_t *p = (threadParam_t*)param;
    int t_id = p->t_id;
    int num_threads = p->num_threads;
    int n = p->n;

    for(int k = 0; k < n; ++k) {
        // t_id 为 0 的线程做除法操作，其它工作线程先等待
        if (t_id == 0) {
            for(int j = k + 1; j < n; j++) {
                m[k][j] = m[k][j] / m[k][k];
            }
            m[k][k] = 1.0;
        }
        // 阻塞，等待完成除法操作
        pthread_barrier_wait(&barrier_DivsionV);

        // 每个线程处理按列划分的部分
        for(int i = k + 1 + t_id; i < n; i += num_threads) {
            for(int j = k + 1; j < n; j++) {
                m[i][j] = m[i][j] - m[k][j] * m[i][k];
            }
            m[i][k] = 0.0;
        }
        // 所有线程一起进入下一轮
        pthread_barrier_wait(&barrier_EliminationV);
    }
    pthread_exit(NULL);

}

// 静态线程+barrier同步，垂直划分（穿插划分）版本
void m_gauss_v(int n, int num_threads) {
    // 初始化 barrier
    pthread_barrier_init(&barrier_DivsionV, NULL, num_threads);
    pthread_barrier_init(&barrier_EliminationV, NULL, num_threads);

    // 创建线程
    pthread_t handles[num_threads]; // 创建对应的 Handle
    threadParam_t param[num_threads]; // 创建对应的线程数据结构
    for(int t_id = 0; t_id < num_threads; t_id++) {
        param[t_id].t_id = t_id;
        param[t_id].n = n;
        param[t_id].num_threads = num_threads;
        pthread_create(&handles[t_id], NULL, threadFunc_v, (void*)&param[t_id]);
    }

    for(int t_id = 0; t_id < num_threads; t_id++) {
        pthread_join(handles[t_id], NULL);
    }

    pthread_barrier_destroy(&barrier_DivsionV);
    pthread_barrier_destroy(&barrier_EliminationV);
}

