#include<iostream>
#include<fstream>
#include<windows.h>
#include<string>
#include<sstream>
#include<omp.h>
#include<pthread.h>
#include<malloc.h>
#include<semaphore.h>
#include <xmmintrin.h> //SSE
#include <emmintrin.h> //SSE2
#include <pmmintrin.h> //SSE3
#include <tmmintrin.h> //SSSE3
#include <smmintrin.h> //SSE4.1
#include <nmmintrin.h> //SSSE4.2
#include <immintrin.h> //AVX、AVX2
using namespace std;
const int n = 10000;
int elimination[n][n] = { 0 };
int eliminated[n][n] = { 0 };
int eliresult[n][n] = { 0 };
//测试样例10 矩阵列数43577，非零消元子39477，被消元行54274
//测试样例7 矩阵列数8399，非零消元子6375，被消元行4535
//测试样例8 矩阵列数23045，非零消元子18748，被消元行14325
//测试样例9 矩阵列数37960，非零消元子29304，被消元行14921
fstream fst1("C:\\Groebner\\5\\被消元行.txt", ios::in | ios::out);//读取被消元行
fstream fst2("C:\\Groebner\\5\\消元子.txt", ios::in | ios::out);//读取消元子
int col = 50000;//列数
int row_eli = 50000;//消元子行数
int row_elied = 50000;//被消元行行数
int index = (col / 32) + 1;
int cnt = 0;
int is_null(int eli[n][n], int index, int row)//判断该行是否为空，不为空返回真值
{
    int count = 0;
    for (int i = 0; i < index; i++)
    {
        if (eli[row][i] != 0)//有不为0的值
        {
            count = 1;
            break;
        }
    }
    return count;
}
int get_big_first(int eli[n][n], int index, int row)//获取该行的最大首项，返回最大首项在在原文件中的列数
{
    for (int i = index - 1; i >= 0; i--)
    {
        if (eli[row][i])
        {
            int temp = eli[row][i];
            for (int j = 31; j >= 0; j--)
            {
                if (temp & (1 << j))
                {
                    return 32 * i + j;
                }
            }
        }
        else
        {
            continue;
        }
    }
    return -1;
}
void Gauss_advance_normal(int row_eli, int row_elied, int index)//特殊高斯消元串行算法
{
    for (int i = 0; i < row_elied; i++)
    {
        while (is_null(eliminated, index, i))//判断被消元行是否为空
        {
            int v = get_big_first(eliminated, index, i);//获取最大首项
            if (is_null(elimination, index, v))
                //判断消元子与被消元行的最大首项匹配的行是否为空
            {
                for (int j = 0; j < index; j++)
                {
                    eliminated[i][j] = eliminated[i][j] ^ elimination[v][j];
                }
            }
            else
            {
                for (int j = 0; j < index; j++)//被消元行升格成消元子
                {
                    elimination[v][j] = eliminated[i][j];
                }
                break;
            }
        }
    }
}
void Gauss_advance_normal_omp(int row_eli, int row_elied, int index)
{
    int thread_count = 8;
    int i;
    int v;
#pragma omp parallel num_threads(thread_count),private(i,v)
#pragma for schedule(guided)
    for (int i = 0; i < row_elied; i++)
    {
        while (is_null(eliminated, index, i))//判断被消元行是否为空
        {
            v = get_big_first(eliminated, index, i);//获取最大首项
            if (is_null(elimination, index, v))//判断消元子与被消元行的最大首项匹配的行是否为空
            {
                //cout<<get_big_first(eliminated, index, i)<<" bfore "<<omp_get_thread_num()<<endl;
                for (int j = 0; j < index; j++)
                {
                    eliminated[i][j] = eliminated[i][j] ^ elimination[v][j];
                }

            }
            else
            {
                #pragma omp critical
                    if (is_null(elimination, index, v)==0)
                    {
                        for (int j = 0; j < index; j++)//被消元行升格成消元子
                        {
                            elimination[v][j] = eliminated[i][j];
                            eliresult[cnt][j] = eliminated[i][j];
                        }
                        cnt++;
                        //cout<<get_big_first(eliminated, index, i)<<" elimination "<<omp_get_thread_num()<<endl;
                    }

            }
        }
    }
}

void Gauss_advance_normal_omp_sse(int row_eli, int row_elied, int index)
{
    int thread_count = 8;
    int i;
    int v;
#pragma omp parallel num_threads(thread_count),private(i,v)
#pragma for schedule(guided)
    for (int i = 0; i < row_elied; i++)
    {
        while (is_null(eliminated, index, i))//判断被消元行是否为空
        {
            v = get_big_first(eliminated, index, i);//获取最大首项
            if (is_null(elimination, index, v))//判断消元子与被消元行的最大首项匹配的行是否为空
            {
                //cout<<get_big_first(eliminated, index, i)<<" bfore "<<omp_get_thread_num()<<endl;
                for (int j = 0; j < index; j++)
                {
                    eliminated[i][j] = eliminated[i][j] ^ elimination[v][j];
                }

            }
            else
            {
                #pragma omp critical
                    if (is_null(elimination, index, v)==0)
                    {
                        for (int j = 0; j < index; j++)//被消元行升格成消元子
                        {
                            elimination[v][j] = eliminated[i][j];
                            eliresult[cnt][j] = eliminated[i][j];
                        }
                        cnt++;
                        //cout<<get_big_first(eliminated, index, i)<<" elimination "<<omp_get_thread_num()<<endl;
                    }

            }
        }
    }
}
typedef struct
{
    int t_id; //线程 id
}threadParam_t;

const int NUM_THREADS = 8;
sem_t sem_leader;

void* funct(void*param)
{
    threadParam_t* p = (threadParam_t*)param;
    int t_id = p->t_id;
    for (int i = t_id; i < row_elied; i+=NUM_THREADS)
    {
        while (is_null(eliminated, index, i))//判断被消元行是否为空
        {
            int v = get_big_first(eliminated, index, i);//获取最大首项
            if (is_null(elimination, index, v))
            //判断消元子与被消元行的最大首项匹配的行是否为空
            {
                int j = 0;
                for (; j + 4 < index; j += 4)//消元行与被消元子的异或运算转换成向量运算
                {
                    __m128i elied = _mm_loadu_si128((__m128i*) & eliminated[i][j]);
                    __m128i eli = _mm_loadu_si128((__m128i*) & elimination[v][j]);
                    __m128i tmp = _mm_xor_si128(elied, eli);
                    _mm_storeu_si128((__m128i*) & eliminated[i][j], tmp);
                }
                for (; j < index; j++)
                {
                    eliminated[i][j] = eliminated[i][j] ^ elimination[v][j];//处理无法向量运算的剩余量
                }
            }
            else
            {
                sem_wait(&sem_leader);
                if (is_null(elimination, index, v)==0)
                {
                    int j = 0;
                    for (; j + 4 < index; j += 4)
                    {
                        __m128i elied = _mm_loadu_si128((__m128i*) & eliminated[i][j]);//升格过程也可以转换成向量运算
                        _mm_storeu_si128((__m128i*) & elimination[v][j], elied);
                        _mm_storeu_si128((__m128i*) & eliresult[cnt][j], elied);

                    }
                    for (; j < index; j++)
                    {
                        elimination[v][j] = eliminated[i][j];//处理无法向量运算的剩余量
                        eliresult[cnt][j] = eliminated[i][j];
                    }
                    cnt++;
                }
                sem_post(&sem_leader);

            }
        }
    }
    pthread_exit(NULL);
}
void Gauss_advance_pthread()
{
    sem_init(&sem_leader, 0, 1);
    pthread_t* handles = (pthread_t*)malloc(NUM_THREADS * sizeof(pthread_t));
    threadParam_t* param = (threadParam_t*)malloc(NUM_THREADS * sizeof(threadParam_t));
    for (int t_id = 0; t_id < NUM_THREADS; t_id++)
    {
        param[t_id].t_id = t_id;
        pthread_create(&handles[t_id], NULL, funct, (void*)&param[t_id]);
    }
    for (int t_id = 0; t_id < NUM_THREADS; t_id++)
    {
        pthread_join(handles[t_id], NULL);
    }
    sem_destroy(&sem_leader);
}
void* funct_sse(void*param)
{
    threadParam_t* p = (threadParam_t*)param;
    int t_id = p->t_id;
    for (int i = t_id; i < row_elied; i+=NUM_THREADS)
    {
        while (is_null(eliminated, index, i))//判断被消元行是否为空
        {
            int v = get_big_first(eliminated, index, i);//获取最大首项
            if (is_null(elimination, index, v))
            //判断消元子与被消元行的最大首项匹配的行是否为空
            {
                int j = 0;
                for (; j + 4 < index; j += 4)//消元行与被消元子的异或运算转换成向量运算
                {
                    __m128i elied = _mm_loadu_si128((__m128i*) & eliminated[i][j]);
                    __m128i eli = _mm_loadu_si128((__m128i*) & elimination[v][j]);
                    __m128i tmp = _mm_xor_si128(elied, eli);
                    _mm_storeu_si128((__m128i*) & eliminated[i][j], tmp);
                }
                for (; j < index; j++)
                {
                    eliminated[i][j] = eliminated[i][j] ^ elimination[v][j];//处理无法向量运算的剩余量
                }
            }
            else
            {
                sem_wait(&sem_leader);
                if (is_null(elimination, index, v)==0)
                {
                    int j = 0;
                    for (; j + 4 < index; j += 4)
                    {
                        __m128i elied = _mm_loadu_si128((__m128i*) & eliminated[i][j]);//升格过程也可以转换成向量运算
                        _mm_storeu_si128((__m128i*) & elimination[v][j], elied);
                        _mm_storeu_si128((__m128i*) & eliresult[cnt][j], elied);

                    }
                    for (; j < index; j++)
                    {
                        elimination[v][j] = eliminated[i][j];//处理无法向量运算的剩余量
                        eliresult[cnt][j] = eliminated[i][j];
                    }
                    cnt++;
                }
                sem_post(&sem_leader);

            }
        }
    }
    pthread_exit(NULL);
}
void Gauss_advance_pthread_sse()
{
    sem_init(&sem_leader, 0, 1);
    pthread_t* handles = (pthread_t*)malloc(NUM_THREADS * sizeof(pthread_t));
    threadParam_t* param = (threadParam_t*)malloc(NUM_THREADS * sizeof(threadParam_t));
    for (int t_id = 0; t_id < NUM_THREADS; t_id++)
    {
        param[t_id].t_id = t_id;
        pthread_create(&handles[t_id], NULL, funct_sse, (void*)&param[t_id]);
    }
    for (int t_id = 0; t_id < NUM_THREADS; t_id++)
    {
        pthread_join(handles[t_id], NULL);
    }
    sem_destroy(&sem_leader);
}
void Gauss_advance_sse(int row_eli, int row_elied, int index)
{
    for (int i = 0; i < row_elied; i++)
    {
        while (is_null(eliminated, index, i))
        {
            int v = get_big_first(eliminated, index, i);
            if (is_null(elimination, index, v))
            {
                int j = 0;
                for (; j + 4 < index; j += 4)//消元行与被消元子的异或运算转换成向量运算
                {
                    __m128i elied = _mm_loadu_si128((__m128i*) & eliminated[i][j]);
                    __m128i eli = _mm_loadu_si128((__m128i*) & elimination[v][j]);
                    __m128i tmp = _mm_xor_si128(elied, eli);
                    _mm_storeu_si128((__m128i*) & eliminated[i][j], tmp);
                }
                for (; j < index; j++)
                {
                    eliminated[i][j] = eliminated[i][j] ^ elimination[v][j];//处理无法向量运算的剩余量
                }
            }
            else
            {
                int j = 0;
                for (; j + 4 < index; j += 4)
                {
                    __m128i elied = _mm_loadu_si128((__m128i*) & eliminated[i][j]);//升格过程也可以转换成向量运算
                    _mm_storeu_si128((__m128i*) & elimination[v][j], elied);
                }
                for (; j < index; j++)
                {
                    elimination[v][j] = eliminated[i][j];//处理无法向量运算的剩余量
                }
                break;
            }
        }
    }
}
void display(int row, int col, int A[n][n])
{
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++)
        {
            cout << A[i][j] << " ";
        }
        cout << endl;
    }
}
void write_eliminated(int row_elied)//被消元行位图存储
{
    string row;
    int x;
    for (int i = 0; i < row_elied; i++)//读取次数
    {
        getline(fst1, row);//每次读取一行
        stringstream  ss(row);
        while (ss >> x)
        {
            int x_index = x / 32;//x位图存储的列数
            int x_offset = x % 32;//x在对应数目中的偏移量
            eliminated[i][x_index] |= (1 << x_offset);//存储
        }
    }
}
void write_elimination(int row_eli)//消元子位图存储
{
    string row;
    int x;
    for (int i = 0; i < row_eli; i++)//读取次数
    {
        getline(fst2, row);
        stringstream  ss(row);
        int temp = 0;
        int count = 0;
        while (ss >> x)
        {
            if (count == 0)//确定存储行数
            {
                count++;
                temp = x;
            }
            int x_index = x / 32;//x位图存储的列数
            int x_offset = x % 32;//x在对应数目中的偏移量
            elimination[temp][x_index] |= (1 << x_offset);//存储
        }
        count = 0;
    }
}
void writeout_result(int row_elied, int index)
{
    cout<<"result:"<<endl;
    for (int i = 0; i < row_elied; i++)//从第一行开始读取
    {
        for (int j = index - 1; j >= 0; j--)//从最大首相所在列开始读取
        {
            for (int k = 31; k >= 0; k--)//从最大首相开始读取
            {
                if (eliresult[i][j] & (1 << k))
                {
                    cout << 32 * j + k << " ";
                }
            }
        }
        cout << endl;
    }
}
int main()
{
    long long freq, head, tail;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);

    write_eliminated(row_elied);//读取文件
    write_elimination(row_eli);//读取文件
 QueryPerformanceCounter((LARGE_INTEGER*)&head);
    //Gauss_advance_pthread_sse();//sse优化算法
    //Gauss_advance_normal(row_eli, row_elied, index);//平凡算法
 //   Gauss_advance_normal_omp_sse(row_eli, row_elied, index);
    Gauss_advance_pthread();
    //writeout_result(row_eli,row_elied);

    QueryPerformanceCounter((LARGE_INTEGER*)&tail);

    double time1 = (tail - head) * 1000 / freq;
    cout << " Gauss_advance_pthread总运行时间：" << time1 << "ms" << endl;

    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    Gauss_advance_pthread_sse();//sse优化算法
    //Gauss_advance_normal(row_eli, row_elied, index);//平凡算法
 //   Gauss_advance_normal_omp_sse(row_eli, row_elied, index);
 //   Gauss_advance_pthread();
    //writeout_result(row_eli,row_elied);

    QueryPerformanceCounter((LARGE_INTEGER*)&tail);

    time1 = (tail - head) * 1000 / freq;
    cout << " Gauss_advance_pthread_sse总运行时间：" << time1 << "ms" << endl;

    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    //Gauss_advance_pthread_sse();//sse优化算法
    //Gauss_advance_normal(row_eli, row_elied, index);//平凡算法
    //   Gauss_advance_normal_omp_sse(row_eli, row_elied, index);
   // Gauss_advance_pthread();
    //writeout_result(row_eli,row_elied);
   Gauss_advance_sse(row_eli, row_elied, index);
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);

    time1 = (tail - head) * 1000 / freq;
    cout << " Gauss_advance_sse总运行时间：" << time1 << "ms" << endl;

}
