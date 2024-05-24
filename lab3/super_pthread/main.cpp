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
#include <immintrin.h> //AVX��AVX2
using namespace std;
const int n = 10000;
int elimination[n][n] = { 0 };
int eliminated[n][n] = { 0 };
int eliresult[n][n] = { 0 };
//��������10 ��������43577��������Ԫ��39477������Ԫ��54274
//��������7 ��������8399��������Ԫ��6375������Ԫ��4535
//��������8 ��������23045��������Ԫ��18748������Ԫ��14325
//��������9 ��������37960��������Ԫ��29304������Ԫ��14921
fstream fst1("C:\\Groebner\\5\\����Ԫ��.txt", ios::in | ios::out);//��ȡ����Ԫ��
fstream fst2("C:\\Groebner\\5\\��Ԫ��.txt", ios::in | ios::out);//��ȡ��Ԫ��
int col = 50000;//����
int row_eli = 50000;//��Ԫ������
int row_elied = 50000;//����Ԫ������
int index = (col / 32) + 1;
int cnt = 0;
int is_null(int eli[n][n], int index, int row)//�жϸ����Ƿ�Ϊ�գ���Ϊ�շ�����ֵ
{
    int count = 0;
    for (int i = 0; i < index; i++)
    {
        if (eli[row][i] != 0)//�в�Ϊ0��ֵ
        {
            count = 1;
            break;
        }
    }
    return count;
}
int get_big_first(int eli[n][n], int index, int row)//��ȡ���е����������������������ԭ�ļ��е�����
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
void Gauss_advance_normal(int row_eli, int row_elied, int index)//�����˹��Ԫ�����㷨
{
    for (int i = 0; i < row_elied; i++)
    {
        while (is_null(eliminated, index, i))//�жϱ���Ԫ���Ƿ�Ϊ��
        {
            int v = get_big_first(eliminated, index, i);//��ȡ�������
            if (is_null(elimination, index, v))
                //�ж���Ԫ���뱻��Ԫ�е��������ƥ������Ƿ�Ϊ��
            {
                for (int j = 0; j < index; j++)
                {
                    eliminated[i][j] = eliminated[i][j] ^ elimination[v][j];
                }
            }
            else
            {
                for (int j = 0; j < index; j++)//����Ԫ���������Ԫ��
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
        while (is_null(eliminated, index, i))//�жϱ���Ԫ���Ƿ�Ϊ��
        {
            v = get_big_first(eliminated, index, i);//��ȡ�������
            if (is_null(elimination, index, v))//�ж���Ԫ���뱻��Ԫ�е��������ƥ������Ƿ�Ϊ��
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
                        for (int j = 0; j < index; j++)//����Ԫ���������Ԫ��
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
        while (is_null(eliminated, index, i))//�жϱ���Ԫ���Ƿ�Ϊ��
        {
            v = get_big_first(eliminated, index, i);//��ȡ�������
            if (is_null(elimination, index, v))//�ж���Ԫ���뱻��Ԫ�е��������ƥ������Ƿ�Ϊ��
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
                        for (int j = 0; j < index; j++)//����Ԫ���������Ԫ��
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
    int t_id; //�߳� id
}threadParam_t;

const int NUM_THREADS = 8;
sem_t sem_leader;

void* funct(void*param)
{
    threadParam_t* p = (threadParam_t*)param;
    int t_id = p->t_id;
    for (int i = t_id; i < row_elied; i+=NUM_THREADS)
    {
        while (is_null(eliminated, index, i))//�жϱ���Ԫ���Ƿ�Ϊ��
        {
            int v = get_big_first(eliminated, index, i);//��ȡ�������
            if (is_null(elimination, index, v))
            //�ж���Ԫ���뱻��Ԫ�е��������ƥ������Ƿ�Ϊ��
            {
                int j = 0;
                for (; j + 4 < index; j += 4)//��Ԫ���뱻��Ԫ�ӵ��������ת������������
                {
                    __m128i elied = _mm_loadu_si128((__m128i*) & eliminated[i][j]);
                    __m128i eli = _mm_loadu_si128((__m128i*) & elimination[v][j]);
                    __m128i tmp = _mm_xor_si128(elied, eli);
                    _mm_storeu_si128((__m128i*) & eliminated[i][j], tmp);
                }
                for (; j < index; j++)
                {
                    eliminated[i][j] = eliminated[i][j] ^ elimination[v][j];//�����޷����������ʣ����
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
                        __m128i elied = _mm_loadu_si128((__m128i*) & eliminated[i][j]);//�������Ҳ����ת������������
                        _mm_storeu_si128((__m128i*) & elimination[v][j], elied);
                        _mm_storeu_si128((__m128i*) & eliresult[cnt][j], elied);

                    }
                    for (; j < index; j++)
                    {
                        elimination[v][j] = eliminated[i][j];//�����޷����������ʣ����
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
        while (is_null(eliminated, index, i))//�жϱ���Ԫ���Ƿ�Ϊ��
        {
            int v = get_big_first(eliminated, index, i);//��ȡ�������
            if (is_null(elimination, index, v))
            //�ж���Ԫ���뱻��Ԫ�е��������ƥ������Ƿ�Ϊ��
            {
                int j = 0;
                for (; j + 4 < index; j += 4)//��Ԫ���뱻��Ԫ�ӵ��������ת������������
                {
                    __m128i elied = _mm_loadu_si128((__m128i*) & eliminated[i][j]);
                    __m128i eli = _mm_loadu_si128((__m128i*) & elimination[v][j]);
                    __m128i tmp = _mm_xor_si128(elied, eli);
                    _mm_storeu_si128((__m128i*) & eliminated[i][j], tmp);
                }
                for (; j < index; j++)
                {
                    eliminated[i][j] = eliminated[i][j] ^ elimination[v][j];//�����޷����������ʣ����
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
                        __m128i elied = _mm_loadu_si128((__m128i*) & eliminated[i][j]);//�������Ҳ����ת������������
                        _mm_storeu_si128((__m128i*) & elimination[v][j], elied);
                        _mm_storeu_si128((__m128i*) & eliresult[cnt][j], elied);

                    }
                    for (; j < index; j++)
                    {
                        elimination[v][j] = eliminated[i][j];//�����޷����������ʣ����
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
                for (; j + 4 < index; j += 4)//��Ԫ���뱻��Ԫ�ӵ��������ת������������
                {
                    __m128i elied = _mm_loadu_si128((__m128i*) & eliminated[i][j]);
                    __m128i eli = _mm_loadu_si128((__m128i*) & elimination[v][j]);
                    __m128i tmp = _mm_xor_si128(elied, eli);
                    _mm_storeu_si128((__m128i*) & eliminated[i][j], tmp);
                }
                for (; j < index; j++)
                {
                    eliminated[i][j] = eliminated[i][j] ^ elimination[v][j];//�����޷����������ʣ����
                }
            }
            else
            {
                int j = 0;
                for (; j + 4 < index; j += 4)
                {
                    __m128i elied = _mm_loadu_si128((__m128i*) & eliminated[i][j]);//�������Ҳ����ת������������
                    _mm_storeu_si128((__m128i*) & elimination[v][j], elied);
                }
                for (; j < index; j++)
                {
                    elimination[v][j] = eliminated[i][j];//�����޷����������ʣ����
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
void write_eliminated(int row_elied)//����Ԫ��λͼ�洢
{
    string row;
    int x;
    for (int i = 0; i < row_elied; i++)//��ȡ����
    {
        getline(fst1, row);//ÿ�ζ�ȡһ��
        stringstream  ss(row);
        while (ss >> x)
        {
            int x_index = x / 32;//xλͼ�洢������
            int x_offset = x % 32;//x�ڶ�Ӧ��Ŀ�е�ƫ����
            eliminated[i][x_index] |= (1 << x_offset);//�洢
        }
    }
}
void write_elimination(int row_eli)//��Ԫ��λͼ�洢
{
    string row;
    int x;
    for (int i = 0; i < row_eli; i++)//��ȡ����
    {
        getline(fst2, row);
        stringstream  ss(row);
        int temp = 0;
        int count = 0;
        while (ss >> x)
        {
            if (count == 0)//ȷ���洢����
            {
                count++;
                temp = x;
            }
            int x_index = x / 32;//xλͼ�洢������
            int x_offset = x % 32;//x�ڶ�Ӧ��Ŀ�е�ƫ����
            elimination[temp][x_index] |= (1 << x_offset);//�洢
        }
        count = 0;
    }
}
void writeout_result(int row_elied, int index)
{
    cout<<"result:"<<endl;
    for (int i = 0; i < row_elied; i++)//�ӵ�һ�п�ʼ��ȡ
    {
        for (int j = index - 1; j >= 0; j--)//��������������п�ʼ��ȡ
        {
            for (int k = 31; k >= 0; k--)//��������࿪ʼ��ȡ
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

    write_eliminated(row_elied);//��ȡ�ļ�
    write_elimination(row_eli);//��ȡ�ļ�
 QueryPerformanceCounter((LARGE_INTEGER*)&head);
    //Gauss_advance_pthread_sse();//sse�Ż��㷨
    //Gauss_advance_normal(row_eli, row_elied, index);//ƽ���㷨
 //   Gauss_advance_normal_omp_sse(row_eli, row_elied, index);
    Gauss_advance_pthread();
    //writeout_result(row_eli,row_elied);

    QueryPerformanceCounter((LARGE_INTEGER*)&tail);

    double time1 = (tail - head) * 1000 / freq;
    cout << " Gauss_advance_pthread������ʱ�䣺" << time1 << "ms" << endl;

    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    Gauss_advance_pthread_sse();//sse�Ż��㷨
    //Gauss_advance_normal(row_eli, row_elied, index);//ƽ���㷨
 //   Gauss_advance_normal_omp_sse(row_eli, row_elied, index);
 //   Gauss_advance_pthread();
    //writeout_result(row_eli,row_elied);

    QueryPerformanceCounter((LARGE_INTEGER*)&tail);

    time1 = (tail - head) * 1000 / freq;
    cout << " Gauss_advance_pthread_sse������ʱ�䣺" << time1 << "ms" << endl;

    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    //Gauss_advance_pthread_sse();//sse�Ż��㷨
    //Gauss_advance_normal(row_eli, row_elied, index);//ƽ���㷨
    //   Gauss_advance_normal_omp_sse(row_eli, row_elied, index);
   // Gauss_advance_pthread();
    //writeout_result(row_eli,row_elied);
   Gauss_advance_sse(row_eli, row_elied, index);
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);

    time1 = (tail - head) * 1000 / freq;
    cout << " Gauss_advance_sse������ʱ�䣺" << time1 << "ms" << endl;

}
