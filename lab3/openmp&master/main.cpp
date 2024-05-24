#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <windows.h>
#include <pmmintrin.h>
#include <omp.h>

using namespace std;

const int N = 4000;

float mat[N][N];
float m[N][N];

// 定义 thread_count 变量
// int thread_count = 4; // 线程数
long long head, tail, freq; // timers

void init_mat(float m[][N], int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < i; j++) {
            m[i][j] = 0;
        }
        m[i][i] = 1.0;
        for (int j = i + 1; j < n; j++) {
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

void reset_mat(float mat[][N], float m[][N], int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            mat[i][j] = m[i][j];
}

void naive_lu(float mat[][N], int n) {
    for (int k = 0; k < n; k++) {
        for (int j = k + 1; j < n; j++)
            mat[k][j] = mat[k][j] / mat[k][k];
        mat[k][k] = 1.0;
        for (int i = k + 1; i < n; i++) {
            for (int j = k + 1; j < n; j++)
                mat[i][j] = mat[i][j] - mat[i][k] * mat[k][j];
            mat[i][k] = 0;
        }
    }
}


void omp_lu_static(float mat[][N], int n, int thread_count) {
    #pragma omp parallel num_threads(thread_count)
    {
        for (int k = 0; k < n; k++) {
            #pragma omp single
            for (int j = k + 1; j < n; j++) {
                mat[k][j] = mat[k][j] / mat[k][k];
            }
            mat[k][k] = 1.0;
            #pragma omp for simd
            for (int i = k + 1; i < n; i++) {
                for (int j = k + 1; j < n; j++) {
                    mat[i][j] = mat[i][j] - mat[i][k] * mat[k][j];
                }
                mat[i][k] = 0;
            }
        }
    }
}

void omp_lu_dynamic(float mat[][N], int n, int thread_count) {
    #pragma omp parallel num_threads(thread_count)
    for (int k = 0; k < n; k++) {
        #pragma omp for schedule(dynamic, 24)
        for (int j = k + 1; j < n; j++) {
            mat[k][j] = mat[k][j] / mat[k][k];
        }
        mat[k][k] = 1.0;
        #pragma omp for schedule(dynamic, 24)
        for (int i = k + 1; i < n; i++) {
            for (int j = k + 1; j < n; j++)
                mat[i][j] = mat[i][j] - mat[i][k] * mat[k][j];
            mat[i][k] = 0;
        }
    }
}

void omp_lu_guided(float mat[][N], int n, int thread_count) {
    #pragma omp parallel num_threads(thread_count)
    for (int k = 0; k < n; k++) {
        #pragma omp for schedule(guided, 24)
        for (int j = k + 1; j < n; j++) {
            mat[k][j] = mat[k][j] / mat[k][k];
        }
        mat[k][k] = 1.0;
        #pragma omp for schedule(guided, 24)
        for (int i = k + 1; i < n; i++) {
            for (int j = k + 1; j < n; j++)
                mat[i][j] = mat[i][j] - mat[i][k] * mat[k][j];
            mat[i][k] = 0;
        }
    }
}

void omp_lu_sse_dynamic(float mat[][N], int n, int thread_count) {
    __m128 t1, t2, t3;
    #pragma omp parallel num_threads(thread_count)
    for (int k = 0; k < n; k++) {
        float temp1[4] = {mat[k][k], mat[k][k], mat[k][k], mat[k][k]};
        t1 = _mm_loadu_ps(temp1);
        int j = k + 1;
        for (j; j < n - 3; j += 4) {
            t2 = _mm_loadu_ps(mat[k] + j);
            t3 = _mm_div_ps(t2, t1);
            _mm_storeu_ps(mat[k] + j, t3);
        }
        int www = j;
        #pragma omp for schedule(dynamic, 24)
        for (int j = www; j < n; j++) {
            mat[k][j] = mat[k][j] / mat[k][k];
        }
        mat[k][k] = 1.0;
        #pragma omp for schedule(dynamic, 24)
        for (int i = k + 1; i < n; i++) {
            float temp2[4] = {mat[i][k], mat[i][k], mat[i][k], mat[i][k]};
            t1 = _mm_loadu_ps(temp2);
            j = k + 1;
            for (j; j <= n - 3; j += 4) {
                t2 = _mm_loadu_ps(mat[i] + j);
                t3 = _mm_loadu_ps(mat[k] + j);
                t3 = _mm_mul_ps(t1, t3);
                t2 = _mm_sub_ps(t2, t3);
                _mm_storeu_ps(mat[i] + j, t2);
            }
            for (j; j < n; j++)
                mat[i][j] = mat[i][j] - mat[i][k] * mat[k][j];
            mat[i][k] = 0;
        }
    }
}
void omp_lu_guided_sse(float mat[][N], int n, int thread_count) {
    __m128 t1, t2, t3;
    #pragma omp parallel num_threads(thread_count)
    for (int k = 0; k < n; k++) {
        float temp1[4] = {mat[k][k], mat[k][k], mat[k][k], mat[k][k]};
        t1 = _mm_loadu_ps(temp1);

        #pragma omp for schedule(guided, 24)
        for (int j = k + 1; j < n - 3; j += 4) {
            t2 = _mm_loadu_ps(&mat[k][j]);
            t3 = _mm_div_ps(t2, t1);
            _mm_storeu_ps(&mat[k][j], t3);
        }
        for (int j = (n / 4) * 4; j < n; j++) {
            mat[k][j] = mat[k][j] / mat[k][k];
        }
        mat[k][k] = 1.0;

        #pragma omp for schedule(guided, 24)
        for (int i = k + 1; i < n; i++) {
            float temp2[4] = {mat[i][k], mat[i][k], mat[i][k], mat[i][k]};
            t1 = _mm_loadu_ps(temp2);
            for (int j = k + 1; j < n - 3; j += 4) {
                t2 = _mm_loadu_ps(&mat[i][j]);
                t3 = _mm_loadu_ps(&mat[k][j]);
                t3 = _mm_mul_ps(t1, t3);
                t2 = _mm_sub_ps(t2, t3);
                _mm_storeu_ps(&mat[i][j], t2);
            }
            for (int j = (n / 4) * 4; j < n; j++) {
                mat[i][j] = mat[i][j] - mat[i][k] * mat[k][j];
            }
            mat[i][k] = 0;
        }
    }
}


void omp_lu_static_sse(float mat[][N], int n, int thread_count) {
    __m128 t1, t2, t3;
    #pragma omp parallel num_threads(thread_count)
    for (int k = 0; k < n; k++) {
        float temp1[4] = {mat[k][k], mat[k][k], mat[k][k], mat[k][k]};
        t1 = _mm_loadu_ps(temp1);

        #pragma omp for schedule(static)
        for (int j = k + 1; j < n - 3; j += 4) {
            t2 = _mm_loadu_ps(&mat[k][j]);
            t3 = _mm_div_ps(t2, t1);
            _mm_storeu_ps(&mat[k][j], t3);
        }
        for (int j = (n / 4) * 4; j < n; j++) {
            mat[k][j] = mat[k][j] / mat[k][k];
        }
        mat[k][k] = 1.0;

        #pragma omp for schedule(static)
        for (int i = k + 1; i < n; i++) {
            float temp2[4] = {mat[i][k], mat[i][k], mat[i][k], mat[i][k]};
            t1 = _mm_loadu_ps(temp2);
            for (int j = k + 1; j < n - 3; j += 4) {
                t2 = _mm_loadu_ps(&mat[i][j]);
                t3 = _mm_loadu_ps(&mat[k][j]);
                t3 = _mm_mul_ps(t1, t3);
                t2 = _mm_sub_ps(t2, t3);
                _mm_storeu_ps(&mat[i][j], t2);
            }
            for (int j = (n / 4) * 4; j < n; j++) {
                mat[i][j] = mat[i][j] - mat[i][k] * mat[k][j];
            }
            mat[i][k] = 0;
        }
    }
}
void omp_static_sse_barrier(float A[N][N], int n,int NUM_THREADS) {
    __m128 va, vx, vaij, vaik, vakj;

    #pragma omp parallel num_threads(NUM_THREADS) private(va, vx, vaij, vaik, vakj)
    {
        for (int k = 0; k < n; k++) {
            // Serial part
            #pragma omp master
            {
                __m128 vt = _mm_set1_ps(A[k][k]);
                int j;
                for (j = k + 1; j < n; j++) {
                    va = _mm_loadu_ps(&(A[k][j]));
                    va = _mm_div_ps(va, vt);
                    _mm_storeu_ps(&(A[k][j]), va);
                }
                for (; j < n; j++) {
                    A[k][j] = A[k][j] * 1.0f / A[k][k];
                }
                A[k][k] = 1.0f;
            }

            // Parallel part
            #pragma omp barrier
            #pragma omp for schedule(static)
            for (int i = k + 1; i < n; i++) {
                vaik = _mm_set1_ps(A[i][k]);
                int j;
                for (j = k + 1; j + 4 <= n; j += 4) {
                    vakj = _mm_loadu_ps(&(A[k][j]));
                    vaij = _mm_loadu_ps(&(A[i][j]));
                    vx = _mm_mul_ps(vakj, vaik);
                    vaij = _mm_sub_ps(vaij, vx);
                    _mm_storeu_ps(&A[i][j], vaij);
                }

                for (; j < n; j++) {
                    A[i][j] = A[i][j] - A[i][k] * A[k][j];
                }

                A[i][k] = 0;
            }
            // Upon exiting the loop, threads implicitly synchronize before proceeding to the next row processing
        }
    }
}
void print_mat(float mat[][N]) {
    if (N > 16)
        return;
    cout << endl;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            cout << mat[i][j] << " ";
        cout << endl;
    }
    cout << endl;
}

int main() {
    QueryPerformanceFrequency((LARGE_INTEGER *)&freq); // similar to CLOCKS_PER_SEC

    // 问题规模
    int i = 500;
    int s = 500;

    for (i = 500; i <= N; i += s) {
        cout << "问题规模： " << i << endl;
        init_mat(m, i);

            reset_mat(mat, m, i);
            QueryPerformanceCounter((LARGE_INTEGER *)&head);// start time
            //for(int x=0;x<5;x++)
            naive_lu(mat, i);
            QueryPerformanceCounter((LARGE_INTEGER *)&tail); // end time
            cout << "naive LU: " << (tail - head) * 1000.0 / (freq) << "ms" << endl;
            //cout << "dynamic OpenMP LU: " << (tail - head) * 1000.0 /(freq*5)<< "ms" << endl;
           // print_mat(mat);
        for (int count = 8; count <= 8; count += 4) {
            cout << "线程数： " << count << endl;


            reset_mat(mat, m, i);
            QueryPerformanceCounter((LARGE_INTEGER *)&head); // start time
           // for(int x=0;x<5;x++)
            omp_lu_static(mat, i, count);
            QueryPerformanceCounter((LARGE_INTEGER *)&tail); // end time
           // cout << "static OpenMP LU: " << (tail - head) * 1000.0 /(freq*5) << "ms" << endl;
            cout << "static OpenMP LU: " << (tail - head) * 1000.0 /(freq) << "ms" << endl;
//           // print_mat(mat);
//
//            reset_mat(mat, m, i);
//            QueryPerformanceCounter((LARGE_INTEGER *)&head); // start time
//            //for(int x=0;x<5;x++)
//            omp_lu_dynamic(mat, i, count);
//            QueryPerformanceCounter((LARGE_INTEGER *)&tail); // end time
//          //  cout << "dynamic OpenMP LU: " << (tail - head) * 1000.0 /(freq*5)<< "ms" << endl;
//              cout << "dynamic OpenMP LU: " << (tail - head) * 1000.0 /(freq)<< "ms" << endl;
//           // print_mat(mat);
//
//            reset_mat(mat, m, i);
//            QueryPerformanceCounter((LARGE_INTEGER *)&head); // start time
//          //  for(int x=0;x<5;x++)
//            omp_lu_guided(mat, i, count);
//            QueryPerformanceCounter((LARGE_INTEGER *)&tail); // end time
//            cout << "guided OpenMP LU: " << (tail - head) * 1000.0 /(freq)<< "ms" << endl;
//           // cout << "guided OpenMP LU: " << (tail - head) * 1000.0 /(freq*5)<< "ms" << endl;
//           // print_mat(mat);
//
            reset_mat(mat, m, i);
            QueryPerformanceCounter((LARGE_INTEGER *)&head); // start time
           // for(int x=0;x<5;x++)
            omp_lu_static_sse(mat, i, count);
            QueryPerformanceCounter((LARGE_INTEGER *)&tail); // end time
            cout << "sse and static OpenMP LU: " << (tail - head) * 1000.0 /(freq)<< "ms" << endl;
           // cout << "sse and static OpenMP LU: " << (tail - head) * 1000.0 /(freq*5)<< "ms" << endl;

//            reset_mat(mat, m, i);
//            QueryPerformanceCounter((LARGE_INTEGER *)&head); // start time
//           // for(int x=0;x<5;x++)
//            omp_static_sse_barrier(mat, i, count);
//            QueryPerformanceCounter((LARGE_INTEGER *)&tail); // end time
//            cout << "sse and static OpenMP  barrier: " << (tail - head) * 1000.0 /(freq)<< "ms" << endl;
           // cout << "sse and static OpenMP LU: " << (tail - head) * 1000.0 /(freq*5)<< "ms" << endl;

//            reset_mat(mat, m, i);
//            QueryPerformanceCounter((LARGE_INTEGER *)&head); // start time
//          //  for(int x=0;x<5;x++)
//            omp_lu_sse_dynamic(mat, i, count);
//            QueryPerformanceCounter((LARGE_INTEGER *)&tail); // end time
//            cout << "sse and dynamic OpenMP LU: " << (tail - head) * 1000.0 / (freq)<< "ms" << endl;
//         //    cout << "sse and dynamic OpenMP LU: " << (tail - head) * 1000.0 / (freq*5)<< "ms" << endl;
//          //  print_mat(mat);



//             reset_mat(mat, m, i);
//            QueryPerformanceCounter((LARGE_INTEGER *)&head); // start time
//           // for(int x=0;x<5;x++)
//            omp_lu_guided_sse(mat, i, count);
//            QueryPerformanceCounter((LARGE_INTEGER *)&tail); // end time
//            cout << "sse and guided OpenMP LU: " << (tail - head) * 1000.0 / (freq) << "ms" << endl;
//          //  cout << "sse and guided OpenMP LU: " << (tail - head) * 1000.0 / (freq*5) << "ms" << endl;
//

        if(i==2000) s=2000;

        }
        cout<<endl;
    }
    cout << endl;
    return 0;
}
