#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <chrono>
#include <omp.h>
#include <arm_neon.h>

using namespace std;

const int N = 2000;

float mat[N][N];
float m[N][N];

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
void omp_lu_neon_dynamic(float A[][N], int n, int NUM_THREADS)
{
    float32x4_t va = vmovq_n_f32(0);
    float32x4_t vx = vmovq_n_f32(0);
    float32x4_t vaij = vmovq_n_f32(0);
    float32x4_t vaik = vmovq_n_f32(0);
    float32x4_t vakj = vmovq_n_f32(0);

    #pragma omp parallel num_threads(NUM_THREADS), private(va, vx, vaij, vaik,vakj)
	for (int k = 0; k < n; k++)
	{
		//串行部分
		#pragma omp single
		{
		    float32x4_t vt=vmovq_n_f32(A[k][k]);
            int j;
			for (j = k + 1; j < n; j++)
			{
				va=vld1q_f32(&(A[k][j]) );
                va= vdivq_f32(va,vt);
                vst1q_f32(&(A[k][j]), va);
			}
			for(; j<n; j++)
            {
                A[k][j]=A[k][j]*1.0 / A[k][k];

            }
            A[k][k] = 1.0;
		}

		//并行部分
		#pragma omp for schedule(dynamic, 14)
		for (int i = k + 1; i < n; i++)
		{
		    vaik=vmovq_n_f32(A[i][k]);
            int j;
			for (j = k + 1; j+4 <= n; j+=4)
			{
				vakj=vld1q_f32(&(A[k][j]));
				vaij=vld1q_f32(&(A[i][j]));
				vx=vmulq_f32(vakj,vaik);
				vaij=vsubq_f32(vaij,vx);

				vst1q_f32(&A[i][j], vaij);
			}

			for(; j<n; j++)
            {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }

			A[i][k] = 0;
		}
		// 离开for循环时，各个线程默认同步，进入下一行的处理
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
    int thread_count = 8; // 设置线程数
    int s = 500;

    for (int i = 500; i <= N; i += s) {
        cout << "问题规模： " << i << endl;
        init_mat(m, i);

        reset_mat(mat, m, i);
        auto start = chrono::high_resolution_clock::now();
        naive_lu(mat, i);
        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double, milli> duration = end - start;
        cout << "naive LU: " << duration.count() << "ms" << endl;

        for (int count = 8; count <= 8; count += 4) {
            cout << "线程数： " << count << endl;

            reset_mat(mat, m, i);
            start = chrono::high_resolution_clock::now();
            omp_lu_neon_dynamic(mat, i, count);
            end = chrono::high_resolution_clock::now();
            duration = end - start;
            cout << "neon and dynamic OpenMP LU: " << duration.count() << "ms" << endl;

        }
        cout << endl;
    }
    return 0;
}
