#include <omp.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <sys/time.h>
#include <immintrin.h> // use SSE/AVX
using namespace std;


/*
unsigned int Act[8399][264] = { 0 };
unsigned int Pas[8399][264] = { 0 };

const int Num = 263;
const int pasNum = 4535;
const int lieNum = 8399;
*/


unsigned int Act[23045][722] = { 0 };
unsigned int Pas[23045][722] = { 0 };

const int Num = 721;
const int pasNum = 14325;
const int lieNum = 23045;


/*
unsigned int Act[37960][1188] = { 0 };
unsigned int Pas[37960][1188] = { 0 };

const int Num = 1187;
const int pasNum = 14921;
const int lieNum = 37960;
*/

/*
unsigned int Act[43577][1363] = { 0 };
unsigned int Pas[54274][1363] = { 0 };

const int Num = 1362;
const int pasNum = 54274;
const int lieNum = 43577;

*/


// Number of threads
const int NUM_THREADS = 7;

// Global variable to determine if the next round should be entered
bool sign;

struct threadParam_t {
    int t_id; // Thread id
};

//测试样例10 矩阵列数43577，非零消元子39477，被消元行54274
//测试样例7 矩阵列数8399，非零消元子6375，被消元行4535
//测试样例8 矩阵列数23045，非零消元子18748，被消元行14325
//测试样例9 矩阵列数37960，非零消元子29304，被消元行14921

// Initializing elimination rows
void init_A() {
    unsigned int a;
    ifstream infile("Groebner/测试样例8 矩阵列数23045，非零消元子18748，被消元行14325/消元子.txt");
    char fin[10000] = { 0 };
    int index;
    while (infile.getline(fin, sizeof(fin))) {
        std::stringstream line(fin);
        int biaoji = 0;
        while (line >> a) {
            if (biaoji == 0) {
                index = a;
                biaoji = 1;
            }
            int k = a % 32;
            int j = a / 32;
            int temp = 1 << k;
            Act[index][Num - 1 - j] += temp;
            Act[index][Num] = 1; // Mark row as non-empty
        }
    }
}

// Initializing passive rows
void init_P() {
    unsigned int a;
    ifstream infile("Groebner/测试样例8 矩阵列数23045，非零消元子18748，被消元行14325/被消元行.txt");
    char fin[10000] = { 0 };
    int index = 0;
    while (infile.getline(fin, sizeof(fin))) {
        std::stringstream line(fin);
        int biaoji = 0;
        while (line >> a) {
            if (biaoji == 0) {
                Pas[index][Num] = a;
                biaoji = 1;
            }
            int k = a % 32;
            int j = a / 32;
            int temp = 1 << k;
            Pas[index][Num - 1 - j] += temp;
        }
        index++;
    }
}

// Function with SIMD and OpenMP
void f_omp1() {
    __m128i va_Pas, va_Act;
    bool sign;
    #pragma omp parallel num_threads(NUM_THREADS) private(va_Pas, va_Act)
    do {
        for (int i = lieNum - 1; i - 8 >= -1; i -= 8) {
            #pragma omp for schedule(static)
            for (int j = 0; j < pasNum; j++) {
                while (Pas[j][Num] <= i && Pas[j][Num] >= i - 7) {
                    int index = Pas[j][Num];
                    if (Act[index][Num] == 1) {
                        for (int k = 0; k + 4 <= Num; k += 4) {
                            va_Pas = _mm_loadu_si128((__m128i*)&Pas[j][k]);
                            va_Act = _mm_loadu_si128((__m128i*)&Act[index][k]);
                            va_Pas = _mm_xor_si128(va_Pas, va_Act);
                            _mm_storeu_si128((__m128i*)&Pas[j][k], va_Pas);
                        }
                        for (int k = (Num / 4) * 4; k < Num; k++) {
                            Pas[j][k] ^= Act[index][k];
                        }
                        int num = 0, S_num = 0;
                        for (num = 0; num < Num; num++) {
                            if (Pas[j][num] != 0) {
                                unsigned int temp = Pas[j][num];
                                while (temp != 0) {
                                    temp = temp >> 1;
                                    S_num++;
                                }
                                S_num += num * 32;
                                break;
                            }
                        }
                        Pas[j][Num] = S_num - 1;
                    } else {
                        break;
                    }
                }
            }
        }
        for (int i = lieNum % 8 - 1; i >= 0; i--) {
            #pragma omp for schedule(static)
            for (int j = 0; j < pasNum; j++) {
                while (Pas[j][Num] == i) {
                    if (Act[i][Num] == 1) {
                        for (int k = 0; k + 4 <= Num; k += 4) {
                            va_Pas = _mm_loadu_si128((__m128i*)&Pas[j][k]);
                            va_Act = _mm_loadu_si128((__m128i*)&Act[i][k]);
                            va_Pas = _mm_xor_si128(va_Pas, va_Act);
                            _mm_storeu_si128((__m128i*)&Pas[j][k], va_Pas);
                        }
                        for (int k = (Num / 4) * 4; k < Num; k++) {
                            Pas[j][k] ^= Act[i][k];
                        }
                        int num = 0, S_num = 0;
                        for (num = 0; num < Num; num++) {
                            if (Pas[j][num] != 0) {
                                unsigned int temp = Pas[j][num];
                                while (temp != 0) {
                                    temp = temp >> 1;
                                    S_num++;
                                }
                                S_num += num * 32;
                                break;
                            }
                        }
                        Pas[j][Num] = S_num - 1;
                    } else {
                        break;
                    }
                }
            }
        }
        #pragma omp single
        {
            sign = false;
            for (int i = 0; i < pasNum; i++) {
                int temp = Pas[i][Num];
                if (temp == -1) {
                    continue;
                }
                if (Act[temp][Num] == 0) {
                    for (int k = 0; k < Num; k++)
                        Act[temp][k] = Pas[i][k];
                    Pas[i][Num] = -1;
                    sign = true;
                }
            }
        }
    } while (sign == true);
}

// Function with OpenMP
void f_omp2() {
    #pragma omp parallel num_threads(NUM_THREADS)
    for (int i = lieNum - 1; i - 8 >= -1; i -= 8) {
        for (int j = 0; j < pasNum; j++) {
            while (Pas[j][Num] <= i && Pas[j][Num] >= i - 7) {
                int index = Pas[j][Num];
                if (Act[index][Num] == 1) {
                    #pragma omp for schedule(static)
                    for (int k = 0; k < Num; k++) {
                        Pas[j][k] ^= Act[index][k];
                    }
                    int num = 0, S_num = 0;
                    for (num = 0; num < Num; num++) {
                        if (Pas[j][num] != 0) {
                            unsigned int temp = Pas[j][num];
                            while (temp != 0) {
                                temp = temp >> 1;
                                S_num++;
                            }
                            S_num += num * 32;
                            break;
                        }
                    }
                    Pas[j][Num] = S_num - 1;
                } else {
                    for (int k = 0; k < Num; k++)
                        Act[index][k] = Pas[j][k];
                    Act[index][Num] = 1;
                    break;
                }
            }
        }
    }
    for (int i = lieNum % 8 - 1; i >= 0; i--) {
        for (int j = 0; j < pasNum; j++) {
            while (Pas[j][Num] == i) {
                if (Act[i][Num] == 1) {
                    #pragma omp for schedule(static)
                    for (int k = 0; k < Num; k++) {
                        Pas[j][k] ^= Act[i][k];
                    }
                    int num = 0, S_num = 0;
                    for (num = 0; num < Num; num++) {
                        if (Pas[j][num] != 0) {
                            unsigned int temp = Pas[j][num];
                            while (temp != 0) {
                                temp = temp >> 1;
                                S_num++;
                            }
                            S_num += num * 32;
                            break;
                        }
                    }
                    Pas[j][Num] = S_num - 1;
                } else {
                    for (int k = 0; k < Num; k++)
                        Act[i][k] = Pas[j][k];
                    Act[i][Num] = 1;
                    break;
                }
            }
        }
    }
}

// Ordinary function without optimizations
void f_ordinary() {
    for (int i = lieNum - 1; i - 8 >= -1; i -= 8) {
        for (int j = 0; j < pasNum; j++) {
            while (Pas[j][Num] <= i && Pas[j][Num] >= i - 7) {
                int index = Pas[j][Num];
                if (Act[index][Num] == 1) {
                    for (int k = 0; k < Num; k++) {
                        Pas[j][k] ^= Act[index][k];
                    }
                    int num = 0, S_num = 0;
                    for (num = 0; num < Num; num++) {
                        if (Pas[j][num] != 0) {
                            unsigned int temp = Pas[j][num];
                            while (temp != 0) {
                                temp = temp >> 1;
                                S_num++;
                            }
                            S_num += num * 32;
                            break;
                        }
                    }
                    Pas[j][Num] = S_num - 1;
                } else {
                    for (int k = 0; k < Num; k++)
                        Act[index][k] = Pas[j][k];
                    Act[index][Num] = 1;
                    break;
                }
            }
        }
    }
    for (int i = lieNum % 8 - 1; i >= 0; i--) {
        for (int j = 0; j < pasNum; j++) {
            while (Pas[j][Num] == i) {
                if (Act[i][Num] == 1) {
                    for (int k = 0; k < Num; k++) {
                        Pas[j][k] ^= Act[i][k];
                    }
                    int num = 0, S_num = 0;
                    for (num = 0; num < Num; num++) {
                        if (Pas[j][num] != 0) {
                            unsigned int temp = Pas[j][num];
                            while (temp != 0) {
                                temp = temp >> 1;
                                S_num++;
                            }
                            S_num += num * 32;
                            break;
                        }
                    }
                    Pas[j][Num] = S_num - 1;
                } else {
                    for (int k = 0; k < Num; k++)
                        Act[i][k] = Pas[j][k];
                    Act[i][Num] = 1;
                    break;
                }
            }
        }
    }
}

void init() {
    init_A();
    init_P();
}

int main() {
    init();
    struct timeval start;
    struct timeval end;
    unsigned long diff;

//    gettimeofday(&start, NULL);
//   //for(int x=0;x<5;x++)
//    f_ordinary();
//    gettimeofday(&end, NULL);
//    diff = 1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec;
//  //  cout << "ordinary: " << diff*0.001 *0.2<< " ms" << endl;
//     cout << "ordinary: " << diff*0.001 << " ms" << endl;


    init();
    gettimeofday(&start, NULL);
   // for(int x=0;x<5;x++)
    f_omp2();
    gettimeofday(&end, NULL);
    diff = 1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec;
    cout << "lab2: " << diff*0.001 << " ms" << endl;
 //   cout << "lab2: " << diff*0.001*0.2 << " ms" << endl;

//    init();
//    gettimeofday(&start, NULL);
//   // for(int x=0;x<5;x++)
//    f_omp1();
//    gettimeofday(&end, NULL);
//    diff = 1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec;
//   // cout << "lab1: " << diff*0.001*0.2 << " ms" << endl;
//    cout << "lab1: " << diff*0.001 << " ms" << endl;

    return 0;
}
