#include <iostream>
#include <fstream>
#include <sstream>
#include <windows.h>

using namespace std;

const int Num = 1187;      // 每行存储需要的整数数量，向上取整
const int pasNum = 14921;   // 被消元行数
const int lieNum = 37960;   // 列数

unsigned int Act[37960][1188] = { 0 };  // 消元子矩阵
unsigned int Pas[14921][1188] = { 0 };   // 被消元行矩阵

// 初始化消元子矩阵
void init_A() {
    ifstream infile("Groebner/测试样例9 矩阵列数37960，非零消元子29304，被消元行14921/消元子.txt");
    if (!infile) {
        cerr << "Error opening file: 消元子.txt" << endl;
        return;
    }

    char fin[10000] = { 0 };
    int index;
    while (infile.getline(fin, sizeof(fin))) {
        stringstream line(fin);
        int mark = 0;
        while (line >> index) {
            if (mark == 0) {
                mark = 1;
                continue; // 跳过行号处理
            }
            int k = index % 32;
            int j = index / 32;
            int temp = 1 << k;//得到一个在第 k 位上为 1，其他位为 0 的二进制数 temp
            Act[index][Num - 1 - j] += temp;
            Act[index][Num] = 1;
        }
    }

    infile.close();
}

// 初始化被消元行矩阵
void init_P() {
    ifstream infile("Groebner/测试样例9 矩阵列数37960，非零消元子29304，被消元行14921/被消元行.txt");
    if (!infile) {
        cerr << "Error opening file: 被消元行.txt" << endl;
        return;
    }

    char fin[10000] = { 0 };
    int index = 0;
    while (infile.getline(fin, sizeof(fin))) {
        stringstream line(fin);
        int mark = 0;
        while (line >> index) {
            if (mark == 0) {
                Pas[index][Num] = index; // 第一个元素作为行号
                mark = 1;
                continue; // 跳过行号处理
            }
            int k = index % 32;
            int j = index / 32;
            int temp = 1 << k;
            Pas[index][Num - 1 - j] += temp;
        }
        index++;
    }

    infile.close();
}

// 普通消元过程
void f_ordinary() {
    for (int i = lieNum - 1; i - 8 >= -1; i -= 8) {
        for (int j = 0; j < pasNum; j++) {
            while (Pas[j][Num] <= i && Pas[j][Num] >= i - 7) {
                int index = Pas[j][Num];
                if (Act[index][Num] == 1) {
                    for (int k = 0; k < Num; k++) {
                        Pas[j][k] = Pas[j][k] ^ Act[index][k];
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
                    for (int k = 0; k < Num; k++) {
                        Act[index][k] = Pas[j][k];
                    }
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
                        Pas[j][k] = Pas[j][k] ^ Act[i][k];
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
                    for (int k = 0; k < Num; k++) {
                        Act[i][k] = Pas[j][k];
                    }
                    Act[i][Num] = 1;
                    break;
                }
            }
        }
    }
}

int main() {
    double seconds;
    __int64 head, tail, freq;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);

    init_A();
    init_P();

    QueryPerformanceCounter((LARGE_INTEGER*)&head); // 开始计时
    f_ordinary();
    QueryPerformanceCounter((LARGE_INTEGER*)&tail); // 结束计时

    seconds = (tail - head) * 1000.0 / freq; // 单位 ms
    cout << seconds << " ms" << endl;

    return 0;
}
