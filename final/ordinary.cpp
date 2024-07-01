#include <iostream>
#include <fstream>
#include <sstream>
#include <windows.h>

using namespace std;

const int Num = 1187;      // ÿ�д洢��Ҫ����������������ȡ��
const int pasNum = 14921;   // ����Ԫ����
const int lieNum = 37960;   // ����

unsigned int Act[37960][1188] = { 0 };  // ��Ԫ�Ӿ���
unsigned int Pas[14921][1188] = { 0 };   // ����Ԫ�о���

// ��ʼ����Ԫ�Ӿ���
void init_A() {
    ifstream infile("Groebner/��������9 ��������37960��������Ԫ��29304������Ԫ��14921/��Ԫ��.txt");
    if (!infile) {
        cerr << "Error opening file: ��Ԫ��.txt" << endl;
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
                continue; // �����кŴ���
            }
            int k = index % 32;
            int j = index / 32;
            int temp = 1 << k;//�õ�һ���ڵ� k λ��Ϊ 1������λΪ 0 �Ķ������� temp
            Act[index][Num - 1 - j] += temp;
            Act[index][Num] = 1;
        }
    }

    infile.close();
}

// ��ʼ������Ԫ�о���
void init_P() {
    ifstream infile("Groebner/��������9 ��������37960��������Ԫ��29304������Ԫ��14921/����Ԫ��.txt");
    if (!infile) {
        cerr << "Error opening file: ����Ԫ��.txt" << endl;
        return;
    }

    char fin[10000] = { 0 };
    int index = 0;
    while (infile.getline(fin, sizeof(fin))) {
        stringstream line(fin);
        int mark = 0;
        while (line >> index) {
            if (mark == 0) {
                Pas[index][Num] = index; // ��һ��Ԫ����Ϊ�к�
                mark = 1;
                continue; // �����кŴ���
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

// ��ͨ��Ԫ����
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

    QueryPerformanceCounter((LARGE_INTEGER*)&head); // ��ʼ��ʱ
    f_ordinary();
    QueryPerformanceCounter((LARGE_INTEGER*)&tail); // ������ʱ

    seconds = (tail - head) * 1000.0 / freq; // ��λ ms
    cout << seconds << " ms" << endl;

    return 0;
}
