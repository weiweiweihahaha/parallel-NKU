#include <pthread.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <semaphore.h>
#include <sys/time.h>

using namespace std;

unsigned int Act[23045][722] = { 0 };
unsigned int Pas[23045][722] = { 0 };

const int Num = 721;
const int pasNum = 14325;
const int lieNum = 23045;

// 线程数定义
int NUM_THREADS = 7;

// 信号量定义
sem_t sem_leader;
sem_t* sem_Next;

// 全局变量定义，用于判断接下来是否进入下一轮
bool sign;

struct threadParam_t {
    int t_id; // 线程 id
};

// 消元子初始化
void init_A() {
    // 每个消元子第一个为1位所在的位置，就是它所在二维数组的行号
    // 例如：消元子（561，...）由Act[561][]存放
    unsigned int a;
    ifstream infile("消元子.txt");
    char fin[10000] = { 0 };
    int index;
    // 从文件中提取行
    while (infile.getline(fin, sizeof(fin))) {
        std::stringstream line(fin);
        int biaoji = 0;

        // 从行中提取单个的数字
        while (line >> a) {
            if (biaoji == 0) {
                // 取每行第一个数字为行标
                index = a;
                biaoji = 1;
            }
            int k = a % 32;
            int j = a / 32;

            int temp = 1 << k;
            Act[index][Num - 1 - j] += temp;
            Act[index][Num] = 1; // 设置该位置记录消元子该行是否为空，为空则是0，否则为1
        }
    }
}

// 被消元行初始化
void init_P() {
    // 直接按照磁盘文件的顺序存，在磁盘文件是第几行，在数组就是第几行
    unsigned int a;
    ifstream infile("被消元行.txt");
    char fin[10000] = { 0 };
    int index = 0;
    // 从文件中提取行
    while (infile.getline(fin, sizeof(fin))) {
        std::stringstream line(fin);
        int biaoji = 0;

        // 从行中提取单个的数字
        while (line >> a) {
            if (biaoji == 0) {
                // 用Pas[ ][263]存放被消元行每行第一个数字，用于之后的消元操作
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

void* threadFunc(void* param) {
    threadParam_t* p = (threadParam_t*)param;
    int t_id = p->t_id;

    do {
        // 不升格地处理被消元行
        for (int i = lieNum - 1; i - 8 >= -1; i -= 8) {
            for (int j = t_id; j < pasNum; j += NUM_THREADS) {
                // 看被消元行有没有首项在此范围内的
                while (Pas[j][Num] <= i && Pas[j][Num] >= i - 7) {
                    int index = Pas[j][Num];

                    if (Act[index][Num] == 1) { // 消元子不为空
                        for (int k = 0; k < Num; k++) {
                            Pas[j][k] = Pas[j][k] ^ Act[index][k];
                        }

                        // 更新Pas[j][18]存的首项值
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
                    } else { // 消元子为空
                        break;
                    }
                }
            }
        }

        for (int i = lieNum % 8 - 1; i >= 0; i--) {
            for (int j = t_id; j < pasNum; j += NUM_THREADS) {
                while (Pas[j][Num] == i) {
                    if (Act[i][Num] == 1) { // 消元子不为空
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
                    } else { // 消元子为空
                        break;
                    }
                }
            }
        }

        if (t_id == 0) {
            for (int i = 0; i < NUM_THREADS - 1; ++i)
                sem_wait(&sem_leader);
        } else {
            sem_post(&sem_leader);
            sem_wait(&sem_Next[t_id - 1]);
        }

        if (t_id == 0) {
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

        if (t_id == 0) {
            for (int i = 0; i < NUM_THREADS - 1; ++i)
                sem_post(&sem_Next[i]);
        }

    } while (sign == true);

    pthread_exit(NULL);
}

int main() {
    init_A();
    init_P();

    // 初始化信号量
    sem_init(&sem_leader, 0, 0);
    sem_Next = new sem_t[NUM_THREADS - 1];
    for (int i = 0; i < NUM_THREADS - 1; ++i)
        sem_init(&sem_Next[i], 0, 0);

    // 计时操作
    struct timeval head, tail;
    double seconds;

    gettimeofday(&head, NULL); // 开始计时

    // 创建线程
    pthread_t* handles = new pthread_t[NUM_THREADS];
    threadParam_t* param = new threadParam_t[NUM_THREADS];

    for (int t_id = 0; t_id < NUM_THREADS; t_id++) {
        param[t_id].t_id = t_id;
        pthread_create(&handles[t_id], NULL, threadFunc, (void*)&param[t_id]);
    }

    for (int t_id = 0; t_id < NUM_THREADS; t_id++)
        pthread_join(handles[t_id], NULL);

    // 销毁所有信号量
    sem_destroy(&sem_leader);
    for (int i = 0; i < NUM_THREADS - 1; ++i)
        sem_destroy(&sem_Next[i]);

    delete[] sem_Next;

    gettimeofday(&tail, NULL); // 结束计时
    seconds = ((tail.tv_sec - head.tv_sec) * 1000000 + (tail.tv_usec - head.tv_usec)) / 1000.0; // 单位 ms
    cout << "time: " << seconds << " ms" << endl;

    delete[] handles;
    delete[] param;

    return 0;
}
