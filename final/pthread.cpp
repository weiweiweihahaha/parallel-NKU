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

// �߳�������
int NUM_THREADS = 7;

// �ź�������
sem_t sem_leader;
sem_t* sem_Next;

// ȫ�ֱ������壬�����жϽ������Ƿ������һ��
bool sign;

struct threadParam_t {
    int t_id; // �߳� id
};

// ��Ԫ�ӳ�ʼ��
void init_A() {
    // ÿ����Ԫ�ӵ�һ��Ϊ1λ���ڵ�λ�ã����������ڶ�ά������к�
    // ���磺��Ԫ�ӣ�561��...����Act[561][]���
    unsigned int a;
    ifstream infile("��Ԫ��.txt");
    char fin[10000] = { 0 };
    int index;
    // ���ļ�����ȡ��
    while (infile.getline(fin, sizeof(fin))) {
        std::stringstream line(fin);
        int biaoji = 0;

        // ��������ȡ����������
        while (line >> a) {
            if (biaoji == 0) {
                // ȡÿ�е�һ������Ϊ�б�
                index = a;
                biaoji = 1;
            }
            int k = a % 32;
            int j = a / 32;

            int temp = 1 << k;
            Act[index][Num - 1 - j] += temp;
            Act[index][Num] = 1; // ���ø�λ�ü�¼��Ԫ�Ӹ����Ƿ�Ϊ�գ�Ϊ������0������Ϊ1
        }
    }
}

// ����Ԫ�г�ʼ��
void init_P() {
    // ֱ�Ӱ��մ����ļ���˳��棬�ڴ����ļ��ǵڼ��У���������ǵڼ���
    unsigned int a;
    ifstream infile("����Ԫ��.txt");
    char fin[10000] = { 0 };
    int index = 0;
    // ���ļ�����ȡ��
    while (infile.getline(fin, sizeof(fin))) {
        std::stringstream line(fin);
        int biaoji = 0;

        // ��������ȡ����������
        while (line >> a) {
            if (biaoji == 0) {
                // ��Pas[ ][263]��ű���Ԫ��ÿ�е�һ�����֣�����֮�����Ԫ����
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
        // ������ش�����Ԫ��
        for (int i = lieNum - 1; i - 8 >= -1; i -= 8) {
            for (int j = t_id; j < pasNum; j += NUM_THREADS) {
                // ������Ԫ����û�������ڴ˷�Χ�ڵ�
                while (Pas[j][Num] <= i && Pas[j][Num] >= i - 7) {
                    int index = Pas[j][Num];

                    if (Act[index][Num] == 1) { // ��Ԫ�Ӳ�Ϊ��
                        for (int k = 0; k < Num; k++) {
                            Pas[j][k] = Pas[j][k] ^ Act[index][k];
                        }

                        // ����Pas[j][18]�������ֵ
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
                    } else { // ��Ԫ��Ϊ��
                        break;
                    }
                }
            }
        }

        for (int i = lieNum % 8 - 1; i >= 0; i--) {
            for (int j = t_id; j < pasNum; j += NUM_THREADS) {
                while (Pas[j][Num] == i) {
                    if (Act[i][Num] == 1) { // ��Ԫ�Ӳ�Ϊ��
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
                    } else { // ��Ԫ��Ϊ��
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

    // ��ʼ���ź���
    sem_init(&sem_leader, 0, 0);
    sem_Next = new sem_t[NUM_THREADS - 1];
    for (int i = 0; i < NUM_THREADS - 1; ++i)
        sem_init(&sem_Next[i], 0, 0);

    // ��ʱ����
    struct timeval head, tail;
    double seconds;

    gettimeofday(&head, NULL); // ��ʼ��ʱ

    // �����߳�
    pthread_t* handles = new pthread_t[NUM_THREADS];
    threadParam_t* param = new threadParam_t[NUM_THREADS];

    for (int t_id = 0; t_id < NUM_THREADS; t_id++) {
        param[t_id].t_id = t_id;
        pthread_create(&handles[t_id], NULL, threadFunc, (void*)&param[t_id]);
    }

    for (int t_id = 0; t_id < NUM_THREADS; t_id++)
        pthread_join(handles[t_id], NULL);

    // ���������ź���
    sem_destroy(&sem_leader);
    for (int i = 0; i < NUM_THREADS - 1; ++i)
        sem_destroy(&sem_Next[i]);

    delete[] sem_Next;

    gettimeofday(&tail, NULL); // ������ʱ
    seconds = ((tail.tv_sec - head.tv_sec) * 1000000 + (tail.tv_usec - head.tv_usec)) / 1000.0; // ��λ ms
    cout << "time: " << seconds << " ms" << endl;

    delete[] handles;
    delete[] param;

    return 0;
}
