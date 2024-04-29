#include<iostream>
#include<fstream>
#include<string>
#include<sstream>
#include<map>
#include<windows.h>
#include<tmmintrin.h>
#include<xmmintrin.h>
#include<emmintrin.h>
#include<pmmintrin.h>
#include<smmintrin.h>
#include<nmmintrin.h>
#include<immintrin.h>
using namespace std;

const int maxsize = 3000;
const int maxrow = 3000; //3000*32>90000 ,������90000*90000�ľ���3000��
const int numBasis = 90000;

map<int, int*>iToBasis;
map<int, int>iToFirst;
map<int, int*>ans;

fstream RowFile("D:/�����/paralleling/ʵ��/lab2/Groebner/��������7 ��������8399��������Ԫ��6375������Ԫ��4535/����Ԫ��.txt", ios::in | ios::out);
fstream BasisFile("D:/�����/paralleling/ʵ��/lab2/Groebner/��������7 ��������8399��������Ԫ��6375������Ԫ��4535/��Ԫ��.txt", ios::in | ios::out);




int gRows[maxrow][maxsize];
int gBasis[numBasis][maxsize];

void reset() {
	memset(gRows, 0, sizeof(gRows));
	memset(gBasis, 0, sizeof(gBasis));
	RowFile.close();
	BasisFile.close();
	RowFile.open("D:/�����/paralleling/ʵ��/lab2/Groebner/��������7 ��������8399��������Ԫ��6375������Ԫ��4535/����Ԫ��.txt", ios::in | ios::out);
	BasisFile.open("D:/�����/paralleling/ʵ��/lab2/Groebner/��������7 ��������8399��������Ԫ��6375������Ԫ��4535/��Ԫ��.txt", ios::in | ios::out);
	iToBasis.clear();
	iToFirst.clear();
	ans.clear();

}

void readBasis() {
	for (int i = 0; i < maxrow; i++) {
		if (BasisFile.eof()) {
			return;
		}
		string tmp;
		bool flag = false;
		int row = 0;
		getline(BasisFile, tmp);
		stringstream s(tmp);
		int pos;
		while (s >> pos) {
			//cout << pos << " ";
			if (!flag) {
				row = pos;
				flag = true;
				iToBasis.insert(pair<int, int*>(row, gBasis[row]));
			}
			int index = pos / 32;
			int offset = pos % 32;
			gBasis[row][index] = gBasis[row][index] | (1 << offset);
		}
		flag = false;
		row = 0;
	}
}

int readRowsFrom(int pos) {
	iToFirst.clear();
	if (RowFile.is_open())
		RowFile.close();
	RowFile.open("D:/�����/paralleling/ʵ��/lab2/Groebner/��������7 ��������8399��������Ԫ��6375������Ԫ��4535/����Ԫ��.txt", ios::in | ios::out);
	memset(gRows, 0, sizeof(gRows));   //����Ϊ0
	string line;
	for (int i = 0; i < pos; i++) {
		getline(RowFile, line);
	}
	for (int i = pos; i < pos + maxsize; i++) {
		int tmp;
		getline(RowFile, line);
		if (line.empty()) {
			cout << "End of File!" << endl;
			return i;   //û�ܶ�ȡ��3000�У����ض�ȡ������
		}
		bool flag = false;
		stringstream s(line);
		while (s >> tmp) {
			if (!flag) {//i-pos���кţ�tmp������
				iToFirst.insert(pair<int, int>(i - pos, tmp));
			}
			int index = tmp / 32;
			int offset = tmp % 32;
			gRows[i - pos][index] = gRows[i - pos][index] | (1 << offset);
			flag = true;
		}
	}
	return -1;  //�ɹ���ȡ3000��

}

void update(int row) {
	bool flag = 0;
	for (int i = maxsize - 1; i >= 0; i--) {
		if (gRows[row][i] == 0)
			continue;
		else {
			if (!flag)
				flag = true;
			int pos = i * 32;
			int offset = 0;
			for (int k = 31; k >= 0; k--) {
				if (gRows[row][i] & (1 << k))
				{
					offset = k;
					break;
				}
			}
			int newfirst = pos + offset;
			iToFirst.erase(row);
			iToFirst.insert(pair<int, int>(row, newfirst));
			break;
		}
	}
	if (!flag) {
		iToFirst.erase(row);
	}
	return;
}

void writeResult(ofstream& out) {
	for (auto it = ans.rbegin(); it != ans.rend(); it++) {
		int* result = it->second;
		int max = it->first / 32 + 1;
		for (int i = max; i >= 0; i--) {
			if (result[i] == 0)
				continue;
			int pos = i * 32;
			//int offset = 0;
			for (int k = 31; k >= 0; k--) {
				if (result[i] & (1 << k)) {
					out << k + pos << " ";
				}
			}
		}
		out << endl;
	}
}

void GE() {
	int begin = 0;
	int flag;
	while (true) {
		flag = readRowsFrom(begin);
		int num = (flag == -1) ? maxsize : flag;
		for (int i = 0; i < num; i++) {
			while (iToFirst.find(i) != iToFirst.end()) {
				int first = iToFirst.find(i)->second;
				if (iToBasis.find(first) != iToBasis.end()) {  //���ڸ���Ԫ��
					int* basis = iToBasis.find(first)->second;
					for (int j = 0; j < maxsize; j++) {
						gRows[i][j] = gRows[i][j] ^ basis[j];

					}
					update(i);
				}
				else {
					for (int j = 0; j < maxsize; j++) {
						gBasis[first][j] = gRows[i][j];
					}
					iToBasis.insert(pair<int, int*>(first, gBasis[first]));
					ans.insert(pair<int, int*>(first, gBasis[first]));
					iToFirst.erase(i);
				}
			}
		}
		if (flag == -1)
			begin += maxsize;
		else
			break;
	}

}

void AVX_GE() {
	int begin = 0;
	int flag;
	while (true) {
		flag = readRowsFrom(begin);
		int num = (flag == -1) ? maxsize : flag;
		for (int i = 0; i < num; i++) {
			while (iToFirst.find(i) != iToFirst.end()) {
				int first = iToFirst.find(i)->second;
				if (iToBasis.find(first) != iToBasis.end()) {  //���ڸ���Ԫ��
					int* basis = iToBasis.find(first)->second;
					int j = 0;
					for (; j + 8 < maxsize; j += 8) {
						__m256i vij = _mm256_loadu_si256((__m256i*) & gRows[i][j]);
						__m256i vj = _mm256_loadu_si256((__m256i*) & basis[j]);
						__m256i vx = _mm256_xor_si256(vij, vj);
						_mm256_storeu_si256((__m256i*) & gRows[i][j], vx);
					}
					for (; j < maxsize; j++) {
						gRows[i][j] = gRows[i][j] ^ basis[j];
					}
					update(i);
				}
				else {
					for (int j = 0; j < maxsize; j++) {
						gBasis[first][j] = gRows[i][j];
					}
					iToBasis.insert(pair<int, int*>(first, gBasis[first]));
					ans.insert(pair<int, int*>(first, gBasis[first]));
					iToFirst.erase(i);
				}
			}
		}
		if (flag == -1)
			begin += maxsize;
		else
			break;
	}
}

void AVX512_GE() {
	int begin = 0;
	int flag;
	while (true) {
		flag = readRowsFrom(begin);
		int num = (flag == -1) ? maxsize : flag;
		for (int i = 0; i < num; i++) {
			while (iToFirst.find(i) != iToFirst.end()) {
				int first = iToFirst.find(i)->second;
				if (iToBasis.find(first) != iToBasis.end()) {  // ���ڸ���Ԫ��
					int* basis = iToBasis.find(first)->second;
					int j = 0;
					for (; j + 16 < maxsize; j += 16) {
						__m512i vij = _mm512_loadu_si512((__m512i*)&gRows[i][j]);
						__m512i vj = _mm512_loadu_si512((__m512i*)&basis[j]);
						__m512i vx = _mm512_xor_si512(vij, vj);
						_mm512_storeu_si512((__m512i*)&gRows[i][j], vx);
					}
					for (; j < maxsize; j++) {
						gRows[i][j] = gRows[i][j] ^ basis[j];
					}
					update(i);
				}
				else {
					for (int j = 0; j < maxsize; j++) {
						gBasis[first][j] = gRows[i][j];
					}
					iToBasis.insert(pair<int, int*>(first, gBasis[first]));
					ans.insert(pair<int, int*>(first, gBasis[first]));
					iToFirst.erase(i);
				}
			}
		}
		if (flag == -1)
			begin += maxsize;
		else
			break;
	}
}

void SSE_GE() {
	int begin = 0;
	int flag;
	while (true) {
		flag = readRowsFrom(begin);
		int num = (flag == -1) ? maxsize : flag;
		for (int i = 0; i < num; i++) {
			while (iToFirst.find(i) != iToFirst.end()) {
				int first = iToFirst.find(i)->second;
				if (iToBasis.find(first) != iToBasis.end()) {  // ���ڸ���Ԫ��
					int* basis = iToBasis.find(first)->second;
					int j = 0;
					for (; j + 4 < maxsize; j += 4) {
						__m128i vij = _mm_loadu_si128((__m128i*) & gRows[i][j]);
						__m128i vj = _mm_loadu_si128((__m128i*) & basis[j]);
						__m128i vx = _mm_xor_si128(vij, vj);
						_mm_storeu_si128((__m128i*) & gRows[i][j], vx);
					}
					for (; j < maxsize; j++) {
						gRows[i][j] = gRows[i][j] ^ basis[j];
					}
					update(i);
				}
				else {
					for (int j = 0; j < maxsize; j++) {
						gBasis[first][j] = gRows[i][j];
					}
					iToBasis.insert(pair<int, int*>(first, gBasis[first]));
					ans.insert(pair<int, int*>(first, gBasis[first]));
					iToFirst.erase(i);
				}
			}
		}
		if (flag == -1)
			begin += maxsize;
		else
			break;
	}
}

int main() {
	double time1 = 0;
	double time_avx = 0;
	double time_avx512 = 0;
	double time_SSE= 0;

	long long head, tail, freq;
	for (int i = 0; i < 1; i++) {
		ofstream out("D:�����/paralleling/ʵ��/lab2/Groebner/��������7 ��������8399��������Ԫ��6375������Ԫ��4535/��Ԫ���.txt");
		ofstream out1("D:/�����/paralleling/ʵ��/lab2/Groebner/��������7 ��������8399��������Ԫ��6375������Ԫ��4535/��Ԫ���(AVX).txt");
		out << "__________" << endl;
		out1 << "__________" << endl;
		QueryPerformanceFrequency((LARGE_INTEGER*)&freq);

		readBasis();
		//writeResult();
		QueryPerformanceCounter((LARGE_INTEGER*)&head);
		GE();
		QueryPerformanceCounter((LARGE_INTEGER*)&tail);
		cout << "Ordinary time:" << (tail - head) * 1000 / freq << "ms" << endl;
		time1 += (tail - head) * 1000 / freq;
		//writeResult(out);

		reset();

		readBasis();
		QueryPerformanceCounter((LARGE_INTEGER*)&head);
		AVX_GE();
		QueryPerformanceCounter((LARGE_INTEGER*)&tail);
		cout << "AVX time:" << (tail - head) * 1000 / freq << "ms" << endl;
		time_avx += (tail - head) * 1000 / freq;
		//writeResult(out1);

		reset();

		readBasis();
		QueryPerformanceCounter((LARGE_INTEGER*)&head);
		AVX512_GE();
		QueryPerformanceCounter((LARGE_INTEGER*)&tail);
		cout << "AVX 512 time:" << (tail - head) * 1000 / freq << "ms" << endl;
		time_avx512 += (tail - head) * 1000 / freq;
		//writeResult(out1);

		reset();

		readBasis();
		QueryPerformanceCounter((LARGE_INTEGER*)&head);
		SSE_GE();
		QueryPerformanceCounter((LARGE_INTEGER*)&tail);
		cout << "SSE time:" << (tail - head) * 1000 / freq << "ms" << endl;
		time_SSE += (tail - head) * 1000 / freq;
		//writeResult(out1);

		reset();
		out.close();
		out1.close();
	}
	cout << "time1:" << time1 / 5 << endl << "timeavx:" << time_avx / 5 << "time_avx512:" << time_avx512 / 5;
	cout << "time_SSE:" << time_SSE / 5;
}