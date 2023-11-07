#include <stdio.h>
#include <iostream>
#include <time.h>
#include <chrono>
#include <stdlib.h>

using namespace std;
typedef long long ll;
typedef unsigned char uchar;
typedef unsigned int uint;
typedef unsigned long long ull;
typedef long double ld;
typedef pair<int, int> pii;
typedef pair<ll, ll> pll;

int MAX_A_I = (1 << 24);

struct uchar4 {
	uchar x, y, z, w;
};

uchar4 make_uchar4(uchar x, uchar y, uchar z, uchar w){
    uchar4 p;
    p.x = x;
    p.y = y;
    p.z = z;
    p.w = w;
    return p;
}




int main()
{


    cin.tie(nullptr);
	ios_base::sync_with_stdio(false);
	cout.precision(10);


	// standart input

	int n;

	cin >> n;

	size_t data_size = n * sizeof(int);

	int* m_data = (int*)malloc(data_size);
	int* m_res = (int*)malloc(data_size);
	int* m_hist = (int*)malloc(MAX_A_I * sizeof(int));
	int maxim = 0;


    //cout << "lists created " << endl;
	for (int i = 0; i < n; i++) {
        cin >> m_data[i];
        maxim = max(m_data[i], maxim);
	}
    cout << "data read" << endl;

    auto start = chrono::steady_clock::now();


    //--------Calculation------//

    for(int i = 0; i < n; i++){
        m_hist[m_data[i]] ++;
    }

    for(int i = 1; i < maxim; i++){
        m_hist[i] += m_hist[i-1];
    }

    for (int i = 0; i < n; i++) {
        //cout << m_hist[m_data[i]] << "!" << endl;
        m_res[--m_hist[m_data[i]]] = m_data[i];
	}
	//cout << "data processed" << endl;

	/*
	for (size_t i = 0; i < n; i++) {
		//data[i] = i;
		cout << m_res[i] << ' ';
	}*/
	//cout << endl;

	auto m_end = chrono::steady_clock::now();
    free(m_data);
	free(m_res);


    auto diff = m_end - start;

    cout << (double)(chrono::duration <double, nano> (diff).count())/(double)1000 << " mks" << endl;

    return 0;
}
