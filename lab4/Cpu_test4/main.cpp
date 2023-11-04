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

	int n, m;

	cin >> n >> m;

	size_t data_size = n * m * sizeof(double);

	double* m_data = (double*)malloc(data_size);
	double* m_res = (double*)malloc(data_size);


    cout << "lists created " << endl;
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < m; j++) {

			cin >> m_data[i * m + j];
		}
	}
    cout << "data read" << endl;

    auto start = chrono::steady_clock::now();


    //--------Calculation------//

    for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
            m_res[i * n + j] = m_data[j * m + i];
		}
	}
	cout << "data processed" << endl;

	/*for (size_t i = 0; i < n; i++) {
		//data[i] = i;
		cout << data[i] << ' ';
	}
	cout << endl;*/

	auto m_end = chrono::steady_clock::now();
    free(m_data);
	free(m_res);


    auto diff = m_end - start;

    cout << (double)(chrono::duration <double, nano> (diff).count())/(double)1000 << " mks" << endl;

    return 0;
}
