#include <stdio.h>
#include <iostream>
#include <time.h>
#include <chrono>
#include <stdlib.h>

using namespace std;

int main()
{


    cin.tie(nullptr);
	ios_base::sync_with_stdio(false);
	cout.precision(10);


	size_t n;
	cin >> n;
	size_t data_size = sizeof(double) * n;

	double* data = (double*)malloc(data_size);

	for (size_t i = 0; i < n; i++) {
		//data[i] = i;
		cin >> data[i];
	}


    auto start = chrono::steady_clock::now();
	for (size_t i = 0; i < n; i++) {
		//data[i] = i;
		data[i] *= data[i];
	}

	/*for (size_t i = 0; i < n; i++) {
		//data[i] = i;
		cout << data[i] << ' ';
	}
	cout << endl;*/

    free(data);

    auto m_end = chrono::steady_clock::now();
    auto diff = m_end - start;

    cout << (double)(chrono::duration <double, nano> (diff).count())/(double)1000 << " mks" << endl;

    return 0;
}
