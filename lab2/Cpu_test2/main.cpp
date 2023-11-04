#include <stdio.h>
#include <iostream>
#include <time.h>
#include <chrono>
#include <stdlib.h>

using namespace std;
typedef long long ll;
typedef unsigned char uchar;
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


	string inp_file, out_file;
	cin >> inp_file ;//>> out_file;

	// file reading

	int w, h;
	FILE* fp = fopen(inp_file.c_str(), "rb");
	fread(&w, sizeof(int), 1, fp);
	fread(&h, sizeof(int), 1, fp);
	uchar4* img_data = (uchar4*)malloc(sizeof(uchar4) * w * h);
	fread(img_data, sizeof(uchar4), w * h, fp);
	fclose(fp);


    auto start = chrono::steady_clock::now();


    //--------Calculation------//

    uchar4 p;
    uchar brightness;
    for (int y = 0; y < h; y ++)
		for (int x = 0; x < w; x ++) {
			p = img_data[y * w + x];
			brightness = (uchar)(0.299 * p.x + 0.587 * p.y + 0.114 * p.z);
			//(0.299*R + 0.587*G + 0.114*B) - percieved brightness
			//(0.2126*R + 0.7152*G + 0.0722*B) - standard brightness
			img_data[y * w + x] = make_uchar4(brightness, brightness, brightness, p.w);

		}



	/*for (size_t i = 0; i < n; i++) {
		//data[i] = i;
		cout << data[i] << ' ';
	}
	cout << endl;*/

	auto m_end = chrono::steady_clock::now();
    free(img_data);


    auto diff = m_end - start;

    cout << (double)(chrono::duration <double, nano> (diff).count())/(double)1000 << " mks" << endl;

    return 0;
}
