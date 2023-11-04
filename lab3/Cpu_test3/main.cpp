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

    uchar4 classes[] = {
		make_uchar4(255, 0, 0, 0),
		make_uchar4(0, 255, 0, 0),
		make_uchar4(0, 0, 255, 0)
	};

    int x, y;
	uchar4 p, cls;
	uint cl_diff, cl_diff_old, pixel_class;
	bool old_g_new, old_le_new; // g - greater than, le - lesser or equal than.

	for (y = 0; y < h; y ++)
		for (x = 0; x < w; x ++) {
			p = img_data[y * w + x];
			cl_diff = 0;
			cl_diff_old = 0 - 1;
			pixel_class = 0;

			for (int cls_ind = 0; cls_ind < 3; cls_ind++) {
				cls = classes[cls_ind];
				cl_diff = (p.x - cls.x) * (p.x - cls.x) + (p.y - cls.y) * (p.y - cls.y) + (p.z - cls.z) * (p.z - cls.z);

				old_g_new = (cl_diff_old > cl_diff);
				old_le_new = (cl_diff_old <= cl_diff);

				pixel_class = pixel_class * old_le_new + cls_ind * old_g_new;
				cl_diff_old = cl_diff_old * old_le_new + cl_diff * old_g_new;
			}

			img_data[y * w + x] = make_uchar4(p.x, p.y, p.z, pixel_class);

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
