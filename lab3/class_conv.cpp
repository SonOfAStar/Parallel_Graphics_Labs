#include <stdio.h>
#include <iostream>

#include <stdlib.h>

using namespace std;

struct uchar4 {
	char x, y, z, w;
};

typedef long long ll;
typedef unsigned char uchar;
typedef unsigned long long ull;
typedef long double ld;
typedef pair<int, int> pii;
typedef pair<ll, ll> pll;


int main(int argc, char** argv) {

	cin.tie(nullptr);
	ios_base::sync_with_stdio(false);
	cout.precision(10);

	// standart input

	string inp_file, out_file;
	cin >> inp_file >> out_file;

	// file reading

	int w, h;
	FILE* fp = fopen(inp_file.c_str(), "rb");
	fread(&w, sizeof(int), 1, fp);
	fread(&h, sizeof(int), 1, fp);
	uchar4* img_data = (uchar4*)malloc(sizeof(uchar4) * w * h);
	fread(img_data, sizeof(uchar4), w * h, fp);
	fclose(fp);


	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			uchar4 p = img_data[i * w + j];
			p.x = (p.w == 0) * 255;
			p.y = (p.w == 1) * 255;
			p.z = (p.w == 2) * 255;
			p.w = 255;
			img_data[i * w + j] = p;
		}
	}


	//file writing

	fp = fopen(out_file.c_str(), "wb");
	fwrite(&w, sizeof(int), 1, fp);
	fwrite(&h, sizeof(int), 1, fp);
	fwrite(img_data, sizeof(uchar4), w * h, fp);
	fclose(fp);

	free(img_data);

	return 0;
}