
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

#include <stdlib.h>
	

#define CSC(call)  												\
do {															\
	cudaError_t res = call;										\
	if (res != cudaSuccess) {									\
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",		\
				__FILE__, __LINE__, cudaGetErrorString(res));	\
		exit(0);												\
	}															\
} while(0)


using namespace std;

typedef long long ll;
typedef unsigned char uchar;
typedef unsigned int uint;
typedef unsigned long long ull;
typedef long double ld;
typedef pair<int, int> pii;
typedef pair<ll, ll> pll;

struct GpuTimer
{
	cudaEvent_t start;
	cudaEvent_t stop;
 
	GpuTimer()
	{
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
	}
 
	~GpuTimer()
	{
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}
 
	void Start()
	{
		cudaEventRecord(start, 0);
	}
 
	void Stop()
	{
		cudaEventRecord(stop, 0);
	}
 
	float Elapsed()
	{
		float elapsed;
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsed, start, stop);
		return elapsed;
	}
};


// CUDA global constants
__constant__ uchar4 p_class_vector[3];


// CUDA kernel
__global__ void kernel(uchar4 * data_vec, int n) {

	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int offsetx = blockDim.x * gridDim.x;
	int x;
	uchar4 p, cls;
	uint diff, diff_old, pixel_class;
	bool old_g_new, old_le_new; // g - greater than, le - lesser or equal than.

	for (x = idx; x < n; x += offsetx) {
		p = data_vec[x];
		diff = 0;
		diff_old = 0 - 1;
		pixel_class = 0;

		for (int cls_ind = 0; cls_ind < 3; cls_ind++) {
			cls = p_class_vector[cls_ind];
			diff = (p.x - cls.x) * (p.x - cls.x) + (p.y - cls.y) * (p.y - cls.y) + (p.z - cls.z) * (p.z - cls.z);
			
			old_g_new = (diff_old > diff);
			old_le_new = (diff_old <= diff);

			pixel_class = pixel_class * old_le_new + cls_ind * old_g_new;
			diff_old = diff_old * old_le_new + diff * old_g_new;
		}

		data_vec[x] = make_uchar4(p.x, p.y, p.z, pixel_class);
		
	}

}


int main(int argc, char** argv) {

	cin.tie(nullptr);
	ios_base::sync_with_stdio(false);
	cout.precision(10);
	GpuTimer timer;

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

	size_t data_size = w * h * sizeof(uchar4);
	//cuda array preparation

	uchar4* img_cuarr;

	CSC(cudaMalloc(&img_cuarr, data_size));

	CSC(cudaMemcpy(img_cuarr, img_data, data_size, cudaMemcpyHostToDevice));

	//cudaMalloc(&cop_data, data_size);
	//cudaMemcpy(cop_data, data, data_size, cudaMemcpyHostToDevice);

	//do magic with constant memry
	uchar4 classes[] = {
		make_uchar4(255, 0, 0, 0),
		make_uchar4(0, 255, 0, 0),
		make_uchar4(0, 0, 255, 0)
	};
	cudaMemcpyToSymbol(p_class_vector, &classes, 3 * sizeof(uchar4));

	//CUDA kernel call
	timer.Start();
	kernel <<< 256, 256 >>> (img_cuarr, w * h);
	timer.Stop();
	CSC(cudaMemcpy(img_data, img_cuarr, w * h * sizeof(uchar4), cudaMemcpyDeviceToHost));

	//free output data
	CSC(cudaFree(img_cuarr));

	//write output
	/*
	fp = fopen(out_file.c_str(), "wb");
	fwrite(&w, sizeof(int), 1, fp);
	fwrite(&h, sizeof(int), 1, fp);
	fwrite(img_data, sizeof(uchar4), w * h, fp);
	fclose(fp);
	*/
	
	cout << timer.Elapsed() << " ms" << endl;
	
	free(img_data);
	return 0;
}