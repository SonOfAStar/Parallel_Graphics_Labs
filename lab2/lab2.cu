
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

__global__ void kernel(uchar4 * out, cudaTextureObject_t texObj, int w, int h) {

	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;
	int offsetx = blockDim.x * gridDim.x;
	int offsety = blockDim.y * gridDim.y;
	int x, y;
	uchar4 p;
	uchar brightness;

	for (y = idy; y < h; y += offsety)
		for (x = idx; x < w; x += offsetx) {
			p = tex2D<uchar4>(texObj, x, y);
			brightness = (uchar)(0.299 * p.x + 0.587 * p.y + 0.114 * p.z);
			//(0.299*R + 0.587*G + 0.114*B) - percieved brightness
			//(0.2126*R + 0.7152*G + 0.0722*B) - standard brightness
			out[y * w + x] = make_uchar4(brightness, brightness, brightness, p.w);
			
		}

}


int main(int argc, char** argv) {

	cin.tie(nullptr);
	ios_base::sync_with_stdio(false);
	cout.precision(3);
	GpuTimer timer;

	// standart input

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

	//cuda array preparation

	cudaArray* texture_arr;
	cudaChannelFormatDesc ch = cudaCreateChannelDesc<uchar4>();
	CSC(cudaMallocArray(&texture_arr, &ch, w, h));

	CSC(cudaMemcpy2DToArray(texture_arr, 0, 0, img_data, 0, w, h, cudaMemcpyHostToDevice));

	// cuda resource prep
	cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = texture_arr;
	
	//texture object creation
	cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.readMode = cudaReadModeElementType;
	texDesc.filterMode = cudaFilterModePoint;

	cudaTextureObject_t tex;
	CSC(cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL));

	//create output data
	uchar4* output_data;
	CSC(cudaMalloc(&output_data, sizeof(uchar4) * w * h));
	timer.Start();
	kernel <<< dim3(1024, 256), dim3(32, 32) >>> (output_data, tex, w, h);
	timer.Stop();
	CSC(cudaMemcpy(img_data, output_data, sizeof(uchar4) * w * h, cudaMemcpyDeviceToHost));

	//free texture, texture_arr, output data
	CSC(cudaFreeArray(texture_arr));
	CSC(cudaFree(output_data));
	CSC(cudaDestroyTextureObject(tex));

	/*fp = fopen(out_file.c_str(), "wb");
	fwrite(&w, sizeof(int), 1, fp);
	fwrite(&h, sizeof(int), 1, fp);
	fwrite(img_data, sizeof(uchar4), w * h, fp);
	fclose(fp);*/

	cout << timer.Elapsed() << " ms" << endl;

	free(img_data);
	return 0;
}