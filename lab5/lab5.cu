
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/scan.h>
#include <thrust/device_ptr.h>

#include <stdio.h>
#include <stdlib.h>

#include <iostream>
	

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

double eps = 1e-10;
int MAX_A_I = (1 << 24);
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



// CUDA histogramm kernel
__global__ void Hist_Kernel(int* g_data_vec, int* g_hist_vec, int n) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int offset_x = blockDim.x * gridDim.x;
	
	while (idx < n ) {
		atomicAdd(g_hist_vec + g_data_vec[idx], 1);
		idx += offset_x;
		__syncthreads();
	}

}


// CUDA sorting kernel
__global__ void Sort_Kernel(int* g_data_vec, int* g_hist_vec, int* g_res_vec, int n) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int offset_x = blockDim.x * gridDim.x;	
	
	while (idx < n ) {
		g_res_vec[atomicAdd(g_hist_vec + g_data_vec[idx], 1)] = g_data_vec[idx];
		idx += offset_x;
		__syncthreads();
	}
	

}


int main(int argc, char** argv) {

	cin.tie(nullptr);
	ios_base::sync_with_stdio(false);
	//cout.precision(0);
	GpuTimer timer;

	// standart input
	
	int n;
	cin >> n;
	
	size_t data_size = n * sizeof(int);

	int* m_data = (int*)malloc(data_size);
	int* m_res = (int*)malloc(data_size);
	
	for (int i = 0; i < n; i++) {
		cin >> m_data[i];
	}
	

	// binary reading
	/*
	freopen(NULL, "rb", stdin);

	int n;
	fread(&n, sizeof(int), 1, stdin);
	
	size_t data_size = n * sizeof(int);

	int* m_data = (int*)malloc(data_size);
	int* m_res = (int*)malloc(data_size);
	
	fread(m_data, sizeof(int), n, stdin);
	*/

	//cuda array preparation

	int* cu_data;
	int* cu_res;

	int* cu_hist;

	CSC(cudaMalloc(&cu_data, data_size));
	CSC(cudaMalloc(&cu_res, data_size));

	CSC(cudaMalloc(&cu_hist, sizeof(int) * MAX_A_I));

	//cuda memory set
	CSC(cudaMemcpy(cu_data, m_data, data_size, cudaMemcpyHostToDevice));	

	CSC(cudaMemset(cu_res, 0, data_size));
	CSC(cudaMemset(cu_hist, 0, sizeof(int) * MAX_A_I));

	timer.Start();
	//CUDA kernels call
	
	int blks = 64;
	int trds = 256;
	
	
	Hist_Kernel <<< blks, trds >>> (cu_data, cu_hist, n);
	CSC(cudaGetLastError());

	//thrust transfigurations
	thrust::device_ptr<int> cu_hist_dp = thrust::device_pointer_cast(cu_hist);
	thrust::exclusive_scan(cu_hist_dp, cu_hist_dp + MAX_A_I, cu_hist_dp);

	Sort_Kernel <<< blks, trds >>> (cu_data, cu_hist, cu_res, n);
	CSC(cudaGetLastError());
	timer.Stop();


	//return to sender
	CSC(cudaMemcpy(m_res, cu_res, data_size, cudaMemcpyDeviceToHost));

	//free output data
	CSC(cudaFree(cu_data));
	CSC(cudaFree(cu_res));

	CSC(cudaFree(cu_hist));
	
	/*
	//-----Checker-----
	for (int i = 0; i < n-1; i++) {
		if(m_res[i] > m_res[i+1]){
			cout << "!";
		}
	}
	cout << endl;
	*/

	cout << timer.Elapsed() << " ms" << endl;
	
	//standart output
	
	
	
	/*for (int i = 0; i < n; i++) {
		cout << m_res[i] << ' ';
	}*/
	
	//binary output
	
	//fwrite(m_res, sizeof(int), n, stdout);
	

	free(m_data);
	free(m_res);

	return 0;
}