
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <chrono>
#include <stdlib.h>



using namespace std;

typedef long long ll;
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

__global__ void kernel(double* input, size_t inp_size) {

	int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int offset = blockDim.x * gridDim.x;
	while (thread_idx < inp_size) {
		input[thread_idx] *= input[thread_idx];
		thread_idx += offset;
	}

}


int main(int argc, char** argv) {

	cin.tie(nullptr);
	ios_base::sync_with_stdio(false);
	cout.precision(10);
	GpuTimer timer;
	


	size_t n;
	cin >> n;
	size_t data_size = sizeof(double) * n;

	double* data = (double*)malloc(data_size);

	for (size_t i = 0; i < n; i++) {
		//data[i] = i;
		cin >> data[i];
	}

	double* cop_data;

	cudaMalloc(&cop_data, data_size);
	cudaMemcpy(cop_data, data, data_size, cudaMemcpyHostToDevice);

	timer.Start();
	kernel <<< 64, 1024 >>> (cop_data, n);
	timer.Stop();
	
	cudaMemcpy(data, cop_data, data_size, cudaMemcpyDeviceToHost);
	
	/*
	for (size_t i = 0; i < n; i++) {
		cout << data[i];
		if (i + 1 < n) cout << ' ';
	}
	cout << endl;
	*/

	cudaFree(cop_data);
	free(data);
	
    //cout.precision(3);
	cout << timer.Elapsed() << " ms" << endl;
    
	
	return 0;
}