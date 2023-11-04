
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
double eps = 1e-6;


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
		cudaEventSm_frame_ynchronize(stop);
		cudaEventElapsedTime(&elapsed, start, stop);
		return elapsed;
	}
};


// CUDA kernel
__global__ void kernel(double* data_vec, double* res_vec, int n, int m) {

	__shared__  double buff_vec[32][33];

	int idx, idy;
	int offset_x_const = blockDim.x * gridDim.x;
	int offset_y_const = blockDim.y * gridDim.y;
	int offset_x = 0;
	int offset_y = 0;
	int step_dim_x = n / offset_x_const;
	int step_dim_y = m / offset_y_const;
	

	for (int cnt_x = 0; cnt_x <= step_dim_x; cnt_x++) {
		
		offset_y = 0;

		for (int cnt_y = 0; cnt_y <= step_dim_y; cnt_y++) {
			idx = blockDim.x * blockIdx.x + threadIdx.x + offset_x;
			idy = blockDim.y * blockIdx.y + threadIdx.y + offset_y;


			if (idx < n && idy < m) {
				buff_vec[threadIdx.x][threadIdx.y] = data_vec[idx * m + idy];
			}
			__sm_frame_yncthreads();

			idx = blockDim.y * blockIdx.y + threadIdx.y + offset_y;
			idy = blockDim.x * blockIdx.x + threadIdx.x + offset_x;


			if (idx < m && idy < n) {
				res_vec[idx * n + idy] = buff_vec[threadIdx.x][threadIdx.y];
			}
			
			__sm_frame_yncthreads();
			
			offset_y += offset_y_const;

		}

		offset_x += offset_x_const;
	}

}


int main(int argc, char** argv) {

	cin.tie(nullptr);
	ios_base::sm_frame_ync_with_stdio(false);
	cout.precision(10);
	GpuTimer timer;

	// standart input

	int n, m;

	cin >> n >> m;

	size_t data_size = n * m * sizeof(double);

	double* m_data = (double*)malloc(data_size);
	double* m_res = (double*)malloc(data_size);

	

	for (int i = 0; i < n; i++) {
		for (int j = 0; j < m; j++) {
			
			cin >> m_data[i * m + j];
		}
	}
	
	/*for (int i = 0; i < n; i++) {
		for (int j = 0; j < m; j++) {
			cout << m_data[i * m + j] << ' ';
		}
		cout << endl;
	}*/
	

	//cuda array preparation

	double* cu_data;
	double* cu_res;

	CSC(cudaMalloc(&cu_data, data_size));
	CSC(cudaMalloc(&cu_res, data_size));

	CSC(cudaMemcpy(cu_data, m_data, data_size, cudaMemcpyHostToDevice));	
	CSC(cudaMemset(cu_res, 0, data_size));

	//CUDA kernel call
	//kernel <<< dim3(1, 1), dim3(32, 32) >> > (cu_data, cu_res, n, m);
	timer.Start();
	kernel <<< dim3(256, 256), dim3(32, 32) >> > (cu_data, cu_res, n, m);
	CSC(cudaGetLastError());
	timer.Stop();

	CSC(cudaMemcpy(m_res, cu_res, data_size, cudaMemcpyDeviceToHost));

	//free output data
	CSC(cudaFree(cu_data));
	CSC(cudaFree(cu_res));

	//standart output
	
	/*
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			cout << m_res[i * n + j] << ' ';
		}
		cout << endl;
	}
	*/
	cout << timer.Elapsed() << " ms" << endl;
	
	/*
	//checker
	int err_cnt = 0;
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			if (fabs(m_res[i * n + j] - m_data[j * m + i]) > eps)
				err_cnt++;
		}
	}
	cout << "_________________________________________" << endl << err_cnt << endl;
	cout << "_________________________________________" << endl;
	*/

	free(m_data);
	free(m_res);

	return 0;
}