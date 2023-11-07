#define _CRT_SECURE_NO_DEPRECATE

#include <stdio.h>
#include <iostream>
#include <cmath>
#include <string>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

using namespace std;

#define _i(i, j) (((j) + 1) * (bl_sz_x + 2) + (i) + 1)
#define _ix(id) (((id) % (bl_cnt_x + 2)) - 1)
#define _iy(id) (((id) / (bl_cnt_y+ 2)) - 1)


#define _ib(i, j) ((j) * bl_cnt_x + (i))

int main(int argc, char* argv[]) {
	//int bl_id, bl_jd, bl_cnt_all, n;

	int bl_id, bl_jd;
	int bl_cnt_x, bl_cnt_y, n;
	int bl_sz_x, bl_sz_y;
	int i, j;
	FILE* f;

	int id, numproc, proc_name_len;
	char proc_name[MPI_MAX_PROCESSOR_NAME];

	double lx, ly, hx, hy, bc_down, bc_up, bc_left, bc_right, m_eps;
	double bl_max_delta, bl_new_delta;
	double first_value;
	double gl_max_delta = 0; // global max_delta for convergence computation
	double* last_layer, * temp, * new_layer, * buff;

	string out_file;



	MPI_Status status;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numproc);
	MPI_Comm_rank(MPI_COMM_WORLD, &id);
	MPI_Get_processor_name(proc_name, &proc_name_len);


	//fprintf(stderr, "proc %d(%d) on %s\n", id, numproc, proc_name);
	//fflush(stderr);

	MPI_Barrier(MPI_COMM_WORLD);

	if (id == 0) {					// Инициализация параметров расчета
		cin >> bl_cnt_x >> bl_cnt_y >> bl_sz_x >> bl_sz_y;
		cin >> out_file;
		cin >> m_eps;
		cin >> lx >> ly >> bc_left >> bc_right >> bc_down >> bc_up >> first_value;

		
		f = fopen(out_file.c_str(), "w");
		if (f == NULL) {
			cerr << "Can not open file!" << endl;
			return 0;
		}
		

		//it = 0;

		n = bl_sz_x > bl_sz_x ? bl_sz_x : bl_sz_y;

		//fprintf(stderr, "Initial data: \n");
		//fprintf(stderr, "n = %d \n", n);
		//fprintf(stderr, "bl_cnt_x = %d, bl_cnt_y = %d, bl_sz_x = %d, bl_sz_y = %d \n", bl_cnt_x, bl_cnt_y, bl_sz_x, bl_sz_y);
		//fprintf(stderr, "lx = %.10f, ly = %.10f,\n bc_down = %.10f, bc_up = %.10f \n", lx, ly, bc_down, bc_up);
		//fprintf(stderr, "bc_left = %.10f, bc_right = %.12f,\n first_value = %.12f, m_eps = %.12f \n", bc_left, bc_right, first_value, m_eps);
		//fflush(stderr);

	}

	MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);			// Передача параметров расчета всем процессам
	MPI_Bcast(&bl_cnt_x, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&bl_cnt_y, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&bl_sz_x, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&bl_sz_y, 1, MPI_INT, 0, MPI_COMM_WORLD);

	MPI_Bcast(&lx, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&ly, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&bc_down, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&bc_up, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&bc_left, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&bc_right, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&first_value, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&m_eps, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	bl_id = id % bl_cnt_x;
	bl_jd = id / bl_cnt_x;

	hx = lx / (bl_sz_x * bl_cnt_x);
	hy = ly / (bl_sz_y * bl_cnt_y);
	hx = 1 / (hx * hx);
	hy = 1 / (hy * hy);

	//fprintf(stderr, "lx = %.10f, ly = %.10f,\n hx = %.10f, hy = %.10f \n", lx, ly, hx, hy);
	//fflush(stderr);

	last_layer = (double*)malloc(sizeof(double) * (bl_sz_x + 2) * (bl_sz_y + 2));
	new_layer = (double*)malloc(sizeof(double) * (bl_sz_x + 2) * (bl_sz_y + 2));
	buff = (double*)malloc(sizeof(double) * (n + 2));

	for (i = -1; i <= bl_sz_x; i++)
		for (j = -1; j <= bl_sz_y; j++) {			// Инициализация блока
			last_layer[_i(i, j)] = first_value;
			new_layer[_i(i, j)] = first_value;

		}


	int buffer_size;
	MPI_Pack_size(n + 2, MPI_DOUBLE, MPI_COMM_WORLD, &buffer_size);
	buffer_size = 8 * (buffer_size + MPI_BSEND_OVERHEAD);
	double* buffer = (double*)malloc(buffer_size);
	MPI_Buffer_attach(buffer, buffer_size);

	double start = MPI_Wtime();

	do {
		MPI_Barrier(MPI_COMM_WORLD);

		//--------------------------------------Sending-data-----------------------------------------------//
		if (bl_id + 1 < bl_cnt_x) {
			for (j = 0; j < bl_sz_y; j++)
				buff[j] = last_layer[_i(bl_sz_x - 1, j)];
			MPI_Bsend(buff, bl_sz_y, MPI_DOUBLE, _ib(bl_id + 1, bl_jd), id, MPI_COMM_WORLD);
		}

		if (bl_jd + 1 < bl_cnt_y) {
			for (i = 0; i < bl_sz_x; i++)
				buff[i] = last_layer[_i(i, bl_sz_y - 1)];
			MPI_Bsend(buff, bl_sz_x, MPI_DOUBLE, _ib(bl_id, bl_jd + 1), id, MPI_COMM_WORLD);
		}

		if (bl_id > 0) {
			for (j = 0; j < bl_sz_y; j++)
				buff[j] = last_layer[_i(0, j)];
			MPI_Bsend(buff, bl_sz_y, MPI_DOUBLE, _ib(bl_id - 1, bl_jd), id, MPI_COMM_WORLD);
		}

		if (bl_jd > 0) {
			for (i = 0; i < bl_sz_x; i++)
				buff[i] = last_layer[_i(i, 0)];
			MPI_Bsend(buff, bl_sz_x, MPI_DOUBLE, _ib(bl_id, bl_jd - 1), id, MPI_COMM_WORLD);
		}
		//--------------------------------------Receiving-data-----------------------------------------------//
		if (bl_id > 0) {
			MPI_Recv(buff, bl_sz_y, MPI_DOUBLE, _ib(bl_id - 1, bl_jd), _ib(bl_id - 1, bl_jd), MPI_COMM_WORLD, &status);
			for (j = 0; j < bl_sz_y; j++)
				last_layer[_i(-1, j)] = buff[j];
		}
		else {
			for (j = 0; j < bl_sz_y; j++)
				last_layer[_i(-1, j)] = bc_left;

		}
		if (bl_jd > 0) {
			MPI_Recv(buff, bl_sz_x, MPI_DOUBLE, _ib(bl_id, bl_jd - 1), _ib(bl_id, bl_jd - 1), MPI_COMM_WORLD, &status);
			for (i = 0; i < bl_sz_x; i++)
				last_layer[_i(i, -1)] = buff[i];
		}
		else {
			for (i = 0; i < bl_sz_x; i++)
				last_layer[_i(i, -1)] = bc_down;
		}
		if (bl_id + 1 < bl_cnt_x) {
			MPI_Recv(buff, bl_sz_y, MPI_DOUBLE, _ib(bl_id + 1, bl_jd), _ib(bl_id + 1, bl_jd), MPI_COMM_WORLD, &status);
			for (j = 0; j < bl_sz_y; j++)
				last_layer[_i(bl_sz_x, j)] = buff[j];
		}
		else {
			for (j = 0; j < bl_sz_y; j++)
				last_layer[_i(bl_sz_x, j)] = bc_right;
		}
		if (bl_jd + 1 < bl_cnt_y) {
			//fprintf(stderr, "Sender data: msg_sz=%d, source=%d, tag=%d\n", bl_sz_x, _ib(bl_id, bl_jd + 1), _ib(bl_id, bl_jd + 1));
			MPI_Recv(buff, bl_sz_x, MPI_DOUBLE, _ib(bl_id, bl_jd + 1), _ib(bl_id, bl_jd + 1), MPI_COMM_WORLD, &status);
			for (i = 0; i < bl_sz_x; i++)
				last_layer[_i(i, bl_sz_y)] = buff[i];
		}
		else {
			for (i = 0; i < bl_sz_x; i++)
				last_layer[_i(i, bl_sz_y)] = bc_up;
		}

		MPI_Barrier(MPI_COMM_WORLD);

		bl_max_delta = 0;

		for (i = 0; i < bl_sz_x; i++)
			for (j = 0; j < bl_sz_y; j++) {
				new_layer[_i(i, j)] = 0.5 * ((last_layer[_i(i + 1, j)] + last_layer[_i(i - 1, j)]) * hx +
					(last_layer[_i(i, j + 1)] + last_layer[_i(i, j - 1)]) * hy) /
					(hx + hy);

				bl_new_delta = fabs(new_layer[_i(i, j)] - last_layer[_i(i, j)]);

				if (bl_new_delta > bl_max_delta)
					bl_max_delta = bl_new_delta;

			}

		//fprintf(stderr, "New delta = %.16f computed\n", bl_max_delta);
		//fflush(stderr);
		//fprintf(stderr, "New layer computed\n");
		//fflush(stderr);

		temp = new_layer;
		new_layer = last_layer;
		last_layer = temp;

		//fprintf(stderr, "New delta = %.16f computed, block %d\n", bl_max_delta, id);
		//fflush(stderr);

		MPI_Allreduce(&bl_max_delta, &gl_max_delta, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);


		if (id == 0) {
			//fprintf(stderr, "MAX delta = %.16f computed, block %d\n\n\n", gl_max_delta, id);
			//fflush(stderr);
			//it++;
			//fprintf(stderr, "it = %d\n", it);
			//fprintf(stderr, "delta = %.12f\n", gl_max_delta);
			//fflush(stderr);	
		}
	} while (m_eps < gl_max_delta);

	double finish = MPI_Wtime();

	if (id == 0) {
		double m_time = (finish - start)* 1e3;
		printf(" %.3f ms", m_time);

	}

	
	
	if (id != 0) {
		for (j = 0; j < bl_sz_y; j++) {
			for (i = 0; i < bl_sz_x; i++)
				buff[i] = last_layer[_i(i, j)];
			MPI_Send(buff, bl_sz_x, MPI_DOUBLE, 0, id, MPI_COMM_WORLD);
		}
	}
	else {
		for (bl_jd = 0; bl_jd < bl_cnt_y; bl_jd++)
			for (j = 0; j < bl_sz_y; j++)
				for (bl_id = 0; bl_id < bl_cnt_x; bl_id++) {
					if (_ib(bl_id, bl_jd) == 0)
						for (i = 0; i < bl_sz_x; i++)
							buff[i] = last_layer[_i(i, j)];
					else
						MPI_Recv(buff, bl_sz_x, MPI_DOUBLE, _ib(bl_id, bl_jd), _ib(bl_id, bl_jd), MPI_COMM_WORLD, &status);
					for (i = 0; i < bl_sz_x; i++)
						fprintf(f,"%.10f ", buff[i]);
						//printf("%.10f ", buff[i]);
					if (bl_id + 1 == bl_cnt_x) {
						fprintf(f,"\n");
						//printf("\n");
						if (j == bl_sz_y)
							fprintf(f,"\n");
							//printf("\n");
					}
				}

		fclose(f);
	}
	
	

	//
	free(new_layer);
	free(last_layer);
	free(buff);
	

	/*
	if (id != 0) {
		for (j = -1; j <= bl_sz_y; j++) {
			for (i = -1; i <= bl_sz_x; i++)
				buff[i + 1] = last_layer[_i(i, j)];
			MPI_Send(buff, bl_sz_x + 2, MPI_DOUBLE, 0, id, MPI_COMM_WORLD);
		}
	}
	else {
		for (bl_jd = 0; bl_jd < bl_cnt_y; bl_jd++)
			for (j = -1; j <= bl_sz_y; j++)
				for (bl_id = 0; bl_id < bl_cnt_x; bl_id++) {
					if (_ib(bl_id, bl_jd) == 0)
						for (i = -1; i <= bl_sz_x; i++)
							buff[i + 1] = last_layer[_i(i, j)];
					else
						MPI_Recv(buff, bl_sz_x + 2, MPI_DOUBLE, _ib(bl_id, bl_jd), _ib(bl_id, bl_jd), MPI_COMM_WORLD, &status);
					for (i = -1; i <= n; i++)
						printf("%.8f ", buff[i + 1]);
					if (bl_id + 1 == bl_cnt_x) {
						printf("\n");
						if (j == n)
							printf("\n");
					}
					else
						printf(" ");
				}
		fclose(f);
	}
	*/

	MPI_Finalize();

	return 0;
}