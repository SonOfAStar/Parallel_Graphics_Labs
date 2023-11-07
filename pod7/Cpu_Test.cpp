#include <stdio.h>
#include <iostream>
#include <cmath>
#include <string>
#include <stdlib.h>
#include <chrono>

using namespace std;

#define _i(i, j) (((j) + 1) * (bl_sz_x + 2) + (i) + 1)
#define _ix(id) (((id) % (bl_cnt_x + 2)) - 1)
#define _iy(id) (((id) / (bl_cnt_y+ 2)) - 1)


#define _ib(i, j) ((j) * bl_cnt_x + (i))

int main(int argc, char* argv[]) {
	//int bl_id, bl_jd, bl_cnt_all, n;

	int bl_id, bl_jd;
	int it = 0;
	int bl_cnt_x, bl_cnt_y, n;
	int bl_sz_x, bl_sz_y;
	int i, j;
	//FILE* f;


	double lx, ly, hx, hy, bc_down, bc_up, bc_left, bc_right, m_eps;
	double bl_max_delta, bl_new_delta;
	double first_value;
	double gl_max_delta = 0; // global max_delta for convergence computation
	double* last_layer, * temp, * new_layer, * buff;

	string out_file;

    cin >> bl_cnt_x >> bl_cnt_y >> bl_sz_x >> bl_sz_y;
    cin >> out_file;
    cin >> m_eps;
    cin >> lx >> ly >> bc_left >> bc_right >> bc_down >> bc_up >> first_value;


    bl_sz_x *= bl_cnt_x;
    bl_sz_y *= bl_cnt_y;

    /*
    f = fopen(out_file.c_str(), "w");
    if (f == NULL) {
        cerr << "Can not open file!" << endl;
        return 0;
    }
    */


    //it = 0;

    n = bl_sz_x > bl_sz_x ? bl_sz_x : bl_sz_y;

    /*
    fprintf(stderr, "Initial data: \n");
    fprintf(stderr, "n = %d \n", n);
    fprintf(stderr, "bl_cnt_x = %d, bl_cnt_y = %d, bl_sz_x = %d, bl_sz_y = %d \n", bl_cnt_x, bl_cnt_y, bl_sz_x, bl_sz_y);
    fprintf(stderr, "lx = %.10f, ly = %.10f,\n bc_down = %.10f, bc_up = %.10f \n", lx, ly, bc_down, bc_up);
    fprintf(stderr, "bc_left = %.10f, bc_right = %.12f,\n first_value = %.12f, m_eps = %.12f \n", bc_left, bc_right, first_value, m_eps);
    fflush(stderr);
    */

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

    auto start = chrono::steady_clock::now();

	do {
        it++;
        for (j = 0; j < bl_sz_y; j++)
            last_layer[_i(-1, j)] = bc_left;

        for (i = 0; i < bl_sz_x; i++)
            last_layer[_i(i, -1)] = bc_down;



		for (j = 0; j < bl_sz_y; j++)
				last_layer[_i(bl_sz_x, j)] = bc_right;

		for (i = 0; i < bl_sz_x; i++)
				last_layer[_i(i, bl_sz_y)] = bc_up;


		bl_max_delta = 0;

		for (i = 0; i < bl_sz_x; i++)
			for (j = 0; j < bl_sz_y; j++) {
				new_layer[_i(i, j)] = 0.5 * ((last_layer[_i(i + 1, j)] + last_layer[_i(i - 1, j)]) * hx +(last_layer[_i(i, j + 1)] + last_layer[_i(i, j - 1)]) * hy) /(hx + hy);

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


	} while (m_eps < bl_max_delta);
    auto m_end = chrono::steady_clock::now();



    printf("finished in %.d iterations \n\n",it);

/*
	for (i = 0; i < bl_sz_x; i++){
			for (j = 0; j < bl_sz_y; j++) {
				//fprintf(f,"%.10f ", last_layer[i]);
				printf("%.10f ", last_layer[_i(i,j)]);
			}
        //fprintf(f,"\n");
        printf("\n");
	}
*/



    auto diff = m_end - start;

    cout << (double)(chrono::duration <double, nano> (diff).count())/(double)1000 << " mks" << endl;

    free(new_layer);
    free(last_layer);
    free(buff);



	return 0;
}
