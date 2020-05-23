/* 
 * Kernel functions for evolution.cu,
 * a simple simulation of life evolution.
 *
 * Developed for CUDA 10.2.
 *
 * (c) 2020, Manuel de Castro Caballero, Pablo Martínez López
 * Developed for an assignment in a Parallel Computing course, 
 * Computer Engineering degree, Universidad de Valladolid (Spain). 
 * Academic year 2019/2020
 */
#define int_urand48( max, seq )	(int)( (long)(max) * glibc_nrand48( seq ) / 2147483648 )


#define GLOBAL_ID threadIdx.x + blockIdx.x *blockDim.x

#ifndef FOOD_T
#define FOOD_T
typedef struct
{
	int pos;
	int food;
} food_t;
#endif

void cell_new_directionXD( Cell *cell ) {
	int angle = int_urand48( INT_2PI, cell->random_seq );
	cell->mov_row = taylor_sin( angle );
	cell->mov_col = taylor_cos( angle );
}

/*
 * Function: Mutation of the movement genes on a new cell
 * 	This function can be changed and/or optimized by the students
 */
void cell_mutationXD( Cell *cell ) {
	/* 1. Select which genes change:
	 	0 Left grows taking part of the Advance part
	 	1 Advance grows taking part of the Left part
	 	2 Advance grows taking part of the Right part
	 	3 Right grows taking part of the Advance part
	*/
	int mutation_type = int_urand48( 4, cell->random_seq );
	/* 2. Select the amount of mutation (up to 50%) */
	int mutation_percentage = int_urand48( PRECISION / 2, cell->random_seq );
	/* 3. Apply the mutation */
	int mutation_value;
	switch( mutation_type ) {
		case 0:
			mutation_value = intfloatMult( cell->choose_mov[1] , mutation_percentage );
			cell->choose_mov[1] -= mutation_value;
			cell->choose_mov[0] += mutation_value;
			break;
		case 1:
			mutation_value = intfloatMult( cell->choose_mov[0] , mutation_percentage );
			cell->choose_mov[0] -= mutation_value;
			cell->choose_mov[1] += mutation_value;
			break;
		case 2:
			mutation_value = intfloatMult( cell->choose_mov[2] , mutation_percentage );
			cell->choose_mov[2] -= mutation_value;
			cell->choose_mov[1] += mutation_value;
			break;
		case 3:
			mutation_value = intfloatMult( cell->choose_mov[1] , mutation_percentage );
			cell->choose_mov[1] -= mutation_value;
			cell->choose_mov[2] += mutation_value;
			break;
	}
	/* 4. Correct potential precision problems */
	cell->choose_mov[2] = PRECISION - cell->choose_mov[1] - cell->choose_mov[0];
}


__global__ void placeFood(int *culture, food_t *food, int count)
{
	int globalPos = GLOBAL_ID;

	if (globalPos < count)
	{
		atomicAdd(&culture[food[globalPos].pos], food[globalPos].food);
	}
}

__global__ void foodDecrease(int *culture, int size)
{
	int globalPos = GLOBAL_ID;

	if (globalPos < size)
	{
		culture[globalPos] -= culture[globalPos] / 20;
	}
}

/*
 * Placeholder function to write to device arrays.
 * This shouldn't be used.
 */
__global__ void addInDeviceArray(int *array, int pos, int value)
{
	if (GLOBAL_ID == 0)
	{
		array[pos] += value;
	}
}

/*
 * 4.4 and 4.5 loops.
 */
__global__ void fourAndFive_loops(int *culture, int *culture_cells, int num_cells, Cell *cells, int *alives, int *steps, int *histories)
{
	/* 4.4. Cell actions */
	//time_start();
	// Space for the list of new cells (maximum number of new cells is num_cells)
	Cell *new_cells = (Cell *)malloc(sizeof(Cell) * num_cells);
#ifdef DEVELOPMENT
	if (new_cells == NULL)
	{
		fprintf(stderr, "-- Error allocating new cells structures for: %d cells\n", num_cells);
		exit(EXIT_FAILURE);
	}
#endif // DEVELOPMENT
	int global_pos = GLOBAL_ID;

	//for (i=0; i<num_cells; i++) {
	if (global_pos < num_cells)
	{
		if (cells[global_pos].alive)
		{

			/*int alive_counter = 0;
			int step_dead_counter = 0; //Reducci
			int history_counter = 0;*/

			/* 4.4.1. Food harvesting */
			int food = culture[global_pos];
			int count = culture_cells[global_pos];

			int my_food = food / count;
			cells[global_pos].storage += my_food;

			/* 4.4.2. Split cell if the conditions are met: Enough maturity and energy */
			if (cells[global_pos].age > 30 && cells[global_pos].storage > ENERGY_NEEDED_TO_SPLIT)
			{
				// Split: Create new cell
				//num_cells_alive ++;
				alives[global_pos]=1;
				//sim_stat.history_total_cells ++;
				histories[global_pos]=1;
				//step_new_cells++;
				steps[global_pos]=1;

				// New cell is a copy of parent cell
				new_cells[global_pos] = cells[global_pos];

				// Split energy stored and update age in both cells
				cells[global_pos].storage /= 2;
				new_cells[global_pos].storage /= 2;
				cells[global_pos].age = 1;
				new_cells[global_pos].age = 1;

				// Random seed for the new cell, obtained using the parent random sequence
				new_cells[global_pos].random_seq[0] = (unsigned short)glibc_nrand48(cells[global_pos].random_seq);
				new_cells[global_pos].random_seq[1] = (unsigned short)glibc_nrand48(cells[global_pos].random_seq);
				new_cells[global_pos].random_seq[2] = (unsigned short)glibc_nrand48(cells[global_pos].random_seq);

				// Both cells start in random directions
				cell_new_directionXD(&cells[global_pos]);
				cell_new_directionXD(&new_cells[global_pos]);

				// Mutations of the movement genes in both cells
				cell_mutationXD(&cells[global_pos]);
				cell_mutationXD(&new_cells[global_pos]);
			}
		}
		culture[global_pos]=0;
	} // End cell actions
	//time_end(time4_4);

	/* 4.5. Clean ancillary data structures */
	//time_start();
	/* 4.5.1. Clean the food consumed by the cells in the culture data structure */
	//for (i = 0; i < num_cells; i++)
	/*{
		if (cells[i].alive)
		{
			culture[i] = 0;
			cudaCheckCall((cudaMemset(&culture[matPos(cells[i].pos_row, cells[i].pos_col)], 0, sizeof(int))));
		}
	}
	//time_end(time4_5);
	*/
}
