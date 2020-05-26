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
#include "int_float.h"
#include<stdio.h>
#include<stdlib.h>
#define int_urand48( max, seq )	(int)( (long)(max) * glibc_nrand48( seq ) / 2147483648 )
#define matPos(exp1, exp2)	(int)(exp1) / PRECISION * columns + (int)(exp2) / PRECISION

#define GLOBAL_ID threadIdx.x + blockIdx.x *blockDim.x

#ifndef FOOD_T
#define FOOD_T
typedef struct
{
	int pos;
	int food;
} food_t;
#endif

__host__ __device__ void cell_new_direction( Cell *cell );
__host__ __device__ void cell_mutation( Cell *cell );

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
__global__ void evolution44_45(int *culture, int *culture_cells, int columns, int num_cells, Cell *cells, int *step_new_cells, unsigned short *seeds)
{
	/* 4.4. Cell actions */
	// Space for the list of new cells (maximum number of new cells is num_cells)
#ifdef DEVELOPMENT
	if (new_cells == NULL)
	{
		fprintf(stderr, "-- Error allocating new cells structures for: %d cells\n", num_cells);
		exit(EXIT_FAILURE);
	}
#endif // DEVELOPMENT
	int global_pos = GLOBAL_ID;
	Cell *my_cell = &cells[global_pos];

	if (global_pos < num_cells)
	{
		/* 4.4.1. Food harvesting */
		int food = culture[matPos(my_cell->pos_row, my_cell->pos_col)];
		int count = culture_cells[matPos(my_cell->pos_row, my_cell->pos_col)];

		int my_food = food / count;
		my_cell->storage += my_food;

		/* 4.4.2. Split cell if the conditions are met: Enough maturity and energy */
		if (my_cell->age > 30 && my_cell->storage > ENERGY_NEEDED_TO_SPLIT)
		{
			// Split: Create new cell
			atomicAdd(step_new_cells, 1);

			// Split energy stored and update age in both cells
			my_cell->storage /= 2;
			my_cell->age = 1;

			// New cell is a copy of parent cell
			cells[num_cells + global_pos] = *my_cell;
			Cell *my_new_cell = &cells[num_cells + global_pos];

			// Random seed for the new cell, obtained using the parent random sequence
			my_new_cell->random_seq[0] = (unsigned short)glibc_nrand48(&seeds[global_pos]);
			my_new_cell->random_seq[1] = (unsigned short)glibc_nrand48(&seeds[global_pos]);
			my_new_cell->random_seq[2] = (unsigned short)glibc_nrand48(&seeds[global_pos]);

			// Both cells start in random directions
			cell_new_direction(my_cell);
			cell_new_direction(my_new_cell);

			// Mutations of the movement genes in both cells
			cell_mutation(my_cell);
			cell_mutation(my_new_cell);
		}		
	} // End cell actions
	__syncthreads();
	culture[matPos(my_cell->pos_row, my_cell->pos_col)] = 0;

	if(global_pos==0)
		printf("antes del 4.7\n");
	// 4.7. Join cell lists: Old and new cells list /
	if (global_pos == 0)
	{
		if ( step_new_cells > 0 ) {
			int free_position = 0;
			for(int i = num_cells + 1; i < 2 * num_cells; i++) {
				if ( cells[i].alive ) {
					if ( free_position != i ) {
						cells[free_position] = cells[i];
					}
					free_position ++;
				}
			}
		}
	}
	if(global_pos==0)
		printf("despues del 4.7\n");
}
