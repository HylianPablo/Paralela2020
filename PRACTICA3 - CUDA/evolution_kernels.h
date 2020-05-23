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
#define GLOBAL_ID	threadIdx.x + blockIdx.x * blockDim.x

#ifndef FOOD_T
#define FOOD_T
typedef struct {
	int pos;
	int food;
} food_t;
#endif

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
