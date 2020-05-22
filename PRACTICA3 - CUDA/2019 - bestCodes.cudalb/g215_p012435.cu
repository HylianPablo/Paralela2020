// User: g215@83.42.84.115 
// ExecutionRequest[P:'extinguish-046.cu',P:1,T:1,args:'',q:'cudalb'] 
// May 15 2019 22:51:53
#include "cputils.h" // Added by tablon
/*
 * Simplified simulation of fire extinguishing
 *
 * Computacion Paralela, Grado en Informatica (Universidad de Valladolid)
 * 2018/2019
 *
 * v1.4
 *
 * (c) 2019 Arturo Gonzalez Escribano
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <cputils.h>

#define RADIUS_TYPE_1		3
#define RADIUS_TYPE_2_3		9
#define THRESHOLD	0.1f
#define MAX_BLQ_DIM	(int) 256
#define X_DIV	(int) 16
#define Y_DIV	(int) 16

/* Structure to store data of an extinguishing team */
typedef struct {
	int x,y;
	int type;
	int target;
} Team;

/* Structure to store data of a fire focal point */
typedef struct {
	int x,y;
	int start;
	int heat;
	int active; // States: 0 Not yet activated; 1 Active; 2 Deactivated by a team
} FocalPoint;

/* Macro function to simplify accessing with two coordinates to a flattened array */
#define accessMat( arr, exp1, exp2 )	arr[ (exp1) * columns + (exp2) ]

#define CUDA_CHECK()  { \
	cudaError_t check = cudaGetLastError(); \
	if ( check != cudaSuccess ) { 		\
		printf("Error.... %s \n", cudaGetErrorString( check ) ); \
		exit( EXIT_FAILURE ); \
	} }

__global__ void test()
{
	printf("Hola, soy el hilo nº X = %d, Y = %d del bloque nº  X = %d, Y = %d.\n", threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y);
}

__global__ void Initialize(float *surfaceKernel, int rows, int columns) {
	// Desactivar hilos que excedan los límites del array de entrada
	int blockID = blockIdx.x + gridDim.x * blockIdx.y;
	int thread_local_id = threadIdx.x + threadIdx.y * blockDim.x;
	int threads_in_block = blockDim.x * blockDim.y;
	int gid = thread_local_id + blockID * threads_in_block;
   	if ( (gid >= (rows-1)*columns) || (gid < columns) || (gid%columns==0) || ((gid+1)%columns==0)) return;

	surfaceKernel[gid] = 0.0;

}

__global__ void UpdateFocal(FocalPoint *focalKernel, int num_focalKernel, float *surfaceKernel, int columns) {

	// Desactivar hilos que excedan los límites del array de entrada
	int blockID = blockIdx.x + gridDim.x * blockIdx.y;
	int thread_local_id = threadIdx.x + threadIdx.y * blockDim.x;
	int threads_in_block = blockDim.x * blockDim.y;
	int gid = thread_local_id + blockID * threads_in_block;
   	if ( gid >= num_focalKernel ) return;

	if ( focalKernel[gid].active != 1 ) return;
	int x = focalKernel[gid].x;
	int y = focalKernel[gid].y;
	accessMat( surfaceKernel, x, y ) = focalKernel[gid].heat;

}

__global__ void CopySurface(float *surfaceKernel, float *surfaceCopyKernel, int rows, int columns) {
	// Desactivar hilos que excedan los límites del array de entrada
	int blockID = blockIdx.x + gridDim.x * blockIdx.y;
	int thread_local_id = threadIdx.x + threadIdx.y * blockDim.x;
	int threads_in_block = blockDim.x * blockDim.y;
	int gid = thread_local_id + blockID * threads_in_block;
   	if ( (gid >= (rows-1)*columns) || (gid < columns) || (gid%columns==0) || ((gid+1)%columns==0)) return;

	surfaceCopyKernel[gid] = surfaceKernel[gid];

}


__global__ void SpreadHeat(float *surfaceKernel, float *surfaceCopyKernel, int rows, int columns) {

	int blockID = blockIdx.x + gridDim.x * blockIdx.y;
	int thread_local_id = threadIdx.x + threadIdx.y * blockDim.x;
	int threads_in_block = blockDim.x * blockDim.y;
	int gid = thread_local_id + blockID * threads_in_block;

   	if ( (gid >= (rows-1)*columns) || (gid < columns) || (gid%columns==0) || ((gid+1)%columns==0)) return;

   	float up = surfaceCopyKernel[gid-columns];
   	float down = surfaceCopyKernel[gid+columns];
   	float left = surfaceCopyKernel[gid-1];
   	float right = surfaceCopyKernel[gid+1];

   	surfaceKernel[gid] = ( up + down + left + right ) / 4;


//	int IDX_Thread = threadIdx.x; 	//Identificación del hilo de la dimensión X dentro del bloque
//	int IDY_Thread = threadIdx.y; 	//Identificación del hilo de la dimensión Y dentro del bloque
//	int IDX_Block = blockIdx.x; 	//Identificación del bloque de la dimensión X dentro del grid
//	int IDY_Block = blockIdx.y; 	//Identificación del bloque de la dimensión Y dentro del grid
//	int shapeBlock_X = blockDim.x;	//Número de hilos por bloque en la dimensión X
//	int shapeBlock_Y = blockDim.y;	//Número de hilos por bloque en la dimensión Y
//
//	int tidX= IDX_Block * shapeBlock_X + IDX_Thread;   // Identificador del hilo general en su respectiva dimension
//	int tidY= IDY_Block * shapeBlock_Y + IDY_Thread;   // Identificador del hilo general en su respectiva dimension
//
//	if ((tidX < rows-1) && (tidX != 0) && (tidY < columns-1) && (tidY != 0)){
//		accessMat( surfaceKernel, tidX, tidY ) = ( 
//			accessMat( surfaceCopyKernel, tidX-1, tidY ) +
//			accessMat( surfaceCopyKernel, tidX+1, tidY ) +
//			accessMat( surfaceCopyKernel, tidX, tidY-1 ) +
//			accessMat( surfaceCopyKernel, tidX, tidY+1 ) ) / 4;
//
//	}
}

__global__ void GlobalReduction(float *surfaceKernel, float *surfaceCopyKernel, int size, int flag, float *global_residualKernelOut, int iter, int step) {
	// Memoria shared
	extern __shared__ float global_residualKernel[];

	// Desactivar hilos que excedan los límites del array de entrada
	int blockID = blockIdx.x + gridDim.x * blockIdx.y;
	int thread_local_id = threadIdx.x + threadIdx.y * blockDim.x;
	int threads_in_block = blockDim.x * blockDim.y;
	int gid = thread_local_id + blockID * threads_in_block;
   	if ( gid >= size ) return;

	int tid = thread_local_id;

	// Cargar dato en memoria shared
	if (flag == 0) {
	//	printf("%03d%02dglobal_residualKernel[tid:%d] = surfaceKernel[gid:%d] - surfaceCopyKernel[gid:%d] -> %f = %f - %f \n", iter, step, tid, gid, gid, (surfaceKernel[ gid ] - surfaceCopyKernel[ gid ]), surfaceKernel[ gid ], surfaceCopyKernel[ gid ]);
		global_residualKernel[ tid ] = surfaceKernel[ gid ] - surfaceCopyKernel[ gid ];

	} else {
		global_residualKernel[ tid ] = global_residualKernelOut[ gid ];
	}

	// Asegurarse que todos los warps del bloque han cargado los datos
	__syncthreads();

	// Generalización: El único bloque del último nivel puede tener menos datos para reducir
	int mysize = blockDim.x;
	if ((gridDim.x - 1) == blockID) {
		if ( size%mysize != 0 )
			mysize = size%mysize;
	}

	// Hacemos la reducción en memoria shared
	for (unsigned int s=mysize/2; s>0; s=s/2)
	{
		if (tid < s) {
			// Hacemos la reducción sacando el maximo
			if (global_residualKernel[tid] < global_residualKernel[tid+s]) {
				global_residualKernel[tid] = global_residualKernel[tid+s];
			}
		}
		__syncthreads();
	}

	// El hilo 0 de cada bloque escribe el resultado final de la reducción
	// en la memoria global del dispositivo pasada por parámetro (g_odata[])
	if (tid == 0){

		global_residualKernelOut[blockID] = global_residualKernel[tid];
		//printf("global_residualKernelOut[blockID: %d] = %f\n", blockID, global_residualKernelOut[blockID]);
	}

}

__global__ void SaveReduction(float *global_residualKernel, float *global_residualAuxKernel, int step, int iter) {
	
	printf("%03d%02d residual Pre-Save: %f\n", iter,step,global_residualAuxKernel[0]);
	printf("%03d%02d residual Pre-Reduction: %f\n", iter,step,global_residualKernel[0]);
	
	if (global_residualKernel[0] > global_residualAuxKernel[0]) {
		global_residualAuxKernel[0] = global_residualKernel[0];
	}

	printf("%03d%02d residual Reduction: %f\n", iter,step,global_residualKernel[0]);
	printf("%03d%02d residual Save: %f\n\n", iter,step,global_residualAuxKernel[0]);
}

__global__ void FocalActivation(FocalPoint *focalKernel, int *num_deactivatedKernel, int iterKernel, int num_focalKernel) {

	int blockID = blockIdx.x + gridDim.x * blockIdx.y;
	int tid = threadIdx.x + threadIdx.y * blockDim.x;
	int threads_in_block = blockDim.x * blockDim.y;
	int gid = tid + blockID * threads_in_block;
	
	if (gid >= num_focalKernel) {
		return;
	}

	num_deactivatedKernel[0] = 0;
	if ( focalKernel[gid].start == iterKernel ) {
		focalKernel[gid].active = 1;
	}
	// Count focal points already deactivated by a team
	if ( focalKernel[gid].active == 2 ) {
		atomicAdd(&num_deactivatedKernel[0],1);
	}
}

__global__ void TeamMovement(Team *teamsKernel, FocalPoint *focalKernel, int num_focalKernel, int num_teamsKernel) {

	int blockID = blockIdx.x + gridDim.x * blockIdx.y;
	int tid = threadIdx.x + threadIdx.y * blockDim.x;
	int threads_in_block = blockDim.x * blockDim.y;
	int gid = tid + blockID * threads_in_block;
	
	if (gid >= num_teamsKernel) {
		return;
	}

	float distance = FLT_MAX;
	int target = -1;
	for( int j=0; j<num_focalKernel; j++ ) {
		if ( focalKernel[j].active != 1 ) continue; // Skip non-active focalKernel points
		float dx = focalKernel[j].x - teamsKernel[gid].x;
		float dy = focalKernel[j].y - teamsKernel[gid].y;
		float local_distance = sqrtf( dx*dx + dy*dy );
		if ( local_distance < distance ) {
			distance = local_distance;
			target = j;
		}
	}
	/* 4.3.2. Annotate target for the next stage */
	teamsKernel[gid].target = target;

	int comp1 = focalKernel[target].x < teamsKernel[gid].x;
	int comp2 = focalKernel[target].x > teamsKernel[gid].x;
	int comp3 = focalKernel[target].y < teamsKernel[gid].y;
	int comp4 = focalKernel[target].y > teamsKernel[gid].y;

	/* 4.3.3. No active focalKernel point to choose, no movement */
	if ( target != -1 ) {
		/* 4.3.4. Move in the focalKernel point direction */
		switch (teamsKernel[gid].type) {
			case 1: 
				// Type 1: Can move in diagonal
				if ( comp1 ) teamsKernel[gid].x--;
				else if ( comp2 ) teamsKernel[gid].x++;
				if ( comp3 ) teamsKernel[gid].y--;
				else if ( comp4 ) teamsKernel[gid].y++;
				break;
			case 2:
				// Type 2: First in horizontal direction, then in vertical direction
				if ( comp3 ) teamsKernel[gid].y--;
				else if ( comp4 ) teamsKernel[gid].y++;
				else if ( comp1 ) teamsKernel[gid].x--;
				else if ( comp2 ) teamsKernel[gid].x++;
				break;
			default:
				// Type 3: First in vertical direction, then in horizontal direction
				if ( comp1 ) teamsKernel[gid].x--;
				else if ( comp2 ) teamsKernel[gid].x++;
				else if ( comp3 ) teamsKernel[gid].y--;
				else if ( comp4 ) teamsKernel[gid].y++;
				break;
		}
	}

	//if (gid==0) {
	//	printf("Numero de equipos : %d \n", num_teamsKernel);
	//}

}

__global__ void TeamAction(Team *teamsKernel, FocalPoint *focalKernel, int num_teamsKernel, int columns, int rows, unsigned int *actionKernel, int iter) {

	int blockID = blockIdx.x + gridDim.x * blockIdx.y;
	int tid = threadIdx.x + threadIdx.y * blockDim.x;
	int threads_in_block = blockDim.x * blockDim.y;
	int gid = tid + blockID * threads_in_block;

	if (gid < num_teamsKernel) {

		/* 4.4.1. Deactivate the target focal point when it is reached */
		int target = teamsKernel[gid].target;
		if ( target != -1 && focalKernel[target].x == teamsKernel[gid].x && focalKernel[target].y == teamsKernel[gid].y 
			&& focalKernel[target].active == 1 )
			focalKernel[target].active = 2;
		/* 4.4.2. Reduce heat in a circle around the team */
		int radius;
		// Influence area of fixed radius depending on type
		if ( teamsKernel[gid].type == 1 ) radius = RADIUS_TYPE_1;
		else radius = RADIUS_TYPE_2_3;
		for( int i=teamsKernel[gid].x-radius; i<=teamsKernel[gid].x+radius; i++ ) {
			for( int j=teamsKernel[gid].y-radius; j<=teamsKernel[gid].y+radius; j++ ) {
				if ( i<1 || i>=rows-1 || j<1 || j>=columns-1 ) continue; // Out of the heated surfaceKernel
				float dx = teamsKernel[gid].x - i;
				float dy = teamsKernel[gid].y - j;
				float distance = sqrtf( dx*dx + dy*dy );
				if ( (distance <= radius)) {
					atomicInc( (&(accessMat(actionKernel, i, j)) ), num_teamsKernel+1);
				}
			}
		}
	}

	__syncthreads();
}

__global__ void Cooldown(unsigned int *actionKernel, float *surfaceKernel, float *surfaceKernelOut, int rows, int columns, int iter) {

	int blockID = blockIdx.x + gridDim.x * blockIdx.y;
	int tid = threadIdx.x + threadIdx.y * blockDim.x;
	int threads_in_block = blockDim.x * blockDim.y;
	int gid = tid + blockID * threads_in_block;

	if (gid >= rows * columns) return;

	float aux;

	aux = surfaceKernel[gid];
	
	for (int i = 0; i < actionKernel[gid]; i++) {
		aux *= 0.75;
	}

	surfaceKernelOut[gid] = aux;

	atomicInc( (&(actionKernel[gid]) ), 0);

	__syncthreads();

}

/*
 * Function: Print usage line in stderr
 */
void show_usage( char *program_name ) {
	fprintf(stderr,"Usage: %s <config_file> | <command_line_args>\n", program_name );
	fprintf(stderr,"\t<config_file> ::= -f <file_name>\n");
	fprintf(stderr,"\t<command_line_args> ::= <rows> <columns> <maxIter> <numTeams> [ <teamX> <teamY> <teamType> ... ] <numFocalPoints> [ <focalX> <focalY> <focalStart> <focalTemperature> ... ]\n");
	fprintf(stderr,"\n");
}

#ifdef DEBUG
/* 
 * Function: Print the current state of the simulation 
 */
void print_status( int iteration, int rows, int columns, float *surface, int num_teams, Team *teams, int num_focal, FocalPoint *focal, float global_residual ) {
	/* 
	 * You don't need to optimize this function, it is only for pretty printing and debugging purposes.
	 * It is not compiled in the production versions of the program.
	 * Thus, it is never used when measuring times in the leaderboard
	 */
	int i,j;

	printf("Iteration: %d\n", iteration );
	printf("+");
	for( j=0; j<columns; j++ ) printf("---");
	printf("+\n");
	for( i=0; i<rows; i++ ) {
		printf("|");
		for( j=0; j<columns; j++ ) {
			char symbol;
			if ( accessMat( surface, i, j ) >= 1000 ) symbol = '*';
			else if ( accessMat( surface, i, j ) >= 100 ) symbol = '0' + (int)(accessMat( surface, i, j )/100);
			else if ( accessMat( surface, i, j ) >= 50 ) symbol = '+';
			else if ( accessMat( surface, i, j ) >= 25 ) symbol = '.';
			else symbol = '0';

			int t;
			int flag_team = 0;
			for( t=0; t<num_teams; t++ ) 
				if ( teams[t].x == i && teams[t].y == j ) { flag_team = 1; break; }
			if ( flag_team ) printf("[%c]", symbol );
			else {
				int f;
				int flag_focal = 0;
				for( f=0; f<num_focal; f++ ) 
					if ( focal[f].x == i && focal[f].y == j && focal[f].active == 1 ) { flag_focal = 1; break; }
				if ( flag_focal ) printf("(%c)", symbol );
				else printf(" %c ", symbol );
			}
		}
		printf("|\n");
	}
	printf("+");
	for( j=0; j<columns; j++ ) printf("---");
	printf("+\n");
	printf("Global residual: %f\n\n", global_residual);
}
#endif

/*
 * MAIN PROGRAM
 */
int main(int argc, char *argv[]) {
	int i,j,t;

	// Simulation data
	int rows, columns, max_iter;
	float *surface, *surfaceCopy;
	int num_teams, num_focal;
	Team *teams;
	FocalPoint *focal;

	/* 1. Read simulation arguments */
	/* 1.1. Check minimum number of arguments */
	if (argc<2) {
		fprintf(stderr,"-- Error in arguments: No arguments\n");
		show_usage( argv[0] );
		exit( EXIT_FAILURE );
	}

	int read_from_file = ! strcmp( argv[1], "-f" );
	/* 1.2. Read configuration from file */
	if ( read_from_file ) {
		/* 1.2.1. Open file */
		if (argc<3) {
			fprintf(stderr,"-- Error in arguments: file-name argument missing\n");
			show_usage( argv[0] );
			exit( EXIT_FAILURE );
		}
		FILE *args = cp_abrir_fichero( argv[2] );
		if ( args == NULL ) {
			fprintf(stderr,"-- Error in file: not found: %s\n", argv[1]);
			exit( EXIT_FAILURE );
		}	

		/* 1.2.2. Read surface and maximum number of iterations */
		int ok;
		ok = fscanf(args, "%d %d %d", &rows, &columns, &max_iter);
		if ( ok != 3 ) {
			fprintf(stderr,"-- Error in file: reading rows, columns, max_iter from file: %s\n", argv[1]);
			exit( EXIT_FAILURE );
		}

		surface = (float *)malloc( sizeof(float) * (size_t)rows * (size_t)columns );
		surfaceCopy = (float *)malloc( sizeof(float) * (size_t)rows * (size_t)columns );

		if ( surface == NULL || surfaceCopy == NULL ) {
			fprintf(stderr,"-- Error allocating: surface structures\n");
			exit( EXIT_FAILURE );
		}

		/* 1.2.3. Teams information */
		ok = fscanf(args, "%d", &num_teams );
		if ( ok != 1 ) {
			fprintf(stderr,"-- Error file, reading num_teams from file: %s\n", argv[1]);
			exit( EXIT_FAILURE );
		}
		teams = (Team *)malloc( sizeof(Team) * (size_t)num_teams );
		if ( teams == NULL ) {
			fprintf(stderr,"-- Error allocating: %d teams\n", num_teams );
			exit( EXIT_FAILURE );
		}
		for( i=0; i<num_teams; i++ ) {
			ok = fscanf(args, "%d %d %d", &teams[i].x, &teams[i].y, &teams[i].type);
			if ( ok != 3 ) {
				fprintf(stderr,"-- Error in file: reading team %d from file: %s\n", i, argv[1]);
				exit( EXIT_FAILURE );
			}
		}

		/* 1.2.4. Focal points information */
		ok = fscanf(args, "%d", &num_focal );
		if ( ok != 1 ) {
			fprintf(stderr,"-- Error in file: reading num_focal from file: %s\n", argv[1]);
			exit( EXIT_FAILURE );
		}
		focal = (FocalPoint *)malloc( sizeof(FocalPoint) * (size_t)num_focal );
		if ( focal == NULL ) {
			fprintf(stderr,"-- Error allocating: %d focal points\n", num_focal );
			exit( EXIT_FAILURE );
		}
		for( i=0; i<num_focal; i++ ) {
			ok = fscanf(args, "%d %d %d %d", &focal[i].x, &focal[i].y, &focal[i].start, &focal[i].heat);
			if ( ok != 4 ) {
				fprintf(stderr,"-- Error in file: reading focal point %d from file: %s\n", i, argv[1]);
				exit( EXIT_FAILURE );
			}
			focal[i].active = 0;
		}
	}
	/* 1.3. Read configuration from arguments */
	else {
		/* 1.3.1. Check minimum number of arguments */
		if (argc<6) {
			fprintf(stderr, "-- Error in arguments: not enough arguments when reading configuration from the command line\n");
			show_usage( argv[0] );
			exit( EXIT_FAILURE );
		}

		/* 1.3.2. Surface and maximum number of iterations */
		rows = atoi( argv[1] );
		columns = atoi( argv[2] );
		max_iter = atoi( argv[3] );

		surface = (float *)malloc( sizeof(float) * (size_t)rows * (size_t)columns );
		surfaceCopy = (float *)malloc( sizeof(float) * (size_t)rows * (size_t)columns );

		/* 1.3.3. Teams information */
		num_teams = atoi( argv[4] );
		teams = (Team *)malloc( sizeof(Team) * (size_t)num_teams );
		if ( teams == NULL ) {
			fprintf(stderr,"-- Error allocating: %d teams\n", num_teams );
			exit( EXIT_FAILURE );
		}
		if ( argc < num_teams*3 + 5 ) {
			fprintf(stderr,"-- Error in arguments: not enough arguments for %d teams\n", num_teams );
			exit( EXIT_FAILURE );
		}
		for( i=0; i<num_teams; i++ ) {
			teams[i].x = atoi( argv[5+i*3] );
			teams[i].y = atoi( argv[6+i*3] );
			teams[i].type = atoi( argv[7+i*3] );
		}

		/* 1.3.4. Focal points information */
		int focal_args = 5 + i*3;
		if ( argc < focal_args+1 ) {
			fprintf(stderr,"-- Error in arguments: not enough arguments for the number of focal points\n");
			show_usage( argv[0] );
			exit( EXIT_FAILURE );
		}
		num_focal = atoi( argv[focal_args] );
		focal = (FocalPoint *)malloc( sizeof(FocalPoint) * (size_t)num_focal );
		if ( teams == NULL ) {
			fprintf(stderr,"-- Error allocating: %d focal points\n", num_focal );
			exit( EXIT_FAILURE );
		}
		if ( argc < focal_args + 1 + num_focal*4 ) {
			fprintf(stderr,"-- Error in arguments: not enough arguments for %d focal points\n", num_focal );
			exit( EXIT_FAILURE );
		}
		for( i=0; i<num_focal; i++ ) {
			focal[i].x = atoi( argv[focal_args+i*4+1] );
			focal[i].y = atoi( argv[focal_args+i*4+2] );
			focal[i].start = atoi( argv[focal_args+i*4+3] );
			focal[i].heat = atoi( argv[focal_args+i*4+4] );
			focal[i].active = 0;
		}

		/* 1.3.5. Sanity check: No extra arguments at the end of line */
		if ( argc > focal_args+i*4+1 ) {
			fprintf(stderr,"-- Error in arguments: extra arguments at the end of the command line\n");
			show_usage( argv[0] );
			exit( EXIT_FAILURE );
		}
	}


#ifdef DEBUG
	/* 1.4. Print arguments */
	printf("Arguments, Rows: %d, Columns: %d, max_iter: %d\n", rows, columns, max_iter);
	printf("Arguments, Teams: %d, Focal points: %d\n", num_teams, num_focal );
	for( i=0; i<num_teams; i++ ) {
		printf("\tTeam %d, position (%d,%d), type: %d\n", i, teams[i].x, teams[i].y, teams[i].type );
	}
	for( i=0; i<num_focal; i++ ) {
		printf("\tFocal_point %d, position (%d,%d), start time: %d, temperature: %d\n", i, 
		focal[i].x,
		focal[i].y,
		focal[i].start,
		focal[i].heat );
	}
#endif // DEBUG

	/* 2. Select GPU and start global timer */
	cudaSetDevice(0);
	cudaDeviceSynchronize();
	double ttotal = cp_Wtime();

/*
 *
 * START HERE: DO NOT CHANGE THE CODE ABOVE THIS POINT
 *
 */

 	int bX; /**< Numero de columnas del bloque*/
	int bY; /**< Numero de filas del bloque*/
	int gX; /**< Numero de columnas del grid*/
	int gY; /**< Numero de filas del grid*/
// 	double time1;				/*-----*/
//	double t1;				/*-----*/
// 	double time2;				/*-----*/
// 	double t2;				/*-----*/
// 	double time2_1;				/*-----*/
// 	double t2_1;				/*-----*/
// 	double time2_2;				/*-----*/
// 	double t2_2;				/*-----*/
// 	double time2_3;				/*-----*/
// 	double t2_3;				/*-----*/
// 	double time2_4;				/*-----*/
// 	double t2_4;				/*-----*/
// 	double time3;				/*-----*/
// 	double t3;				/*-----*/
// 	double time4;				/*-----*/
// 	double t4;				/*-----*/

	if (rows<=X_DIV){
		bX=rows;	
		gX=1;	
	} else {
		bX=X_DIV;	
		gX=rows/X_DIV;
		if (rows%X_DIV != 0) gX++;	
 	}

 	if (columns <= Y_DIV) {
		bY=columns;	
		gY=1;	
 	} else {
		bY=Y_DIV;	
		gY=columns/Y_DIV;	
		if (columns%Y_DIV != 0) gY++;
	}

 	dim3 bloqSurfaceGpuFunc(bX,bY); 
	dim3 gridSurfaceGpuFunc(gX,gY);

	if (num_teams<=Y_DIV){
		bY=num_teams;
		bX=1;
		gX=1;
		gY=1;
	} else if (num_teams>MAX_BLQ_DIM) {
		bX=X_DIV;	
		bY=Y_DIV;
		gY=num_teams/MAX_BLQ_DIM;
		if (num_teams%Y_DIV != 0) gY++;
		gX=1;
 	} else {	
		bY=Y_DIV;
		bX=num_teams/Y_DIV;
		if (num_teams%Y_DIV != 0) bX++;
		gX=1;
		gY=1;
 	}

 	dim3 bloqTeamsGpuFunc(bX,bY); 
	dim3 gridTeamsGpuFunc(gX,gY);

	if (num_focal<=Y_DIV){
		bY=num_focal;
		bX=1;
		gX=1;
		gY=1;
	} else if (num_focal>MAX_BLQ_DIM) {
		bX=X_DIV;	
		bY=Y_DIV;
		gY=num_focal/MAX_BLQ_DIM;
		if (num_focal%Y_DIV != 0) gY++;
		gX=1;
 	} else {	
		bY=Y_DIV;
		bX=num_focal/Y_DIV;
		if (num_focal%Y_DIV != 0) bX++;
		gX=1;
		gY=1;
 	}

 	dim3 bloqFocalGpuFunc(bX,bY); 
	dim3 gridFocalGpuFunc(gX,gY);

	unsigned int sizeSurface = (size_t)rows*(size_t)columns * sizeof(float);
	unsigned int sizeFocal = (size_t)num_focal * sizeof(FocalPoint);
	unsigned int sizeTeams = (size_t)num_teams * sizeof(Team);

	float *surfaceKernel;
	float *surfaceCopyKernel;
	cudaMalloc((void**) &surfaceKernel, sizeSurface );
	cudaMalloc((void**) &surfaceCopyKernel, sizeSurface);

	FocalPoint *focalKernel;     
	cudaMalloc((void**) &focalKernel, sizeFocal);

	Team *teamsKernel;
	cudaMalloc((void**) &teamsKernel, sizeTeams);

	float * global_residualKernel;
	cudaMalloc((void**) &global_residualKernel, ((size_t)gX * (size_t)gY*sizeof(float)));

	float * global_residualAuxKernel;
	cudaMalloc((void**) &global_residualAuxKernel, (sizeof(float)));

	int * num_deactivatedKernel;
	cudaMalloc((void**) &num_deactivatedKernel, (sizeof(int)));

	unsigned int * actionKernel;
	cudaMalloc((void**) &actionKernel, ( (size_t)rows * (size_t)columns * sizeof(int) ));

//	float *pinned_surface, *pinned_surfaceCopy;
//	Team *pinned_teams;
//	FocalPoint *pinned_focal;

//	cudaMallocHost((void**)&pinned_surface, sizeSurface); // host pinned
//	cudaMallocHost((void**)&pinned_surfaceCopy, sizeSurface); // host pinned
//	cudaMallocHost((void**)&pinned_teams, sizeTeams); // host pinned
//	cudaMallocHost((void**)&pinned_focal, sizeFocal); // host pinned
//
//	memcpy(pinned_surface, surface, sizeSurface);
//	memcpy(pinned_surfaceCopy, surfaceCopy, sizeSurface);
//	memcpy(pinned_teams, teams, sizeTeams);
//	memcpy(pinned_focal, focal, sizeFocal);

	cudaMemcpy (focalKernel, focal, sizeFocal, cudaMemcpyHostToDevice);
	cudaMemcpy (surfaceKernel, surface, sizeSurface, cudaMemcpyHostToDevice);
	cudaMemcpy (teamsKernel, teams, sizeTeams, cudaMemcpyHostToDevice);

	/* 3. Initialize surface */

	Initialize<<<gridSurfaceGpuFunc,bloqSurfaceGpuFunc>>>(surfaceKernel,rows,columns);

	/* 4. Simulation */
	int iter;
	int flag_stability = 0;
	int num_deactivated = 0;
	for( iter=0; iter<max_iter && ! flag_stability; iter++ ) {

		/* 4.1. Activate focal points */

//		cudaDeviceSynchronize();				/*-----*/
//		time1 = cp_Wtime();				/*-----*/

//
//		cudaDeviceSynchronize();				/*-----*/
//		t1 += cp_Wtime() - time1;				/*-----*/
//		cudaMemcpy (focalKernel, focal, sizeFocal, cudaMemcpyHostToDevice);
		if (num_deactivated != num_focal) {
			FocalActivation<<<gridFocalGpuFunc, bloqFocalGpuFunc>>>(focalKernel, num_deactivatedKernel, iter, num_focal);

//		for( i=0; i<num_focal; i++ ) {
//			if ( focal[i].start == iter ) {
//				focal[i].active = 1;
//			}
//			// Count focal points already deactivated by a team
//			if ( focal[i].active == 2 ) num_deactivated++;
//		}

//		cudaMemcpy(focal, focalKernel, sizeFocal, cudaMemcpyDeviceToHost);
//		CUDA_CHECK();
			cudaMemcpy(&num_deactivated, &num_deactivatedKernel[0], sizeof(int), cudaMemcpyDeviceToHost);
		}

		/* 4.2. Propagate heat (10 steps per each team movement) */
		float global_residual = 0.0f;
		float global_residual_aux = 0.0f;
		int step;
//		if (first_activation) {

//			cudaDeviceSynchronize();				/*-----*/
//			time2 = cp_Wtime();				/*-----*/
//			cudaMemcpy (focalKernel, focal, sizeFocal, cudaMemcpyHostToDevice);
//			cudaMemcpy (surfaceKernel, surface, sizeSurface, cudaMemcpyHostToDevice);
			for( step=0; step<10; step++ )	{
				int flag = 0;
				/* 4.2.1. Update heat on active focal points */

//				cudaDeviceSynchronize();				/*-----*/
//				time2_1 = cp_Wtime();				/*-----*/

				UpdateFocal<<<gridFocalGpuFunc,bloqFocalGpuFunc>>>(focalKernel, num_focal, surfaceKernel, columns);
//				CUDA_CHECK();

//				cudaDeviceSynchronize();				/*-----*/
//				t2_1 += cp_Wtime() - time2_1;				/*-----*/

				/* 4.2.2. Copy values of the surface in ancillary structure (Skip borders) */

//				cudaDeviceSynchronize();				/*-----*/
//				time2_2 = cp_Wtime();				/*-----*/

				CopySurface<<<gridSurfaceGpuFunc, bloqSurfaceGpuFunc>>>(surfaceKernel, surfaceCopyKernel, rows, columns);
//				CUDA_CHECK();

//				cudaDeviceSynchronize();				/*-----*/
//				t2_2 += cp_Wtime() - time2_2;				/*-----*/

				/* 4.2.3. Update surface values (skip borders) */

//				cudaDeviceSynchronize();				/*-----*/
//				time2_3 = cp_Wtime();				/*-----*/

				SpreadHeat<<<gridSurfaceGpuFunc, bloqSurfaceGpuFunc>>>(surfaceKernel, surfaceCopyKernel, rows, columns);
//				CUDA_CHECK();
//				cudaDeviceSynchronize();				/*-----*/
//				t2_3 += cp_Wtime() - time2_3;				/*-----*/

				/* 4.2.4. Compute the maximum residual difference (absolute value) */

//				cudaDeviceSynchronize();				/*-----*/
//				time2_4 = cp_Wtime();				/*-----*/

				if (num_deactivated == num_focal) {

					int tmp_matrix_size = rows*columns;
					int sharedMemorySize = MAX_BLQ_DIM * sizeof(float);
					while(tmp_matrix_size>1){
						int baseNumBlocks = tmp_matrix_size/MAX_BLQ_DIM;

						int additionalBlock;
						if(baseNumBlocks != 0){
							additionalBlock = 0;
						} else{
							additionalBlock = 1;
						}
						int numBlocks = baseNumBlocks + additionalBlock;
						// Dentro de cada nivel, cada bloque realiza una reducción en árbol binario
						GlobalReduction<<< numBlocks, MAX_BLQ_DIM, sharedMemorySize >>>(surfaceKernel, surfaceCopyKernel, tmp_matrix_size, flag, global_residualKernel, iter, step);
//						CUDA_CHECK();
						// Actualizar tmp_matrix_size al numero de bloques de la iteracion anterior
						tmp_matrix_size = numBlocks;
						flag++;
					}


					cudaMemcpy (&global_residual_aux, &global_residualKernel[0], sizeof(float), cudaMemcpyDeviceToHost);
//					SaveReduction<<<1,1>>>(global_residualKernel, global_residualAuxKernel, step, iter);
//
//					CUDA_CHECK();


					if (global_residual < global_residual_aux) {
						global_residual = global_residual_aux;
					}
	
				}


//				cudaDeviceSynchronize();				/*-----*/
//				t2_4 += cp_Wtime() - time2_4;				/*-----*/
			}

//			cudaDeviceSynchronize();				/*-----*/
//			t2 += cp_Wtime() - time2;				/*-----*/

//		}

//		if (num_deactivated == num_focal) {
//
//			GetMax<<<1,1>>>(global_residualAuxKernel, global_residualAuxKernel,iter,step);
//			cudaMemcpy (&global_residual, &global_residualAuxKernel[0], sizeof(float), cudaMemcpyDeviceToHost);	

//			if (iter == 916) {
//				printf("global = %f\n", global_residual);
//			}	
//		}

		/* If the global residual is lower than THRESHOLD, we have reached enough stability, stop simulation at the end of this iteration */

		if( num_deactivated == num_focal && global_residual < THRESHOLD ) flag_stability = 1;

		if (num_deactivated != num_focal) {


//		if (first_activation) {

			/* 4.3. Move teams */
//			cudaDeviceSynchronize();				/*-----*/
//			time3 = cp_Wtime();				/*-----*/

//			cudaMemcpy (teamsKernel, teams, sizeTeams, cudaMemcpyHostToDevice);

			TeamMovement<<<gridTeamsGpuFunc,bloqTeamsGpuFunc>>>(teamsKernel,focalKernel,num_focal,num_teams);
//			CUDA_CHECK();

//			cudaMemcpy (teams, teamsKernel, sizeTeams, cudaMemcpyDeviceToHost);
			/* 4.4. Team actions */

//			TeamAction<<<gridSurfaceGpuFunc, bloqSurfaceGpuFunc, sharedMemorySize>>>(teamsKernel, focalKernel, num_teams, columns, rows, surfaceKernel, surfaceKernel, num_teams);


		}
		
		TeamAction<<<gridTeamsGpuFunc, bloqTeamsGpuFunc>>>(teamsKernel, focalKernel, num_teams, columns, rows, actionKernel, iter);
		
		Cooldown<<<gridSurfaceGpuFunc, bloqSurfaceGpuFunc>>>(actionKernel, surfaceKernel, surfaceKernel, rows, columns, iter);

//			cudaDeviceSynchronize();				/*-----*/
//			t4 += cp_Wtime() - time4;				/*-----*/
//		}


#ifdef DEBUG
		/* 4.5. DEBUG: Print the current state of the simulation at the end of each iteration */
		print_status( iter, rows, columns, surface, num_teams, teams, num_focal, focal, global_residual );
#endif // DEBUG

	}


	cudaMemcpy (surface , surfaceKernel, sizeSurface, cudaMemcpyDeviceToHost);

	cudaFree(surfaceKernel);
	cudaFree(surfaceCopyKernel);
	cudaFree(teamsKernel);
	cudaFree(focalKernel);
	cudaFree(global_residualKernel);
	cudaFree(num_deactivatedKernel);
	cudaFree(global_residualAuxKernel);
	cudaFree(actionKernel);
//	cudaFreeHost(pinned_surface);
// 	cudaFreeHost(pinned_surfaceCopy);
// 	cudaFreeHost(pinned_teams);
// 	cudaFreeHost(pinned_focal);
	
/*
 *
 * STOP HERE: DO NOT CHANGE THE CODE BELOW THIS POINT
 *
 */

	/* 5. Stop global time */
	cudaDeviceSynchronize();
	ttotal = cp_Wtime() - ttotal;

	/* 6. Output for leaderboard */
	printf("\n");
	/* 6.1. Total computation time */
	printf("Time: %lf\n", ttotal);
//	printf("Time 1: %lf\n", t1);				/*-----*/
//	printf("Time 2: %lf\n", t2);				/*-----*/
//	printf("Time 2_1: %lf\n", t2_1);				/*-----*/
//	printf("Time 2_2: %lf\n", t2_2);				/*-----*/
//	printf("Time 2_3: %lf\n", t2_3);				/*-----*/
//	printf("Time 2_4: %lf\n", t2_4);				/*-----*/
//	printf("Time 3: %lf\n", t3);				/*-----*/
//	printf("Time 4: %lf\n", t4);				/*-----*/
	/* 6.2. Results: Number of iterations, position of teams, residual heat on the focal points */
	printf("Result: %d", iter);
	/*
	for (i=0; i<num_teams; i++)
		printf(" %d %d", teams[i].x, teams[i].y );
	*/
	for (i=0; i<num_focal; i++)
		printf(" %.6f", accessMat( surface, focal[i].x, focal[i].y ) );
	printf("\n");

	/* 7. Free resources */	
	free( teams );
	free( focal );
	free( surface );
	free( surfaceCopy );

	/* 8. End */
	return 0;
}
