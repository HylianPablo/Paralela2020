// User: g110@88.10.254.19 
// ExecutionRequest[P:'extinguishing.cu',P:1,T:1,args:'',q:'cudalb'] 
// May 16 2019 19:56:59
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
 * Funciones kernel
 */
 /* Punto 3 */
__global__ void initialize_surface_and_surfaceCopy_kernel(float *surface, float *surfaceCopy, int columns) {
	int gid = (gridDim.x * blockIdx.y + blockIdx.x) * (blockDim.x * blockDim.y) + (blockDim.x * threadIdx.y + threadIdx.x);
	int g_row = blockIdx.y;
	int g_column = blockIdx.x * blockDim.x + threadIdx.x;
	if (g_column >= columns) return;
	accessMat(surface, g_row, g_column) = 0.0f;
	accessMat(surfaceCopy, g_row, g_column) = 0.0f;
}

/* Punto 4.2.1 */
__global__ void update_heat_on_active_focal_kernel(FocalPoint *focal, float *surface, int num_focal, int columns) {
	int gid = (gridDim.x * blockIdx.y + blockIdx.x) * (blockDim.x * blockDim.y) + (blockDim.x * threadIdx.y + threadIdx.x);
	int g_row = blockIdx.y;
	int g_column = blockIdx.x * blockDim.x + threadIdx.x;
	if (g_column >= columns) return;

	for (int i = 0; i<num_focal; i++) {
		if (focal[i].active == 1) {
			if (focal[i].x == g_row && focal[i].y == g_column) {
				accessMat(surface, g_row, g_column) = focal[i].heat;
			}
		}
	}
}

__global__ void update_heat_on_active_focal_kernel_v2(FocalPoint *focal, float *surface, int num_focal, int columns) {
	int gid = blockDim.x * blockIdx.x + threadIdx.x;
	if (gid >= num_focal) return;

	if (focal[gid].active == 1) {
		accessMat(surface, focal[gid].x, focal[gid].y) = focal[gid].heat;
	}
}

/* Punto 4.2.2 */
__global__ void copy_surface_on_surfaceCopy_kernel(float *surface, float *surfaceCopy, int rows, int columns) {
	int gid = (gridDim.x * blockIdx.y + blockIdx.x) * (blockDim.x * blockDim.y) + (blockDim.x * threadIdx.y + threadIdx.x);
	int g_row = blockIdx.y;
	int g_column = blockIdx.x * blockDim.x + threadIdx.x;
	if (g_column >= columns) return;


	if (g_row == 0 || g_row == rows-1 || g_column == 0 || g_column == columns-1) return; //skip borders
	accessMat(surfaceCopy, g_row, g_column) = accessMat(surface, g_row, g_column);	
}

/* Punto 4.2.3 */
__global__ void update_surface_values_kernel(float *surface, float *surfaceCopy, int rows, int columns) {
	int gid = (gridDim.x * blockIdx.y + blockIdx.x) * (blockDim.x * blockDim.y) + (blockDim.x * threadIdx.y + threadIdx.x);
	int g_row = blockIdx.y;
	int g_column = blockIdx.x * blockDim.x + threadIdx.x;
	if (g_column >= columns) return;


	if (g_row == 0 || g_row == rows-1 || g_column == 0 || g_column == columns-1) return; //skip borders
	accessMat(surface, g_row, g_column) = (
		accessMat(surfaceCopy, g_row-1, g_column) +
		accessMat(surfaceCopy, g_row+1, g_column) +
		accessMat(surfaceCopy, g_row, g_column-1) +
		accessMat(surfaceCopy, g_row, g_column+1) ) / 4;
}

/* Punto 4.2.4 */
__global__ void calcula_global_residual(float *surface, float *surfaceCopy, float *globales, int size) {
	int gid = threadIdx.x + blockIdx.x * blockDim.x;
	if (gid >= size) return;
	globales[gid] = fabs(surface[gid] - surfaceCopy[gid]);
}

__global__ void reduce_global_residual(float *globales_out, float *globales_in, int size) {
	int gid = threadIdx.x + blockDim.x * blockIdx.x;
	if (gid > (size/2)) return;

	if ( globales_in[ gid + (size/2) ] > globales_in[ gid ] ) 
		globales_out[ gid ] = globales_in[ gid + (size/2) ];

	if ( size%2 != 0 && gid == 0 )
		if ( globales_in[ size - 1 ] > globales_in[ gid ] ) 
			globales_out[ gid ] = globales_in[ size - 1 ];
}

/* Punto 4.4 */
__device__ float atomicMul(float *address, float val) {
	int *address_as_ull = (int*)address; 
	int old = *address_as_ull, assumed; 
	do { 
			assumed = old; 
		old = atomicCAS(address_as_ull, assumed, __float_as_int(val * __int_as_float(assumed))); 
	} while (assumed != old); 
	return __int_as_float(old);
}

__global__ void team_actions_v2(float *surface, FocalPoint *focal, Team *teams, int num_teams, int rows, int columns) {
	int gid = blockIdx.x * blockDim.x + threadIdx.x;

	if (gid >= num_teams) return;
	int my_target = teams[gid].target;
	if (my_target != -1 && focal[my_target].x == teams[gid].x && focal[my_target].y == teams[gid].y && focal[my_target].active == 1) {
		focal[my_target].active = 2;
	}

	int radius;
	if (teams[gid].type == 1) radius = RADIUS_TYPE_1;
	else radius = RADIUS_TYPE_2_3;

	for (int my_i = teams[gid].x-radius; my_i <= teams[gid].x+radius; my_i++) {
		for(int my_j = teams[gid].y-radius; my_j <= teams[gid].y+radius; my_j++) {
			if (my_i < 1 || my_i >= rows-1 || my_j < 1 || my_j >= columns-1) continue;
			float dx = teams[gid].x - my_i;
			float dy = teams[gid].y - my_j;
			float distance = sqrtf( dx*dx + dy*dy );
			if ( distance <= radius ) {
				float team_efficiency = 0.75f;
				atomicMul(&accessMat(surface, my_i, my_j), team_efficiency);
				//accessMat( surface, i, j ) = accessMat( surface, i, j ) * ( 1 - 0.25 ); // Team efficiency factor
			}
		}
	}
}

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

	/* Declaracion de tamaños */
	int threads_per_block = 1024;
	int sharedMemorySize = threads_per_block * sizeof(float);
	int fixed_columns = columns;
   /* Para el padding
	int nearestUpperPow2 = pow(2, ceil(log2((double) columns) ));
	fixed_columns = nearestUpperPow2;
   */
	int blocks_grid_x = fixed_columns / threads_per_block;
	if (blocks_grid_x * threads_per_block < fixed_columns)
		blocks_grid_x++; // para el padding
	int blocks_grid_y = rows;
	dim3 grid_dim(blocks_grid_x, blocks_grid_y); // un grid con blocks_grid_y filas y blocks_grid_x columnas
	dim3 blocks_dim(threads_per_block); // bloques de 1 fila y threads_per_block columnas

	float *d_surface;
	unsigned int surface_size = (size_t)rows * (size_t)columns * sizeof(float);
	int surface_boxes = rows * columns;
	cudaMalloc((void**)&d_surface, surface_size);

	float *d_surfaceCopy;
	cudaMalloc((void**)&d_surfaceCopy, surface_size);

	float *d_surface_aux_copy;

	FocalPoint *d_focal;
	unsigned int focal_size = (size_t)num_focal * sizeof(FocalPoint);
	cudaMalloc((void**)&d_focal, focal_size);

	float *d_globales;
	cudaMalloc((void**)&d_globales, surface_size);

	Team *d_teams;
	unsigned int teams_size = (size_t)num_teams * sizeof(Team);
	cudaMalloc((void**)&d_teams, teams_size);


	/* 3. Initialize surface */
	initialize_surface_and_surfaceCopy_kernel<<<grid_dim, blocks_dim>>>(d_surface, d_surfaceCopy, columns);
	
	/* 4. Simulation */
	int iter;
	int flag_stability = 0;
	int first_activation = 0;
	for( iter=0; iter<max_iter && ! flag_stability; iter++ ) {

		/* 4.1. Activate focal points */
		int num_deactivated = 0;
		for( i=0; i<num_focal; i++ ) {
			if ( focal[i].start == iter ) {
				focal[i].active = 1;
				if ( ! first_activation ) first_activation = 1;
			}
			// Count focal points already deactivated by a team
			if ( focal[i].active == 2 ) num_deactivated++;
		}

		/* 4.2. Propagate heat (10 steps per each team movement) */
		float global_residual = 0.0f;
		int step;
		cudaMemcpy(d_focal, focal, focal_size, cudaMemcpyHostToDevice);
		//cudaMemcpy(d_surface, surface, surface_size, cudaMemcpyHostToDevice); // -- Creo que sobra
		//cudaMemcpy(d_surfaceCopy, surfaceCopy, surface_size, cudaMemcpyHostToDevice); // -- Creo que sobra
		for( step=0; step<10; step++ )	{
			/* 4.2.1. Update heat on active focal points */
			int num_blocks_focal = num_focal / threads_per_block;
			if (num_blocks_focal * threads_per_block < num_focal) num_blocks_focal++;
			update_heat_on_active_focal_kernel_v2<<<num_blocks_focal, threads_per_block>>>(d_focal, d_surface, num_focal, columns);
			//update_heat_on_active_focal_kernel<<<grid_dim, blocks_dim>>>(d_focal, d_surface, num_focal, columns);

			//cudaDeviceSynchronize(); // -- Quizas haga falta usarlo

			/* 4.2.2. Copy values of the surface in ancillary structure (Skip borders) */
			//copy_surface_on_surfaceCopy_kernel<<<grid_dim, blocks_dim>>>(d_surface, d_surfaceCopy, rows, columns);
			d_surface_aux_copy = d_surface;
			d_surface = d_surfaceCopy;
			d_surfaceCopy = d_surface_aux_copy;

			//cudaDeviceSynchronize(); // -- Quizas haga falta usarlo
			
			/* 4.2.3. Update surface values (skip borders) */
			update_surface_values_kernel<<<grid_dim, blocks_dim>>>(d_surface, d_surfaceCopy, rows, columns);

			/* 4.2.4. Compute the maximum residual difference (absolute value) */
			if (step == 0) {
				int numBlocks = ((rows * columns) / threads_per_block) + 1;
				calcula_global_residual<<<numBlocks, threads_per_block>>>(d_surface, d_surfaceCopy, d_globales, (rows*columns));
				int redSize;
				for (redSize = (rows*columns); redSize > 1; redSize /= 2) {
					numBlocks = (redSize / 2) / threads_per_block;
					if ((redSize / 2) % threads_per_block != 0) numBlocks++;

					reduce_global_residual<<<numBlocks, threads_per_block>>>(d_globales, d_globales, redSize);
					//cudaDeviceSynchronize();
				}
				cudaMemcpy(&global_residual, d_globales, sizeof(float), cudaMemcpyDeviceToHost);
			}
		}
		//cudaMemcpy(surface, d_surface, surface_size, cudaMemcpyDeviceToHost);
		//cudaMemcpy(surfaceCopy, d_surfaceCopy, surface_size, cudaMemcpyDeviceToHost); // -- Creo que sobra
		/* If the global residual is lower than THRESHOLD, we have reached enough stability, stop simulation at the end of this iteration */
		if( num_deactivated == num_focal && global_residual < THRESHOLD ) flag_stability = 1;

		int num_blocks_teams = num_teams / threads_per_block;
		if (num_blocks_teams * threads_per_block < num_teams) num_blocks_teams++;

		/* 4.3. Move teams */
		for( t=0; t<num_teams; t++ ) {
			// 4.3.1. Choose nearest focal point
			float distance = FLT_MAX;
			int target = -1;
			for( j=0; j<num_focal; j++ ) {
				if ( focal[j].active != 1 ) continue; // Skip non-active focal points
				float dx = focal[j].x - teams[t].x;
				float dy = focal[j].y - teams[t].y;
				float local_distance = sqrtf( dx*dx + dy*dy );
				if ( local_distance < distance ) {
					distance = local_distance;
					target = j;
				}
			}
			// 4.3.2. Annotate target for the next stage
			teams[t].target = target;

			// 4.3.3. No active focal point to choose, no movement
			if ( target == -1 ) continue; 

			// 4.3.4. Move in the focal point direction
			if ( teams[t].type == 1 ) { 
				// Type 1: Can move in diagonal
				if ( focal[target].x < teams[t].x ) teams[t].x--;
				if ( focal[target].x > teams[t].x ) teams[t].x++;
				if ( focal[target].y < teams[t].y ) teams[t].y--;
				if ( focal[target].y > teams[t].y ) teams[t].y++;
			}
			else if ( teams[t].type == 2 ) { 
				// Type 2: First in horizontal direction, then in vertical direction
				if ( focal[target].y < teams[t].y ) teams[t].y--;
				else if ( focal[target].y > teams[t].y ) teams[t].y++;
				else if ( focal[target].x < teams[t].x ) teams[t].x--;
				else if ( focal[target].x > teams[t].x ) teams[t].x++;
			}
			else {
				// Type 3: First in vertical direction, then in horizontal direction
				if ( focal[target].x < teams[t].x ) teams[t].x--;
				else if ( focal[target].x > teams[t].x ) teams[t].x++;
				else if ( focal[target].y < teams[t].y ) teams[t].y--;
				else if ( focal[target].y > teams[t].y ) teams[t].y++;
			}
		}
		
		/* 4.4. Team actions */
		cudaMemcpy(d_teams, teams, teams_size, cudaMemcpyHostToDevice);
		team_actions_v2<<<num_blocks_teams, threads_per_block>>>(d_surface, d_focal, d_teams, num_teams, rows, columns);
		cudaMemcpy(focal, d_focal, focal_size, cudaMemcpyDeviceToHost);

#ifdef DEBUG
		/* 4.5. DEBUG: Print the current state of the simulation at the end of each iteration */
		print_status( iter, rows, columns, surface, num_teams, teams, num_focal, focal, global_residual );
#endif // DEBUG
	}
	cudaMemcpy(surface, d_surface, surface_size, cudaMemcpyDeviceToHost);
	cudaFree(d_focal);
	cudaFree(d_surface);
	cudaFree(d_surfaceCopy);
	cudaFree(d_teams);
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
	printf("Time: %lf\n", ttotal );
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
