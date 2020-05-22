// User: g212@157.88.139.133 
// ExecutionRequest[P:'la_venda_ya_cayo.cu',P:1,T:1,args:'',q:'cudalb'] 
// May 16 2019 19:47:16
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

__global__ void gpuFunc_UpdateFocalHeat(float *surface, FocalPoint *focal, int columns) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if ( focal[tid].active != 1 ) return;
	int x = focal[tid].x;
	int y = focal[tid].y;
	accessMat( surface, x, y ) = focal[tid].heat;
}

__global__ void gpuFunc_UpdateSurfaceHeat(float *surfaceCopy, float *surface, int rows, int columns)
{
	//int blockSize = blockDim.x * blockDim.y;
    //int threadblock = (threadIdx.y * blockDim.x + threadIdx.x);
	//int position = (blockIdx.y * gridDim.x * blockSize + blockIdx.x * blockSize + threadblock);

	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < 1 || j < 1 || i > rows-2 || j > columns-2)
		return;

	accessMat( surface, i, j ) = (
		accessMat( surfaceCopy, i-1, j ) +
		accessMat( surfaceCopy, i+1, j ) +
		accessMat( surfaceCopy, i, j-1 ) +
		accessMat( surfaceCopy, i, j+1 ) ) / 4;
}

__global__ void gpuFunc_CalculateDiff(float *surfaceCopy, float *surface, float* global_residual_red, int rows, int columns)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < 1 || j < 1 || i > rows-2 || j > columns-2)
		return;

    global_residual_red[(i-1) * (columns - 2) + j-1] = fabs( accessMat( surface, i, j ) - accessMat( surfaceCopy, i, j ) );
}

__global__ void kReduce( float *global_residual, int size ) {
    // Calcular id global
    //int i = blockIdx.y * blockDim.y + threadIdx.y;
	//int j = blockIdx.x * blockDim.x + threadIdx.x;

    //int tid = i * columns + j;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    //if (size == 15937 && blockIdx.x < 2)
    //    printf("%d-%d\n", tid, tid + size / 2);

    if ( tid >= size / 2 )
        return;

    // Reducir elemento en tid con su pareja
    global_residual[tid] = max(global_residual[tid], global_residual[tid + (size / 2)]);

    // Si size es impar, alguien reduce el elemento extra
    if ( size % 2 != 0 && tid == size  - 1)
        global_residual[tid] = max(global_residual[tid], global_residual[tid + (size / 2) + 1]);
}

__global__ void reduce_kernel_global(const float *g_idata, float *g_odata, int size)
{
	// Cálculo del identificador global del hilo
	int gid = threadIdx.x + blockIdx.x * blockDim.x;

	// Condición para evitar hilos ociosos por no tener pares asignados
	if ( gid >= size / 2 ) return;

	// Reducción del elemento en la posición gid con su pareja
	g_odata[ gid ] = max(g_idata[ gid ], g_idata[ gid + size/2 ]);
    if ( size % 2 != 0 && gid == 0 ) // ¿Quién se encarga de hacer la reducción?
       g_odata[ gid ] = max(g_idata[ gid ], g_idata[ size - 1 ]); // ¿Dónde está el elemento desparejado?
}

__global__ void reduce_kernel(const float* g_idata, float* g_odata, int size){
    extern __shared__ float tmp[];
    int gid = threadIdx.x + blockIdx.x * blockDim.x;
    if ( gid >= size )
        return;

    int tid = threadIdx.x;
    tmp[ tid ] = g_idata[ gid ];
    __syncthreads();
    // Ojo al último bloque de cualquier nivel
    int mysize = blockDim.x;
    if ( (blockIdx.x == gridDim.x-1) && ( (blockDim.x * gridDim.x - size) > 0))  {
        mysize = blockDim.x-(blockDim.x*gridDim.x-size);
    }
    // Reducir datos del bloque en memoria compartida
    for(int redSize=mysize; redSize>1; redSize/=2)  {
        /* reducción por parejas en memoria compartida */
        tmp[tid] = max(tmp[tid], tmp[tid + redSize/2]);
        // redSize impar en algún nivel...
        if ( (redSize % 2 != 0) && tid == 0 )
            tmp[0] = max(tmp[0], tmp[redSize-1]); // ¡elemento suelto!
        __syncthreads();
    }
    // Un hilo del bloque escribe en memoria global
    if ( tid == 0 )
        g_odata[ blockIdx.x ] = tmp[ tid ];
}

__global__ void reduceKernel(float *input, float *output) {
    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    int tid = threadIdx.x;

    for (unsigned int s = blockDim.x / 2; s > 0; s>>=1) {
        if (tid < s)
            input[myId] = max(input[myId], input[myId+s]);
        __syncthreads();
    }

    if (tid == 0)
        output[blockIdx.x] = input[myId];
}

__global__ void max_reduce(const float *d_array, float *d_max, const size_t elements)
{
    extern __shared__ float shared[];

    int tid = threadIdx.x;
    int gid = (blockDim.x * blockIdx.x) + tid;
    shared[tid] = -FLT_MAX;  // 1

    if (gid < elements)
        shared[tid] = d_array[gid];
    __syncthreads();

    for (unsigned int s=blockDim.x/2; s>0; s>>=1)
    {
        if (tid < s && gid < elements)
            shared[tid] = max(shared[tid], shared[tid + s]);  // 2
        __syncthreads();
    }
    // what to do now?
    // option 1: save block result and launch another kernel
    if (tid == 0)
        d_max[blockIdx.x] = shared[tid]; // 3
}

__global__ void gpuFunc_MoveTeams(Team *teams, FocalPoint *focal, int num_teams, int num_focal)
{
    int j, t;

    t = blockIdx.x * blockDim.x + threadIdx.x;

    if (t >= num_teams)
        return;

    /* 4.3.1. Choose nearest focal point */
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
    /* 4.3.2. Annotate target for the next stage */
    teams[t].target = target;

    /* 4.3.3. No active focal point to choose, no movement */
    if ( target == -1 ) return;

    /* 4.3.4. Move in the focal point direction */
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

	float *aux, *d_surface, *d_surfaceCopy, *d_global_residual_red;
    Team *d_teams;
    FocalPoint *d_focal;

    int surfaceSize = sizeof(float) * rows * columns;
    int surfaceNoBorderSize = sizeof(float) * (size_t)(rows - 2) * (size_t)(columns - 2);

    //global_residual_red = (float *)  malloc(surfaceNoBorderSize);

	cudaMalloc( (void**) &d_surface, surfaceSize );
	cudaMalloc( (void**) &d_surfaceCopy, surfaceSize );

    cudaMalloc( (void**) &d_global_residual_red, surfaceNoBorderSize );

    cudaMalloc( (void**) &d_teams, sizeof(Team) * num_teams );
    cudaMalloc( (void**) &d_focal, sizeof(FocalPoint) * num_focal );

    cudaDeviceProp dp;
    cudaGetDeviceProperties(&dp, 0);

    int threads = dp.maxThreadsPerMultiProcessor;
    int blocks = 16;

    int threads_per_block = threads / blocks;
    int candidate;
    bool c_found;
    do {
        blocks /= 2;
        candidate = threads / blocks;
        c_found = candidate > dp.maxThreadsPerBlock;
        if (!c_found)
            threads_per_block = candidate;
    } while(!c_found);

    int block_y = 64;
    int block_x = threads_per_block / block_y;

    int grid_x = columns / block_x;
    int grid_y = rows / block_y;

    if (columns % block_x != 0)
        grid_x++;
    if (rows % block_y != 0)
        grid_y++;

    dim3 gridShape_UpdateSurfaceHeat(grid_y, grid_x);
    dim3 blockShape_UpdateSurfaceHeat(block_y, block_x);

    int team_grid_x = num_teams / threads_per_block;
    if (num_teams % threads_per_block != 0)
        team_grid_x++;

    dim3 gridShape_MoveTeams(1, team_grid_x);
    dim3 blockShape_MoveTeams(1, threads_per_block);


    int nBlocks_HeatFocal = ceil((double)num_focal / threads_per_block);

	/* 3. Initialize surface */
    //memset(surface, 0, surfaceSize);
    //memset(surfaceCopy, 0, surfaceSize);
	for( i=0; i<rows; i++ )
		for( j=0; j<columns; j++ )
			accessMat( surface, i, j ) = 0.0;
            accessMat( surfaceCopy, i, j ) = 0.0;

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

		// Uso first activation
		if (first_activation == 0) continue;

        cudaMemcpy(d_surface, surface, sizeof(float) * rows * columns, cudaMemcpyHostToDevice);
        cudaMemcpy(d_surfaceCopy, surfaceCopy, sizeof(float) * rows * columns, cudaMemcpyHostToDevice);
        cudaMemcpy(d_focal, focal, sizeof(FocalPoint) * num_focal, cudaMemcpyHostToDevice);

		/* 4.2. Propagate heat (10 steps per each team movement) */
		float global_residual = 0.0f;
		int step;
		for( step=0; step<10; step++ )	{
			/* 4.2.1. Update heat on active focal points */
            gpuFunc_UpdateFocalHeat <<< nBlocks_HeatFocal, threads_per_block >>> (d_surface, d_focal, columns);

			/* 4.2.2. Copy values of the surface in ancillary structure (Skip borders) */
			aux = d_surface;
			d_surface = d_surfaceCopy;
			d_surfaceCopy = aux;

			/* 4.2.3. Update surface values (skip borders) */
			gpuFunc_UpdateSurfaceHeat<<<gridShape_UpdateSurfaceHeat, blockShape_UpdateSurfaceHeat>>>(d_surfaceCopy, d_surface, rows, columns);

			/* 4.2.4. Compute the maximum residual difference (absolute value) */
            if (step != 0) continue;

            // Diferencia
            gpuFunc_CalculateDiff <<< gridShape_UpdateSurfaceHeat, blockShape_UpdateSurfaceHeat >>> (d_surfaceCopy, d_surface, d_global_residual_red, rows, columns);

            // Arbol de reduccion
            int redSize;
            for (redSize = (rows - 2) * (columns - 2); redSize>1; redSize /= 2) {
                //int threads = threads_per_block;

			    int numBlocks = ceil((double)redSize/threads_per_block);
			    // Reducción por niveles en la GPU
			    reduce_kernel_global<<< numBlocks, threads_per_block >>>( d_global_residual_red, d_global_residual_red, redSize );
		    }
		}

        cudaMemcpy(focal, d_focal, sizeof(FocalPoint) * num_focal, cudaMemcpyDeviceToHost);
        cudaMemcpy(surface, d_surface, sizeof(float) * rows * columns, cudaMemcpyDeviceToHost);
        cudaMemcpy(surfaceCopy, d_surfaceCopy, sizeof(float) * rows * columns, cudaMemcpyDeviceToHost);
        cudaMemcpy(&global_residual, d_global_residual_red, sizeof(float), cudaMemcpyDeviceToHost);

		/* If the global residual is lower than THRESHOLD, we have reached enough stability, stop simulation at the end of this iteration */
		if( num_deactivated == num_focal && global_residual < THRESHOLD ) flag_stability = 1;

		/* 4.3. Move teams */
		if( num_deactivated < num_focal) // si no hay focos activos, no hace falta calcular el movimiento
        for( t=0; t<num_teams; t++ ) {
			/* 4.3.1. Choose nearest focal point */
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
			/* 4.3.2. Annotate target for the next stage */
			teams[t].target = target;

			/* 4.3.3. No active focal point to choose, no movement */
			if ( target == -1 ) continue;

			/* 4.3.4. Move in the focal point direction */
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
		for( t=0; t<num_teams; t++ ) {
			/* 4.4.1. Deactivate the target focal point when it is reached */
			int target = teams[t].target;
			if ( target != -1 && focal[target].x == teams[t].x && focal[target].y == teams[t].y
				&& focal[target].active == 1 )
				focal[target].active = 2;

			/* 4.4.2. Reduce heat in a circle around the team */
			int radius;
			// Influence area of fixed radius depending on type
			if ( teams[t].type == 1 ) radius = RADIUS_TYPE_1;
			else radius = RADIUS_TYPE_2_3;
			for( i=teams[t].x-radius; i<=teams[t].x+radius; i++ ) {
				for( j=teams[t].y-radius; j<=teams[t].y+radius; j++ ) {
					if ( i<1 || i>=rows-1 || j<1 || j>=columns-1 ) continue; // Out of the heated surface
					float dx = teams[t].x - i;
					float dy = teams[t].y - j;
					float distance = sqrtf( dx*dx + dy*dy );
					if ( distance <= radius ) {
						accessMat( surface, i, j ) = accessMat( surface, i, j ) * ( 1 - 0.25 ); // Team efficiency factor
					}
				}
			}
		}

#ifdef DEBUG
		/* 4.5. DEBUG: Print the current state of the simulation at the end of each iteration */
		print_status( iter, rows, columns, surface, num_teams, teams, num_focal, focal, global_residual );
#endif // DEBUG
	}

	cudaFree(d_surface);
	cudaFree(d_surfaceCopy);

    cudaFree(d_global_residual_red);

    cudaFree(d_teams);
    cudaFree(d_focal);

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
