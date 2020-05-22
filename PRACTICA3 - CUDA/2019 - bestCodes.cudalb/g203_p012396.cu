// User: g203@37.11.243.244 
// ExecutionRequest[P:'MueveTuCu.cu',P:1,T:1,args:'',q:'cudalb'] 
// May 15 2019 22:22:05
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

/*CUDA*/
#define BLOQUEx 16
#define	BLOQUEy 32
#define	BLOQUETEAMS 256

#define THREADS BLOQUEx*BLOQUEy
/*CUDA */

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

__device__ __constant__ int rows_d;
__device__ __constant__ int columns_d;
__device__ __constant__ int numfocal_d;
__device__ __constant__ int numteams_d;
__device__ __constant__ int size_d;
__device__ __constant__ int iter_d;
__device__ __constant__ int RADIUS_TYPE_1_d;
__device__ __constant__ int RADIUS_TYPE_2_3_d;
__device__  int num_deactivated_d;
__device__  int first_activation_d;
//__device__ void warpReduce(volatile float* aux_d, int ij);

 __global__ void kernelini( float* surface_d)
{
	
	int ij=(blockIdx.y*blockDim.y+threadIdx.y)*columns_d+blockIdx.x*blockDim.x+threadIdx.x;
	// Cálculo del identificador global del hilo
	if(ij > -1 && ij<rows_d*columns_d){
	 surface_d[ij]=0.0f;
	}

		
}
__global__ void actualizaCalor(float *surface,int i,int j, int heat){
	surface[i*columns_d+j]=heat;
}
__global__ void kernel2( float* surface_d, float* surfaceCopy_d)
{
	
	int i=blockIdx.y * blockDim.y + threadIdx.y;
	int j=blockIdx.x * blockDim.x + threadIdx.x;
	if(i > 0 && i<rows_d-1 &&
		j > 0 && j<columns_d-1){
	surface_d[i*columns_d+j] = (surfaceCopy_d[(i-1)*columns_d+j]+surfaceCopy_d[(i+1)*columns_d+j]+surfaceCopy_d[i*columns_d+(j-1)]+surfaceCopy_d[i*columns_d+(j+1)] ) / 4;
	}

		
}

__global__ void max(float* aux_d,int redSize,float* global_residual_d)
{
	
	// Cálculo del identificador global del hilo
	int ij=(blockIdx.y*blockDim.y+threadIdx.y)*columns_d+blockIdx.x*blockDim.x+threadIdx.x;
	  if ( ij >=redSize/2 ) return;

				// Reducción del elemento en la posición gid con su pareja	
				if(ij > -1 && ij<rows_d*columns_d){
						if (aux_d[ij]<aux_d[ij+(redSize/2)]){

						aux_d[ij]=aux_d[ij+(redSize/2)];
							
						}
				
				
				if ( redSize%2 != 0 && ij == 0 ) {
			   		if (aux_d[0]<aux_d[redSize-1])
			   		{	
			    		aux_d[0]=aux_d[redSize-1];
			    	}
				}
					
				if(ij==0){
					global_residual_d[0]=aux_d[0];
				}
			}

}




__global__ void kernel3( float* surface_d, float* surfaceCopy_d,float* aux_d,float* global_residual_d)
{


	// Cálculo del identificador global del hilo
	int ij=(blockIdx.y*blockDim.y+threadIdx.y)*columns_d+blockIdx.x*blockDim.x+threadIdx.x;
	if(ij > -1 && ij<rows_d*columns_d){
	if ( fabs( surface_d[ij] - surfaceCopy_d[ij] ) > global_residual_d[0] ) {
		
	aux_d[ij] =  fabs( surface_d[ij] - surfaceCopy_d[ij] );
	
	}
	

}
}

__global__ void kernelAtomic(FocalPoint *focal_d)
{
int ij=(blockIdx.y*blockDim.y+threadIdx.y)*columns_d+blockIdx.x*blockDim.x+threadIdx.x;
if(ij<numfocal_d){

			if ( focal_d[ij].start == iter_d ) {
				focal_d[ij].active = 1;
				if ( ! first_activation_d ) first_activation_d = 1;
			}
			// Count focal points already deactivated by a team
			if ( focal_d[ij].active == 2 ) atomicAdd(&num_deactivated_d, 1);

}
}

__global__ void kernelMove(Team *team_d,FocalPoint *focal_d)
{
	int t = (blockIdx.y*blockDim.y+threadIdx.y)*columns_d+blockIdx.x*blockDim.x+threadIdx.x;
	int j;
	if(t<numteams_d){
		// 4.3.1. Choose nearest focal point 
			float distance = FLT_MAX;
			int target = -1;
			for( j=0; j<numfocal_d; j++ ) {
				if ( focal_d[j].active != 1 ) continue; // Skip non-active focal points
				float dx = focal_d[j].x - team_d[t].x;
				float dy = focal_d[j].y - team_d[t].y;
				float local_distance = sqrtf( dx*dx + dy*dy );

				if ( local_distance < distance ) {
					distance = local_distance;
					target = j;
				}
			}
			//4.3.2. Annotate target for the next stage 
			team_d[t].target = target;

			// 4.3.3. No active focal point to choose, no movement 
			if ( target == -1 ) return; 
		//if ( target != -1 ){
			// 4.3.4. Move in the focal point direction 
			if ( team_d[t].type == 1 ) { 
				
			//	printf(" team T=%d F=%d  team_d[t].x %d team_d[t].y %d team_d[t].type== %d iter=%d \n",t,j,team_d[t].x,team_d[t].y,team_d[t].type,iter_d);
				// Type 1: Can move in diagonal
				if ( focal_d[target].x < team_d[t].x ) team_d[t].x--;
				if ( focal_d[target].x > team_d[t].x ) team_d[t].x++;
				if ( focal_d[target].y < team_d[t].y ) team_d[t].y--;
				if ( focal_d[target].y > team_d[t].y ) team_d[t].y++;
			}
			else if ( team_d[t].type == 2 ) { 
			
				// Type 2: First in horizontal direction, then in vertical direction
				//printf("no entro\n");
				if ( focal_d[target].y < team_d[t].y ) team_d[t].y--;
				else if ( focal_d[target].y > team_d[t].y ) team_d[t].y++;
				else if ( focal_d[target].x < team_d[t].x ) team_d[t].x--;
				else if ( focal_d[target].x > team_d[t].x ) team_d[t].x++;
			}
			else {
				
			
				// Type 3: First in vertical direction, then in horizontal direction
				if ( focal_d[target].x < team_d[t].x ) team_d[t].x--;
				else if ( focal_d[target].x > team_d[t].x ) team_d[t].x++;
				else if ( focal_d[target].y < team_d[t].y ) team_d[t].y--;
				else if ( focal_d[target].y > team_d[t].y ) team_d[t].y++;
			}
		}
	
}
__global__ void Reduce1(float *surface_d,int x,int y,int radius){

	int i=x-radius+threadIdx.y;
	int j=y-radius+threadIdx.x;
	if(threadIdx.x>=7 || threadIdx.y >=7) return;
	if ( i<1 || i>=rows_d-1 || j<1 || j>=columns_d-1  ) return; // Out of the heated surface
	float dx = x - i;
	float dy = y - j;
	float distance = dx*dx + dy*dy ;
	if ( distance <= radius*radius ) {
		surface_d[i*columns_d+j]*=0.75; // Team efficiency factor
	}


}

__global__ void Reduce2(float *surface_d,int x,int y,int radius){

	int i=x-radius+threadIdx.y;
	int j=y-radius+threadIdx.x;
	if(threadIdx.x>=19 || threadIdx.y >=19) return;
	if ( i<1 || i>=rows_d-1 || j<1 || j>=columns_d-1  ) return; // Out of the heated surface
	float dx = x - i;
	float dy = y - j;
	float distance = dx*dx + dy*dy ;
	if ( distance <= radius*radius ) {
		surface_d[i*columns_d+j]*=0.75; // Team efficiency factor
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


 
	cudaError_t error;
	float *global_residual_d;
	float *surface_d;
	float *surfaceCopy_d;
	float *aux_d;
	Team *team_d;
	FocalPoint *focal_d;
	float *puntero;
	int tama=rows*columns;
	int threads=THREADS;
	int bloqueTeams=BLOQUETEAMS;
	int gridEnX, gridEnY,gridTeams;

	error=cudaMalloc( (void**) &team_d, sizeof(Team)* (size_t)num_teams);
	error=cudaMalloc( (void**) &focal_d, sizeof(FocalPoint) * (size_t)num_focal);
	error=cudaMalloc( (void**) &surface_d, sizeof(float) * (int) tama);
	error=cudaMalloc( (void**) &aux_d, sizeof(float) * (int) tama);
	error=cudaMalloc( (void**) &surfaceCopy_d, sizeof(float) * (int) tama);
	error=cudaMalloc( (void**) &global_residual_d, sizeof(float) * (int) 1);
	if ( error != cudaSuccess )
	printf("ErrCUDA 1: %s\n", cudaGetErrorString( error ) );

	
	if ( error != cudaSuccess )
	printf("ErrCUDA 2: %s\n", cudaGetErrorString( error ) );
	
	cudaMemcpyToSymbolAsync(rows_d,&rows, sizeof(int),0,cudaMemcpyHostToDevice);
	cudaMemcpyToSymbolAsync(columns_d,&columns, sizeof(int),0,cudaMemcpyHostToDevice);
	cudaMemcpyToSymbolAsync(size_d,&tama, sizeof(int),0,cudaMemcpyHostToDevice);
	cudaMemcpyToSymbolAsync(numfocal_d,&num_focal, sizeof(int),0,cudaMemcpyHostToDevice);
	cudaMemcpyToSymbolAsync(numteams_d,&num_teams, sizeof(int),0,cudaMemcpyHostToDevice);

	

	gridTeams= num_teams/bloqueTeams;
	if (num_teams%bloqueTeams!=0) gridTeams++;

	gridEnX= columns/BLOQUEx;
	if (columns%BLOQUEy!=0) gridEnX++;
	gridEnY= rows/BLOQUEy;
	if (rows%BLOQUEy!=0) gridEnY++;

	dim3 bloqShapeGpu(BLOQUEx,BLOQUEy);
	dim3 gridShapeGpu(gridEnX,gridEnY);
	dim3 Reducegrid(1,1);
	dim3 Reduce1bloq(8,8);
	
	dim3 Reduce2bloq(32,32);
	/* 3. Initialize surface */

	//error=cudaMemcpy(surface_d,surface, sizeof(float) * tama,cudaMemcpyHostToDevice);
	kernelini<<<gridShapeGpu,bloqShapeGpu>>>(&surface_d[rows*columns*0]);
	//error=cudaMemcpy(surface,surface_d, sizeof(float) * tama,cudaMemcpyDeviceToHost);

	
	for( i=0; i<rows; i++ )
		for( j=0; j<columns; j++ )
			accessMat( surface, i, j ) = 0.0;
		
	

	
	/* 4. Simulation */
	int iter;
	int flag_stability = 0;
	int first_activation = 0;
	for( iter=0; iter<max_iter && ! flag_stability; iter++ ) {
	cudaMemcpyToSymbolAsync(iter_d,&iter, sizeof(int),0,cudaMemcpyHostToDevice);//PRUEBAS######################
		/* 4.1. Activate focal points */
		int num_deactivated = 0;
		
		error=cudaMemcpy(focal_d,focal,  sizeof(FocalPoint) * (size_t)num_focal,cudaMemcpyHostToDevice);
		cudaMemcpyToSymbolAsync(num_deactivated_d,&num_deactivated,sizeof(int),0,cudaMemcpyHostToDevice);
		cudaMemcpyToSymbolAsync(first_activation_d,&first_activation,sizeof(int),0,cudaMemcpyHostToDevice);
		kernelAtomic<<<gridShapeGpu,bloqShapeGpu>>>(focal_d);
		cudaMemcpyFromSymbolAsync(&num_deactivated, num_deactivated_d,sizeof(int), 0, cudaMemcpyDeviceToHost);
		cudaMemcpyFromSymbolAsync(&first_activation, first_activation_d,sizeof(int), 0, cudaMemcpyDeviceToHost);
		error=cudaMemcpy(focal,focal_d, sizeof(FocalPoint) * (size_t)num_focal,cudaMemcpyDeviceToHost);
		if(!first_activation) continue;
		/*
		for( i=0; i<num_focal; i++ ) {
			if ( focal[i].start == iter ) {
				focal[i].active = 1;
				if ( ! first_activation ) first_activation = 1;
			}
			// Count focal points already deactivated by a team
			if ( focal[i].active == 2 ) num_deactivated++;
		}
		//cambiazo
		*/
		/* 4.2. Propagate heat (10 steps per each team movement) */
		float *global_residual ;
		if ((global_residual=(float *) calloc(1, sizeof(int)))==NULL){		//vector A
		printf("error\n");
		exit (-1);
	}
		global_residual[0]=0.0f;
		int step;
		for( step=0; step<10; step++ )	{
			/* 4.2.1. Update heat on active focal points */

			for( i=0; i<num_focal; i++ ) {
				if ( focal[i].active != 1 ) continue;
				int x = focal[i].x;
				int y = focal[i].y;
				actualizaCalor<<<1,1>>>(surface_d,x,y,focal[i].heat);
				
			//	accessMat( surface, x, y ) = focal[i].heat;
				
			}
		
			
			/* 4.2.2. Copy values of the surface in ancillary structure (Skip borders) */
			/*
			for( i=1; i<rows-1; i++ )
				for( j=1; j<columns-1; j++ )
					accessMat( surfaceCopy, i, j ) = accessMat( surface, i, j );
					*/
			puntero=surface_d;
			surface_d=surfaceCopy_d;
			surfaceCopy_d=puntero;
 
			/* 4.2.3. Update surface values (skip borders) */
			//error=cudaMemcpy(surface_d,surface, sizeof(float) * tama,cudaMemcpyHostToDevice);
			//error=cudaMemcpy(surfaceCopy_d,surfaceCopy, sizeof(float) * tama,cudaMemcpyHostToDevice);

			kernel2<<<gridShapeGpu,bloqShapeGpu>>>(&surface_d[rows*columns*0],&surfaceCopy_d[rows*columns*0]);
			//if(step==0)error=cudaMemcpy(surface,surface_d, sizeof(float) * tama,cudaMemcpyDeviceToHost);
			//if(step==0)error=cudaMemcpy(surfaceCopy,surfaceCopy_d, sizeof(float) * tama,cudaMemcpyDeviceToHost);
			
			/* 4.2.4. Compute the maximum residual difference (absolute value) */

		
			if( num_deactivated == num_focal && step==0){

			if(step==0)error=cudaMemcpy(aux_d,surface, sizeof(float) * tama,cudaMemcpyHostToDevice);
			if(step==0)error=cudaMemcpy(global_residual_d,global_residual, sizeof(float)*(int)1,cudaMemcpyHostToDevice);
		
			kernel3<<<gridShapeGpu,bloqShapeGpu>>>(&surface_d[rows*columns*0],&surfaceCopy_d[rows*columns*0],aux_d,global_residual_d);
				    // ¿Cuántos elementos se reducen?

				
				for (int redSize = tama; redSize>=1; redSize /= 2) 
				{
					int numBlocks=(redSize+(threads)-1)/threads;
					max<<<numBlocks,threads>>>(aux_d,redSize,global_residual_d);
				}	
			}
			if(step==9)error=cudaMemcpy(global_residual,global_residual_d, sizeof(float) ,cudaMemcpyDeviceToHost);
		

		/*	
			if( num_deactivated == num_focal && step==0){

			for( i=1; i<rows-1; i++ )
				for( j=1; j<columns-1; j++ ) 
					if ( fabs( accessMat( surface, i, j ) - accessMat( surfaceCopy, i, j ) ) > global_residual[0] ) {
						global_residual[0] = fabs( accessMat( surface, i, j ) - accessMat( surfaceCopy, i, j ) );
					}
			}
*/
		
			

		}//fin del for gordo
		/* If the global residual is lower than THRESHOLD, we have reached enough stability, stop simulation at the end of this iteration */
		if( num_deactivated == num_focal && global_residual[0] < THRESHOLD ) flag_stability = 1;

		/* 4.3. Move teams */
		//if(num_deactivated!=num_focal)//cambiazo
			
			error=cudaMemcpy(team_d,teams,  sizeof(Team) * (size_t)num_teams,cudaMemcpyHostToDevice);
			error=cudaMemcpy(focal_d,focal, sizeof(FocalPoint) * (size_t)num_focal,cudaMemcpyHostToDevice);
			if(num_deactivated!=num_focal)
			kernelMove<<<gridTeams,bloqueTeams>>>(team_d,focal_d);
			error=cudaMemcpy(teams,team_d,  sizeof(Team) * (size_t)num_teams,cudaMemcpyDeviceToHost);
			error=cudaMemcpy(focal,focal_d, sizeof(FocalPoint) * (size_t)num_focal,cudaMemcpyDeviceToHost);

		/* 4.4. Team actions */

	
		for( t=0; t<num_teams; t++ ) {
			// 4.4.1. Deactivate the target focal point when it is reached 
			int target = teams[t].target;
			if ( target != -1 && focal[target].x == teams[t].x && focal[target].y == teams[t].y 
				&& focal[target].active == 1 )
				focal[target].active = 2;

			// 4.4.2. Reduce heat in a circle around the team 
			int radius;
			// Influence area of fixed radius depending on type
			if ( teams[t].type == 1 )
			{
			 radius = RADIUS_TYPE_1;
			Reduce1<<<Reducegrid,Reduce1bloq>>>(surface_d,teams[t].x,teams[t].y,radius);
			}
			else 
			{
				radius = RADIUS_TYPE_2_3;
				Reduce2<<<Reducegrid,Reduce2bloq>>>(surface_d,teams[t].x,teams[t].y,radius);
			}
		
		
	

		}
					

#ifdef DEBUG
		/* 4.5. DEBUG: Print the current state of the simulation at the end of each iteration */
		print_status( iter, rows, columns, surface, num_teams, teams, num_focal, focal, global_residual[0] );
#endif // DEBUG
	}
	error=cudaMemcpy(surface,surface_d, sizeof(float) * tama,cudaMemcpyDeviceToHost);
	
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
