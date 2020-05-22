// User: g102@157.88.139.133 
// ExecutionRequest[P:'equis.cu',P:1,T:1,args:'',q:'cudalb'] 
// May 16 2019 19:44:01
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
#include <cstdlib>
#include <iostream>
#include <cputils.h>

#define RADIUS_TYPE_1		3
#define RADIUS_TYPE_2_3		9
#define THRESHOLD	0.1f

#define CX 8
#define CY 8

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
#define max(a,b) \
({ __typeof__ (a) _a = (a); \
	__typeof__ (b) _b = (b); \
	_a > _b ? _a : _b; })

__device__ float atomicMaxf(float* address, float val)
{
int *address_as_int =(int*)address;
int old = *address_as_int, assumed;
while (val > __int_as_float(old)) {
assumed = old;
old = atomicCAS(address_as_int, assumed,
__float_as_int(val));
}
return __int_as_float(old);
}

__global__ void gpuFunc_mostrarHilosBloques(/* arguments */) {
	if(blockIdx.x==9){
	printf("Hilo x: %d Hilo y: %d Bloque x: %d Bloque y: %d Dimension bloque x: %d Dimension bloque y: %d, Dimension grid x: %d, Dimension grid y: %d\n",threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y,gridDim.x,gridDim.y);
}
}


__global__ void gpuFunc_init(float *surface,int items){

  int position = ((blockIdx.y*gridDim.x)+blockIdx.x)*(blockDim.x*blockDim.y)+((threadIdx.y*blockDim.x)+threadIdx.x);
  if(position<items){
    surface[position]=0.0;
  }
}

__global__ void gpuFunc_init_int(unsigned int *surface,int items){

  int position = ((blockIdx.y*gridDim.x)+blockIdx.x)*(blockDim.x*blockDim.y)+((threadIdx.y*blockDim.x)+threadIdx.x);
  if(position<items){
    surface[position]=0;
  }
}


__global__ void gpuFunc_ActualizarSinBordes(float *surfaceaux, float *surfaceCopyaux, int rows, int columns, int items)
{
int position = ((blockIdx.y*gridDim.x)+blockIdx.x)*(blockDim.x*blockDim.y)+((threadIdx.y*blockDim.x)+threadIdx.x);

if(position > columns && position < (items - columns) && ((position%columns) != 0) && ((position%columns) != (columns-1)))
{
surfaceaux[position] =
(surfaceCopyaux[position-columns] +
surfaceCopyaux[position+columns] +
surfaceCopyaux[position-1] +
surfaceCopyaux[position+1])/4;
}

}

__global__ void gpuFunc_ActualizarCalorFocos(float *surfaceaux, int num_focal, FocalPoint *focalaux,int columns)
{
	int position = ((blockIdx.y*gridDim.x)+blockIdx.x)*(blockDim.x*blockDim.y)+((threadIdx.y*blockDim.x)+threadIdx.x);
//puede ser el if menor o igual
	if((position<num_focal)&&(focalaux[position].active == 1)){
		int x = focalaux[position].x;
		int y = focalaux[position].y;
		accessMat( surfaceaux, x, y ) = focalaux[position].heat;
	}

}
__global__ void gpuFunc_IntercambioPunteros(float *surfaceaux,float *surfaceCopyaux, float *aux){
	aux=surfaceaux;
	surfaceaux=surfaceCopyaux;
	surfaceCopyaux=aux;
}

__global__ void gpuFunc_MaximResidual(float *surfaceaux, float *surfaceCopyaux, float *auxresidual,int items)
{
	int position = ((blockIdx.y*gridDim.x)+blockIdx.x)*(blockDim.x*blockDim.y)+((threadIdx.y*blockDim.x)+threadIdx.x);
if(position<items){

		auxresidual[position] =  surfaceaux[position] - surfaceCopyaux[position] ;
	//	printf("global_residual: %f\n", auxresidual[position] );

}
}

__global__ void gpuFunc_ClearMem(float *d_max){
d_max[0]=0.0f;
}

__global__ void gpuFunc_ReductionMax(float *auxresidual, float *auxresidualcopy, int items)
{
	extern __shared__ float shared[];
	int tid = threadIdx.x;
	int myId = threadIdx.x +blockDim.x*blockIdx.x;
	//int position = ((blockIdx.y*gridDim.x)+blockIdx.x)*(blockDim.x*blockDim.y)+((threadIdx.y*blockDim.x)+threadIdx.x);


	shared[tid] = 0.0f;


	while (myId < items){
	shared[tid] = max(shared[tid], auxresidual[myId]);
	myId += gridDim.x*blockDim.x;
	}__syncthreads();

	myId = (blockDim.x * blockIdx.x) + tid;  // 1
	    for (unsigned int s=blockDim.x/2; s>0; s>>=1)
	    {
	        if (tid < s && myId < items)
	            shared[tid] = max(shared[tid], shared[tid + s]);
	        __syncthreads();
	    }
	// what to do now?

	// option 2: use atomics
//	printf("global_residual: %f\n",shared[0] );
	if (tid == 0){
	atomicMaxf(auxresidualcopy, shared[0]);
	}

	}






/* 4.3 kernel*/
__global__ void gpuFunc_SelectTarget(FocalPoint *focalaux,Team *teamsaux, int num_teams, int num_focal){

	float distance = FLT_MAX;
	int target = -1;
	int position = ((blockIdx.y*gridDim.x)+blockIdx.x)*(blockDim.x*blockDim.y)+((threadIdx.y*blockDim.x)+threadIdx.x);
	if(position<num_teams){
		for( int j=0; j<num_focal; j++ ) {
			if ( focalaux[j].active != 1 ) continue; // Skip non-active focal points
			float dx = focalaux[j].x - teamsaux[position].x;
			float dy = focalaux[j].y - teamsaux[position].y;
			float local_distance = sqrtf( dx*dx + dy*dy );
			if ( local_distance < distance ) {
				distance = local_distance;
				target = j;
			}
	}
	teamsaux[position].target = target;

	/* 4.3.3. No active focal point to choose, no movement */
	if ( target != -1 ){

	/* 4.3.4. Move in the focal point direction */
	if ( teamsaux[position].type == 1 ) {
		// Type 1: Can move in diagonal
		if ( focalaux[target].x < teamsaux[position].x ) teamsaux[position].x--;
		if ( focalaux[target].x > teamsaux[position].x ) teamsaux[position].x++;
		if ( focalaux[target].y < teamsaux[position].y ) teamsaux[position].y--;
		if ( focalaux[target].y > teamsaux[position].y ) teamsaux[position].y++;
	}
	else if ( teamsaux[position].type == 2 ) {
		// Type 2: First in horizontal direction, then in vertical direction
		if ( focalaux[target].y < teamsaux[position].y ) teamsaux[position].y--;
		else if ( focalaux[target].y > teamsaux[position].y ) teamsaux[position].y++;
		else if ( focalaux[target].x < teamsaux[position].x ) teamsaux[position].x--;
		else if ( focalaux[target].x > teamsaux[position].x ) teamsaux[position].x++;
	}
	else {
		// Type 3: First in vertical direction, then in horizontal direction
		if ( focalaux[target].x < teamsaux[position].x ) teamsaux[position].x--;
		else if ( focalaux[target].x > teamsaux[position].x ) teamsaux[position].x++;
		else if ( focalaux[target].y < teamsaux[position].y ) teamsaux[position].y--;
		else if ( focalaux[target].y > teamsaux[position].y ) teamsaux[position].y++;
	}
}
}
}

/*4.4 TeamAction*/
__global__ void gpuFunc_TeamAction(Team *teamsaux, FocalPoint *focalaux, int num_teams,unsigned int *surfaceCambios,int rows, int columns){
	int position = ((blockIdx.y*gridDim.x)+blockIdx.x)*(blockDim.x*blockDim.y)+((threadIdx.y*blockDim.x)+threadIdx.x);
	//surfaceCambios[position]=0;
	if(position<num_teams){
		int target = teamsaux[position].target;
		if(target != -1 && focalaux[target].x == teamsaux[position].x && focalaux[target].y == teamsaux[position].y && focalaux[target].active == 1 )
				focalaux[target].active = 2;


		int radius;
		// Influence area of fixed radius depending on type
		if ( teamsaux[position].type == 1 ) radius = RADIUS_TYPE_1;
		else radius = RADIUS_TYPE_2_3;
		for( int i=teamsaux[position].x-radius; i<=teamsaux[position].x+radius; i++ ) {
			for(int j=teamsaux[position].y-radius; j<=teamsaux[position].y+radius; j++ ) {
				if ( i<1 || i>=rows-1 || j<1 || j>=columns-1 ) continue; // Out of the heated surface
				float dx = teamsaux[position].x - i;
				float dy = teamsaux[position].y - j;
				float distance = sqrtf( dx*dx + dy*dy );
				if ( distance <= radius ) {
					atomicInc(surfaceCambios+(i*columns+j),num_teams+1); // Team efficiency factor
				}
			}
		}
	}
}


__global__ void gpuFunc_Apagar(unsigned int *surfaceCambios,float *surfaceaux, int items){
	int position = ((blockIdx.y*gridDim.x)+blockIdx.x)*(blockDim.x*blockDim.y)+((threadIdx.y*blockDim.x)+threadIdx.x);
	if(position<items){
		int i = surfaceCambios[position];
		for(int j=0;j<i;j++){
			surfaceaux[position]=(surfaceaux[position]*( 1 - 0.25 ));
		}
		surfaceCambios[position]=0;
	}


}
/*for( t=0; t<num_teams; t++ ) {



	int target = teams[t].target;
	if ( target != -1 && focal[target].x == teams[t].x && focal[target].y == teams[t].y
		&& focal[target].active == 1 )
		focal[target].active = 2;


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
}*/




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

int blockx, blocky, gridx, gridy;
 if (rows<=CX){
			 blockx=rows;
			 gridx=1;
	 } else {
			 blockx=CX;
			 gridx=rows/blockx;
			 if (rows%CX != 0) gridx++;
		}

		if (columns <=CY) {
			 blocky=columns;
			 gridy=1;
		} else {
			 blocky=CY;
			 gridy=columns/CY;
			 if (columns%CY != 0) gridy++;
		}
dim3 bloqueShape(CX*CY,1);
dim3 gridShape(gridx*gridy,1);
/*surface = (float *)malloc( sizeof(float) * (size_t)rows * (size_t)columns );
surfaceCopy = (float *)malloc( sizeof(float) * (size_t)rows * (size_t)columns );
*/
//gpuFunc_mostrarHilosBloques<<<gridShape,bloqueShape>>>();
int flagfoco=0;
float *surfaceaux;
float *surfaceCopyaux;
float *aux;
float *d_max;
unsigned int *surfaceCambios;
float *auxresidual;
Team *teamsaux;
FocalPoint *focalaux;
cudaError_t err;
size_t memoriaShared=(CX*CY)*sizeof(float);
int items=rows*columns;
err=cudaMalloc((void**)&surfaceaux, sizeof(float)*rows*columns);
printf("%s\n",cudaGetErrorString(err));

err=cudaMalloc(&surfaceCopyaux, sizeof(float)*rows*columns);
printf("%s\n",cudaGetErrorString(err));

err=cudaMalloc(&aux, sizeof(float)*rows*columns);
printf("%s\n",cudaGetErrorString(err));

err=cudaMalloc(&d_max, sizeof(float));
printf("%s\n",cudaGetErrorString(err));

err=cudaMalloc(&auxresidual, sizeof(float)*rows*columns);
printf("%s\n",cudaGetErrorString(err));

err=cudaMalloc(&focalaux, sizeof(FocalPoint) * num_focal );
printf("%s\n",cudaGetErrorString(err));

err=cudaMalloc(&teamsaux, sizeof(Team) * num_teams );
printf("%s\n",cudaGetErrorString(err));

err=cudaMalloc(&surfaceCambios, sizeof(unsigned int)*rows*columns);
printf("%s\n",cudaGetErrorString(err));
/*err=cudaMemcpy(focalaux,focal,sizeof(FocalPoint)*num_focal);
printf("%s\n",cudaGetErrorString(err));*/
	/* 3. Initialize surface */
		gpuFunc_init<<<gridShape,bloqueShape>>>(surfaceaux,items);
		gpuFunc_init_int<<<gridShape,bloqueShape>>>(surfaceCambios,items);
err=cudaMemcpy(surface,surfaceaux,sizeof(float)*rows*columns,cudaMemcpyDeviceToHost);
printf("%s\n",cudaGetErrorString(err));

err=cudaMemcpy(teamsaux,teams,sizeof(Team)*num_teams,cudaMemcpyHostToDevice);
printf("%s\n",cudaGetErrorString(err));






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
				flagfoco=1;
			}
			// Count focal points already deactivated by a team
			if ( focal[i].active == 2 ) num_deactivated++;
		}
		if(flagfoco==0) continue;
		/* 4.2. Propagate heat (10 steps per each team movement) */
		err=cudaMemcpy(focalaux,focal,sizeof(FocalPoint)*num_focal,cudaMemcpyHostToDevice);
		//err=cudaMemcpy(surfaceaux,surface,sizeof(float)*rows*columns,cudaMemcpyHostToDevice);

		float global_residual = 0.0f;
		int step;

		for( step=0; step<10; step++ )	{
			/* 4.2.1. Update heat on active focal points */


			gpuFunc_ActualizarCalorFocos<<<gridShape,bloqueShape>>>(surfaceaux,num_focal,focalaux,columns);


			/* 4.2.2. Copy values of the surface in ancillary structure (Skip borders) */

			aux=surfaceaux;
			surfaceaux=surfaceCopyaux;
			surfaceCopyaux=aux;


			/* 4.2.3. Update surface values (skip borders) */

			gpuFunc_ActualizarSinBordes<<<gridShape,bloqueShape>>>(surfaceaux,surfaceCopyaux,rows,columns,items);


			/* 4.2.4. Compute the maximum residual difference (absolute value) */

			if(num_deactivated==num_focal&&(step==0||step==9)){

			gpuFunc_MaximResidual<<<gridShape,bloqueShape>>>(surfaceaux,surfaceCopyaux,auxresidual,items);

			gpuFunc_ReductionMax<<<gridShape,bloqueShape,memoriaShared>>>(auxresidual,d_max,items);
			}


		}
		//cudaMemcpy(focal,focalaux,sizeof(FocalPoint)*num_focal,cudaMemcpyDeviceToHost);
		/* If the global residual is lower than THRESHOLD, we have reached enough stability, stop simulation at the end of this iteration */
		cudaMemcpy(&global_residual,&d_max[0],sizeof(float),cudaMemcpyDeviceToHost);
	//	cudaMemcpy(surface,surfaceaux,sizeof(float)*rows*columns,cudaMemcpyDeviceToHost);
	//	cudaMemcpy(surfaceCopy,surfaceCopyaux,sizeof(float)*rows*columns,cudaMemcpyDeviceToHost);
		gpuFunc_ClearMem<<<1,1>>>(d_max);

		if( num_deactivated == num_focal && global_residual < THRESHOLD ) flag_stability = 1;

		/* 4.3. Move teams */

			/* 4.3.1. Choose nearest focal point */


		if(num_deactivated!=num_focal){
		gpuFunc_SelectTarget<<<gridShape,bloqueShape>>>(focalaux,teamsaux,num_teams,num_focal);
		}
		//cudaMemcpy(teams,teamsaux,sizeof(Team)*num_teams,cudaMemcpyDeviceToHost);
		/* 4.4. Team actions */
		gpuFunc_TeamAction<<<gridShape,bloqueShape>>>(teamsaux,focalaux,num_teams,surfaceCambios,rows,columns);
		gpuFunc_Apagar<<<gridShape,bloqueShape>>>(surfaceCambios,surfaceaux,items);
		/*for( t=0; t<num_teams; t++ ) {



			int target = teams[t].target;
			if ( target != -1 && focal[target].x == teams[t].x && focal[target].y == teams[t].y
				&& focal[target].active == 1 )
				focal[target].active = 2;


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
		}*/
		err=cudaMemcpy(focal,focalaux,sizeof(FocalPoint)*num_focal,cudaMemcpyDeviceToHost);
		//cudaMemcpy(teams,teamsaux,sizeof(Team)*num_teams,cudaMemcpyDeviceToHost);


#ifdef DEBUG
		/* 4.5. DEBUG: Print the current state of the simulation at the end of each iteration */
		print_status( iter, rows, columns, surface, num_teams, teams, num_focal, focal, global_residual );
#endif // DEBUG
	}
cudaMemcpy(surface,surfaceaux,sizeof(float)*rows*columns,cudaMemcpyDeviceToHost);
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
	cudaFree(auxresidual);
	cudaFree(d_max);
	cudaFree(surfaceCambios);
	cudaFree(surfaceaux);
	cudaFree(surfaceCopyaux);
	cudaFree(aux);
	cudaFree(focalaux);
	free( teams );
	free( focal );
	free( surface );
	free( surfaceCopy );
	cudaDeviceReset();
	/* 8. End */
	return 0;
}
