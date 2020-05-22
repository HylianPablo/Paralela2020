// User: g106@157.88.139.133 
// ExecutionRequest[P:'pipo14.0.cu',P:1,T:1,args:'',q:'cudalb'] 
// May 16 2019 17:50:56
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
 
 
 __device__ float atomicMult(float* address, double val)
 {
	 /*
	 unsigned long long int* address_as_ull =
							   (unsigned long long int*)address;
	 unsigned long long int old = *address_as_ull, assumed;*/
	 int* address_as_int = (int*)address;
	 int old = *address_as_int, assumed; 
	 do {
		 
			 assumed = old;
			 old = atomicCAS(address_as_int, assumed, __float_as_int(val * __float_as_int(assumed)));
			 } while (assumed != old); return __int_as_float(old);
 }
 
 __global__ void kernel_enciendePosConFoco(FocalPoint *focal,int num_focal, float* surface,int columnas_surface){
	
	 /* Fórmula para calcular la posición */ 
	 int pos_x = blockDim.x*blockIdx.x+ threadIdx.x; // Elemento del vector como función de hilo y bloque
	 
	 
	 /* Cálculo de la suma correspondiente */
	 if(pos_x<num_focal){
		 if ( focal[pos_x].active == 1 ){
		 int x = focal[pos_x].x;
		 int y = focal[pos_x].y;
		 surface[ x * columnas_surface + y ] = focal[pos_x].heat;
		 }
	 };
	
 }
 
 __global__ void kernelEnciendeFoco(float *surface,int i,int j, int columns , int heat){
	 
	 surface[i*columns+j]=heat;
	 
 }
 
 __global__ void kernelApagarFocos(FocalPoint* focal, Team* teams,int num_teams){
	
	
	 
 
	
	 // Fórmula para calcular la posición 
	 int pos_x = blockDim.x*blockIdx.x+threadIdx.x; // Elemento del vector como función de hilo y bloque
	 
 
	 if (pos_x<num_teams){
		 int target = teams[pos_x].target;
		 if ( target != -1 && focal[target].x == teams[pos_x].x && focal[target].y == teams[pos_x].y 
			 && focal[target].active == 1 )
			 focal[target].active = 2;
	 }
 }
 __global__ void kernelActivaFoco(FocalPoint *focal,int num_focal,int iter){
	 
	 int id=threadIdx.x+blockDim.x*blockIdx.x;
		 
	 if(id<num_focal  )
		 if(focal[id].start == iter)
			 focal[id].active = 1;
 }
 
	 
 
 
 __global__ void kernelApagarFuego(float *surface,Team *teams,int t,int radius,int rows,int columns){
	 int x = teams[t].x;
	 int y = teams[t].y;
	 int i=x-radius+threadIdx.y;
	 int j=y-radius+threadIdx.x;
	 
 
		 if ( i<1 || i>=rows-1 || j<1 || j>=columns-1  ) return; // Out of the heated surface
		 float dx = x - i;
		 float dy = y - j;
		 float distance = dx*dx + dy*dy ;
		 if ( distance <= radius*radius ) {
			 surface[i*columns+j]*=0.75;
			 
		 }
 
 
 }
 
 __global__ void kernelMoverEquipo(Team *teams,FocalPoint *focal, int num_teams,int num_focal, int iter){
 
		 int id=threadIdx.x+blockDim.x*blockIdx.x;
		 int j;
		 if(id>=num_teams) return;
 
		 float distance = FLT_MAX;
		 int target = -1;
		 for( j=0; j<num_focal; j++ ) {
			
			 if ( focal[j].active == 1 ){ // Skip non-active focal points
			 float dx = focal[j].x - teams[id].x;
			 float dy = focal[j].y - teams[id].y;
			 float local_distance = dx*dx + dy*dy ;
			 if ( local_distance < distance ) {
				 distance = local_distance;
				 target = j;
			 }
			}
			
		 }
		
		 /* 4.3.2. Annotate target for the next stage */
		 teams[id].target = target;
		 
		 /* 4.3.3. No active focal point to choose, no movement */
		 if ( target == -1 ) return;
		
		 /* 4.3.4. Move in the focal point direction */
		 if ( teams[id].type == 1 ) {
			 // Type 1: Can move in diagonal
			 if ( focal[target].x < teams[id].x ) teams[id].x--;
			 if ( focal[target].x > teams[id].x ) teams[id].x++;
			 if ( focal[target].y < teams[id].y ) teams[id].y--;
			 if ( focal[target].y > teams[id].y ) teams[id].y++;
			// printf("Tipo 1 post %d %d %d iter:%d\n",0,focal[0].x,focal[0].y,iter);
		 }
		 else if ( teams[id].type == 2 ) {
			 // Type 2: First in horizontal direction, then in vertical direction
			 //printf("Tipo 2 %d %d %d iter:%d\n",0,focal[0].x,focal[0].y,iter);
			 if ( focal[target].y < teams[id].y ) teams[id].y--;
			 else if ( focal[target].y > teams[id].y ) teams[id].y++;
			 else if ( focal[target].x < teams[id].x ) teams[id].x--;
			 else if ( focal[target].x > teams[id].x ) teams[id].x++;
		 }
		 else {
			 // Type 3: First in vertical direction, then in horizontal direction
			 //printf("Tipo 3 %d %d %d iter:%d\n",0,focal[0].x,focal[0].y,iter);
			 if ( focal[target].x < teams[id].x ) teams[id].x--;
			 else if ( focal[target].x > teams[id].x ) teams[id].x++;
			 else if ( focal[target].y < teams[id].y ) teams[id].y--;
			 else if ( focal[target].y > teams[id].y ) teams[id].y++;
		 }
 }
 
 
 
 __global__ void kernelPropagarFuegos(float *surface,float *surfaceCopy,int columns,int rows){
	 int i=threadIdx.y+blockDim.y*blockIdx.y;
	 int j=threadIdx.x+blockDim.x*blockIdx.x;
   
	 if (i >= rows-1 || i==0 || j>= columns-1 || j==0) return;
 
	 surface[i*columns+j]=(
		 surfaceCopy[(i-1)*columns+j]+
		 surfaceCopy[(i+1)*columns+j]+
		 surfaceCopy[i*columns+j-1]+
		 surfaceCopy[i*columns+j+1])/4;
 }
 
 //__device__ int flag_residual_gpu;
 //por algun motivo si flag_residual_gpu es una variable device pasan cosas raras
 __global__ void resta_reduce(int* flag_residual_gpu,const float* surface,const float* surfaceCopy,const int d_in_len){
	 if(flag_residual_gpu[0]==1)return;
	 int glbl_tid = blockDim.x * blockIdx.x + threadIdx.x;
	 
	 if (glbl_tid < d_in_len)
		 if (fabs(surfaceCopy[glbl_tid]-surface[glbl_tid])>=THRESHOLD)
			 flag_residual_gpu[0]=1;
 }
 
 /*
  * MAIN PROGRAM
  */
 int main(int argc, char *argv[]) {
	 int i,t;
 
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
 
	  
	 float *superficieGPU, *superficieCopiaGPU;
	 FocalPoint *focosGPU;
	 Team *equiposGPU;
 
 cudaMalloc((void **)&superficieGPU,sizeof(float)*rows*columns);
 cudaMalloc((void **) &superficieCopiaGPU,sizeof(float)*rows*columns);
 cudaMalloc((void **) &equiposGPU,sizeof(Team)*num_teams);
	 
	 cudaMalloc((void **) &focosGPU,sizeof(FocalPoint)*num_focal);
	 
	
	 cudaMemcpy(equiposGPU,teams,sizeof(Team)*num_teams,cudaMemcpyHostToDevice);
	 cudaMemcpy(focosGPU,focal,sizeof(FocalPoint)*num_focal,cudaMemcpyHostToDevice);
	 int sizeGridX = (int)ceil(columns/32.0);
	 int sizeGridY =(int) ceil(rows/32.0);
	 dim3 bloqShapeGpuFunc(sizeGridX,sizeGridY);
	 dim3 gridShapeGpuFunc(32,32);
 
	 int tamBlockX= 64;
	 int tamBlockY= 4;
	 int tamGridX2= columns/tamBlockX;
	 if (columns%tamBlockX!=0) tamGridX2++;
	 int tamGridY2= rows/tamBlockY;
	 if (rows%tamBlockY!=0) tamGridY2++;
 
 
	 dim3 blockSize2(tamBlockX,tamBlockY);
	 dim3 gridSize2(tamGridX2,tamGridY2);
	 
	 int teamHilosPorBloque,teamsBloquesPorGrid;
	 if(num_teams>1024){
	  teamHilosPorBloque = 1024;
	  teamsBloquesPorGrid = (int) ceil(num_teams/(float)teamHilosPorBloque);
	 }
	 else{
		 teamHilosPorBloque = num_teams;
		 teamsBloquesPorGrid = 1;
	 }
 
	 
	 int focosHilosPorBloque;
	 int focosBloquesPorGrid;
	 if(num_focal>1024) {
		focosHilosPorBloque = 1024;
		focosBloquesPorGrid = (int) ceil(num_teams/(float)focosHilosPorBloque);
	 }else{
		focosHilosPorBloque = num_focal;
		focosBloquesPorGrid = 1;
	 }
 
	 int reduceNumeroDeBloques =(int) ceil(rows*columns/1024.0);
 
	 cudaMemset(superficieGPU,0,rows*columns);
	 cudaMemset(superficieCopiaGPU,0,rows*columns);
	 
	 
	 /* 3. Initialize surface */
	 
 
	 /* 4. Simulation */
	 int iter;
	 int flag_stability = 0;
	 int flag_deactivate = 0;
	 for( iter=0; iter<max_iter && ! flag_stability; iter++ ) {
		 
		 /* 4.1. Activate focal points */
		 int num_deactivated = 0;
	 
		
		 kernelActivaFoco<<<focosBloquesPorGrid,focosHilosPorBloque>>>(focosGPU,num_focal,iter);
		 cudaMemcpy(focal,focosGPU,sizeof(FocalPoint)*num_focal,cudaMemcpyDeviceToHost);
		 for( i=0; i<num_focal; i++ ) {
			 if ( focal[i].active == 2 ) num_deactivated++;
		 }
 
 
		 /* 4.2. Propagate heat (10 steps per each team movement) */
		 
		 int step;
		 
 
		
	 
		 
		 /* 4.2.1. Update heat on active focal points */
		 kernel_enciendePosConFoco<<<focosBloquesPorGrid,focosHilosPorBloque>>>(focosGPU,num_focal,superficieGPU,columns);
 
			 float *surfaceAux;
			 surfaceAux = superficieCopiaGPU;
			 superficieCopiaGPU = superficieGPU;
			 superficieGPU = surfaceAux;
 
		 /*	float *surfaceAux;
			 surfaceAux = surfaceCopy;
			 surfaceCopy = surface;
			 surface =surfaceAux;*/
 
			 
	 kernelPropagarFuegos<<<gridSize2,blockSize2>>>(superficieGPU,superficieCopiaGPU,columns,rows);
 
			 /* 4.2.4. Compute the maximum residual difference (absolute value) */
			 
			 
			
 
	 if( num_deactivated == num_focal) flag_deactivate = 1;
		 for(step=0; step<9; step++ )	{
			 /* 4.2.1. Update heat on active focal points */
			 kernel_enciendePosConFoco<<<focosBloquesPorGrid,focosHilosPorBloque>>>(focosGPU,num_focal,superficieGPU,columns);
 
			 float *surfaceAux;
			 surfaceAux = superficieCopiaGPU;
			 superficieCopiaGPU = superficieGPU;
			 superficieGPU = surfaceAux;
 
			 kernelPropagarFuegos<<<gridSize2,blockSize2>>>(superficieGPU,superficieCopiaGPU,columns,rows);

		 }
 
		 /* 4.3. Move teams */
			

			
			 kernelMoverEquipo<<<teamsBloquesPorGrid,teamHilosPorBloque>>>(equiposGPU,focosGPU,num_teams,num_focal,iter);
			
	

		 kernelApagarFocos<<<teamsBloquesPorGrid,teamHilosPorBloque>>>(focosGPU,equiposGPU,num_teams);
		
		 /* 4.4. Team actions */
			 for( t=0; t<num_teams; t++ ) {
			 
 
			 /* 4.4.2. Reduce heat in a circle around the team */
			 int radius;
			 // Influence area of fixed radius depending on type
			 if ( teams[t].type == 1 ) {radius = RADIUS_TYPE_1;
				 dim3 gridRed(1,1);
				 dim3 blockRed(7,7);
				 kernelApagarFuego<<<gridRed,blockRed>>>(superficieGPU,equiposGPU,t,radius,rows,columns);
			 }
			 else {radius = RADIUS_TYPE_2_3;
			 dim3 gridRed(1,1);
			 dim3 blockRed(19,19);
			 kernelApagarFuego<<<gridRed,blockRed>>>(superficieGPU,equiposGPU,t,radius,rows,columns);
			 }
		 }

		 if(flag_deactivate){
			 iter++;
			 break;
		 }
	 
		 
		 #ifdef DEBUG
				 /* 4.5. DEBUG: Print the current state of the simulation at the end of each iteration */
			 cudaMemcpy(surface,superficieGPU,sizeof(float)*rows*columns,cudaMemcpyDeviceToHost);
			 print_status( iter, rows, columns, surface, num_teams, teams, num_focal, focal, global_residual );
		 #endif // DEBUG
 
	 }
	 for( ;iter<max_iter && ! flag_stability; iter++ ) {
		 
	

		/* 4.2. Propagate heat (10 steps per each team movement) */
		
		int step;
		

		int flag_residual=0;//para quedarnos con el del 1er step
	
		
		

			float *surfaceAux;
			surfaceAux = superficieCopiaGPU;
			superficieCopiaGPU = superficieGPU;
			superficieGPU = surfaceAux;

		/*	float *surfaceAux;
			surfaceAux = surfaceCopy;
			surfaceCopy = surface;
			surface =surfaceAux;*/

			
	kernelPropagarFuegos<<<gridSize2,blockSize2>>>(superficieGPU,superficieCopiaGPU,columns,rows);

			/* 4.2.4. Compute the maximum residual difference (absolute value) */
			
			
			int* flag_residual_gpu;
			cudaMalloc(&flag_residual_gpu,sizeof(int));
			cudaMemset(flag_residual_gpu,0,sizeof(int));
		
			resta_reduce<<<reduceNumeroDeBloques,1024>>>(flag_residual_gpu,superficieGPU,superficieCopiaGPU,rows*columns);
cudaMemcpy(&flag_residual, flag_residual_gpu, sizeof(int), cudaMemcpyDeviceToHost);
		

if(   !flag_residual) flag_stability = 1;
		for(step=0; step<9; step++ )	{
			/* 4.2.1. Update heat on active focal points */
		

			float *surfaceAux;
			surfaceAux = superficieCopiaGPU;
			superficieCopiaGPU = superficieGPU;
			superficieGPU = surfaceAux;

			/*float *surfaceAux;
			surfaceAux = surfaceCopy;
			surfaceCopy = surface;
			surface =surfaceAux;*/

			kernelPropagarFuegos<<<gridSize2,blockSize2>>>(superficieGPU,superficieCopiaGPU,columns,rows);
		
		
			
		}

	//	cudaMemcpy(superficieGPU,surface,sizeof(float)*rows*columns,cudaMemcpyHostToDevice);
	//		cudaMemcpy(superficieCopiaGPU,surfaceCopy,sizeof(float)*rows*columns,cudaMemcpyHostToDevice);
		/* If the global residual is lower than THRESHOLD, we have reached enough stability, stop simulation at the end of this iteration */
		
		/* 4.3. Move teams */
			
		//cudaMemcpy(teams,equiposGPU,sizeof(Team)*num_teams,cudaMemcpyDeviceToHost);
		
	
   
		/* 4.4. Team actions */
			for( t=0; t<num_teams; t++ ) {
			

			/* 4.4.2. Reduce heat in a circle around the team */
			int radius;
			// Influence area of fixed radius depending on type
			if ( teams[t].type == 1 ) {radius = RADIUS_TYPE_1;
				dim3 gridRed(1,1);
				dim3 blockRed(7,7);
				kernelApagarFuego<<<gridRed,blockRed>>>(superficieGPU,equiposGPU,t,radius,rows,columns);
			}
			else {radius = RADIUS_TYPE_2_3;
			dim3 gridRed(1,1);
			dim3 blockRed(19,19);
			kernelApagarFuego<<<gridRed,blockRed>>>(superficieGPU,equiposGPU,t,radius,rows,columns);
			}
		}
	
		
		#ifdef DEBUG
				/* 4.5. DEBUG: Print the current state of the simulation at the end of each iteration */
			cudaMemcpy(surface,superficieGPU,sizeof(float)*rows*columns,cudaMemcpyDeviceToHost);
			print_status( iter, rows, columns, surface, num_teams, teams, num_focal, focal, global_residual );
		#endif // DEBUG

	}
 cudaMemcpy(surface,superficieGPU,sizeof(float)*rows*columns,cudaMemcpyDeviceToHost);
 
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
 