// User: g101@157.88.125.192 
// ExecutionRequest[P:'pepelotas.cu',P:1,T:1,args:'',q:'cudalb'] 
// May 16 2019 13:43:40
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

 __global__ void copia(float *surfaceGPU, float *surfaceCopyGPU,int rows,int columns){

	int id = (blockIdx.y * blockDim.y + threadIdx.y) * gridDim.x * blockDim.x + blockIdx.x*blockDim.x+threadIdx.x;
    int i = id/columns;
    int j = id%columns;

	if (id<(rows*columns)  && j<columns-1 && j > 0 && i > 0 && i<rows-1){
		surfaceCopyGPU[id]=surfaceGPU[id];
	}
}

//Funcion para reducir el calor. (i,j):Coordenadas del equipo antincendios. radius:Radio de accion.
__device__ float atomicMul(float* address, float val)
{
  int* address_as_int = (int*)address;
  int old = *address_as_int, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_int, assumed, __float_as_int(val *
__int_as_float(assumed)));
 } while (assumed != old); return __int_as_float(old);
}

__global__ void updateheat(float *surfaceGPU, FocalPoint *focalGPU,int rows,int columns, int num_focal){

	int id = (blockIdx.y * blockDim.y + threadIdx.y) * gridDim.x * blockDim.x + blockIdx.x*blockDim.x+threadIdx.x;

    if (id < num_focal){
        if ( focalGPU[id].active == 1 ) {
            int x = focalGPU[id].x;
            int y = focalGPU[id].y;
            accessMat(surfaceGPU,x,y) = focalGPU[id].heat;
        }
    }
}

__global__ void inicializa(float *surfaceGPU,float *surfaceCopyGPU,int rows,int columns){
    int id = (blockIdx.y * blockDim.y + threadIdx.y) * gridDim.x * blockDim.x + blockIdx.x*blockDim.x+threadIdx.x;
    int tamano = rows*columns;
	if (id<tamano){
        surfaceGPU[id]=0.0f;
		surfaceCopyGPU[id]=0.0f;
	}
}

__global__ void partefinal(float *surfaceGPU,float *surfaceCopyGPU,int rows,int columns,Team *teamsGPU, FocalPoint *focalGPU, int num_teams){
    int id = (blockIdx.y * blockDim.y + threadIdx.y) * gridDim.x * blockDim.x + blockIdx.x*blockDim.x+threadIdx.x;

    if (id<num_teams){
      int target = teamsGPU[id].target;
      if ( target != -1 && focalGPU[target].x == teamsGPU[id].x && focalGPU[target].y == teamsGPU[id].y
          && focalGPU[target].active == 1 )
          focalGPU[target].active = 2;

      // 4.4.2. Reduce heat in a circle around the team
      int radius;
      // Influence area of fixed radius depending on type
      if ( teamsGPU[id].type == 1 ) radius = RADIUS_TYPE_1;
      else radius = RADIUS_TYPE_2_3;


      for(int i=teamsGPU[id].x-radius; i<=teamsGPU[id].x+radius; i++ ) {
          for(int j=teamsGPU[id].y-radius; j<=teamsGPU[id].y+radius; j++ ) {
              if ( i<1 || i>=rows-1 || j<1 || j>=columns-1 ) continue; // Out of the heated surface

              float distance = (teamsGPU[id].x - i)*(teamsGPU[id].x - i) + (teamsGPU[id].y - j)*(teamsGPU[id].y - j) ;
              if ( distance <= radius*radius ) {
                  atomicMul(&surfaceGPU[i*(columns)+j],0.75);
                  //accessMat( surfaceGPU, i, j ) = accessMat( surfaceGPU, i, j ) * ( 0.75 ); // Team efficiency factor
              }
          }
      }
  }
}


__global__ void itera(float *surfaceGPU,float *surfaceCopyGPU,int filas, int columns){
	int id = (blockIdx.y * blockDim.y + threadIdx.y) * gridDim.x * blockDim.x + blockIdx.x*blockDim.x+threadIdx.x;
	//int tamano = filas * columns;

    int i = id/columns;
	int j = id%columns;

	if (i > 0 && i<filas-1 && j>0 && j<columns-1) {
		accessMat( surfaceGPU, i, j ) = (
			accessMat( surfaceCopyGPU, i-1, j ) +
			accessMat( surfaceCopyGPU, i+1, j ) +
			accessMat( surfaceCopyGPU, i, j-1 ) +
			accessMat( surfaceCopyGPU, i, j+1 ) ) / 4;
	}
}

template <unsigned int blockSize>
__device__ void warpReduce(volatile float* sdata,int tid) {
    if (blockSize >= 64)  sdata[tid] = fmax(sdata[tid],sdata[tid + 32]);
    if (blockSize >= 32) sdata[tid] = fmax(sdata[tid],sdata[tid + 16]);
    if (blockSize >= 16) sdata[tid] = fmax(sdata[tid],sdata[tid +  8]);
    if (blockSize >= 8) sdata[tid] = fmax(sdata[tid],sdata[tid +  4]);
    if (blockSize >= 4) sdata[tid] = fmax(sdata[tid],sdata[tid +  2]);
    if (blockSize >= 2) sdata[tid] = fmax(sdata[tid],sdata[tid +  1]);
}

template <unsigned int blockSize>
__global__ void reduce_kernel(float *surfaceGPU,float *surfaceCopyGPU,float *global_residualGPU ,int filas, int columns)
{
    int gid = (blockIdx.y * blockDim.y + threadIdx.y) * gridDim.x * blockDim.x + blockIdx.x*blockDim.x+threadIdx.x;

    int i = gid/columns;
	int j = gid%columns;
	// Memoria shared
    extern __shared__ float tmp[];

	int blq = blockIdx.x+blockIdx.y*gridDim.x;
    // Cargar dato en memoria shared
    int tid = threadIdx.x+threadIdx.y*blockDim.x;


	// Desactivar hilos que excedan los límites del array de entrada
	if ( gid < (filas*columns) ) {

        float resta1 = surfaceGPU[gid] - surfaceCopyGPU[gid];
        float resta2 = surfaceGPU[gid+columns] - surfaceCopyGPU[gid+columns];

        if (resta1<0) {
            resta1 = resta1*(-1);
        }
        if (resta2<0) {
            resta2 = resta2*(-1);
        }
        tmp[ tid ] = fmax(resta1,resta2);
        }

    else{
        tmp[tid] = 0.0;
        }

		__syncthreads();

        // Hacemos la reducción en memoria shared
        for(unsigned int s=blockDim.x*blockDim.y/2; s>32; s>>=1) {
            if (tid < s) {
                //sdata[tid] += sdata[tid + s];
                tmp[tid] = fmax(tmp[tid],tmp[tid+s]);
            }
            __syncthreads();
        }
        if (tid < 32) {
            warpReduce<blockSize>(tmp, tid);
        }
        if (blockSize >= 512) {
            if (tid < 256) {
                tmp[tid] = fmax(tmp[tid],tmp[tid+256]);

                }
                 __syncthreads();
            }
        if (blockSize >= 256) {
                if (tid < 128) {
                    tmp[tid] = fmax(tmp[tid],tmp[tid+128]);
                }
                __syncthreads();
            }
        if (blockSize >= 128) {
                if (tid <  64) {
                    tmp[tid] = fmax(tmp[tid],tmp[tid+64]);
                }
                 __syncthreads();
            }
        if (tid < 32)warpReduce<blockSize>(tmp, tid);
/*
v4
igual
        for(unsigned int s=blockDim.x*blockDim.y/2; s>32; s>>=1) {
            if (tid < s) {
                //sdata[tid] += sdata[tid + s];
                tmp[tid] = fmax(tmp[tid],tmp[tid+s]);
            }
            __syncthreads();
        }
        if (tid < 32) {
            warpReduce(tmp, tid);
        }

v3
igual o mejor

        for(unsigned int s=blockDim.x*blockDim.y/2; s>0; s>>=1) {
            if (tid < s) {
                //sdata[tid] += sdata[tid + s];
                tmp[tid] = fmax(tmp[tid],tmp[tid+s]);
            }
            __syncthreads();
        }
v2
esta va igual o algo mejor

        for(unsigned int s=1; s < blockDim.x*blockDim.y; s *= 2) {
            int index = 2 * s * tid;
            if (index < blockDim.x*blockDim.y) {
                tmp[index] = fmax(tmp[index],tmp[index+s]);
            }
            __syncthreads();
        }


v1
ESTA VA CHIDO
        for(unsigned int s=1; s < blockDim.x*blockDim.y; s *= 2) {
            if(tid % (2*s) == 0){
                tmp[tid] = fmax(tmp[tid],tmp[tid+s]);
            }
            __syncthreads();
        }
*/
		// El hilo 0 de cada bloque escribe el resultado final de la reducción
		// en la memoria global del dispositivo pasada por parámetro (g_odata[])
		if (tid == 0)
            global_residualGPU[blq] = tmp[0];



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
     float* surfaceGPU;
     float* surfaceCopyGPU;

     cudaError_t error ;

    error = cudaMalloc((void**) &surfaceGPU, sizeof(float) * (size_t)rows*(size_t)columns);
    if (error != cudaSuccess){
		printf("ERROR %s \n",cudaGetErrorString(error));
	}
    error = cudaMalloc((void**) &surfaceCopyGPU, sizeof(float) * (size_t)rows*(size_t)columns);
    if (error != cudaSuccess){
		printf("ERROR %s \n",cudaGetErrorString(error));
    }

    	int tamBlockX = 32;
	int tamBlockY = 8;
	int tamGridX,tamGridY;

	tamGridY = columns / tamBlockY ;
	tamGridX = rows / tamBlockX ;

    if (columns % tamBlockY!=0){
        tamGridY++;
    }
    if (rows % tamBlockX!=0){
        tamGridX++;
    }

	dim3 tamBloque(tamBlockX,tamBlockY);
    dim3 tamGrid(tamGridX,tamGridY);


     /* 3. Initialize surface */
    /*
     for( i=0; i<rows; i++ )
         for( j=0; j<columns; j++ )
             accessMat( surface, i, j ) = 0.0;
             */
   // cudaMemset(surfaceGPU, 0, sizeof(float) *(size_t)rows*(size_t)columns);
    //cudaMemset(surfaceCopyGPU, 0, sizeof(float) *(size_t)rows*(size_t)columns);
    //inicializa<<<tamGrid,tamBloque>>>(surfaceGPU,surfaceCopyGPU,rows,columns);

    /*
    error = cudaMemcpy(surface,surfaceGPU,sizeof(float) *rows*columns,cudaMemcpyDeviceToHost);
    if (error != cudaSuccess){
		printf("ERROR %s \n",cudaGetErrorString(error));
	}
    error = cudaMemcpy(surfaceCopy,surfaceCopyGPU,sizeof(float) * rows*columns,cudaMemcpyDeviceToHost);
    if (error != cudaSuccess){
		printf("ERROR %s \n",cudaGetErrorString(error));
	}
    */

     /* 4. Simulation */
     int iter;
     int flag_stability = 0;
     int first_activation = 0;

     FocalPoint *focalGPU;

     Team *teamsGPU;

     cudaMalloc((void**) &teamsGPU, sizeof(Team) * (size_t)num_teams);



     cudaMalloc((void**) &focalGPU, sizeof(FocalPoint) * (size_t)num_focal);

     float *global_residualGPU;

     cudaMalloc((void**) &global_residualGPU, sizeof(float) *  (size_t)tamGridX*(size_t)tamGridY);

     float *global_residualLocal = (float *)malloc( sizeof(float)  * (size_t)tamGridX*(size_t)tamGridY );
    
  
   int init_iter = 9999999;
  for( i=0; i<num_focal; i++ ) {
             if ( focal[i].start < init_iter ) {
                 init_iter = focal[i].start;
             }
     }

     for( iter=init_iter; ! flag_stability && iter<max_iter ; iter++ ) {

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

         cudaMemcpy(focalGPU,focal,sizeof(FocalPoint) * num_focal,cudaMemcpyHostToDevice);

         //cudaMemcpy(surfaceGPU,surface,sizeof(float) *rows*columns,cudaMemcpyHostToDevice);

         for( step=0; step<10; step++ )	{
             /* 4.2.1. Update heat on active focal points */




             updateheat<<<tamGrid,tamBloque>>>(surfaceGPU,focalGPU,rows,columns,num_focal);

            /* for( i=0; i<num_focal; i++ ) {
                 if ( focal[i].active != 1 ) continue;
                 int x = focal[i].x;
                 int y = focal[i].y;
                 accessMat( surface, x, y ) = focal[i].heat;
             }
             */

             /* 4.2.2. Copy values of the surface in ancillary structure (Skip borders) */
             /*

             for( i=1; i<rows-1; i++ )
                 for( j=1; j<columns-1; j++ )
                     accessMat( surfaceCopy, i, j ) = accessMat( surface, i, j );
                     */

             //copia<<<tamGrid,tamBloque>>>(surfaceGPU,surfaceCopyGPU,rows,columns);

             float *copia = surfaceGPU;
             surfaceGPU = surfaceCopyGPU;
             surfaceCopyGPU = copia;


             itera<<<tamGrid,tamBloque>>>(surfaceGPU,surfaceCopyGPU,rows,columns);






             /* 4.2.3. Update surface values (skip borders) */
            /*
             for( i=1; i<rows-1; i++ )
                 for( j=1; j<columns-1; j++ )
                     accessMat( surface, i, j ) = (
                         accessMat( surfaceCopy, i-1, j ) +
                         accessMat( surfaceCopy, i+1, j ) +
                         accessMat( surfaceCopy, i, j-1 ) +
                         accessMat( surfaceCopy, i, j+1 ) ) / 4;
 */

             /* 4.2.4. Compute the maximum residual difference (absolute value) */

            if (step== 0 && num_deactivated==num_focal ) {
                //reduce_kernel<<<tamGrid,tamBloque,sizeof(float)*tamBlockX*tamBlockY>>>(surfaceGPU,surfaceCopyGPU,global_residualGPU , rows,  columns);
                int blockSize = tamBlockX*tamBlockY;
                //reduce_kernel<<<tamGrid,tamBloque,sizeof(float)*tamBlockX*tamBlockY>>>(surfaceGPU,surfaceCopyGPU,global_residualGPU , rows,  columns);
                switch (tamBlockX*tamBlockY) {
                    case 1024:
                        reduce_kernel<1024><<<tamGrid,tamBloque,sizeof(float)*tamBlockX*tamBlockY>>>(surfaceGPU,surfaceCopyGPU,global_residualGPU , rows,  columns);
                        break;
                    case 512:
                        reduce_kernel<512><<<tamGrid,tamBloque,sizeof(float)*tamBlockX*tamBlockY>>>(surfaceGPU,surfaceCopyGPU,global_residualGPU , rows,  columns);
                        break;
                    case 256:
                        reduce_kernel<256><<<tamGrid,tamBloque,sizeof(float)*tamBlockX*tamBlockY>>>(surfaceGPU,surfaceCopyGPU,global_residualGPU , rows,  columns);
                        break;
                    case 128:
                        reduce_kernel<128><<<tamGrid,tamBloque,sizeof(float)*tamBlockX*tamBlockY>>>(surfaceGPU,surfaceCopyGPU,global_residualGPU , rows,  columns);
                        break;
                    case 64:
                        reduce_kernel<64><<<tamGrid,tamBloque,sizeof(float)*tamBlockX*tamBlockY>>>(surfaceGPU,surfaceCopyGPU,global_residualGPU , rows,  columns);
                        break;
                    case 32:
                        reduce_kernel<32><<<tamGrid,tamBloque,sizeof(float)*tamBlockX*tamBlockY>>>(surfaceGPU,surfaceCopyGPU,global_residualGPU , rows,  columns);
                        break;
                    case 16:
                        reduce_kernel<16><<<tamGrid,tamBloque,sizeof(float)*tamBlockX*tamBlockY>>>(surfaceGPU,surfaceCopyGPU,global_residualGPU , rows,  columns);
                        break;
                    case 8:
                        reduce_kernel<8><<<tamGrid,tamBloque,sizeof(float)*tamBlockX*tamBlockY>>>(surfaceGPU,surfaceCopyGPU,global_residualGPU , rows,  columns);
                        break;
                    case 4:
                        reduce_kernel<4><<<tamGrid,tamBloque,sizeof(float)*tamBlockX*tamBlockY>>>(surfaceGPU,surfaceCopyGPU,global_residualGPU , rows,  columns);
                        break;
                    case 2:
                        reduce_kernel<2><<<tamGrid,tamBloque,sizeof(float)*tamBlockX*tamBlockY>>>(surfaceGPU,surfaceCopyGPU,global_residualGPU , rows,  columns);
                        break;
                    case 1:
                        reduce_kernel<1><<<tamGrid,tamBloque,sizeof(float)*tamBlockX*tamBlockY>>>(surfaceGPU,surfaceCopyGPU,global_residualGPU , rows,  columns);
                        break;
                    }


                error = cudaGetLastError();
                if ( error != cudaSuccess )
                    printf("Error calcular maximo: %s\n", cudaGetErrorString( error ) );


                error = cudaMemcpy(global_residualLocal,global_residualGPU,sizeof(float) *tamGridX*tamGridY,cudaMemcpyDeviceToHost);
                if ( error != cudaSuccess )
                    printf("Error copiar maximo: %s\n", cudaGetErrorString( error ) );

                for (i = 0; i< (tamGridX*tamGridY); i++){
			if (global_residual>=THRESHOLD){
			break;
			}
                    if (global_residualLocal[i]> global_residual){
                        global_residual = global_residualLocal[i];
                    }
                }
            }


			/*for( i=1; i<rows-1; i++ )
				for( j=1; j<columns-1; j++ )
					if ( fabs( accessMat( surface, i, j ) - accessMat( surfaceCopy, i, j ) ) > global_residual ) {
						global_residual = fabs( accessMat( surface, i, j ) - accessMat( surfaceCopy, i, j ) );
					}*/


         }

      //   cudaMemcpy(surface,surfaceGPU,sizeof(float) *rows*columns,cudaMemcpyDeviceToHost);

         /* If the global residual is lower than THRESHOLD, we have reached enough stability, stop simulation at the end of this iteration */
         if( num_deactivated == num_focal && global_residual < THRESHOLD ) flag_stability = 1;

         /* 4.3. Move teams */
         if((num_deactivated!=num_focal || !first_activation))
         for( t=0; t<num_teams; t++ ) {
             /* 4.3.1. Choose nearest focal point */
             float distance = FLT_MAX;
             int target = -1;
             for( j=0; j<num_focal; j++ ) {
                 if ( focal[j].active != 1 ) continue; // Skip non-active focal points
                    float local_distance  = ( (focal[j].x - teams[t].x)*(focal[j].x - teams[t].x) + (focal[j].y - teams[t].y)*(focal[j].y - teams[t].y) );

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
         cudaMemcpy(teamsGPU,teams ,sizeof(Team) *num_teams,cudaMemcpyHostToDevice);
         partefinal<<<tamGrid,tamBloque>>>(surfaceGPU,surfaceCopyGPU, rows, columns, teamsGPU, focalGPU, num_teams);
         cudaMemcpy(focal,focalGPU,sizeof(FocalPoint) *num_focal,cudaMemcpyDeviceToHost);

         /*
         for( t=0; t<num_teams; t++ ) {
             //4.4.1. Deactivate the target focal point when it is reached
             int target = teams[t].target;
             if ( target != -1 && focal[target].x == teams[t].x && focal[target].y == teams[t].y
                 && focal[target].active == 1 )
                 focal[target].active = 2;

             // 4.4.2. Reduce heat in a circle around the team
             int radius;
             // Influence area of fixed radius depending on type
             if ( teams[t].type == 1 ) radius = RADIUS_TYPE_1;
             else radius = RADIUS_TYPE_2_3;


             for( i=teams[t].x-radius; i<=teams[t].x+radius; i++ ) {
                 for( j=teams[t].y-radius; j<=teams[t].y+radius; j++ ) {
                     if ( i<1 || i>=rows-1 || j<1 || j>=columns-1 ) continue; // Out of the heated surface

                     float distance = (teams[t].x - i)*(teams[t].x - i) + (teams[t].y - j)*(teams[t].y - j) ;
                     if ( distance <= radius*radius ) {
                         accessMat( surface, i, j ) = accessMat( surface, i, j ) * ( 0.75 ); // Team efficiency factor
                     }
                 }
             }
         }*/

 #ifdef DEBUG
         /* 4.5. DEBUG: Print the current state of the simulation at the end of each iteration */
         print_status( iter, rows, columns, surface, num_teams, teams, num_focal, focal, global_residual );
 #endif // DEBUG
     }
     cudaMemcpy(surface,surfaceGPU,sizeof(float) *rows*columns,cudaMemcpyDeviceToHost);

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
