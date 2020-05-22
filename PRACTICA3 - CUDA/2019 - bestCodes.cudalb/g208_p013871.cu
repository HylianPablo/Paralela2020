// User: g208@88.5.104.77 
// ExecutionRequest[P:'extinguishing.cu',P:1,T:1,args:'',q:'cudalb'] 
// May 16 2019 18:11:20
#include "cputils.h" // Added by tablon
/*
 * Simplified simulation of fire extinguishing
 *
 * Computacion Paralela, Grado en Informatica (Universidad de Valladolid)
 * 2018/2019
 *
 * Crhistian de la Puerta Verdejo
 * Aitor Ojeda Bilbao
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
	int inicio;
	int calor;
	int activo; // States: 0 Not yet activated; 1 activo; 2 Deactivated by a team
} FocalPoint;

/* Macro function to simplify accessing with two coordinates to a flattened array */
#define accessMat( arr, exp1, exp2 )	arr[ (exp1) * columns + (exp2) ]

/*cada vez que se ejecuta el programa, va comprobando cada posicion y la actualiza*/
__global__ void actualiza_interfaz(float *surface, float *surfaceCopy,int rows, int columns){
	int idX=threadIdx.y+blockDim.y*blockIdx.y;
	int idY=threadIdx.x+blockDim.x*blockIdx.x;
  
	if (idX >= rows-1 || idX==0 || idY>= columns-1 || idY==0) return; // final de la matriz

	/*lo calculamos en una auxiliar para actualizar*/
	surface[idX*columns+idY]=(
		surfaceCopy[(idX-1)*columns+idY]+
		surfaceCopy[(idX+1)*columns+idY]+
		surfaceCopy[idX*columns+idY-1]+
		surfaceCopy[idX*columns+idY+1])/4;
	
}

/*calcula el numero de equipos y su presencia en los focos y como se actualiza su posicion segun se actualiza la matriz*/
__global__ void movimiento_equipos(Team *teams,FocalPoint *focal, int num_teams,int num_focal){

		int id=threadIdx.x+blockDim.x*blockIdx.x;
		int j;
		if(id>num_teams) return;

		float distancia = FLT_MAX;
		int target = -1;
		for( j=0; j<num_focal; j++ ) {
			if ( focal[j].activo != 1 ) continue; // termina si no hay focos activos
			float dx = focal[j].x - teams[id].x;
			float dy = focal[j].y - teams[id].y;
			float local_distance = sqrtf( dx*dx + dy*dy );
			if ( local_distance < distancia ) {
				distancia = local_distance;
				target = j;
			}
		}
		/* 4.3.2. Annotate target for the next stage */
		teams[id].target = target;

		/* 4.3.3. No activo focal point to choose, no movement */
		if ( target == -1 ) return;

		/* 4.3.4. Move in the focal point direction */
		if ( teams[id].type == 1 ) {
			// 1: se puede mover en diagonal
			if ( focal[target].x < teams[id].x ) teams[id].x--;
			if ( focal[target].x > teams[id].x ) teams[id].x++;
			if ( focal[target].y < teams[id].y ) teams[id].y--;
			if ( focal[target].y > teams[id].y ) teams[id].y++;
		}
		else if ( teams[id].type == 2 ) {
			//2: primero se mueve en horizontal y despues vertical
			if ( focal[target].y < teams[id].y ) teams[id].y--;
			else if ( focal[target].y > teams[id].y ) teams[id].y++;
			else if ( focal[target].x < teams[id].x ) teams[id].x--;
			else if ( focal[target].x > teams[id].x ) teams[id].x++;
		}
		else {
			// 3: primero en vertical, luego en horizontal
			if ( focal[target].x < teams[id].x ) teams[id].x--;
			else if ( focal[target].x > teams[id].x ) teams[id].x++;
			else if ( focal[target].y < teams[id].y ) teams[id].y--;
			else if ( focal[target].y > teams[id].y ) teams[id].y++;
		}

}

/*actualiza la posiciones de los focos de calor*/
__global__ void actualiza_calor(float *surface,int i,int j, int columns , int calor){
	surface[i*columns+j]=calor;
}



/*va reduciendo la cantidad de calor de los focos en funcion de la eficiencia de los equipos de extincion*/
__global__ void reduce_calor(float *surface,int x,int y,int radius,int rows,int columns){

	int i=x-radius+threadIdx.y;
	int j=y-radius+threadIdx.x;
	if(threadIdx.x>18 || threadIdx.y >18) return; // si las posiciones son mayores a 18 termina


		if ( i<1 || i>=rows-1 || j<1 || j>=columns-1  ) return; //sale del rango de la matriz
		float dx = x - i;
		float dy = y - j;
		float distancia = dx*dx + dy*dy ;
		if ( distancia <= radius*radius ) {
			surface[i*columns+j]*=0.75; // factor de eficiencia del equipo de extincion
		}


}

/*reducion auxiliar del calor de los focos de incendio*/
__global__ void reduce_calorAuxiliar(float *surface,int x,int y,int radius,int rows,int columns){

	int i=x-radius+threadIdx.y;
	int j=y-radius+threadIdx.x;
	if(threadIdx.x>6 || threadIdx.y >6) return; //si los valores son mayores que 6 termina

		if ( i<1 || i>=rows-1 || j<1 || j>=columns-1  ) return; // Out of the heated surface
		float dx = x - i;
		float dy = y - j;
		float distancia = dx*dx + dy*dy ;
		if ( distancia <= radius*radius ) {
			surface[i*columns+j]*=0.75; // factor de eficiencia de los equipos
		}


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
					if ( focal[f].x == i && focal[f].y == j && focal[f].activo == 1 ) { flag_focal = 1; break; }
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
			ok = fscanf(args, "%d %d %d %d", &focal[i].x, &focal[i].y, &focal[i].inicio, &focal[i].calor);
			if ( ok != 4 ) {
				fprintf(stderr,"-- Error in file: reading focal point %d from file: %s\n", i, argv[1]);
				exit( EXIT_FAILURE );
			}
			focal[i].activo = 0;
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
			focal[i].inicio = atoi( argv[focal_args+i*4+3] );
			focal[i].calor = atoi( argv[focal_args+i*4+4] );
			focal[i].activo = 0;
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
		printf("\tFocal_point %d, position (%d,%d), inicio time: %d, temperature: %d\n", i,
		focal[i].x,
		focal[i].y,
		focal[i].inicio,
		focal[i].calor );
	}
#endif

	/* 2. Select GPU and inicio global timer */
	cudaSetDevice(0);
	cudaDeviceSynchronize();
	double ttotal = cp_Wtime();

/*
 *
 * START HERE: DO NOT CHANGE THE CODE ABOVE THIS POINT
 *
 */


	float *gpuMatriz, *gpuMatrizAux;
	FocalPoint *gpuFocal;
	Team *gpuEquipo;

	cudaMalloc((void **)&gpuMatriz,sizeof(float)*rows*columns);
	
	cudaMalloc((void **) &gpuMatrizAux,sizeof(float)*rows*columns);
	
	cudaMalloc((void **) &gpuEquipo,sizeof(Team)*num_teams);
	
	cudaMemcpy(gpuEquipo,teams,sizeof(Team)*num_teams,cudaMemcpyHostToDevice);

	cudaMalloc((void **) &gpuFocal,sizeof(FocalPoint)*num_focal);
	
	int tamBloqueX= 64;
	int tamBloqueY= 4;
	int tamGridX, tamGridY;
	int tamBloqueEquipos=256;
	int tamGridEquipos;

	tamGridEquipos= num_teams/tamBloqueEquipos;
	if (num_teams%tamBloqueEquipos!=0) tamGridEquipos++; 
	tamGridX= columns/tamBloqueX;
	if (columns%tamBloqueX!=0) tamGridX++; 
	tamGridY= rows/tamBloqueY;
	if (rows%tamBloqueY!=0) tamGridY++; 

	dim3 blockSize(tamBloqueX,tamBloqueY);
	dim3 gridSize(tamGridX,tamGridY);
	#ifdef DEBUG
	printf("tamGx %d tamGy %d\n",tamGridX,tamGridY);
	#endif


	/* 3. Initialize surface */
	for( i=0; i<rows; i++ )
		for( j=0; j<columns; j++ )
			accessMat( surface, i, j ) = 0.0;

	/* 4. Simulation */
	int iter;
	int flag_stability = 0;
	int first_activation = 0;
	for( iter=0; iter<max_iter && ! flag_stability; iter++ ) {

		/* 4.1. Activate focal points */
		int num_deactivated = 0;
		for( i=0; i<num_focal; i++ ) {
			if ( focal[i].inicio == iter ) {
				focal[i].activo = 1;
				if ( ! first_activation ) first_activation = 1;
			}

			if ( focal[i].activo == 2 ) num_deactivated++;
		}

		/* 4.2. Propagate heat (10 steps per each team movement) */
		float global_residual = 0.0f;
		int step;


		for( step=0; step<10; step++ )	{
			/* 4.2.1. Update heat on activo focal points */
			for( i=0; i<num_focal; i++ ) {
				if ( focal[i].activo != 1 ) continue;
				int x = focal[i].x;
				int y = focal[i].y;
				actualiza_calor<<<1,1>>>(gpuMatriz,x,y,columns,focal[i].calor);
				//cudaGetLastError();
				
			
			}


			/* 4.2.2. Copy values of the surface in ancillary structure (Skip borders) */
		
			float *aux=gpuMatriz;
			gpuMatriz=gpuMatrizAux;
			gpuMatrizAux=aux;

			/* 4.2.3. Update surface values (skip borders) */
			actualiza_interfaz<<<gridSize,blockSize>>>(gpuMatriz,gpuMatrizAux,rows,columns);
			//cudaGetLastError();
			

			/* 4.2.4. Compute the maximum residual difference (absolute value) */

			if(step==0 && num_deactivated==num_focal){
				cudaMemcpy(surface,gpuMatriz,sizeof(float)*rows*columns,cudaMemcpyDeviceToHost);
				
				cudaMemcpy(surfaceCopy,gpuMatrizAux,sizeof(float)*rows*columns,cudaMemcpyDeviceToHost);
				
			for( i=1; i<rows-1; i++ )
				for( j=1; j<columns-1; j++ )
					if ( fabs( accessMat( surface, i, j ) - accessMat( surfaceCopy, i, j ) ) > global_residual ) {
						global_residual = fabs( accessMat( surface, i, j ) - accessMat( surfaceCopy, i, j ) );
					}
			}

		}

		/* If the global residual is lower than THRESHOLD, we have reached enough stability, stop simulation at the end of this iteration */
		if( num_deactivated == num_focal && global_residual < THRESHOLD ) flag_stability = 1;
		/* 4.3. Move teams */
		
			cudaMemcpy(gpuFocal,focal,sizeof(FocalPoint)*num_focal,cudaMemcpyHostToDevice);
			movimiento_equipos<<<tamGridEquipos,tamBloqueEquipos>>>(gpuEquipo,gpuFocal,num_teams,num_focal);
			cudaGetLastError();
			

		cudaMemcpy(teams,gpuEquipo,sizeof(Team)*num_teams,cudaMemcpyDeviceToHost);
		

		/* 4.4. Team actions */
		
		for( t=0; t<num_teams; t++ ) {
			/* 4.4.1. Deactivate the target focal point when it is reached */
			int target = teams[t].target;
			if ( target != -1 && focal[target].x == teams[t].x && focal[target].y == teams[t].y
				&& focal[target].activo == 1 )
				focal[target].activo = 2;

			/* 4.4.2. Reduce calor in a circle around the team */
			int radius;
			
			if ( teams[t].type == 1 ) {radius = RADIUS_TYPE_1;
				dim3 gridRed(1,1);
				dim3 blockRed(8,8);
				reduce_calorAuxiliar<<<gridRed,blockRed>>>(gpuMatriz,teams[t].x,teams[t].y,radius,rows,columns);
			}
			else {radius = RADIUS_TYPE_2_3;
			dim3 gridRed(1,1);
			dim3 blockRed(32,32);
			reduce_calor<<<gridRed,blockRed>>>(gpuMatriz,teams[t].x,teams[t].y,radius,rows,columns);
			}
		}

		#ifdef DEBUG
				/* 4.5. DEBUG: Print the current state of the simulation at the end of each iteration */
				cudaMemcpy(surface,gpuMatriz,sizeof(float)*rows*columns,cudaMemcpyDeviceToHost);
				print_status( iter, rows, columns, surface, num_teams, teams, num_focal, focal, global_residual );
		#endif

	}
	cudaMemcpy(surface,gpuMatriz,sizeof(float)*rows*columns,cudaMemcpyDeviceToHost);
	

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
	/* 6.2. Results: Number of iterations, position of teams, residual calor on the focal points */
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
