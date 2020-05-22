// User: g213@157.88.125.142 
// ExecutionRequest[P:'LaBuena.cu',P:1,T:1,args:'',q:'cudalb'] 
// May 16 2019 14:01:02
#include "cputils.h" // Added by tablon
 /* Simplified simulation of fire extinguishing
 *
 * Computacion Paralela, Grado en Informatica (Universidad de Valladolid)
 * 2018/2019
 *
 * v1.4
 *
 * (c) 2019 Arturo Gonzalez Escribano
 * Cambios producidos por Rodrigo Beltran y Alvaro Villa
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <cputils.h>
#define accessMat( arr, exp1, exp2 )    arr[ (exp1) * columns + (exp2) ]
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
} focoPoint;

/* Macro function to simplify accessing with two coordinates to a flattened array */

//Funcion que se encarga de inicializar las matrices que le pasamos de parametros.
__global__ void inicializar(float *surface,int rows,int columns){
	int x=threadIdx.y+blockDim.y*blockIdx.y;
	int y=threadIdx.x+blockDim.x*blockIdx.x;
	if (x >= rows || y>= columns){
	       	return;
	}
	surface[x*columns+y]=0;
}

//Funcion que se encarga de actualizar los valores de la matriz. Calcula el calor
//y lo introduce en la matriz principal
__global__ void actualizar(float *surface, float *surfaceCopy,int rows, int columns){
        int x=threadIdx.y+blockDim.y*blockIdx.y;
        int y=threadIdx.x+blockDim.x*blockIdx.x;
        if (x >= rows-1 || x==0 || y>= columns-1 || y==0) return;
        surface[x*columns+y]=(
                surfaceCopy[(x-1)*columns+y]+
                surfaceCopy[(x+1)*columns+y]+
                surfaceCopy[x*columns+y-1]+
                surfaceCopy[x*columns+y+1])/4;
}

//Esta funcion se encarga de propagar el calor (actualizandolo)
__global__ void propagar(float *surface,int i,int j, int columns , int heat){
	surface[i*columns+j]=heat;
}

//Esta funcion se encarga de reducir la cantidad de calor (apaga) para los equipos 2 y 3 
__global__ void apagarEquipo23(float *surface,int x,int y,int radius,int rows,int columns){
	int i=x-radius+threadIdx.y;
	int j=y-radius+threadIdx.x;
	if(threadIdx.x>=19 || threadIdx.y >=19) return;
		if ( i<1 || i>=rows-1 || j<1 || j>=columns-1  ) return; // Out of the heated surface
		float dx = x - i;
		float dy = y - j;
		float distance = dx*dx + dy*dy ;
		if ( distance <= radius*radius ) {
			surface[i*columns+j]*=0.75; // Team efficiency factor
		}


}

//Esta funcion se encarga de reducir la cantidad de calor (apaga) para el equipo 1
__global__ void apagarEquipo1(float *surface,int x,int y,int radius,int rows,int columns){
	int i=x-radius+threadIdx.y;
	int j=y-radius+threadIdx.x;
	if(threadIdx.x>=7 || threadIdx.y >=7) return;
		if ( i<1 || i>=rows-1 || j<1 || j>=columns-1  ) return; // Out of the heated surface
		float dx = x - i;
		float dy = y - j;
		float distance = dx*dx + dy*dy ;
		if ( distance <= radius*radius ) {
			surface[i*columns+j]*=0.75; // Team efficiency factor
		}
}

//Esta funcion es la indicada para mover los equipos a lo largo de la matriz para 
//que se arrimen al foco y apaguen el fuego
__global__ void moverEquipos(Team *teams,focoPoint *focal, int num_teams,int num_focal){
		int j;
		int id=threadIdx.x+blockDim.x*blockIdx.x;
		if(id>num_teams){
		       	return;
		}
		float distance = FLT_MAX;
		int target = -1;
		for( j=0; j<num_focal; j++ ) {
			if ( focal[j].active != 1 ) continue; // Skip non-active focal points
			float dx = focal[j].x - teams[id].x;
			float dy = focal[j].y - teams[id].y;
			float local_distance = sqrtf( dx*dx + dy*dy );
			if ( local_distance < distance ) {
				distance = local_distance;
				target = j;
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
		}
		else if ( teams[id].type == 2 ) {
			// Type 2: First in horizontal direction, then in vertical direction
			if ( focal[target].y < teams[id].y ) teams[id].y--;
			else if ( focal[target].y > teams[id].y ) teams[id].y++;
			else if ( focal[target].x < teams[id].x ) teams[id].x--;
			else if ( focal[target].x > teams[id].x ) teams[id].x++;
		}
		else {
			// Type 3: First in vertical direction, then in horizontal direction
			if ( focal[target].x < teams[id].x ) teams[id].x--;
			else if ( focal[target].x > teams[id].x ) teams[id].x++;
			else if ( focal[target].y < teams[id].y ) teams[id].y--;
			else if ( focal[target].y > teams[id].y ) teams[id].y++;
		}

}

/*
 * Function: Print usage line in stderr
 */

void show_usage( char *program_name ) {
	fprintf(stderr,"Usage: %s <config_file> | <command_line_args>\n", program_name );
	fprintf(stderr,"\t<config_file> ::= -f <file_name>\n");
	fprintf(stderr,"\t<command_line_args> ::= <rows> <columns> <maxIter> <numTeams> [ <teamX> <teamY> <teamType> ... ] <numfocoPoints> [ <focalX> <focalY> <focalStart> <focalTemperature> ... ]\n");
	fprintf(stderr,"\n");
}

#ifdef DEBUG
/*
 * Function: Print the current state of the simulation
 */
void print_status( int iteration, int rows, int columns, float *surface, int num_teams, Team *teams, int num_focal, focoPoint *focal, float global_residual ) {
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
	focoPoint *focal;

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

		/* 1.2.4. foco points information */
		ok = fscanf(args, "%d", &num_focal );
		if ( ok != 1 ) {
			fprintf(stderr,"-- Error in file: reading num_focal from file: %s\n", argv[1]);
			exit( EXIT_FAILURE );
		}
		focal = (focoPoint *)malloc( sizeof(focoPoint) * (size_t)num_focal );
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

		/* 1.3.4. foco points information */
		int focal_args = 5 + i*3;
		if ( argc < focal_args+1 ) {
			fprintf(stderr,"-- Error in arguments: not enough arguments for the number of focal points\n");
			show_usage( argv[0] );
			exit( EXIT_FAILURE );
		}
		num_focal = atoi( argv[focal_args] );
		focal = (focoPoint *)malloc( sizeof(focoPoint) * (size_t)num_focal );
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
	printf("Arguments, Teams: %d, foco points: %d\n", num_teams, num_focal );
	for( i=0; i<num_teams; i++ ) {
		printf("\tTeam %d, position (%d,%d), type: %d\n", i, teams[i].x, teams[i].y, teams[i].type );
	}
	for( i=0; i<num_focal; i++ ) {
		printf("\tfoco_point %d, position (%d,%d), start time: %d, temperature: %d\n", i,
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

	float *Surface, *SurfaceCopy;
	focoPoint *foco;
	Team *equipo;
	 
	cudaMalloc((void **)&Surface,sizeof(float)*rows*columns);
	cudaMalloc((void **) &SurfaceCopy,sizeof(float)*rows*columns);
	cudaMalloc((void **) &foco,sizeof(focoPoint)*num_focal);
	cudaMalloc((void **) &equipo,sizeof(Team)*num_teams);
	cudaMemcpy(equipo,teams,sizeof(Team)*num_teams,cudaMemcpyHostToDevice);
	
	//Establecemos los valores de los tamanos de bloque
	int tBlockX= 64;
	int tBlockY= 4;
	int tBlockTeams=256;

	int tamGrx, tamGry,tamGridTeams;

	tamGridTeams= num_teams/tBlockTeams;
	if (num_teams%tBlockTeams!=0) tamGridTeams++;

	tamGrx= columns/tBlockX;
	if (columns%tBlockX!=0) tamGrx++;
	tamGry= rows/tBlockY;
	if (rows%tBlockY!=0) tamGry++;

	//Establecemos los tamanos del grid y de bloque
	dim3 gridSize(tamGrx,tamGry);
	dim3 blockSize(tBlockX,tBlockY);
	
	//Aqui inicializamos las matrices principales y la copia llamando a la funcion que inicializa
        //asi que posteriormente ya nos ahorramos volver a inicializar.	
	inicializar<<<blockSize,gridSize>>>(Surface,rows,columns);
	inicializar<<<blockSize,gridSize>>>(SurfaceCopy,rows,columns);
	
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

		for( step=0; step<10; step++ )	{
			/* 4.2.1. Update heat on active focal points */
			for( i=0; i<num_focal; i++ ) {
				if ( focal[i].active != 1 ) continue;
				int x = focal[i].x;
				int y = focal[i].y;
				propagar<<<1,1>>>(Surface,x,y,columns,focal[i].heat);
			}


			/* 4.2.2. Copy values of the surface in ancillary structure (Skip borders) */
			float *aux=Surface;
			Surface=SurfaceCopy;
			SurfaceCopy=aux;

			/* 4.2.3. Update surface values (skip borders) */
			actualizar<<<gridSize,blockSize>>>(Surface,SurfaceCopy,rows,columns);

			/* 4.2.4. Compute the maximum residual difference (absolute value) */

			if(step==0 && num_deactivated==num_focal){
				cudaMemcpy(surface,Surface,sizeof(float)*rows*columns,cudaMemcpyDeviceToHost);
				cudaMemcpy(surfaceCopy,SurfaceCopy,sizeof(float)*rows*columns,cudaMemcpyDeviceToHost);
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

			cudaMemcpy(foco,focal,sizeof(focoPoint)*num_focal,cudaMemcpyHostToDevice);
			moverEquipos<<<tamGridTeams,tBlockTeams>>>(equipo,foco,num_teams,num_focal);
			cudaMemcpy(teams,equipo,sizeof(Team)*num_teams,cudaMemcpyDeviceToHost);

		/* 4.4. Team actions */
		for( t=0; t<num_teams; t++ ) {
			/* 4.4.1. Deactivate the target focal point when it is reached */
			int target = teams[t].target;
			if ( target != -1 && focal[target].x == teams[t].x && focal[target].y == teams[t].y
				&& focal[target].active == 1 )
				focal[target].active = 2;

			/* 4.4.2. Reduce heat in a circle around the team */
			int radius;
			if ( teams[t].type == 1 ) {radius = RADIUS_TYPE_1;
				dim3 gridRed(1,1);
				dim3 blockRed(8,8);
				apagarEquipo1<<<gridRed,blockRed>>>(Surface,teams[t].x,teams[t].y,radius,rows,columns);
			}
			else {radius = RADIUS_TYPE_2_3;
			dim3 gridRed(1,1);
			dim3 blockRed(32,32);
			apagarEquipo23<<<gridRed,blockRed>>>(Surface,teams[t].x,teams[t].y,radius,rows,columns);
			}
		}

		#ifdef DEBUG
				/* 4.5. DEBUG: Print the current state of the simulation at the end of each iteration */
				print_status( iter, rows, columns, surface, num_teams, teams, num_focal, focal, global_residual );
		#endif // DEBUG

	}
	cudaMemcpy(surface,Surface,sizeof(float)*rows*columns,cudaMemcpyDeviceToHost);
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
