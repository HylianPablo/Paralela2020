// User: g111@83.42.87.254 
// ExecutionRequest[P:'OrodeaAlLago.c',P:1,T:1,args:'',q:'mpilb'] 
// Apr 03 2019 22:42:52
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
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>
#include<float.h>
#include<mpi.h>
#include"cputils.h"

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
 * MAIN PROGRAM
 */
int main(int argc, char *argv[]) {
	int i,j,t;

	// Simulation data
	int rows, columns, max_iter;
	int num_teams, num_focal;
	Team *teams;
	FocalPoint *focal;

	/* 0. Initialize MPI */
	int rank;
	MPI_Init( &argc, &argv );
	MPI_Comm_rank( MPI_COMM_WORLD, &rank );

	/* 1. Read simulation arguments */
	/* 1.1. Check minimum number of arguments */
	if (argc<2) {
		fprintf(stderr,"-- Error in arguments: No arguments\n");
		show_usage( argv[0] );
		MPI_Abort( MPI_COMM_WORLD,  EXIT_FAILURE );
	}

	int read_from_file = ! strcmp( argv[1], "-f" );
	/* 1.2. Read configuration from file */
	if ( read_from_file ) {
		/* 1.2.1. Open file */
		if (argc<3) {
			fprintf(stderr,"-- Error in arguments: file-name argument missing\n");
			show_usage( argv[0] );
			MPI_Abort( MPI_COMM_WORLD,  EXIT_FAILURE );
		}
		FILE *args = cp_abrir_fichero( argv[2] );
		if ( args == NULL ) {
			fprintf(stderr,"-- Error in file: not found: %s\n", argv[1]);
			MPI_Abort( MPI_COMM_WORLD,  EXIT_FAILURE );
		}	

		/* 1.2.2. Read surface size and maximum number of iterations */
		int ok;
		ok = fscanf(args, "%d %d %d", &rows, &columns, &max_iter);
		if ( ok != 3 ) {
			fprintf(stderr,"-- Error in file: reading rows, columns, max_iter from file: %s\n", argv[1]);
			MPI_Abort( MPI_COMM_WORLD,  EXIT_FAILURE );
		}

		/* 1.2.3. Teams information */
		ok = fscanf(args, "%d", &num_teams );
		if ( ok != 1 ) {
			fprintf(stderr,"-- Error file, reading num_teams from file: %s\n", argv[1]);
			MPI_Abort( MPI_COMM_WORLD,  EXIT_FAILURE );
		}
		teams = (Team *)malloc( sizeof(Team) * (size_t)num_teams );
		if ( teams == NULL ) {
			fprintf(stderr,"-- Error allocating: %d teams\n", num_teams );
			MPI_Abort( MPI_COMM_WORLD,  EXIT_FAILURE );
		}
		for( i=0; i<num_teams; i++ ) {
			ok = fscanf(args, "%d %d %d", &teams[i].x, &teams[i].y, &teams[i].type);
			if ( ok != 3 ) {
				fprintf(stderr,"-- Error in file: reading team %d from file: %s\n", i, argv[1]);
				MPI_Abort( MPI_COMM_WORLD,  EXIT_FAILURE );
			}
		}

		/* 1.2.4. Focal points information */
		ok = fscanf(args, "%d", &num_focal );
		if ( ok != 1 ) {
			fprintf(stderr,"-- Error in file: reading num_focal from file: %s\n", argv[1]);
			MPI_Abort( MPI_COMM_WORLD,  EXIT_FAILURE );
		}
		focal = (FocalPoint *)malloc( sizeof(FocalPoint) * (size_t)num_focal );
		if ( focal == NULL ) {
			fprintf(stderr,"-- Error allocating: %d focal points\n", num_focal );
			MPI_Abort( MPI_COMM_WORLD,  EXIT_FAILURE );
		}
		for( i=0; i<num_focal; i++ ) {
			ok = fscanf(args, "%d %d %d %d", &focal[i].x, &focal[i].y, &focal[i].start, &focal[i].heat);
			if ( ok != 4 ) {
				fprintf(stderr,"-- Error in file: reading focal point %d from file: %s\n", i, argv[1]);
				MPI_Abort( MPI_COMM_WORLD,  EXIT_FAILURE );
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
			MPI_Abort( MPI_COMM_WORLD,  EXIT_FAILURE );
		}

		/* 1.3.2. Surface size and maximum number of iterations */
		rows = atoi( argv[1] );
		columns = atoi( argv[2] );
		max_iter = atoi( argv[3] );

		/* 1.3.3. Teams information */
		num_teams = atoi( argv[4] );
		teams = (Team *)malloc( sizeof(Team) * (size_t)num_teams );
		if ( teams == NULL ) {
			fprintf(stderr,"-- Error allocating: %d teams\n", num_teams );
			MPI_Abort( MPI_COMM_WORLD,  EXIT_FAILURE );
		}
		if ( argc < num_teams*3 + 5 ) {
			fprintf(stderr,"-- Error in arguments: not enough arguments for %d teams\n", num_teams );
			MPI_Abort( MPI_COMM_WORLD,  EXIT_FAILURE );
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
			MPI_Abort( MPI_COMM_WORLD,  EXIT_FAILURE );
		}
		num_focal = atoi( argv[focal_args] );
		focal = (FocalPoint *)malloc( sizeof(FocalPoint) * (size_t)num_focal );
		if ( teams == NULL ) {
			fprintf(stderr,"-- Error allocating: %d focal points\n", num_focal );
			MPI_Abort( MPI_COMM_WORLD,  EXIT_FAILURE );
		}
		if ( argc < focal_args + 1 + num_focal*4 ) {
			fprintf(stderr,"-- Error in arguments: not enough arguments for %d focal points\n", num_focal );
			MPI_Abort( MPI_COMM_WORLD,  EXIT_FAILURE );
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
			MPI_Abort( MPI_COMM_WORLD,  EXIT_FAILURE );
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

	/* 2. Start global timer */
	MPI_Barrier(MPI_COMM_WORLD);
	double ttotal = cp_Wtime();

	// MPI Version: To store the results of residual heat on focal points
	float *residualHeat = (float*)malloc( sizeof(float) * (size_t)num_focal );
	if ( residualHeat == NULL ) {
		fprintf(stderr,"-- Error allocating: residualHeat\n");
		MPI_Abort( MPI_COMM_WORLD,  EXIT_FAILURE );
	}

/*
 *
 * START HERE: DO NOT CHANGE THE CODE ABOVE THIS POINT
 *
 */
	float *copyHeat = (float*)malloc( sizeof(float) * (size_t)num_focal );
	if ( copyHeat == NULL ) {
		fprintf(stderr,"-- Error allocating: residualHeat\n");
		MPI_Abort( MPI_COMM_WORLD,  EXIT_FAILURE );
	}
	int nprocs;
	MPI_Status status;
	MPI_Request req;
	MPI_Comm_size( MPI_COMM_WORLD, &nprocs );

	int iter;
	
	int my_rows;
	int resto_rows = rows % nprocs;

	if(resto_rows != 0) {
		if(rank != nprocs - 1){
			my_rows = (int) (rows / nprocs);
		} else {
			my_rows = (int) (rows / nprocs) + resto_rows;
		}
	} else {
		my_rows = (int) (rows / nprocs);
	}
	
	int my_begin_rows;
	int my_end_rows;

	int my_rowsTam;
	if(rank == 0 ) {
		my_rowsTam = my_rows + 1; 
		my_begin_rows = (rows/nprocs) * rank;
		my_end_rows = my_begin_rows + my_rowsTam - 1;
	} else if (rank == nprocs - 1) {
		my_rowsTam = my_rows + 1;
		my_begin_rows = (rows/nprocs) * rank - 1;
		my_end_rows = my_begin_rows + my_rowsTam - 1;
	} else {
		my_rowsTam = my_rows + 2;
		my_begin_rows = (rows/nprocs) * rank - 1;
		my_end_rows = my_begin_rows + my_rowsTam - 1;
	}

	//printf("Rows, begin and end of process %d: %d %d %d %d\n",rank,my_rows,my_rowsTam,my_begin_rows, my_end_rows);

	float *surface = (float *)malloc( sizeof(float) * (size_t)my_rowsTam * (size_t)columns );
	float *surfaceCopy = (float *)malloc( sizeof(float) * (size_t)my_rowsTam * (size_t)columns );
	if ( surface == NULL || surfaceCopy == NULL ) {
		fprintf(stderr,"-- Error allocating: surface structures\n");
		MPI_Abort( MPI_COMM_WORLD,  EXIT_FAILURE );
	}

	/* 3. Initialize surface */
	for( i=0; i<my_rowsTam; i++ ){
		for( j=0; j<columns; j++ ){
			accessMat( surface, i, j ) = 0.0;
			accessMat( surfaceCopy , i , j) = 0.0;
		}
	}

	/* 4. Simulation */
	int flag_stability = 0;
	//int first_activation = 0;
	iter = focal[0].start;
	for( i=1; i<num_focal; i++ ) {
         	iter = (iter < focal[i].start) ? iter : focal[i].start;
	}

	for( ; iter<max_iter && ! flag_stability; iter++ ) {

		/* 4.1. Activate focal points */
		int num_deactivated = 0;
		for( i=0; i<num_focal; i++ ) {
			if ( focal[i].start == iter ) {
				focal[i].active = 1;
				//if ( ! first_activation ) first_activation = 1;
			}
			// Count focal points already deactivated by a team
			if ( focal[i].active == 2 ) num_deactivated++;
		}

		/* 4.2. Propagate heat (10 steps per each team movement) */
		float global_residual = 0.0f;
		float max_residual = 0.0f;
		int step;
		float* temp;

		
		for( step=0; step<10; step++ )	{
			/* 4.2.1. Update heat on active focal points */
			for( i=0; i<num_focal; i++ ) {
				if ( focal[i].active != 1 ) continue;
				int x = focal[i].x;
				//if(my_begin_rows - 1 <= x && my_end_rows + 1 >= x){
				if(my_begin_rows <= x && my_end_rows >= x) {
					//printf("Proceso %d --> My_Begin %d My_End %d x %d\n",rank,my_begin_rows,my_end_rows,x);
					int y = focal[i].y;
					accessMat( surface, x - my_begin_rows, y ) = focal[i].heat;
				}
			}

			/* 4.2.2. Copy values of the surface in ancillary structure (NO Skip borders) */
			temp = surface;
			surface = surfaceCopy;
			surfaceCopy = temp;
			//for( i=0; i<my_rowsTam; i++ )
				//for( j=0; j<columns; j++ )
					//accessMat( surfaceCopy, i, j ) = accessMat( surface, i, j );

			/* 4.2.3. Update surface values (skip borders) */
			for( i=1; i<my_rowsTam-1; i++ )
				for( j=1; j<columns-1; j++ ){
					accessMat( surface, i, j ) = ( 
						accessMat( surfaceCopy, i-1, j ) +
						accessMat( surfaceCopy, i+1, j ) +
						accessMat( surfaceCopy, i, j-1 ) +
						accessMat( surfaceCopy, i, j+1 ) ) / 4;
				
					if(step ==0){
						if ( fabs( accessMat( surface, i, j ) - accessMat( surfaceCopy, i, j ) > global_residual ) ) {
							global_residual = fabs( accessMat( surface, i, j ) - accessMat( surfaceCopy, i, j ) );
						}
					}
				}
					
			if(rank == 0){
				MPI_Send(&surface[(my_rowsTam - 2)*columns],columns,MPI_FLOAT,rank+1,1,MPI_COMM_WORLD);
				MPI_Recv(&surface[(my_rowsTam - 1)*columns],columns, MPI_FLOAT, rank+1,10, MPI_COMM_WORLD,&status);
			} 

			if(rank != 0 && rank != nprocs - 1) {
				MPI_Send(&surface[1*columns],columns,MPI_FLOAT,rank - 1,10,MPI_COMM_WORLD);
				MPI_Send(&surface[(my_rowsTam - 2)*columns],columns,MPI_FLOAT,rank+1,1,MPI_COMM_WORLD);
				MPI_Recv(&surface[0],columns, MPI_FLOAT, rank - 1,1, MPI_COMM_WORLD,&status);
				MPI_Recv(&surface[(my_rowsTam - 1)*columns],columns, MPI_FLOAT, rank+1,10, MPI_COMM_WORLD,&status);
			} 

			if(rank == nprocs -1){
				MPI_Send(&surface[1*columns],columns,MPI_FLOAT,rank - 1,10,MPI_COMM_WORLD);
				MPI_Recv(&surface[0],columns, MPI_FLOAT, rank - 1, 1, MPI_COMM_WORLD,&status);
			}
			
			/* 4.2.4. Compute the maximum residual difference (absolute value) */
			//for( i=1; i<my_rowsTam-1; i++ )
				//for( j=1; j<columns-1; j++ ) 
					//if ( fabs( accessMat( surface, i, j ) - accessMat( surfaceCopy, i, j ) > global_residual ) ) {
					//	global_residual = fabs( accessMat( surface, i, j ) - accessMat( surfaceCopy, i, j ) );
					//}

			/* Only in step 0 Calculates Maximum Residual */
			if(step == 0){	
				MPI_Allreduce(&global_residual,&max_residual,1,MPI_FLOAT,MPI_MAX,MPI_COMM_WORLD);
				global_residual = max_residual;
			}

			// Prueba --> Global Residual
			//if(rank == 0) printf("Iteracion: %d --> Global Residual: %f\n",iter,global_residual);
		}

		/* If the global residual is lower than THRESHOLD, we have reached enough stability, stop simulation at the end of this iteration */
		/* It evaluates More en in the first condition than in the second one */
		if( global_residual < THRESHOLD && num_deactivated == num_focal ) flag_stability = 1;

		/* 4.3. Move teams */
		for( t=0; t<num_teams; t++ ) {
			/* 4.3.1. Choose nearest focal point */
			float distance = FLT_MAX;
			int target = -1;
			for( j=0; j<num_focal; j++ ) {
				if ( focal[j].active != 1 ) continue; // Skip non-active focal points
				float dx = focal[j].x - teams[t].x;
				float dy = focal[j].y - teams[t].y;
				float local_distance = dx*dx + dy*dy; 
				if ( local_distance < distance ) {
					distance = local_distance;
					target = j;
				}
			}
			/* 4.3.2. Annotate target for the next stage */
			teams[t].target = target;

			/* 4.3.3. No active focal point to choose, no movement */
			if ( target == -1 ) continue; 

			switch(teams[t].type){
                                // Type 1: Can move in diagonal
                        	case 1:
                                	if ( focal[target].x < teams[t].x ) teams[t].x--;
                                	else if ( focal[target].x > teams[t].x ) teams[t].x++;
                                	if ( focal[target].y < teams[t].y ) teams[t].y--;
                                	else if ( focal[target].y > teams[t].y ) teams[t].y++;
                                	break;
                       	 	case 2:
                        		// Type 2: First in horizontal direction, then in vertical direction
                                	if ( focal[target].y < teams[t].y ) teams[t].y--;
                                	else if ( focal[target].y > teams[t].y ) teams[t].y++;
                                	else if ( focal[target].x < teams[t].x ) teams[t].x--;
                               	 else if ( focal[target].x > teams[t].x ) teams[t].x++;
       		                	break;
                        	default:
                                	// Type 3: First in vertical direction, then in horizontal direction
                                	if ( focal[target].x < teams[t].x ) teams[t].x--;
                                	else if ( focal[target].x > teams[t].x ) teams[t].x++;
                                	else if ( focal[target].y < teams[t].y ) teams[t].y--;
                        	        else if ( focal[target].y > teams[t].y ) teams[t].y++;
                        }
			/* 4.3.4. Move in the focal point direction */
			/*if ( teams[t].type == 1 ) { 
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
			}*/
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
					//if(i < (my_begin_rows-1) || i > (my_end_rows + 1)) continue;
					if(my_begin_rows <= i && my_end_rows >= i){
						float dx = teams[t].x - i;
						float dy = teams[t].y - j;
						float distance = dx*dx + dy*dy ; // En vez hacer la raiz en un cuadrado, elevas al cuadrado a ambos 
						if ( distance <= radius*radius ) {
							accessMat( surface, i - my_begin_rows, j ) = accessMat( surface, i - my_begin_rows, j ) * 0.75;
							//printf("Proceso %d --> My_Begin %d My_End %d dx %f\n",rank,my_begin_rows,my_end_rows,dx);
							//printf("Proceso %d --> Real: %d Parcial %d\n",rank,i,i - my_begin_rows + 1);
						}
					}
				}
			}
		}

#ifdef DEBUG
		/* 4.5. DEBUG: Print the current state of the simulation at the end of each iteration */
		print_status( iter, rows, columns, surface, num_teams, teams, num_focal, focal, global_residual );
#endif // DEBUG
	}

	//float *copyHeat = (float*)malloc( sizeof(float) * (size_t)num_focal );

	// MPI Version: Store the results of residual heat on focal points
	for (i=0; i<num_focal; i++){
		residualHeat[i] = 0.0f;
		if(focal[i].x >= my_begin_rows && focal[i].x <= my_end_rows){
			residualHeat[i] = accessMat( surface, focal[i].x - my_begin_rows, focal[i].y );
		}
	}

	MPI_Allreduce(residualHeat,copyHeat,num_focal,MPI_FLOAT,MPI_MAX,MPI_COMM_WORLD);
	residualHeat = copyHeat;

	// MPI Version: Store the results of residual heat on focal points
	//float vect = accessMat( surface, focal[i].x, focal[i].y );
	//MPI_Reduce(&residualHeat,&residualHeat,num_focal,MPI_FLOAT,MPI_SUM,0,MPI_COMM_WORLD);

	free( surface );
	free( surfaceCopy );

/*
 *
 * STOP HERE: DO NOT CHANGE THE CODE BELOW THIS POINT
 *
 */

	/* 5. Stop global time */
	MPI_Barrier(MPI_COMM_WORLD);
	ttotal = cp_Wtime() - ttotal;

	/* 6. Output for leaderboard */
	if ( rank == 0 ) {
		printf("\n");
		/* 6.1. Total computation time */
		printf("Time: %lf\n", ttotal );
		/* 6.2. Results: Number of iterations, residual heat on the focal points */
		printf("Result: %d", iter);
		for (i=0; i<num_focal; i++)
			printf(" %.6f", residualHeat[i] );
		printf("\n");
	}

	/* 7. Free resources */	
	free( teams );
	free( focal );

	/* 8. End */
	MPI_Finalize();
	return 0;
}
