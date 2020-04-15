// User: g113@88.11.186.52 
// ExecutionRequest[P:'extinguishing.c',P:1,T:1,args:'',q:'mpilb'] 
// Apr 04 2019 19:47:22
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
#include<cputils.h>

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


	int iter;

	const int RADIUS_TYPE_1_SQUARED = RADIUS_TYPE_1*RADIUS_TYPE_1;
	const int RADIUS_TYPE_2_3_SQUARED = RADIUS_TYPE_2_3*RADIUS_TYPE_2_3;
// MPI Version: Eliminate this conditional to start doing the work in parallel
//if ( rank == 0 ) {
	int num_procs;
	MPI_Comm_size( MPI_COMM_WORLD, &num_procs );

	int global_rows = rows;

	int surplus = rows%num_procs;
	//Rows + 2 so additional space is allocated for receiving data from the other processes
	rows = rows/num_procs + 2;
	//Divide the surplus rows among the processes
	for(i=0; i<surplus; i++){
		if(rank == i){
			rows++;
		}
	}
	//Store the real number of rows the process is working with, not counting the extra 2
	int my_rows = rows - 2;
	//We also need to store the index of the first row assigned to the process as well as the last one
	int first_row, last_row;

	for(i=0; i<num_procs; i++){
		if(rank == i){
			if(i < surplus){
				//All the previous processes will have the same number of rows assigned as this one
				first_row = rank * (my_rows);
				last_row = first_row + my_rows - 1;
			}else{
				//The first n processes with n = surplus will have 1 more row assigned than this one
				first_row = surplus * (my_rows + 1) + (rank - surplus) * (my_rows);
				last_row = first_row + my_rows - 1;
			}
			
		}
	}

	float *surface = (float *)malloc( sizeof(float) * (size_t)rows * (size_t)columns );
	float *surfaceCopy = (float *)malloc( sizeof(float) * (size_t)rows * (size_t)columns );
	float *temp = (float *)malloc( sizeof(float) * (size_t)rows * (size_t)columns );
	if ( surface == NULL || surfaceCopy == NULL || temp == NULL) {
		fprintf(stderr,"-- Error allocating: surface structures\n");
		MPI_Abort( MPI_COMM_WORLD,  EXIT_FAILURE );
	}
	/*
	printf("FIRST rank: %d\n", rank);
	printf("rows: %d\n", rows);
	printf("num_procs: %d\n", num_procs);
	*/
	//If there are more processes than rows (This doesn't work it seems)
	/*if(num_procs > rows){
		for(i=global_rows; i<num_procs; i++){
			if(rank == i){
				rows = 0;
				first_row = -1;
				last_row = -1;
			}
		}
		//The extra processes will essentially not exist
		num_procs = global_rows;
	}*/
	/*
	printf("LAST rank: %d\n", rank);
	printf("rows: %d\n", rows);
	printf("num_procs: %d\n", num_procs);
	MPI_Finalize();
	return 0;
	*/

	/* 3. Initialize surface */
	for( i=0; i<rows; i++ )
		for( j=0; j<columns; j++ )
			accessMat( surface, i, j ) = 0.0;

	//Initialize surface copy
	for( i=0; i<rows; i++ )
		for( j=0; j<columns; j++ )
			accessMat( surfaceCopy, i, j ) = 0.0;

	/* 4. Simulation */
	int flag_stability = 0;
	int first_activation = 0;
	int num_deactivated = 0;
	for( iter=0; iter<max_iter && ! flag_stability; iter++ ) {

		/* 4.1. Activate focal points */
		//int num_deactivated = 0;
		for( i=0; i<num_focal; i++ ) {
			if ( focal[i].start == iter ) {
				focal[i].active = 1;
				if ( ! first_activation ) first_activation = 1;
			}
			// Count focal points already deactivated by a team
			//if ( focal[i].active == 2 ) num_deactivated++;
		}

		
		if(first_activation){
			/* 4.2. Propagate heat (10 steps per each team movement) */
			float global_residual = 0.0f;
			int step;
			//Determine whether the lines to send have changed since the last time they were sent
			int first_line_changed = 1;
			int last_line_changed = 1;
			int signal_message = 0;
			int signal_receive = 0;
			//For the asynchronous receptions
			MPI_Request last_line;
			MPI_Request first_line;
			MPI_Request last_line_signal;
			MPI_Request first_line_signal;

			for( step=0; step<10; step++ )	{
				/* 4.2.1. Update heat on active focal points */
				for( i=0; i<num_focal; i++ ) {
					if ( focal[i].active != 1 ) continue;
					int x = focal[i].x;
					int y = focal[i].y;
					//accessMat( surface, x, y ) = focal[i].heat;
					if(x >= first_row && x <= last_row){
						accessMat( surface, (x - first_row + 1), y ) = focal[i].heat;
						if(x == first_row){
							first_line_changed = 1;
						}else if(x == last_row){
							last_line_changed = 1;
						}
					}
				}

				/* 4.2.2. Copy values of the surface in ancillary structure (Skip borders) */
				//for( i=1; i<rows-1; i++ )
				//	for( j=1; j<columns-1; j++ )
				//		accessMat( surfaceCopy, i, j ) = accessMat( surface, i, j );

				//Send the first line to the previous process and the last line to the next one, the first
				//and last processes have different versions of this
				//We also need to receive lines from those processes, using nonblocking receive operations
				//before sending the lines avoids a deadlock

				if(rank == 0){
					//First process
					//If there is only 1 process, dont send or receive anything
					if(num_procs != 1){
						//Determine whether to send the last line
						if(last_line_changed){
							//Indicate to the receiving process that a line is going to be sent
							signal_message = 1;
							MPI_Isend(&signal_message, 1, MPI_INT, rank + 1, 1, MPI_COMM_WORLD, &last_line_signal);	
							//Send Last line
							MPI_Isend(surface + (my_rows) * columns, columns, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, &last_line);	
						}else{
							//Indicate to the receiving process that their line is still valid
							signal_message = 0;
							MPI_Isend(&signal_message, 1, MPI_INT, rank + 1, 1, MPI_COMM_WORLD, &last_line_signal);
						}
						//This message indicates whether a line has been sent by the next process
						MPI_Recv(&signal_receive, 1, MPI_INT, rank + 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
						if(signal_receive == 1){
							//Receive the line from the next process
							MPI_Recv(surface + (rows-1) * columns, columns, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
						}
						//The process receives the signal this one sent
						MPI_Wait(&last_line_signal, MPI_STATUS_IGNORE);
						//Wait for the receptions
						if(last_line_changed){
							//The process receives the line this one sent
							MPI_Wait(&last_line, MPI_STATUS_IGNORE);
						}
						
						/*
						//Send Last line
						MPI_Isend(surface + (my_rows) * columns, columns, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, &last_line);
						//Receive the line from the next process
						MPI_Recv(surface + (rows-1) * columns, columns, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
						//Wait for the receptions
						MPI_Wait(&last_line, MPI_STATUS_IGNORE);
						*/
					}
				}else if(rank == num_procs-1){
					//Last process
					//Determine whether to send the first line
					if(first_line_changed){
						//Indicate to the receiving process that a line is going to be sent
						signal_message = 1;
						MPI_Isend(&signal_message, 1, MPI_INT, rank - 1, 1, MPI_COMM_WORLD, &first_line_signal);	
						//Send First line
						MPI_Isend(surface + columns, columns, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, &first_line);	
					}else{
						//Indicate to the receiving process that their line is still valid
						signal_message = 0;
						MPI_Isend(&signal_message, 1, MPI_INT, rank - 1, 1, MPI_COMM_WORLD, &first_line_signal);
					}
					//This message indicates whether a line has been sent by the previous process
					MPI_Recv(&signal_receive, 1, MPI_INT, rank - 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					if(signal_receive == 1){
						//Receive the line from the previous process
						MPI_Recv(surface, columns, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					}
					//Wait for the receptions
					//The process receives the signal this one sent
					MPI_Wait(&first_line_signal, MPI_STATUS_IGNORE);
					if(first_line_changed){
						//The process receives the line this one sent
						MPI_Wait(&first_line, MPI_STATUS_IGNORE);
					}

					/*
					//Send First line
					MPI_Isend(surface + columns, columns, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, &first_line);
					//Receive the line from the previous process
					MPI_Recv(surface, columns, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					//Wait for the receptions
					MPI_Wait(&first_line, MPI_STATUS_IGNORE);
					*/

				}else/*if(rows != 0)*/{
					//General case

					//Determine whether to send the first line
					if(first_line_changed){
						//Indicate to the receiving process that a line is going to be sent
						signal_message = 1;
						MPI_Isend(&signal_message, 1, MPI_INT, rank - 1, 1, MPI_COMM_WORLD, &first_line_signal);	
						//Send First line
						MPI_Isend(surface + columns, columns, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, &first_line);
					}else{
						//Indicate to the receiving process that their line is still valid
						signal_message = 0;
						MPI_Isend(&signal_message, 1, MPI_INT, rank - 1, 1, MPI_COMM_WORLD, &first_line_signal);
					}

					//Determine whether to send the last line
					if(last_line_changed){
						//Indicate to the receiving process that a line is going to be sent
						signal_message = 1;
						MPI_Isend(&signal_message, 1, MPI_INT, rank + 1, 1, MPI_COMM_WORLD, &last_line_signal);	
						//Send Last line
						MPI_Isend(surface + (my_rows) * columns, columns, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, &last_line);
					}else{
						//Indicate to the receiving process that their line is still valid
						signal_message = 0;
						MPI_Isend(&signal_message, 1, MPI_INT, rank + 1, 1, MPI_COMM_WORLD, &last_line_signal);
					}

					//This message indicates whether a line has been sent by the previous process
					MPI_Recv(&signal_receive, 1, MPI_INT, rank - 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					if(signal_receive == 1){
						//Receive the line from the previous process
						MPI_Recv(surface, columns, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					}

					//This message indicates whether a line has been sent by the next process
					MPI_Recv(&signal_receive, 1, MPI_INT, rank + 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					if(signal_receive == 1){
						//Receive the line from the next process
						MPI_Recv(surface + (rows-1) * columns, columns, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					}
					
					//Wait for the receptions
					//The process receives the signal this one sent
					MPI_Wait(&first_line_signal, MPI_STATUS_IGNORE);
					MPI_Wait(&last_line_signal, MPI_STATUS_IGNORE);
					if(first_line_changed){
						//The process receives the line this one sent
						MPI_Wait(&first_line, MPI_STATUS_IGNORE);
					}
					if(last_line_changed){
						//The process receives the line this one sent
						MPI_Wait(&last_line, MPI_STATUS_IGNORE);
					}
					
				}

				//Set the flags to indicate the lines haven't changed since the last time they were sent
				first_line_changed = 0;
				last_line_changed = 0;

				/* 4.2.3. Update surface values (skip borders) */
				if(rank == 0){
					/*Uncomment to support one-threaded executions
					if(num_procs == 1){
						for( i=2; i<rows-2; i++ )
							for( j=1; j<columns-1; j++ )
								accessMat( surfaceCopy, i, j ) = ( 
									accessMat( surface, i-1, j ) +
									accessMat( surface, i+1, j ) +
									accessMat( surface, i, j-1 ) +
									accessMat( surface, i, j+1 ) ) / 4;
					*/
						/* 4.2.4. Compute the maximum residual difference (absolute value) */
					/*
						if(num_deactivated == num_focal && step == 0){
							for( i=2; i<rows-2; i++ )
								for( j=1; j<columns-1; j++ ) 
									if ( fabs( accessMat( surfaceCopy, i, j ) - accessMat( surface, i, j ) > global_residual ) ) {
										global_residual = fabs( accessMat( surfaceCopy, i, j ) - accessMat( surface, i, j ) );
									}
						}
					}else{
					*/	
						//Changes on the last line are special cases as it is also checked whether it changes
						for( j=1; j<columns-1; j++ ){
							//Last line
							int index_last = rows-2;
							float previous_value = accessMat( surfaceCopy, index_last, j );
							float next_value = ( 
									accessMat( surface, index_last-1, j ) +
									accessMat( surface, index_last+1, j ) +
									accessMat( surface, index_last, j-1 ) +
									accessMat( surface, index_last, j+1 ) ) / 4;
							if(previous_value != next_value){
								last_line_changed = 1;
								accessMat( surfaceCopy, index_last, j ) = next_value;
							}
						}

						last_line_changed = 1;
						for( i=2; i<rows-1; i++ )
							for( j=1; j<columns-1; j++ )
								accessMat( surfaceCopy, i, j ) = ( 
									accessMat( surface, i-1, j ) +
									accessMat( surface, i+1, j ) +
									accessMat( surface, i, j-1 ) +
									accessMat( surface, i, j+1 ) ) / 4;
						/* 4.2.4. Compute the maximum residual difference (absolute value) */
						if(num_deactivated == num_focal && step == 0){
							for( i=2; i<rows-1; i++ )
								for( j=1; j<columns-1; j++ ) 
									if ( fabs( accessMat( surfaceCopy, i, j ) - accessMat( surface, i, j ) > global_residual ) ) {
										global_residual = fabs( accessMat( surfaceCopy, i, j ) - accessMat( surface, i, j ) );
									}
							//Reduce to get the global maximum
							float global_max;
							MPI_Allreduce(&global_residual, &global_max, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
							global_residual = global_max;
							//MPI_Barrier(MPI_COMM_WORLD);
						}

					//}
										
				}else if(rank == num_procs-1){
					//Changes on the first line are special cases as it is also checked whether it changes
					for( j=1; j<columns-1; j++ ){
						//First line
						float previous_value = accessMat( surfaceCopy, 1, j );
						float next_value = ( 
								accessMat( surface, 0, j ) +
								accessMat( surface, 2, j ) +
								accessMat( surface, 1, j-1 ) +
								accessMat( surface, 1, j+1 ) ) / 4;
						if(previous_value != next_value){
							first_line_changed = 1;
							accessMat( surfaceCopy, 1, j ) = next_value;
						}
					}
					for( i=2; i<rows-2; i++ )
						for( j=1; j<columns-1; j++ )
							accessMat( surfaceCopy, i, j ) = ( 
								accessMat( surface, i-1, j ) +
								accessMat( surface, i+1, j ) +
								accessMat( surface, i, j-1 ) +
								accessMat( surface, i, j+1 ) ) / 4;
					/* 4.2.4. Compute the maximum residual difference (absolute value) */
					if(num_deactivated == num_focal && step == 0){
						for( i=1; i<rows-2; i++ )
							for( j=1; j<columns-1; j++ ) 
								if ( fabs( accessMat( surfaceCopy, i, j ) - accessMat( surface, i, j ) > global_residual ) ) {
									global_residual = fabs( accessMat( surfaceCopy, i, j ) - accessMat( surface, i, j ) );
								}
						//Reduce to get the global maximum
						float global_max;
						MPI_Allreduce(&global_residual, &global_max, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
						global_residual = global_max;
						//MPI_Barrier(MPI_COMM_WORLD);
					}
					
				}else/*if(rows != 0)*/{
					//Changes on the first and last lines are special cases as they also check whether they change
					for( j=1; j<columns-1; j++ ){
						//First line
						float previous_value = accessMat( surfaceCopy, 1, j );
						float next_value = ( 
								accessMat( surface, 0, j ) +
								accessMat( surface, 2, j ) +
								accessMat( surface, 1, j-1 ) +
								accessMat( surface, 1, j+1 ) ) / 4;
						if(previous_value != next_value){
							first_line_changed = 1;
							accessMat( surfaceCopy, 1, j ) = next_value;
						}
						//Last line
						int index_last = rows-2;
						previous_value = accessMat( surfaceCopy, index_last, j );
						next_value = ( 
								accessMat( surface, index_last-1, j ) +
								accessMat( surface, index_last+1, j ) +
								accessMat( surface, index_last, j-1 ) +
								accessMat( surface, index_last, j+1 ) ) / 4;
						if(previous_value != next_value){
							last_line_changed = 1;
							accessMat( surfaceCopy, index_last, j ) = next_value;
						}
					}
					for( i=2; i<rows-2; i++ )
						for( j=1; j<columns-1; j++ )
							accessMat( surfaceCopy, i, j ) = ( 
								accessMat( surface, i-1, j ) +
								accessMat( surface, i+1, j ) +
								accessMat( surface, i, j-1 ) +
								accessMat( surface, i, j+1 ) ) / 4;
					/* 4.2.4. Compute the maximum residual difference (absolute value) */
					if(num_deactivated == num_focal && step == 0){
						for( i=1; i<rows-1; i++ )
							for( j=1; j<columns-1; j++ ) 
								if ( fabs( accessMat( surfaceCopy, i, j ) - accessMat( surface, i, j ) > global_residual ) ) {
									global_residual = fabs( accessMat( surfaceCopy, i, j ) - accessMat( surface, i, j ) );
								}
						//Reduce to get the global maximum
						float global_max;
						MPI_Allreduce(&global_residual, &global_max, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
						global_residual = global_max;
						//MPI_Barrier(MPI_COMM_WORLD);
					}	
				}

				temp = surface;
				surface = surfaceCopy;
				surfaceCopy = temp;
			}
			
			
			/* If the global residual is lower than THRESHOLD, we have reached enough stability, stop simulation at the end of this iteration */
			if( num_deactivated == num_focal){
				if( global_residual < THRESHOLD ){
					flag_stability = 1;
				}
				
			}else{
				/* 4.3. Move teams */
				for( t=0; t<num_teams; t++ ) {
					/* 4.3.1. Choose nearest focal point */
					int distance = 2147483647;
					int target = -1;
					for( j=0; j<num_focal; j++ ) {
						if ( focal[j].active != 1 ) continue; // Skip non-active focal points
						int dx = focal[j].x - teams[t].x;
						int dy = focal[j].y - teams[t].y;
						int local_distance = dx*dx + dy*dy;
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
			}

			
			/* 4.4. Team actions */
			for( t=0; t<num_teams; t++ ) {
				/* 4.4.1. Deactivate the target focal point when it is reached */
				int target = teams[t].target;
				if ( target != -1 && focal[target].x == teams[t].x && focal[target].y == teams[t].y 
					&& focal[target].active == 1 ){
						focal[target].active = 2;
						num_deactivated++;
				}

				/* 4.4.2. Reduce heat in a circle around the team */
				// Influence area of fixed radius depending on type
				if ( teams[t].type == 1 ){
					for( i=teams[t].x-RADIUS_TYPE_1; i<=teams[t].x+RADIUS_TYPE_1; i++ ) {
						if(i >= first_row && i <= last_row){
							for( j=teams[t].y-RADIUS_TYPE_1; j<=teams[t].y+RADIUS_TYPE_1; j++ ) {
							//Only make changes if the targeted point is inside the portion of the matrix
							//this process has
							
								if ( i<1 || i>=global_rows-1 || j<1 || j>=columns-1 ) continue; // Out of the heated surface
								int dx = teams[t].x - i;
								int dy = teams[t].y - j;
								if ( dx*dx + dy*dy <= RADIUS_TYPE_1_SQUARED) {
									accessMat( surface, (i - first_row + 1), j ) = accessMat( surface, (i - first_row + 1), j ) * 0.75; // Team efficiency factor
								}
							}
						}
					}
				}else{
					for( i=teams[t].x-RADIUS_TYPE_2_3; i<=teams[t].x+RADIUS_TYPE_2_3; i++ ) {
						if(i >= first_row && i <= last_row){
							for( j=teams[t].y-RADIUS_TYPE_2_3; j<=teams[t].y+RADIUS_TYPE_2_3; j++ ) {
							//Only make changes if the targeted point is inside the portion of the matrix
							//this process has
							
								if ( i<1 || i>=global_rows-1 || j<1 || j>=columns-1 ) continue; // Out of the heated surface
								int dx = teams[t].x - i;
								int dy = teams[t].y - j;
								if ( dx*dx + dy*dy <= RADIUS_TYPE_2_3_SQUARED) {
									accessMat( surface, (i - first_row + 1), j ) = accessMat( surface, (i - first_row + 1), j ) * 0.75; // Team efficiency factor
								}
							}
						}
					}
				}
			}
		}
	}

	for (i=0; i<num_focal; i++){
		residualHeat[i] = 0;
	}
	float *residualHeatAll;
	if(rank == 0){
		residualHeatAll = (float*)malloc( sizeof(float) * (size_t)(num_focal * num_procs) );
		if ( residualHeatAll == NULL ) {
			fprintf(stderr,"-- Error allocating: residualHeat\n");
			MPI_Abort( MPI_COMM_WORLD,  EXIT_FAILURE );
		}
	}
	// MPI Version: Store the results of residual heat on focal points
	for (i=0; i<num_focal; i++){
		if(focal[i].x >= first_row && focal[i].x <= last_row){
			residualHeat[i] = accessMat( surface, (focal[i].x - first_row + 1), focal[i].y );
		}
	}

	//Send all arrays to the process 0
	MPI_Gather(residualHeat, num_focal, MPI_FLOAT, residualHeatAll, num_focal, MPI_FLOAT, 0, MPI_COMM_WORLD);
	//Put the non-zero elements in residualHeat
	if(rank == 0){
		for(i=0; i<(num_focal * num_procs); i++){
			if(residualHeatAll[i]>0){
				residualHeat[i%num_focal] = residualHeatAll[i];
			}
		}	
	}
	
	free( surface );
	free( surfaceCopy );

// MPI Version: Eliminate this conditional-end to start doing the work in parallel
//}
/*
                     ,-.                       
                     `\ `.                     
                      (`- `.                   
           __          `c.- \                  
          (_ `---..__   `\"_ \                 
            `(`-.__  `-._ (_c |                
              `--.____ ( `,-  | |/             
                  `---(___| ,---/         ,.   
                    ,-;::,'|_/\-|       ,' /   
            _.._    ||   `-'\@/@|_      | ||-. 
    _.--.--'    `-. \`--/,-._/    `\_,--',/|,-\
  .',- _     _.__  `-:_|/   (      /      )|-'|
 /_/_,;-  ,-'    `-._ _)_,-,-,___,'---'\___.-' 
 |___,,.  | _    _.-'' ((_{_// /               
 \.-'  | (-' \.-'       `---'`'                
    ,-'\__>  /|         /                      
  .',-'/|.   \ \        |                      
 / / _|``.`.  `/\_     /|  ,,.                 
/  \  |  |  `-.`(_),. /((((   )                
|   `--\ `.`=. `-._`'|__|  `''                 
|  ,'  |`-|   T``'',----.                      
|      |  `.__|-,-',----.|                     
\     / ,-' _,-;,-'      |                     
 `---'  | ,;-''       _,'                      
        `//       _,-'                         
         \____.-''  
						                              __
						                            ,'   `.
						                           ,  .    `.
						                           |         `.
						                           |   .    .  \
						                           |   .    .   `.
						                           |       .      \
						                           |        .      `.
						                           |       .         \
						                           |       .          `
						                           `.            . .   `
						                            |                   ;
						                            |      .    .   .   .
						                            |   . .       .   .  ; 
						                            |  . . .         .    ;
						                            |    .       .        .
						                            `.   .                 ;
						                             |     ..   .          .
						                             |      .               ; 
						                             |       . .      ..    .
						                             |       .    .  .  .    ;
						                        .;''`()      .  .  ..    .    ;
						                       /  ~~-.>_..   . .              `;
						                    __/ @@)qb'~;-,.  .                 |
						                  ,'  ~~ ( ,qb'|_.=              . .   |
						                  |      ;~|`qh   `-.           . . .  `
						                  |     ,dnb ch.     `-.             .  |
						                  `-.__- qHHb=qd        `-.          .  |
						                  JKKK =='7LL` qd          `-.          |
						                  dU |  \       qd            `-.   ..  |
						                 dU  |.  `-      qd..          . ` .    |
						              /~\U   ||        _._v' `          ; ;     |
						              L/|~,  `/      .  | |`-'  -_        `, .  ,
						                |/   ||      .   `'    ; ,`-.      ;   ;
						                     |`,     .        _.' .' |     ; . |
						                     |`.;~-_._____..-; _,'   |     ;   ;
						                     `.| ~-;,--,.;_..-~    O |     `, / 
						                       | O | /\ /       O  _,;      ;/ 
						                       '\_/| \/ \  O   _.-;  .`      ; 
						                      |... ,-~~'.;__.-;..... ..`-._.-
						                      |... ;-~~`.\:    ..... ....|
						                      |... |     ]:    ..... ....|
						                      `....;      `;   .~~;  ...,;
						                      `....;      `;  .' .  ,....'
						                       `,..`,      ,; ; . `'... ;
						                        `,..;      |;.'. |..., , 
						                          `,;,     |,\..'I   .' 
						                          ,' `,      \|,'~--~ 
						                          '    ,     || ;
						                          `._  |    =|/ ,
						                             ~-|    ;  ;
						                          ,--~~`.   ; '
						                          ;:'~~  ~-~` ;
						                          `-.________,'
*/						              
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