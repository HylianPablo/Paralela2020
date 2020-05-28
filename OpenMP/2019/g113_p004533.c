// User: g113@88.4.70.174 
// ExecutionRequest[P:'DEJAVU.c',P:1,T:1,args:'',q:'openmplb'] 
// Mar 15 2019 17:34:25
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
#include<cputils.h>
#include<omp.h>

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
/*CHANGE the base argument in this function takes the base index for the row we 
want to access, allowing us to only calculate that index once for each row of the
matrix*/
#define accessMat2( arr, base, exp2 )	arr[ (base) + (exp2) ]


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

	/* 2. Start global timer */
	double ttotal = cp_Wtime();
	
/*
 *
 * START HERE: DO NOT CHANGE THE CODE ABOVE THIS POINT
 *
 */	
	#ifdef PROFILER
		/*CHANGE This variable is used to measure the times from
		each section*/
		double section_timer;
		double subsection_timer;
		/*CHANGE these variables are used to store the time from each
		section between iterations*/
		double focal_point_time = 0;
		double heat_propagation_time = 0;
		double team_movement_time = 0;
		double team_actions_time = 0;
		double heat_propagation_1 = 0;
		double heat_propagation_2 = 0;
		double heat_propagation_3 = 0;
		double heat_propagation_4 = 0;
	#endif
	/*
	int row_base = 0;
	int row_base_prev = 0;
	int row_base_next = 0;
	*/
	float *surface_pointer;
	float *copy_pointer;

	/* 3. Initialize surface */
	#ifdef PROFILER
		section_timer = omp_get_wtime();
	#endif
	#pragma omp parallel for 
	for( i=0; i<rows; i++ )
		for( j=0; j<columns; j++ )
			accessMat( surface, i, j ) = 0;

	#ifdef PROFILER
		section_timer = omp_get_wtime() - section_timer;
		printf("Initialization time: %lf\n", section_timer);
	#endif

	/* 4. Simulation */
	int iter;
	//int flag_stability = 0;
	int first_activation = 0;
	float global_residual = 0.0;
	int num_deactivated = 0;
	int treshold_surpassed = 0;
	
	const int RADIUS_TYPE_1_SQUARED = RADIUS_TYPE_1*RADIUS_TYPE_1;
	const int RADIUS_TYPE_2_3_SQUARED = RADIUS_TYPE_2_3*RADIUS_TYPE_2_3;
	int columns_not_even = columns%2;

	for( iter=0; iter<max_iter; iter++ ) {

		/* 4.1. Activate focal points */
		#ifdef PROFILER
			section_timer = omp_get_wtime();
		#endif

		num_deactivated = 0;
		#pragma omp parallel for reduction(+:num_deactivated)
		for( i=0; i<num_focal; i++ ) {
			if ( focal[i].start == iter ) {
				focal[i].active = 1;
				if ( ! first_activation ) first_activation = 1;
			}
			// Count focal points already deactivated by a team
			if ( focal[i].active == 2 ) num_deactivated++;
		}
		#ifdef PROFILER
			focal_point_time += omp_get_wtime() - section_timer;
		#endif

		if(first_activation){
			break;
		}

		#ifdef DEBUG
				/* 4.5. DEBUG: Print the current state of the simulation at the end of each iteration */
				print_status( iter, rows, columns, surface, num_teams, teams, num_focal, focal, global_residual );
		#endif // DEBUG
	}

	int all_deactivated = 0;

	for( ; iter<max_iter; iter++ ) {
		/* 4.1. Activate focal points */
		#ifdef PROFILER
			section_timer = omp_get_wtime();
		#endif

		num_deactivated = 0;
		#pragma omp parallel for reduction(+:num_deactivated)
		for( i=0; i<num_focal; i++ ) {
			if ( focal[i].start == iter ) {
				focal[i].active = 1;
				if ( ! first_activation ) first_activation = 1;
			}
			// Count focal points already deactivated by a team
			if ( focal[i].active == 2 ) num_deactivated++;
		}
		#ifdef PROFILER
			focal_point_time += omp_get_wtime() - section_timer;
		#endif
		/* 4.2. Propagate heat (10 steps per each team movement) */
		#pragma omp parallel shared(surface, surfaceCopy) private(i, j)
		{
			#ifdef PROFILER
				#pragma omp master
				section_timer = omp_get_wtime();
			#endif

			global_residual = 0.0;
			treshold_surpassed = 0;
			int step;
			for( step=0; step<10; step++ ){
				if(step%2 == 0){
					/* 4.2.1. Update heat on active focal points */
					#ifdef PROFILER
						#pragma omp master
						subsection_timer = omp_get_wtime();
					#endif

					for( i=0; i<num_focal; i++ ) {
						if ( focal[i].active != 1 ) continue;
						accessMat( surface, focal[i].x, focal[i].y ) = focal[i].heat;
					}

					#ifdef PROFILER
						#pragma omp master
						heat_propagation_1 += omp_get_wtime() - subsection_timer;
					#endif

					/* 4.2.2. Copy values of the surface in ancillary structure (Skip borders) */
					/* 4.2.3. Update surface values (skip borders) */
					/* I merged both loops */
					#ifdef PROFILER
						#pragma omp master
						subsection_timer = omp_get_wtime();
					#endif

					#pragma omp for
					for( i=1; i<rows-1; i++ ){
						surface_pointer = surface + (i * columns);
						copy_pointer = surfaceCopy + (i * columns);
						for( j=1; j<columns-1; j++ ){
							surface_pointer++;
							copy_pointer++;
							*copy_pointer = ( 
								*(surface_pointer - columns) +
								*(surface_pointer + columns) +
								*(surface_pointer - 1) +
								*(surface_pointer + 1) ) / 4;
						}
					}
					#ifdef PROFILER
						#pragma omp master
						heat_propagation_3 += omp_get_wtime() - subsection_timer;
					#endif

				}else{
					/* 4.2.1. Update heat on active focal points */
					#ifdef PROFILER
						#pragma omp master
						subsection_timer = omp_get_wtime();
					#endif

					for( i=0; i<num_focal; i++ ) {
						if ( focal[i].active != 1 ) continue;
						accessMat( surfaceCopy, focal[i].x, focal[i].y ) = focal[i].heat;
					}
					#ifdef PROFILER
						#pragma omp master
						heat_propagation_1 += omp_get_wtime() - subsection_timer;
					#endif

					/* 4.2.2. Copy values of the surface in ancillary structure (Skip borders) */
					/* 4.2.3. Update surface values (skip borders) */
					/* I merged both loops */
					#ifdef PROFILER
						#pragma omp master
						subsection_timer = omp_get_wtime();
					#endif

					#pragma omp for
					for( i=1; i<rows-1; i++ ){
						copy_pointer = surface + (i * columns);
						surface_pointer = surfaceCopy + (i * columns);
						for( j=1; j<columns-1; j++ ){
							surface_pointer++;
							copy_pointer++;
							*copy_pointer = ( 
								*(surface_pointer - columns) +
								*(surface_pointer + columns) +
								*(surface_pointer - 1) +
								*(surface_pointer + 1) ) / 4;
						}
					}
					#ifdef PROFILER
						#pragma omp master
						heat_propagation_3 += omp_get_wtime() - subsection_timer;
					#endif
				}
			}
			#ifdef PROFILER
				#pragma omp master
				heat_propagation_time += omp_get_wtime() - section_timer;
			#endif
		}

		/* 4.3. Move teams */
		#ifdef PROFILER
			section_timer = omp_get_wtime();
		#endif

		#pragma omp parallel for private(j) 
		for( t=0; t<num_teams; t++ ) {
			/* 4.3.1. Choose nearest focal point */
			int distance = 2147483647;
			int target = -1;

			for( j=0; j<num_focal; j++ ) {
				if ( focal[j].active != 1 ) continue; // Skip non-active focal points
				//int dx = focal[j].x - teams[t].x;
				//int dy = focal[j].y - teams[t].y;
				//float local_distance = sqrtf( dx*dx + dy*dy );
				int local_distance = (focal[j].x - teams[t].x)*(focal[j].x - teams[t].x) + (focal[j].y - teams[t].y)*(focal[j].y - teams[t].y);
				//int local_distance = dx*dx + dy*dy;
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
		#ifdef PROFILER
			team_movement_time += omp_get_wtime() - section_timer;
		#endif
		

		/* 4.4. Team actions */
		#ifdef PROFILER
			section_timer = omp_get_wtime();
		#endif

		#pragma omp parallel for private(i, j)
		for( t=0; t<num_teams; t++ ) {
			/* 4.4.1. Deactivate the target focal point when it is reached */
			int target = teams[t].target;
			if ( target != -1 && focal[target].x == teams[t].x && focal[target].y == teams[t].y 
				&& focal[target].active == 1 ){
				focal[target].active = 2;
				if(num_deactivated + 1 == num_focal){
					all_deactivated = 1;
				}
				//#pragma omp atomic
				//num_deactivated++;
			}

			/* 4.4.2. Reduce heat in a circle around the team */
			// Influence area of fixed radius depending on type
			if ( teams[t].type == 1 ){
				for( i=teams[t].x-RADIUS_TYPE_1; i<=teams[t].x+RADIUS_TYPE_1; i++ ) {
					for( j=teams[t].y-RADIUS_TYPE_1; j<=teams[t].y+RADIUS_TYPE_1; j++ ) {
						if ( i<1 || i>=rows-1 || j<1 || j>=columns-1 ) continue; // Out of the heated surface
						//int dx = teams[t].x - i;
						//int dy = teams[t].y - j;
						//float distance = dx*dx + dy*dy;
						if ( (teams[t].x - i)*(teams[t].x - i) + (teams[t].y - j)*(teams[t].y - j) <= RADIUS_TYPE_1_SQUARED ) {
							#pragma omp atomic
							accessMat( surface, i, j ) = accessMat( surface, i, j ) * ( 0.75 ); // Team efficiency factor
						}
					}
				}
			}else{
				for( i=teams[t].x-RADIUS_TYPE_2_3; i<=teams[t].x+RADIUS_TYPE_2_3; i++ ) {
					for( j=teams[t].y-RADIUS_TYPE_2_3; j<=teams[t].y+RADIUS_TYPE_2_3; j++ ) {
						if ( i<1 || i>=rows-1 || j<1 || j>=columns-1 ) continue; // Out of the heated surface
						//int dx = teams[t].x - i;
						//int dy = teams[t].y - j;
						//float distance = dx*dx + dy*dy;
						if ( (teams[t].x - i)*(teams[t].x - i) + (teams[t].y - j)*(teams[t].y - j) <= RADIUS_TYPE_2_3_SQUARED ) {
							#pragma omp atomic
							accessMat( surface, i, j ) = accessMat( surface, i, j ) * ( 0.75 ); // Team efficiency factor
						}
					}
				}
			}
			
		}
		#ifdef PROFILER
			team_actions_time += omp_get_wtime() - section_timer;
		#endif

		if(all_deactivated){
			iter++;
			break;
		}
	}

	for( ; iter<max_iter; iter++ ) {
		/* 4.2. Propagate heat (10 steps per each team movement) */
		#pragma omp parallel shared(surface, surfaceCopy) private(i, j)
		{
			#ifdef PROFILER
				#pragma omp master
				section_timer = omp_get_wtime();
			#endif

			global_residual = 0.0;
			treshold_surpassed = 0;
			int step;
			for( step=0; step<10; step++ ){
				if(step%2 == 0){
					/* 4.2.1. Update heat on active focal points */
					#ifdef PROFILER
						#pragma omp master
						subsection_timer = omp_get_wtime();
					#endif

					#ifdef PROFILER
						#pragma omp master
						heat_propagation_1 += omp_get_wtime() - subsection_timer;
					#endif

					/* 4.2.2. Copy values of the surface in ancillary structure (Skip borders) */
					/* 4.2.3. Update surface values (skip borders) */
					/* I merged both loops */
					#ifdef PROFILER
						#pragma omp master
						subsection_timer = omp_get_wtime();
					#endif

					#pragma omp for
					for( i=1; i<rows-1; i++ ){
						surface_pointer = surface + (i * columns);
						copy_pointer = surfaceCopy + (i * columns);
						for( j=1; j<columns-1; j++ ){
							surface_pointer++;
							copy_pointer++;
							*copy_pointer = ( 
								*(surface_pointer - columns) +
								*(surface_pointer + columns) +
								*(surface_pointer - 1) +
								*(surface_pointer + 1) ) / 4;
						}
					}
					#ifdef PROFILER
						#pragma omp master
						heat_propagation_3 += omp_get_wtime() - subsection_timer;
					#endif

					/* 4.2.4. Compute the maximum residual difference (absolute value) */
					#ifdef PROFILER
						#pragma omp master
						subsection_timer = omp_get_wtime();
					#endif

					if(!treshold_surpassed){
						#pragma omp for nowait reduction(max:global_residual)
						for( i=1; i<rows-1; i++ ){
							for( j=1; j<columns-1; j++ ){
								if ( accessMat( surfaceCopy, i, j ) - accessMat( surface, i, j ) > global_residual ) {
									global_residual = accessMat( surfaceCopy, i, j ) - accessMat( surface, i, j );
									if(global_residual >= THRESHOLD){
										treshold_surpassed = 1;
										break;
									}
								}
							}
						}
					}

					#ifdef PROFILER
						#pragma omp master
						heat_propagation_4 += omp_get_wtime() - subsection_timer;
					#endif

				}else{
					/* 4.2.1. Update heat on active focal points */
					#ifdef PROFILER
						#pragma omp master
						subsection_timer = omp_get_wtime();
					#endif

					for( i=0; i<num_focal; i++ ) {
						if ( focal[i].active != 1 ) continue;
						accessMat( surfaceCopy, focal[i].x, focal[i].y ) = focal[i].heat;
					}
					#ifdef PROFILER
						#pragma omp master
						heat_propagation_1 += omp_get_wtime() - subsection_timer;
					#endif

					/* 4.2.2. Copy values of the surface in ancillary structure (Skip borders) */
					/* 4.2.3. Update surface values (skip borders) */
					/* I merged both loops */
					#ifdef PROFILER
						#pragma omp master
						subsection_timer = omp_get_wtime();
					#endif

					#pragma omp for
					for( i=1; i<rows-1; i++ ){
						copy_pointer = surface + (i * columns);
						surface_pointer = surfaceCopy + (i * columns);
						for( j=1; j<columns-1; j++ ){
							surface_pointer++;
							copy_pointer++;
							*copy_pointer = ( 
								*(surface_pointer - columns) +
								*(surface_pointer + columns) +
								*(surface_pointer - 1) +
								*(surface_pointer + 1) ) / 4;
						}
					}
					#ifdef PROFILER
						#pragma omp master
						heat_propagation_3 += omp_get_wtime() - subsection_timer;
					#endif

					/* 4.2.4. Compute the maximum residual difference (absolute value) */
					#ifdef PROFILER
						#pragma omp master
						subsection_timer = omp_get_wtime();
					#endif
					
					if(!treshold_surpassed){
						#pragma omp for nowait reduction(max:global_residual)
						for( i=1; i<rows-1; i++ ){
							for( j=1; j<columns-1; j++ ){
								if ( accessMat( surface, i, j ) - accessMat( surfaceCopy, i, j ) > global_residual ) {
									global_residual = accessMat( surface, i, j ) - accessMat( surfaceCopy, i, j );
									if(global_residual >= THRESHOLD){
										treshold_surpassed = 1;
										break;
									}
								}
							}
						}
					}

					#ifdef PROFILER
						#pragma omp master
						heat_propagation_4 += omp_get_wtime() - subsection_timer;
					#endif
				}
			}
			#ifdef PROFILER
				#pragma omp master
				heat_propagation_time += omp_get_wtime() - section_timer;
			#endif
		}

		/* If the global residual is lower than THRESHOLD, we have reached enough stability, stop simulation at the end of this iteration */
		
		if( global_residual < THRESHOLD ){
			//flag_stability = 1;
			/* 4.4. Team actions */
			#ifdef PROFILER
				section_timer = omp_get_wtime();
			#endif

			#pragma omp parallel for private(i, j)
			for( t=0; t<num_teams; t++ ) {
				/* 4.4.1. Deactivate the target focal point when it is reached */
				int target = teams[t].target;
				if ( target != -1 && focal[target].x == teams[t].x && focal[target].y == teams[t].y 
					&& focal[target].active == 1 ){
					focal[target].active = 2;
					//#pragma omp atomic
					//num_deactivated++;
				}

				/* 4.4.2. Reduce heat in a circle around the team */
				// Influence area of fixed radius depending on type
				if ( teams[t].type == 1 ){
					for( i=teams[t].x-RADIUS_TYPE_1; i<=teams[t].x+RADIUS_TYPE_1; i++ ) {
						for( j=teams[t].y-RADIUS_TYPE_1; j<=teams[t].y+RADIUS_TYPE_1; j++ ) {
							if ( i<1 || i>=rows-1 || j<1 || j>=columns-1 ) continue; // Out of the heated surface
							//int dx = teams[t].x - i;
							//int dy = teams[t].y - j;
							//float distance = dx*dx + dy*dy;
							if ( (teams[t].x - i)*(teams[t].x - i) + (teams[t].y - j)*(teams[t].y - j) <= RADIUS_TYPE_1_SQUARED ) {
								#pragma omp atomic
								accessMat( surface, i, j ) = accessMat( surface, i, j ) * ( 0.75 ); // Team efficiency factor
							}
						}
					}
				}else{
					for( i=teams[t].x-RADIUS_TYPE_2_3; i<=teams[t].x+RADIUS_TYPE_2_3; i++ ) {
						for( j=teams[t].y-RADIUS_TYPE_2_3; j<=teams[t].y+RADIUS_TYPE_2_3; j++ ) {
							if ( i<1 || i>=rows-1 || j<1 || j>=columns-1 ) continue; // Out of the heated surface
							//int dx = teams[t].x - i;
							//int dy = teams[t].y - j;
							//float distance = dx*dx + dy*dy;
							if ( (teams[t].x - i)*(teams[t].x - i) + (teams[t].y - j)*(teams[t].y - j) <= RADIUS_TYPE_2_3_SQUARED ) {
								#pragma omp atomic
								accessMat( surface, i, j ) = accessMat( surface, i, j ) * ( 0.75 ); // Team efficiency factor
							}
						}
					}
				}
			}
			#ifdef PROFILER
				team_actions_time += omp_get_wtime() - section_timer;
			#endif
			//Since the for is not checked again, this variable doesn't update
			iter++;
			break;
		}

		/* 4.4. Team actions */
		#ifdef PROFILER
			section_timer = omp_get_wtime();
		#endif

		#pragma omp parallel for private(i, j)
		for( t=0; t<num_teams; t++ ) {
			/* 4.4.1. Deactivate the target focal point when it is reached */
			int target = teams[t].target;
			if ( target != -1 && focal[target].x == teams[t].x && focal[target].y == teams[t].y 
				&& focal[target].active == 1 ){
				focal[target].active = 2;
			}

			/* 4.4.2. Reduce heat in a circle around the team */
			// Influence area of fixed radius depending on type
			if ( teams[t].type == 1 ){
				for( i=teams[t].x-RADIUS_TYPE_1; i<=teams[t].x+RADIUS_TYPE_1; i++ ) {
					for( j=teams[t].y-RADIUS_TYPE_1; j<=teams[t].y+RADIUS_TYPE_1; j++ ) {
						if ( i<1 || i>=rows-1 || j<1 || j>=columns-1 ) continue; // Out of the heated surface
						//int dx = teams[t].x - i;
						//int dy = teams[t].y - j;
						//float distance = dx*dx + dy*dy;
						if ( (teams[t].x - i)*(teams[t].x - i) + (teams[t].y - j)*(teams[t].y - j) <= RADIUS_TYPE_1_SQUARED ) {
							#pragma omp atomic
							accessMat( surface, i, j ) = accessMat( surface, i, j ) * ( 0.75 ); // Team efficiency factor
						}
					}
				}
			}else{
				for( i=teams[t].x-RADIUS_TYPE_2_3; i<=teams[t].x+RADIUS_TYPE_2_3; i++ ) {
					for( j=teams[t].y-RADIUS_TYPE_2_3; j<=teams[t].y+RADIUS_TYPE_2_3; j++ ) {
						if ( i<1 || i>=rows-1 || j<1 || j>=columns-1 ) continue; // Out of the heated surface
						//int dx = teams[t].x - i;
						//int dy = teams[t].y - j;
						//float distance = dx*dx + dy*dy;
						if ( (teams[t].x - i)*(teams[t].x - i) + (teams[t].y - j)*(teams[t].y - j) <= RADIUS_TYPE_2_3_SQUARED ) {
							#pragma omp atomic
							accessMat( surface, i, j ) = accessMat( surface, i, j ) * ( 0.75 ); // Team efficiency factor
						}
					}
				}
			}
			
		}
		#ifdef PROFILER
			team_actions_time += omp_get_wtime() - section_timer;
		#endif
	}


///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////
//////////     ///////////     ////////////////////////////////////////////////////////
//////////  // ///////////  // ////////////////////////////////////////////////////////
//////////  // ///////////  // ////////////////////////////////////////////////////////
//////////  // ///////////  // /////////////////////////////// ////////////////////////
//////////////////////////////////////////////////////////// //////////////////////////
/////////////////////////////////////////////////////           ///////////////////////
//////////////////////////////////////////////////   ////  ////  //////////////////////
////////    /////////////////    ///////////////   /////////////  /////////////////////
/////////    //////////////    ////////////////  ///////////////  /////////////////////
///////////    /////////     /////////////////  ///////////////  //////////////////////
/////////////            //   ////////////////  /////////////  ////////////////////////
//////////////////////    //   ////////////////  //////////  ///////<---Manzana////////
////////////////////////   //  /////////////////          /////////////////////////////
//////////////////////////    /////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

/*
 *
 * STOP HERE: DO NOT CHANGE THE CODE BELOW THIS POINT
 *
 */

	/* 5. Stop global time */
	ttotal = cp_Wtime() - ttotal;
	/////////////////////////////////////////////////////////////////
	#ifdef PROFILER
		printf("Focal point activation time: %lf [%lf%%]\n", focal_point_time, focal_point_time/ttotal * 100);
		printf("Heat propagation time: %lf [%lf%%]\n", heat_propagation_time, heat_propagation_time/ttotal * 100);
		printf("\t4.2.1 time: %lf [%lf%%]\n", heat_propagation_1, heat_propagation_1/ttotal * 100);
		printf("\t4.2.2 time: %lf [%lf%%]\n", heat_propagation_2, heat_propagation_2/ttotal * 100);
		printf("\t4.2.3 time: %lf [%lf%%]\n", heat_propagation_3, heat_propagation_3/ttotal * 100);
		printf("\t4.2.4 time: %lf [%lf%%]\n", heat_propagation_4, heat_propagation_4/ttotal * 100);
		printf("Team movement time: %lf [%lf%%]\n", team_movement_time, team_movement_time/ttotal * 100);
		printf("Team actions time: %lf [%lf%%]\n", team_actions_time, team_actions_time/ttotal * 100);
	#endif
	/////////////////////////////////////////////////////////////////

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
