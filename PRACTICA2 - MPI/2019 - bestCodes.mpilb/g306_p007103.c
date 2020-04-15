// User: g306@146.158.176.103 
// ExecutionRequest[P:'elworbu.c',P:1,T:1,args:'',q:'mpilb'] 
// Apr 04 2019 20:08:33
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
 	float *residualHeat_aux = (float*)malloc( sizeof(float) * (size_t)num_focal );
 	memset(residualHeat_aux, 0, num_focal*sizeof(float));	// Para que no de problemas al reducir

	int iter;
	int num_procs, resto, num_deactivated;
	int total_rows = rows;

	// Numero de procesos
	MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
	char primero = rank == 0;
	char ultimo = rank == (num_procs-1);
	char n = primero + ultimo;

	// Particionar matriz por filas
	if (rows >= num_procs) {
		resto = rows%num_procs;
		rows = rows/num_procs;
		if (rank < resto)
			rows++;
	} else {
		MPI_Abort( MPI_COMM_WORLD,  0 );
	}


	// Focos dentro de la particion
	int *focal_local = (int *)malloc( sizeof(int) * (size_t)num_focal );

	int initial_row = rank*rows;
	if (rank >= resto && resto != 0)
		initial_row+=resto;
	int final_row = initial_row + rows - 1;

	int num_focal_local = 0;
	for (i=0; i<num_focal; i++)
		if (focal[i].x >= initial_row && focal[i].x <= final_row) {
			focal_local[num_focal_local] = i;
			num_focal_local++;
		}

	if (!n)
		rows+=2;
	else
		rows++;

	float *surface, *surfaceCopy, *aux;
	surface = (float *)malloc( sizeof(float) * (size_t)(rows) * (size_t)columns );
	surfaceCopy = (float *)malloc( sizeof(float) * (size_t)(rows) * (size_t)columns );

	if ( surface == NULL || surfaceCopy == NULL) {
		fprintf(stderr,"-- Error allocating: surface structures\n");
		MPI_Abort( MPI_COMM_WORLD,  EXIT_FAILURE );
	}

	// 3
	// Inicializar matrices
	for( i=0; i<rows; i++ )
		for( j=0; j<columns; j++ )
			accessMat( surface, i, j ) = 0.0;

	for( i=0; i<rows; i++ )
		for( j=0; j<columns; j++ )
			accessMat( surfaceCopy, i, j ) = 0.0;

	// 4
	int flag_stability = 0;
	int treshold_surpassed;
	for( iter=0; iter<max_iter && ! flag_stability; iter++ ) {

		// 4.1
		num_deactivated = 0;
		int num_activated = 0;
		treshold_surpassed = 0;
		for( i=0; i<num_focal; i++ ) {
			if ( focal[i].start == iter ) {
				focal[i].active = 1;
				num_activated++;
			}
			if ( focal[i].active == 2 ) {
				num_deactivated++;
			}
		}

		// 4.2
		float global_residual = 0.0f;
		float global_residual_aux;
		int step;
		for( step=0; step<10; step++ )	{
			// 4.2.1
			for( i=0; i<num_focal_local; i++ ) {
				if ( focal[focal_local[i]].active != 1 ) continue;
				int x = focal[focal_local[i]].x - initial_row + 1 - primero;
				int y = focal[focal_local[i]].y;
				accessMat( surface, x, y ) = focal[focal_local[i]].heat;
			}

			// 4.2.2
			/*for( i=1; i<rows-1; i++ )
				for( j=1; j<columns-1; j++ )
					accessMat( surfaceCopy, i, j ) = accessMat( surface, i, j );*/
			aux = surface;
			surface = surfaceCopy;
			surfaceCopy = aux;
			for (i=0; i<num_focal_local; i++) {
				FocalPoint *focal_i = &focal[focal_local[i]];
				if (focal_i->x == 0 || focal_i->x == total_rows-1 || focal_i->y == 0 || focal_i->y == columns-1) {
						accessMat( surfaceCopy, focal_i->x - initial_row + 1 - primero, focal_i->y ) = 0.0f;
						accessMat( surface, focal_i->x - initial_row + 1 - primero, focal_i->y ) = focal_i->heat;
				}
			}

			// Enviar
			if (!primero)
				MPI_Send( (void *)&accessMat( surfaceCopy, 1, 0), columns, MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD);
			if (!ultimo)
				MPI_Send( (void *)&accessMat( surfaceCopy, rows-2, 0), columns, MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD);
			// Recibir
			if (!primero)
				MPI_Recv( (void *)&accessMat( surfaceCopy, 0, 0), columns, MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			if (!ultimo)
				MPI_Recv( (void *)&accessMat( surfaceCopy, rows-1, 0), columns, MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

			// 4.2.3
			// Primera y ultima particion no tiene frontera, pero tienen borde
			for( i=1; i<rows-1; i++ )
				for( j=1; j<columns-1; j++ )
					accessMat( surface, i, j ) = (
						accessMat( surfaceCopy, i-1, j ) +
						accessMat( surfaceCopy, i+1, j ) +
						accessMat( surfaceCopy, i, j-1 ) +
						accessMat( surfaceCopy, i, j+1 ) ) / 4;
			// 4.2.4
			/*for( i=1; i<rows-1; i++ )
				for( j=1; j<columns-1; j++ )
					if ( accessMat( surface, i, j ) - accessMat( surfaceCopy, i, j ) > global_residual ) {
						global_residual = accessMat( surface, i, j ) - accessMat( surfaceCopy, i, j );
					}*/
			if(!treshold_surpassed || ((num_deactivated + num_activated) == 0)) {
				for( i=1; i<rows-1; i++ ){
					for( j=1; j<columns-1; j++ ){
						if ( fabs( accessMat( surface, i, j ) - accessMat( surfaceCopy, i, j ) > global_residual ) ) {
							global_residual = fabs( accessMat( surface, i, j ) - accessMat( surfaceCopy, i, j ) );
							if(global_residual >= THRESHOLD){
								treshold_surpassed = 1;
								break;
							}
						}
					}
				}
			}

			//MPI_Allreduce( (void *)&global_residual, (void *)&global_residual_aux, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
		}

		MPI_Allreduce( (void *)&global_residual, (void *)&global_residual_aux, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);

		if( num_deactivated/*_aux*/ == num_focal && global_residual_aux < THRESHOLD ) flag_stability = 1;

		// 4.3
		//if (((num_activated - num_deactivated) > 0) || (num_deactivated != num_focal)) {
			for( t=0; t<num_teams; t++ ) {
				Team *team_t = &teams[t];
				// 4.3.1
				float distance = FLT_MAX;
				int target = -1;
				for( j=0; j<num_focal; j++ ) {
					if ( focal[j].active != 1 ) continue;
					float dx = focal[j].x - team_t->x;
					float dy = focal[j].y - team_t->y;
					float local_distance = sqrtf( dx*dx + dy*dy );
					if ( local_distance < distance ) {
						distance = local_distance;
						target = j;
					}
				}
				// 4.3.2
				team_t->target = target;
	
				// 4.3.3
				if ( target == -1 ) continue;
				// 4.3.4
				switch ( teams[t].type) {
					case 1:
						if		( focal[target].x < team_t->x )	team_t->x--;
						else if	( focal[target].x > team_t->x )	team_t->x++;
						if		( focal[target].y < team_t->y )	team_t->y--;
						else if	( focal[target].y > team_t->y )	team_t->y++;
						break;
					case 2:
						if		( focal[target].y < team_t->y )	team_t->y--;
						else if	( focal[target].y > team_t->y )	team_t->y++;
						else if	( focal[target].x < team_t->x )	team_t->x--;
						else if	( focal[target].x > team_t->x )	team_t->x++;
						break;
					default:
						if		( focal[target].x < team_t->x )	team_t->x--;
						else if	( focal[target].x > team_t->x )	team_t->x++;
						else if	( focal[target].y < team_t->y )	team_t->y--;
						else if	( focal[target].y > team_t->y )	team_t->y++;
						break;
				}
			}
		//}

		for( t=0; t<num_teams; t++ ) {
			// 4.4.1
			Team *team_t = &teams[t];
			int target = team_t->target;
			if ( target != -1 && focal[target].x == team_t->x && focal[target].y == team_t->y && focal[target].active == 1 )
				focal[target].active = 2;

			// 4.4.2
			int radius;
			if ( team_t->type == 1 ) radius = RADIUS_TYPE_1;
			else radius = RADIUS_TYPE_2_3;
			for( i=team_t->x-radius; i<=team_t->x+radius; i++ ) {
				// Si esta dentro de la particion
				if (i >= initial_row && i <= final_row ) {
					for( j=team_t->y-radius; j<=team_t->y+radius; j++ ) {
						if ( i<1 || j<1 || j>=columns-1 ) continue;
						float dx = team_t->x - i;
						float dy = team_t->y - j;
						float distance = sqrtf( dx*dx + dy*dy );
						if ( distance <= radius ) {
							int x = i - initial_row + 1 - primero;
							accessMat( surface, x, j ) = accessMat( surface, x, j ) * ( 0.75 );
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

	// MPI Version: Store the results of residual heat on focal points
	for( i=0; i<num_focal_local; i++ ) {
		int x = focal[focal_local[i]].x - initial_row + 1 - primero;
		int y = focal[focal_local[i]].y;
		residualHeat_aux[focal_local[i]] = accessMat( surface, x, y);
	}

	// Sincronizar residualHeat
	MPI_Reduce((void *)residualHeat_aux, (void *)residualHeat, num_focal, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

	/*if (rank == 0) {
		for( i=0; i<num_focal; i++ ) {
			FocalPoint *focal_i = &focal[i];
			if (focal_i->x == 0 || focal_i->x == total_rows-1 || focal_i->y == 0 || focal_i->y == columns-1)
				residualHeat[i] = focal_i->heat;
		}
	}*/

	free( surface );
	free( surfaceCopy );
	free( residualHeat_aux );
	free( focal_local );

// MPI Version: Eliminate this conditional-end to start doing the work in parallel
//}


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
	return 0;
}
