// User: g106@94.73.61.116 
// ExecutionRequest[P:'pipo.c',P:1,T:1,args:'',q:'openmplb'] 
// Mar 14 2019 18:49:11
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
	float *surfaceAuxiliar, *surfaceIntercambio;
	surfaceAuxiliar = (float *)malloc( sizeof(float) * (size_t)rows * (size_t)columns );

	/* 3. Initialize surface */
	#pragma omp parallel for private(j)  
	for( i=0; i<rows; i++ )
		for( j=0; j<columns; j++ )
			accessMat( surface, i, j ) = 0.0;
	

	/* 4. Simulation */
	int iter;
	int flag_stability = 0;
	int flag_deactivate = 0;
	
	
	//int first_activation = 0;
	for( iter=0; iter<max_iter && ! flag_stability; iter++ ) {
		float global_residual = 0.0f;
		if (!flag_deactivate){
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

			if (num_deactivated==num_focal)
				flag_deactivate=1;

			/* 4.2. Propagate heat (10 steps per each team movement) */
			
			int step;
			
			for( step=0; step<10; step++ )	{


				

				/* 4.2.1. Update heat on active focal points */
				for( i=0; i<num_focal; i++ ) {
					
					if ( focal[i].active != 1 ) continue;
					int x = focal[i].x;
					int y = focal[i].y; 
					accessMat( surface, x, y ) = focal[i].heat;
				}

				
				/* 4.2.2. Copy values of the surface in ancillary structure (Skip borders) */
				/*#pragma omp parallel for
				for( i=1; i<rows-1; i++ )
					for( j=1; j<columns-1; j++ )
						accessMat( surfaceCopy, i, j ) = accessMat( surface, i, j ); */

				/* 4.2.3. Update surface values (skip borders) */
				#pragma omp parallel for private(j) 
				for( i=1; i<rows-1; i++ )
					for( j=1; j<columns-1; j++ )
						accessMat( surfaceCopy, i, j ) = ( 
							accessMat( surface, i-1, j ) +
							accessMat( surface, i+1, j ) +
							accessMat( surface, i, j-1 ) +
							accessMat( surface, i, j+1 ) ) / 4;

			surfaceIntercambio=surface;
			surface=surfaceCopy;
			surfaceCopy=surfaceIntercambio;
			

			}

			/* 4.3. Move teams */ 
			#pragma omp parallel for if(num_teams>150) private(j)
			for( t=0; t<num_teams; t++ ) { 
				/* 4.3.1. Choose nearest focal point */
				float distance = FLT_MAX;
				int target = -1;
			
				
				for( j=0; j<num_focal; j++ ) {
					
					if ( focal[j].active != 1 ) continue; // Skip non-active focal points
					float dx = focal[j].x - teams[t].x;
					float dy = focal[j].y - teams[t].y;
					float local_distance = sqrtf( dx*dx + dy*dy );
					if ( local_distance < distance ) {
						distance = local_distance;
						target = j;
					}
				}
				
				/* 4.3.2. Annotate target for the next stage */
				teams[t].target = target;

				Team equipo = teams[t];
				FocalPoint foco = focal[target];

				/* 4.3.3. No active focal point to choose, no movement */
				if ( target == -1 ) continue;
				/* 4.3.4. Move in the focal point direction */
				if ( teams[t].type == 1 ) { 
					// Type 1: Can move in diagonal
					if ( foco.x < equipo.x ) teams[t].x--;
					else if ( foco.x > equipo.x ) teams[t].x++;
					if ( foco.y < equipo.y ) teams[t].y--;
					else if ( foco.y > equipo.y ) teams[t].y++;
				}
				else if ( teams[t].type == 2 ) { 
					// Type 2: First in horizontal direction, then in vertical direction
					if ( foco.y < equipo.y ) teams[t].y--;
					else if ( foco.y > equipo.y ) teams[t].y++;
					else if ( foco.x < equipo.x ) teams[t].x--;
					else if ( foco.x > equipo.x ) teams[t].x++;
				}
				else {
					// Type 3: First in vertical direction, then in horizontal direction
					if ( foco.x < equipo.x ) teams[t].x--;
					else if ( foco.x > equipo.x ) teams[t].x++;
					else if ( foco.y < equipo.y ) teams[t].y--;
					else if ( foco.y > equipo.y ) teams[t].y++;
				}
			}
			
			/* 4.4. Team actions */
					/**
					 * int target = teams[t].target;
			if ( target != -1 && focal[target].x == teams[t].x && focal[target].y == teams[t].y 
				&& focal[target].active == 1 )
				focal[target].active = 2;
				*/
			#pragma omp parallel for private(i,j)
			for( t=0; t<num_teams; t++ ) {
				int target = teams[t].target;
			if ( target != -1 && focal[target].x == teams[t].x && focal[target].y == teams[t].y 
				&& focal[target].active == 1 )
				focal[target].active = 2;
			/* 4.4.1. Deactivate the target focal point when it is reached */

			/* 4.4.2. Reduce heat in a circle around the team */
			int radius;
			i=teams[t].x;
			j=teams[t].y;
			// Influence area of fixed radius depending on type
			if ( teams[t].type == 1 ) {
			//radius = RADIUS_TYPE_1;
			//else radius = RADIUS_TYPE_2_3;
			/** Podemos cargarnos los bucles para cargarnos las comprobaciones */
			
				if(i-3>=1 && j+0>=1 && i-3<rows-1 && j+0<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-3, j+0 ) = accessMat( surface, i-3, j+0 ) * ( 0.75 );
				if(i-2>=1 && j-2>=1 && i-2<rows-1 && j-2<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-2, j-2 ) = accessMat( surface, i-2, j-2 ) * ( 0.75 );
				if(i-2>=1 && j-1>=1 && i-2<rows-1 && j-1<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-2, j-1 ) = accessMat( surface, i-2, j-1 ) * ( 0.75 );
				if(i-2>=1 && j+0>=1 && i-2<rows-1 && j+0<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-2, j+0 ) = accessMat( surface, i-2, j+0 ) * ( 0.75 );
				if(i-2>=1 && j+1>=1 && i-2<rows-1 && j+1<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-2, j+1 ) = accessMat( surface, i-2, j+1 ) * ( 0.75 );
				if(i-2>=1 && j+2>=1 && i-2<rows-1 && j+2<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-2, j+2 ) = accessMat( surface, i-2, j+2 ) * ( 0.75 );
				if(i-1>=1 && j-2>=1 && i-1<rows-1 && j-2<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-1, j-2 ) = accessMat( surface, i-1, j-2 ) * ( 0.75 );
				if(i-1>=1 && j-1>=1 && i-1<rows-1 && j-1<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-1, j-1 ) = accessMat( surface, i-1, j-1 ) * ( 0.75 );
				if(i-1>=1 && j+0>=1 && i-1<rows-1 && j+0<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-1, j+0 ) = accessMat( surface, i-1, j+0 ) * ( 0.75 );
				if(i-1>=1 && j+1>=1 && i-1<rows-1 && j+1<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-1, j+1 ) = accessMat( surface, i-1, j+1 ) * ( 0.75 );
				if(i-1>=1 && j+2>=1 && i-1<rows-1 && j+2<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-1, j+2 ) = accessMat( surface, i-1, j+2 ) * ( 0.75 );
				if(i+0>=1 && j-3>=1 && i+0<rows-1 && j-3<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+0, j-3 ) = accessMat( surface, i+0, j-3 ) * ( 0.75 );
				if(i+0>=1 && j-2>=1 && i+0<rows-1 && j-2<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+0, j-2 ) = accessMat( surface, i+0, j-2 ) * ( 0.75 );
				if(i+0>=1 && j-1>=1 && i+0<rows-1 && j-1<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+0, j-1 ) = accessMat( surface, i+0, j-1 ) * ( 0.75 );
				if(i+0>=1 && j+0>=1 && i+0<rows-1 && j+0<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+0, j+0 ) = accessMat( surface, i+0, j+0 ) * ( 0.75 );
				if(i+0>=1 && j+1>=1 && i+0<rows-1 && j+1<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+0, j+1 ) = accessMat( surface, i+0, j+1 ) * ( 0.75 );
				if(i+0>=1 && j+2>=1 && i+0<rows-1 && j+2<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+0, j+2 ) = accessMat( surface, i+0, j+2 ) * ( 0.75 );
				if(i+0>=1 && j+3>=1 && i+0<rows-1 && j+3<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+0, j+3 ) = accessMat( surface, i+0, j+3 ) * ( 0.75 );
				if(i+1>=1 && j-2>=1 && i+1<rows-1 && j-2<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+1, j-2 ) = accessMat( surface, i+1, j-2 ) * ( 0.75 );
				if(i+1>=1 && j-1>=1 && i+1<rows-1 && j-1<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+1, j-1 ) = accessMat( surface, i+1, j-1 ) * ( 0.75 );
				if(i+1>=1 && j+0>=1 && i+1<rows-1 && j+0<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+1, j+0 ) = accessMat( surface, i+1, j+0 ) * ( 0.75 );
				if(i+1>=1 && j+1>=1 && i+1<rows-1 && j+1<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+1, j+1 ) = accessMat( surface, i+1, j+1 ) * ( 0.75 );
				if(i+1>=1 && j+2>=1 && i+1<rows-1 && j+2<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+1, j+2 ) = accessMat( surface, i+1, j+2 ) * ( 0.75 );
				if(i+2>=1 && j-2>=1 && i+2<rows-1 && j-2<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+2, j-2 ) = accessMat( surface, i+2, j-2 ) * ( 0.75 );
				if(i+2>=1 && j-1>=1 && i+2<rows-1 && j-1<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+2, j-1 ) = accessMat( surface, i+2, j-1 ) * ( 0.75 );
				if(i+2>=1 && j+0>=1 && i+2<rows-1 && j+0<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+2, j+0 ) = accessMat( surface, i+2, j+0 ) * ( 0.75 );
				if(i+2>=1 && j+1>=1 && i+2<rows-1 && j+1<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+2, j+1 ) = accessMat( surface, i+2, j+1 ) * ( 0.75 );
				if(i+2>=1 && j+2>=1 && i+2<rows-1 && j+2<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+2, j+2 ) = accessMat( surface, i+2, j+2 ) * ( 0.75 );
				if(i+3>=1 && j+0>=1 && i+3<rows-1 && j+0<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+3, j+0 ) = accessMat( surface, i+3, j+0 ) * ( 0.75 );
			
			}
			else{
				if(i-9>=1 && j+0>=1 && i-9<rows-1 && j+0<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-9, j+0 ) = accessMat( surface, i-9, j+0 ) * ( 0.75 );
				if(i-8>=1 && j-4>=1 && i-8<rows-1 && j-4<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-8, j-4 ) = accessMat( surface, i-8, j-4 ) * ( 0.75 );
				if(i-8>=1 && j-3>=1 && i-8<rows-1 && j-3<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-8, j-3 ) = accessMat( surface, i-8, j-3 ) * ( 0.75 );
				if(i-8>=1 && j-2>=1 && i-8<rows-1 && j-2<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-8, j-2 ) = accessMat( surface, i-8, j-2 ) * ( 0.75 );
				if(i-8>=1 && j-1>=1 && i-8<rows-1 && j-1<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-8, j-1 ) = accessMat( surface, i-8, j-1 ) * ( 0.75 );
				if(i-8>=1 && j+0>=1 && i-8<rows-1 && j+0<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-8, j+0 ) = accessMat( surface, i-8, j+0 ) * ( 0.75 );
				if(i-8>=1 && j+1>=1 && i-8<rows-1 && j+1<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-8, j+1 ) = accessMat( surface, i-8, j+1 ) * ( 0.75 );
				if(i-8>=1 && j+2>=1 && i-8<rows-1 && j+2<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-8, j+2 ) = accessMat( surface, i-8, j+2 ) * ( 0.75 );
				if(i-8>=1 && j+3>=1 && i-8<rows-1 && j+3<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-8, j+3 ) = accessMat( surface, i-8, j+3 ) * ( 0.75 );
				if(i-8>=1 && j+4>=1 && i-8<rows-1 && j+4<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-8, j+4 ) = accessMat( surface, i-8, j+4 ) * ( 0.75 );
				if(i-7>=1 && j-5>=1 && i-7<rows-1 && j-5<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-7, j-5 ) = accessMat( surface, i-7, j-5 ) * ( 0.75 );
				if(i-7>=1 && j-4>=1 && i-7<rows-1 && j-4<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-7, j-4 ) = accessMat( surface, i-7, j-4 ) * ( 0.75 );
				if(i-7>=1 && j-3>=1 && i-7<rows-1 && j-3<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-7, j-3 ) = accessMat( surface, i-7, j-3 ) * ( 0.75 );
				if(i-7>=1 && j-2>=1 && i-7<rows-1 && j-2<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-7, j-2 ) = accessMat( surface, i-7, j-2 ) * ( 0.75 );
				if(i-7>=1 && j-1>=1 && i-7<rows-1 && j-1<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-7, j-1 ) = accessMat( surface, i-7, j-1 ) * ( 0.75 );
				if(i-7>=1 && j+0>=1 && i-7<rows-1 && j+0<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-7, j+0 ) = accessMat( surface, i-7, j+0 ) * ( 0.75 );
				if(i-7>=1 && j+1>=1 && i-7<rows-1 && j+1<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-7, j+1 ) = accessMat( surface, i-7, j+1 ) * ( 0.75 );
				if(i-7>=1 && j+2>=1 && i-7<rows-1 && j+2<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-7, j+2 ) = accessMat( surface, i-7, j+2 ) * ( 0.75 );
				if(i-7>=1 && j+3>=1 && i-7<rows-1 && j+3<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-7, j+3 ) = accessMat( surface, i-7, j+3 ) * ( 0.75 );
				if(i-7>=1 && j+4>=1 && i-7<rows-1 && j+4<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-7, j+4 ) = accessMat( surface, i-7, j+4 ) * ( 0.75 );
				if(i-7>=1 && j+5>=1 && i-7<rows-1 && j+5<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-7, j+5 ) = accessMat( surface, i-7, j+5 ) * ( 0.75 );
				if(i-6>=1 && j-6>=1 && i-6<rows-1 && j-6<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-6, j-6 ) = accessMat( surface, i-6, j-6 ) * ( 0.75 );
				if(i-6>=1 && j-5>=1 && i-6<rows-1 && j-5<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-6, j-5 ) = accessMat( surface, i-6, j-5 ) * ( 0.75 );
				if(i-6>=1 && j-4>=1 && i-6<rows-1 && j-4<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-6, j-4 ) = accessMat( surface, i-6, j-4 ) * ( 0.75 );
				if(i-6>=1 && j-3>=1 && i-6<rows-1 && j-3<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-6, j-3 ) = accessMat( surface, i-6, j-3 ) * ( 0.75 );
				if(i-6>=1 && j-2>=1 && i-6<rows-1 && j-2<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-6, j-2 ) = accessMat( surface, i-6, j-2 ) * ( 0.75 );
				if(i-6>=1 && j-1>=1 && i-6<rows-1 && j-1<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-6, j-1 ) = accessMat( surface, i-6, j-1 ) * ( 0.75 );
				if(i-6>=1 && j+0>=1 && i-6<rows-1 && j+0<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-6, j+0 ) = accessMat( surface, i-6, j+0 ) * ( 0.75 );
				if(i-6>=1 && j+1>=1 && i-6<rows-1 && j+1<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-6, j+1 ) = accessMat( surface, i-6, j+1 ) * ( 0.75 );
				if(i-6>=1 && j+2>=1 && i-6<rows-1 && j+2<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-6, j+2 ) = accessMat( surface, i-6, j+2 ) * ( 0.75 );
				if(i-6>=1 && j+3>=1 && i-6<rows-1 && j+3<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-6, j+3 ) = accessMat( surface, i-6, j+3 ) * ( 0.75 );
				if(i-6>=1 && j+4>=1 && i-6<rows-1 && j+4<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-6, j+4 ) = accessMat( surface, i-6, j+4 ) * ( 0.75 );
				if(i-6>=1 && j+5>=1 && i-6<rows-1 && j+5<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-6, j+5 ) = accessMat( surface, i-6, j+5 ) * ( 0.75 );
				if(i-6>=1 && j+6>=1 && i-6<rows-1 && j+6<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-6, j+6 ) = accessMat( surface, i-6, j+6 ) * ( 0.75 );
				if(i-5>=1 && j-7>=1 && i-5<rows-1 && j-7<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-5, j-7 ) = accessMat( surface, i-5, j-7 ) * ( 0.75 );
				if(i-5>=1 && j-6>=1 && i-5<rows-1 && j-6<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-5, j-6 ) = accessMat( surface, i-5, j-6 ) * ( 0.75 );
				if(i-5>=1 && j-5>=1 && i-5<rows-1 && j-5<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-5, j-5 ) = accessMat( surface, i-5, j-5 ) * ( 0.75 );
				if(i-5>=1 && j-4>=1 && i-5<rows-1 && j-4<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-5, j-4 ) = accessMat( surface, i-5, j-4 ) * ( 0.75 );
				if(i-5>=1 && j-3>=1 && i-5<rows-1 && j-3<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-5, j-3 ) = accessMat( surface, i-5, j-3 ) * ( 0.75 );
				if(i-5>=1 && j-2>=1 && i-5<rows-1 && j-2<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-5, j-2 ) = accessMat( surface, i-5, j-2 ) * ( 0.75 );
				if(i-5>=1 && j-1>=1 && i-5<rows-1 && j-1<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-5, j-1 ) = accessMat( surface, i-5, j-1 ) * ( 0.75 );
				if(i-5>=1 && j+0>=1 && i-5<rows-1 && j+0<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-5, j+0 ) = accessMat( surface, i-5, j+0 ) * ( 0.75 );
				if(i-5>=1 && j+1>=1 && i-5<rows-1 && j+1<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-5, j+1 ) = accessMat( surface, i-5, j+1 ) * ( 0.75 );
				if(i-5>=1 && j+2>=1 && i-5<rows-1 && j+2<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-5, j+2 ) = accessMat( surface, i-5, j+2 ) * ( 0.75 );
				if(i-5>=1 && j+3>=1 && i-5<rows-1 && j+3<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-5, j+3 ) = accessMat( surface, i-5, j+3 ) * ( 0.75 );
				if(i-5>=1 && j+4>=1 && i-5<rows-1 && j+4<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-5, j+4 ) = accessMat( surface, i-5, j+4 ) * ( 0.75 );
				if(i-5>=1 && j+5>=1 && i-5<rows-1 && j+5<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-5, j+5 ) = accessMat( surface, i-5, j+5 ) * ( 0.75 );
				if(i-5>=1 && j+6>=1 && i-5<rows-1 && j+6<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-5, j+6 ) = accessMat( surface, i-5, j+6 ) * ( 0.75 );
				if(i-5>=1 && j+7>=1 && i-5<rows-1 && j+7<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-5, j+7 ) = accessMat( surface, i-5, j+7 ) * ( 0.75 );
				if(i-4>=1 && j-8>=1 && i-4<rows-1 && j-8<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-4, j-8 ) = accessMat( surface, i-4, j-8 ) * ( 0.75 );
				if(i-4>=1 && j-7>=1 && i-4<rows-1 && j-7<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-4, j-7 ) = accessMat( surface, i-4, j-7 ) * ( 0.75 );
				if(i-4>=1 && j-6>=1 && i-4<rows-1 && j-6<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-4, j-6 ) = accessMat( surface, i-4, j-6 ) * ( 0.75 );
				if(i-4>=1 && j-5>=1 && i-4<rows-1 && j-5<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-4, j-5 ) = accessMat( surface, i-4, j-5 ) * ( 0.75 );
				if(i-4>=1 && j-4>=1 && i-4<rows-1 && j-4<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-4, j-4 ) = accessMat( surface, i-4, j-4 ) * ( 0.75 );
				if(i-4>=1 && j-3>=1 && i-4<rows-1 && j-3<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-4, j-3 ) = accessMat( surface, i-4, j-3 ) * ( 0.75 );
				if(i-4>=1 && j-2>=1 && i-4<rows-1 && j-2<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-4, j-2 ) = accessMat( surface, i-4, j-2 ) * ( 0.75 );
				if(i-4>=1 && j-1>=1 && i-4<rows-1 && j-1<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-4, j-1 ) = accessMat( surface, i-4, j-1 ) * ( 0.75 );
				if(i-4>=1 && j+0>=1 && i-4<rows-1 && j+0<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-4, j+0 ) = accessMat( surface, i-4, j+0 ) * ( 0.75 );
				if(i-4>=1 && j+1>=1 && i-4<rows-1 && j+1<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-4, j+1 ) = accessMat( surface, i-4, j+1 ) * ( 0.75 );
				if(i-4>=1 && j+2>=1 && i-4<rows-1 && j+2<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-4, j+2 ) = accessMat( surface, i-4, j+2 ) * ( 0.75 );
				if(i-4>=1 && j+3>=1 && i-4<rows-1 && j+3<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-4, j+3 ) = accessMat( surface, i-4, j+3 ) * ( 0.75 );
				if(i-4>=1 && j+4>=1 && i-4<rows-1 && j+4<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-4, j+4 ) = accessMat( surface, i-4, j+4 ) * ( 0.75 );
				if(i-4>=1 && j+5>=1 && i-4<rows-1 && j+5<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-4, j+5 ) = accessMat( surface, i-4, j+5 ) * ( 0.75 );
				if(i-4>=1 && j+6>=1 && i-4<rows-1 && j+6<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-4, j+6 ) = accessMat( surface, i-4, j+6 ) * ( 0.75 );
				if(i-4>=1 && j+7>=1 && i-4<rows-1 && j+7<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-4, j+7 ) = accessMat( surface, i-4, j+7 ) * ( 0.75 );
				if(i-4>=1 && j+8>=1 && i-4<rows-1 && j+8<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-4, j+8 ) = accessMat( surface, i-4, j+8 ) * ( 0.75 );
				if(i-3>=1 && j-8>=1 && i-3<rows-1 && j-8<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-3, j-8 ) = accessMat( surface, i-3, j-8 ) * ( 0.75 );
				if(i-3>=1 && j-7>=1 && i-3<rows-1 && j-7<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-3, j-7 ) = accessMat( surface, i-3, j-7 ) * ( 0.75 );
				if(i-3>=1 && j-6>=1 && i-3<rows-1 && j-6<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-3, j-6 ) = accessMat( surface, i-3, j-6 ) * ( 0.75 );
				if(i-3>=1 && j-5>=1 && i-3<rows-1 && j-5<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-3, j-5 ) = accessMat( surface, i-3, j-5 ) * ( 0.75 );
				if(i-3>=1 && j-4>=1 && i-3<rows-1 && j-4<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-3, j-4 ) = accessMat( surface, i-3, j-4 ) * ( 0.75 );
				if(i-3>=1 && j-3>=1 && i-3<rows-1 && j-3<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-3, j-3 ) = accessMat( surface, i-3, j-3 ) * ( 0.75 );
				if(i-3>=1 && j-2>=1 && i-3<rows-1 && j-2<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-3, j-2 ) = accessMat( surface, i-3, j-2 ) * ( 0.75 );
				if(i-3>=1 && j-1>=1 && i-3<rows-1 && j-1<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-3, j-1 ) = accessMat( surface, i-3, j-1 ) * ( 0.75 );
				if(i-3>=1 && j+0>=1 && i-3<rows-1 && j+0<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-3, j+0 ) = accessMat( surface, i-3, j+0 ) * ( 0.75 );
				if(i-3>=1 && j+1>=1 && i-3<rows-1 && j+1<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-3, j+1 ) = accessMat( surface, i-3, j+1 ) * ( 0.75 );
				if(i-3>=1 && j+2>=1 && i-3<rows-1 && j+2<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-3, j+2 ) = accessMat( surface, i-3, j+2 ) * ( 0.75 );
				if(i-3>=1 && j+3>=1 && i-3<rows-1 && j+3<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-3, j+3 ) = accessMat( surface, i-3, j+3 ) * ( 0.75 );
				if(i-3>=1 && j+4>=1 && i-3<rows-1 && j+4<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-3, j+4 ) = accessMat( surface, i-3, j+4 ) * ( 0.75 );
				if(i-3>=1 && j+5>=1 && i-3<rows-1 && j+5<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-3, j+5 ) = accessMat( surface, i-3, j+5 ) * ( 0.75 );
				if(i-3>=1 && j+6>=1 && i-3<rows-1 && j+6<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-3, j+6 ) = accessMat( surface, i-3, j+6 ) * ( 0.75 );
				if(i-3>=1 && j+7>=1 && i-3<rows-1 && j+7<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-3, j+7 ) = accessMat( surface, i-3, j+7 ) * ( 0.75 );
				if(i-3>=1 && j+8>=1 && i-3<rows-1 && j+8<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-3, j+8 ) = accessMat( surface, i-3, j+8 ) * ( 0.75 );
				if(i-2>=1 && j-8>=1 && i-2<rows-1 && j-8<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-2, j-8 ) = accessMat( surface, i-2, j-8 ) * ( 0.75 );
				if(i-2>=1 && j-7>=1 && i-2<rows-1 && j-7<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-2, j-7 ) = accessMat( surface, i-2, j-7 ) * ( 0.75 );
				if(i-2>=1 && j-6>=1 && i-2<rows-1 && j-6<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-2, j-6 ) = accessMat( surface, i-2, j-6 ) * ( 0.75 );
				if(i-2>=1 && j-5>=1 && i-2<rows-1 && j-5<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-2, j-5 ) = accessMat( surface, i-2, j-5 ) * ( 0.75 );
				if(i-2>=1 && j-4>=1 && i-2<rows-1 && j-4<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-2, j-4 ) = accessMat( surface, i-2, j-4 ) * ( 0.75 );
				if(i-2>=1 && j-3>=1 && i-2<rows-1 && j-3<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-2, j-3 ) = accessMat( surface, i-2, j-3 ) * ( 0.75 );
				if(i-2>=1 && j-2>=1 && i-2<rows-1 && j-2<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-2, j-2 ) = accessMat( surface, i-2, j-2 ) * ( 0.75 );
				if(i-2>=1 && j-1>=1 && i-2<rows-1 && j-1<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-2, j-1 ) = accessMat( surface, i-2, j-1 ) * ( 0.75 );
				if(i-2>=1 && j+0>=1 && i-2<rows-1 && j+0<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-2, j+0 ) = accessMat( surface, i-2, j+0 ) * ( 0.75 );
				if(i-2>=1 && j+1>=1 && i-2<rows-1 && j+1<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-2, j+1 ) = accessMat( surface, i-2, j+1 ) * ( 0.75 );
				if(i-2>=1 && j+2>=1 && i-2<rows-1 && j+2<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-2, j+2 ) = accessMat( surface, i-2, j+2 ) * ( 0.75 );
				if(i-2>=1 && j+3>=1 && i-2<rows-1 && j+3<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-2, j+3 ) = accessMat( surface, i-2, j+3 ) * ( 0.75 );
				if(i-2>=1 && j+4>=1 && i-2<rows-1 && j+4<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-2, j+4 ) = accessMat( surface, i-2, j+4 ) * ( 0.75 );
				if(i-2>=1 && j+5>=1 && i-2<rows-1 && j+5<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-2, j+5 ) = accessMat( surface, i-2, j+5 ) * ( 0.75 );
				if(i-2>=1 && j+6>=1 && i-2<rows-1 && j+6<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-2, j+6 ) = accessMat( surface, i-2, j+6 ) * ( 0.75 );
				if(i-2>=1 && j+7>=1 && i-2<rows-1 && j+7<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-2, j+7 ) = accessMat( surface, i-2, j+7 ) * ( 0.75 );
				if(i-2>=1 && j+8>=1 && i-2<rows-1 && j+8<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-2, j+8 ) = accessMat( surface, i-2, j+8 ) * ( 0.75 );
				if(i-1>=1 && j-8>=1 && i-1<rows-1 && j-8<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-1, j-8 ) = accessMat( surface, i-1, j-8 ) * ( 0.75 );
				if(i-1>=1 && j-7>=1 && i-1<rows-1 && j-7<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-1, j-7 ) = accessMat( surface, i-1, j-7 ) * ( 0.75 );
				if(i-1>=1 && j-6>=1 && i-1<rows-1 && j-6<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-1, j-6 ) = accessMat( surface, i-1, j-6 ) * ( 0.75 );
				if(i-1>=1 && j-5>=1 && i-1<rows-1 && j-5<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-1, j-5 ) = accessMat( surface, i-1, j-5 ) * ( 0.75 );
				if(i-1>=1 && j-4>=1 && i-1<rows-1 && j-4<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-1, j-4 ) = accessMat( surface, i-1, j-4 ) * ( 0.75 );
				if(i-1>=1 && j-3>=1 && i-1<rows-1 && j-3<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-1, j-3 ) = accessMat( surface, i-1, j-3 ) * ( 0.75 );
				if(i-1>=1 && j-2>=1 && i-1<rows-1 && j-2<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-1, j-2 ) = accessMat( surface, i-1, j-2 ) * ( 0.75 );
				if(i-1>=1 && j-1>=1 && i-1<rows-1 && j-1<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-1, j-1 ) = accessMat( surface, i-1, j-1 ) * ( 0.75 );
				if(i-1>=1 && j+0>=1 && i-1<rows-1 && j+0<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-1, j+0 ) = accessMat( surface, i-1, j+0 ) * ( 0.75 );
				if(i-1>=1 && j+1>=1 && i-1<rows-1 && j+1<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-1, j+1 ) = accessMat( surface, i-1, j+1 ) * ( 0.75 );
				if(i-1>=1 && j+2>=1 && i-1<rows-1 && j+2<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-1, j+2 ) = accessMat( surface, i-1, j+2 ) * ( 0.75 );
				if(i-1>=1 && j+3>=1 && i-1<rows-1 && j+3<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-1, j+3 ) = accessMat( surface, i-1, j+3 ) * ( 0.75 );
				if(i-1>=1 && j+4>=1 && i-1<rows-1 && j+4<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-1, j+4 ) = accessMat( surface, i-1, j+4 ) * ( 0.75 );
				if(i-1>=1 && j+5>=1 && i-1<rows-1 && j+5<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-1, j+5 ) = accessMat( surface, i-1, j+5 ) * ( 0.75 );
				if(i-1>=1 && j+6>=1 && i-1<rows-1 && j+6<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-1, j+6 ) = accessMat( surface, i-1, j+6 ) * ( 0.75 );
				if(i-1>=1 && j+7>=1 && i-1<rows-1 && j+7<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-1, j+7 ) = accessMat( surface, i-1, j+7 ) * ( 0.75 );
				if(i-1>=1 && j+8>=1 && i-1<rows-1 && j+8<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-1, j+8 ) = accessMat( surface, i-1, j+8 ) * ( 0.75 );
				if(i+0>=1 && j-9>=1 && i+0<rows-1 && j-9<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+0, j-9 ) = accessMat( surface, i+0, j-9 ) * ( 0.75 );
				if(i+0>=1 && j-8>=1 && i+0<rows-1 && j-8<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+0, j-8 ) = accessMat( surface, i+0, j-8 ) * ( 0.75 );
				if(i+0>=1 && j-7>=1 && i+0<rows-1 && j-7<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+0, j-7 ) = accessMat( surface, i+0, j-7 ) * ( 0.75 );
				if(i+0>=1 && j-6>=1 && i+0<rows-1 && j-6<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+0, j-6 ) = accessMat( surface, i+0, j-6 ) * ( 0.75 );
				if(i+0>=1 && j-5>=1 && i+0<rows-1 && j-5<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+0, j-5 ) = accessMat( surface, i+0, j-5 ) * ( 0.75 );
				if(i+0>=1 && j-4>=1 && i+0<rows-1 && j-4<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+0, j-4 ) = accessMat( surface, i+0, j-4 ) * ( 0.75 );
				if(i+0>=1 && j-3>=1 && i+0<rows-1 && j-3<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+0, j-3 ) = accessMat( surface, i+0, j-3 ) * ( 0.75 );
				if(i+0>=1 && j-2>=1 && i+0<rows-1 && j-2<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+0, j-2 ) = accessMat( surface, i+0, j-2 ) * ( 0.75 );
				if(i+0>=1 && j-1>=1 && i+0<rows-1 && j-1<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+0, j-1 ) = accessMat( surface, i+0, j-1 ) * ( 0.75 );
				if(i+0>=1 && j+0>=1 && i+0<rows-1 && j+0<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+0, j+0 ) = accessMat( surface, i+0, j+0 ) * ( 0.75 );
				if(i+0>=1 && j+1>=1 && i+0<rows-1 && j+1<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+0, j+1 ) = accessMat( surface, i+0, j+1 ) * ( 0.75 );
				if(i+0>=1 && j+2>=1 && i+0<rows-1 && j+2<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+0, j+2 ) = accessMat( surface, i+0, j+2 ) * ( 0.75 );
				if(i+0>=1 && j+3>=1 && i+0<rows-1 && j+3<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+0, j+3 ) = accessMat( surface, i+0, j+3 ) * ( 0.75 );
				if(i+0>=1 && j+4>=1 && i+0<rows-1 && j+4<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+0, j+4 ) = accessMat( surface, i+0, j+4 ) * ( 0.75 );
				if(i+0>=1 && j+5>=1 && i+0<rows-1 && j+5<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+0, j+5 ) = accessMat( surface, i+0, j+5 ) * ( 0.75 );
				if(i+0>=1 && j+6>=1 && i+0<rows-1 && j+6<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+0, j+6 ) = accessMat( surface, i+0, j+6 ) * ( 0.75 );
				if(i+0>=1 && j+7>=1 && i+0<rows-1 && j+7<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+0, j+7 ) = accessMat( surface, i+0, j+7 ) * ( 0.75 );
				if(i+0>=1 && j+8>=1 && i+0<rows-1 && j+8<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+0, j+8 ) = accessMat( surface, i+0, j+8 ) * ( 0.75 );
				if(i+0>=1 && j+9>=1 && i+0<rows-1 && j+9<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+0, j+9 ) = accessMat( surface, i+0, j+9 ) * ( 0.75 );
				if(i+1>=1 && j-8>=1 && i+1<rows-1 && j-8<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+1, j-8 ) = accessMat( surface, i+1, j-8 ) * ( 0.75 );
				if(i+1>=1 && j-7>=1 && i+1<rows-1 && j-7<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+1, j-7 ) = accessMat( surface, i+1, j-7 ) * ( 0.75 );
				if(i+1>=1 && j-6>=1 && i+1<rows-1 && j-6<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+1, j-6 ) = accessMat( surface, i+1, j-6 ) * ( 0.75 );
				if(i+1>=1 && j-5>=1 && i+1<rows-1 && j-5<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+1, j-5 ) = accessMat( surface, i+1, j-5 ) * ( 0.75 );
				if(i+1>=1 && j-4>=1 && i+1<rows-1 && j-4<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+1, j-4 ) = accessMat( surface, i+1, j-4 ) * ( 0.75 );
				if(i+1>=1 && j-3>=1 && i+1<rows-1 && j-3<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+1, j-3 ) = accessMat( surface, i+1, j-3 ) * ( 0.75 );
				if(i+1>=1 && j-2>=1 && i+1<rows-1 && j-2<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+1, j-2 ) = accessMat( surface, i+1, j-2 ) * ( 0.75 );
				if(i+1>=1 && j-1>=1 && i+1<rows-1 && j-1<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+1, j-1 ) = accessMat( surface, i+1, j-1 ) * ( 0.75 );
				if(i+1>=1 && j+0>=1 && i+1<rows-1 && j+0<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+1, j+0 ) = accessMat( surface, i+1, j+0 ) * ( 0.75 );
				if(i+1>=1 && j+1>=1 && i+1<rows-1 && j+1<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+1, j+1 ) = accessMat( surface, i+1, j+1 ) * ( 0.75 );
				if(i+1>=1 && j+2>=1 && i+1<rows-1 && j+2<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+1, j+2 ) = accessMat( surface, i+1, j+2 ) * ( 0.75 );
				if(i+1>=1 && j+3>=1 && i+1<rows-1 && j+3<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+1, j+3 ) = accessMat( surface, i+1, j+3 ) * ( 0.75 );
				if(i+1>=1 && j+4>=1 && i+1<rows-1 && j+4<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+1, j+4 ) = accessMat( surface, i+1, j+4 ) * ( 0.75 );
				if(i+1>=1 && j+5>=1 && i+1<rows-1 && j+5<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+1, j+5 ) = accessMat( surface, i+1, j+5 ) * ( 0.75 );
				if(i+1>=1 && j+6>=1 && i+1<rows-1 && j+6<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+1, j+6 ) = accessMat( surface, i+1, j+6 ) * ( 0.75 );
				if(i+1>=1 && j+7>=1 && i+1<rows-1 && j+7<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+1, j+7 ) = accessMat( surface, i+1, j+7 ) * ( 0.75 );
				if(i+1>=1 && j+8>=1 && i+1<rows-1 && j+8<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+1, j+8 ) = accessMat( surface, i+1, j+8 ) * ( 0.75 );
				if(i+2>=1 && j-8>=1 && i+2<rows-1 && j-8<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+2, j-8 ) = accessMat( surface, i+2, j-8 ) * ( 0.75 );
				if(i+2>=1 && j-7>=1 && i+2<rows-1 && j-7<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+2, j-7 ) = accessMat( surface, i+2, j-7 ) * ( 0.75 );
				if(i+2>=1 && j-6>=1 && i+2<rows-1 && j-6<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+2, j-6 ) = accessMat( surface, i+2, j-6 ) * ( 0.75 );
				if(i+2>=1 && j-5>=1 && i+2<rows-1 && j-5<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+2, j-5 ) = accessMat( surface, i+2, j-5 ) * ( 0.75 );
				if(i+2>=1 && j-4>=1 && i+2<rows-1 && j-4<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+2, j-4 ) = accessMat( surface, i+2, j-4 ) * ( 0.75 );
				if(i+2>=1 && j-3>=1 && i+2<rows-1 && j-3<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+2, j-3 ) = accessMat( surface, i+2, j-3 ) * ( 0.75 );
				if(i+2>=1 && j-2>=1 && i+2<rows-1 && j-2<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+2, j-2 ) = accessMat( surface, i+2, j-2 ) * ( 0.75 );
				if(i+2>=1 && j-1>=1 && i+2<rows-1 && j-1<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+2, j-1 ) = accessMat( surface, i+2, j-1 ) * ( 0.75 );
				if(i+2>=1 && j+0>=1 && i+2<rows-1 && j+0<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+2, j+0 ) = accessMat( surface, i+2, j+0 ) * ( 0.75 );
				if(i+2>=1 && j+1>=1 && i+2<rows-1 && j+1<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+2, j+1 ) = accessMat( surface, i+2, j+1 ) * ( 0.75 );
				if(i+2>=1 && j+2>=1 && i+2<rows-1 && j+2<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+2, j+2 ) = accessMat( surface, i+2, j+2 ) * ( 0.75 );
				if(i+2>=1 && j+3>=1 && i+2<rows-1 && j+3<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+2, j+3 ) = accessMat( surface, i+2, j+3 ) * ( 0.75 );
				if(i+2>=1 && j+4>=1 && i+2<rows-1 && j+4<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+2, j+4 ) = accessMat( surface, i+2, j+4 ) * ( 0.75 );
				if(i+2>=1 && j+5>=1 && i+2<rows-1 && j+5<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+2, j+5 ) = accessMat( surface, i+2, j+5 ) * ( 0.75 );
				if(i+2>=1 && j+6>=1 && i+2<rows-1 && j+6<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+2, j+6 ) = accessMat( surface, i+2, j+6 ) * ( 0.75 );
				if(i+2>=1 && j+7>=1 && i+2<rows-1 && j+7<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+2, j+7 ) = accessMat( surface, i+2, j+7 ) * ( 0.75 );
				if(i+2>=1 && j+8>=1 && i+2<rows-1 && j+8<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+2, j+8 ) = accessMat( surface, i+2, j+8 ) * ( 0.75 );
				if(i+3>=1 && j-8>=1 && i+3<rows-1 && j-8<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+3, j-8 ) = accessMat( surface, i+3, j-8 ) * ( 0.75 );
				if(i+3>=1 && j-7>=1 && i+3<rows-1 && j-7<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+3, j-7 ) = accessMat( surface, i+3, j-7 ) * ( 0.75 );
				if(i+3>=1 && j-6>=1 && i+3<rows-1 && j-6<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+3, j-6 ) = accessMat( surface, i+3, j-6 ) * ( 0.75 );
				if(i+3>=1 && j-5>=1 && i+3<rows-1 && j-5<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+3, j-5 ) = accessMat( surface, i+3, j-5 ) * ( 0.75 );
				if(i+3>=1 && j-4>=1 && i+3<rows-1 && j-4<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+3, j-4 ) = accessMat( surface, i+3, j-4 ) * ( 0.75 );
				if(i+3>=1 && j-3>=1 && i+3<rows-1 && j-3<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+3, j-3 ) = accessMat( surface, i+3, j-3 ) * ( 0.75 );
				if(i+3>=1 && j-2>=1 && i+3<rows-1 && j-2<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+3, j-2 ) = accessMat( surface, i+3, j-2 ) * ( 0.75 );
				if(i+3>=1 && j-1>=1 && i+3<rows-1 && j-1<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+3, j-1 ) = accessMat( surface, i+3, j-1 ) * ( 0.75 );
				if(i+3>=1 && j+0>=1 && i+3<rows-1 && j+0<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+3, j+0 ) = accessMat( surface, i+3, j+0 ) * ( 0.75 );
				if(i+3>=1 && j+1>=1 && i+3<rows-1 && j+1<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+3, j+1 ) = accessMat( surface, i+3, j+1 ) * ( 0.75 );
				if(i+3>=1 && j+2>=1 && i+3<rows-1 && j+2<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+3, j+2 ) = accessMat( surface, i+3, j+2 ) * ( 0.75 );
				if(i+3>=1 && j+3>=1 && i+3<rows-1 && j+3<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+3, j+3 ) = accessMat( surface, i+3, j+3 ) * ( 0.75 );
				if(i+3>=1 && j+4>=1 && i+3<rows-1 && j+4<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+3, j+4 ) = accessMat( surface, i+3, j+4 ) * ( 0.75 );
				if(i+3>=1 && j+5>=1 && i+3<rows-1 && j+5<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+3, j+5 ) = accessMat( surface, i+3, j+5 ) * ( 0.75 );
				if(i+3>=1 && j+6>=1 && i+3<rows-1 && j+6<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+3, j+6 ) = accessMat( surface, i+3, j+6 ) * ( 0.75 );
				if(i+3>=1 && j+7>=1 && i+3<rows-1 && j+7<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+3, j+7 ) = accessMat( surface, i+3, j+7 ) * ( 0.75 );
				if(i+3>=1 && j+8>=1 && i+3<rows-1 && j+8<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+3, j+8 ) = accessMat( surface, i+3, j+8 ) * ( 0.75 );
				if(i+4>=1 && j-8>=1 && i+4<rows-1 && j-8<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+4, j-8 ) = accessMat( surface, i+4, j-8 ) * ( 0.75 );
				if(i+4>=1 && j-7>=1 && i+4<rows-1 && j-7<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+4, j-7 ) = accessMat( surface, i+4, j-7 ) * ( 0.75 );
				if(i+4>=1 && j-6>=1 && i+4<rows-1 && j-6<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+4, j-6 ) = accessMat( surface, i+4, j-6 ) * ( 0.75 );
				if(i+4>=1 && j-5>=1 && i+4<rows-1 && j-5<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+4, j-5 ) = accessMat( surface, i+4, j-5 ) * ( 0.75 );
				if(i+4>=1 && j-4>=1 && i+4<rows-1 && j-4<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+4, j-4 ) = accessMat( surface, i+4, j-4 ) * ( 0.75 );
				if(i+4>=1 && j-3>=1 && i+4<rows-1 && j-3<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+4, j-3 ) = accessMat( surface, i+4, j-3 ) * ( 0.75 );
				if(i+4>=1 && j-2>=1 && i+4<rows-1 && j-2<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+4, j-2 ) = accessMat( surface, i+4, j-2 ) * ( 0.75 );
				if(i+4>=1 && j-1>=1 && i+4<rows-1 && j-1<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+4, j-1 ) = accessMat( surface, i+4, j-1 ) * ( 0.75 );
				if(i+4>=1 && j+0>=1 && i+4<rows-1 && j+0<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+4, j+0 ) = accessMat( surface, i+4, j+0 ) * ( 0.75 );
				if(i+4>=1 && j+1>=1 && i+4<rows-1 && j+1<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+4, j+1 ) = accessMat( surface, i+4, j+1 ) * ( 0.75 );
				if(i+4>=1 && j+2>=1 && i+4<rows-1 && j+2<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+4, j+2 ) = accessMat( surface, i+4, j+2 ) * ( 0.75 );
				if(i+4>=1 && j+3>=1 && i+4<rows-1 && j+3<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+4, j+3 ) = accessMat( surface, i+4, j+3 ) * ( 0.75 );
				if(i+4>=1 && j+4>=1 && i+4<rows-1 && j+4<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+4, j+4 ) = accessMat( surface, i+4, j+4 ) * ( 0.75 );
				if(i+4>=1 && j+5>=1 && i+4<rows-1 && j+5<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+4, j+5 ) = accessMat( surface, i+4, j+5 ) * ( 0.75 );
				if(i+4>=1 && j+6>=1 && i+4<rows-1 && j+6<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+4, j+6 ) = accessMat( surface, i+4, j+6 ) * ( 0.75 );
				if(i+4>=1 && j+7>=1 && i+4<rows-1 && j+7<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+4, j+7 ) = accessMat( surface, i+4, j+7 ) * ( 0.75 );
				if(i+4>=1 && j+8>=1 && i+4<rows-1 && j+8<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+4, j+8 ) = accessMat( surface, i+4, j+8 ) * ( 0.75 );
				if(i+5>=1 && j-7>=1 && i+5<rows-1 && j-7<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+5, j-7 ) = accessMat( surface, i+5, j-7 ) * ( 0.75 );
				if(i+5>=1 && j-6>=1 && i+5<rows-1 && j-6<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+5, j-6 ) = accessMat( surface, i+5, j-6 ) * ( 0.75 );
				if(i+5>=1 && j-5>=1 && i+5<rows-1 && j-5<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+5, j-5 ) = accessMat( surface, i+5, j-5 ) * ( 0.75 );
				if(i+5>=1 && j-4>=1 && i+5<rows-1 && j-4<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+5, j-4 ) = accessMat( surface, i+5, j-4 ) * ( 0.75 );
				if(i+5>=1 && j-3>=1 && i+5<rows-1 && j-3<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+5, j-3 ) = accessMat( surface, i+5, j-3 ) * ( 0.75 );
				if(i+5>=1 && j-2>=1 && i+5<rows-1 && j-2<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+5, j-2 ) = accessMat( surface, i+5, j-2 ) * ( 0.75 );
				if(i+5>=1 && j-1>=1 && i+5<rows-1 && j-1<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+5, j-1 ) = accessMat( surface, i+5, j-1 ) * ( 0.75 );
				if(i+5>=1 && j+0>=1 && i+5<rows-1 && j+0<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+5, j+0 ) = accessMat( surface, i+5, j+0 ) * ( 0.75 );
				if(i+5>=1 && j+1>=1 && i+5<rows-1 && j+1<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+5, j+1 ) = accessMat( surface, i+5, j+1 ) * ( 0.75 );
				if(i+5>=1 && j+2>=1 && i+5<rows-1 && j+2<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+5, j+2 ) = accessMat( surface, i+5, j+2 ) * ( 0.75 );
				if(i+5>=1 && j+3>=1 && i+5<rows-1 && j+3<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+5, j+3 ) = accessMat( surface, i+5, j+3 ) * ( 0.75 );
				if(i+5>=1 && j+4>=1 && i+5<rows-1 && j+4<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+5, j+4 ) = accessMat( surface, i+5, j+4 ) * ( 0.75 );
				if(i+5>=1 && j+5>=1 && i+5<rows-1 && j+5<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+5, j+5 ) = accessMat( surface, i+5, j+5 ) * ( 0.75 );
				if(i+5>=1 && j+6>=1 && i+5<rows-1 && j+6<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+5, j+6 ) = accessMat( surface, i+5, j+6 ) * ( 0.75 );
				if(i+5>=1 && j+7>=1 && i+5<rows-1 && j+7<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+5, j+7 ) = accessMat( surface, i+5, j+7 ) * ( 0.75 );
				if(i+6>=1 && j-6>=1 && i+6<rows-1 && j-6<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+6, j-6 ) = accessMat( surface, i+6, j-6 ) * ( 0.75 );
				if(i+6>=1 && j-5>=1 && i+6<rows-1 && j-5<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+6, j-5 ) = accessMat( surface, i+6, j-5 ) * ( 0.75 );
				if(i+6>=1 && j-4>=1 && i+6<rows-1 && j-4<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+6, j-4 ) = accessMat( surface, i+6, j-4 ) * ( 0.75 );
				if(i+6>=1 && j-3>=1 && i+6<rows-1 && j-3<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+6, j-3 ) = accessMat( surface, i+6, j-3 ) * ( 0.75 );
				if(i+6>=1 && j-2>=1 && i+6<rows-1 && j-2<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+6, j-2 ) = accessMat( surface, i+6, j-2 ) * ( 0.75 );
				if(i+6>=1 && j-1>=1 && i+6<rows-1 && j-1<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+6, j-1 ) = accessMat( surface, i+6, j-1 ) * ( 0.75 );
				if(i+6>=1 && j+0>=1 && i+6<rows-1 && j+0<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+6, j+0 ) = accessMat( surface, i+6, j+0 ) * ( 0.75 );
				if(i+6>=1 && j+1>=1 && i+6<rows-1 && j+1<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+6, j+1 ) = accessMat( surface, i+6, j+1 ) * ( 0.75 );
				if(i+6>=1 && j+2>=1 && i+6<rows-1 && j+2<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+6, j+2 ) = accessMat( surface, i+6, j+2 ) * ( 0.75 );
				if(i+6>=1 && j+3>=1 && i+6<rows-1 && j+3<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+6, j+3 ) = accessMat( surface, i+6, j+3 ) * ( 0.75 );
				if(i+6>=1 && j+4>=1 && i+6<rows-1 && j+4<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+6, j+4 ) = accessMat( surface, i+6, j+4 ) * ( 0.75 );
				if(i+6>=1 && j+5>=1 && i+6<rows-1 && j+5<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+6, j+5 ) = accessMat( surface, i+6, j+5 ) * ( 0.75 );
				if(i+6>=1 && j+6>=1 && i+6<rows-1 && j+6<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+6, j+6 ) = accessMat( surface, i+6, j+6 ) * ( 0.75 );
				if(i+7>=1 && j-5>=1 && i+7<rows-1 && j-5<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+7, j-5 ) = accessMat( surface, i+7, j-5 ) * ( 0.75 );
				if(i+7>=1 && j-4>=1 && i+7<rows-1 && j-4<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+7, j-4 ) = accessMat( surface, i+7, j-4 ) * ( 0.75 );
				if(i+7>=1 && j-3>=1 && i+7<rows-1 && j-3<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+7, j-3 ) = accessMat( surface, i+7, j-3 ) * ( 0.75 );
				if(i+7>=1 && j-2>=1 && i+7<rows-1 && j-2<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+7, j-2 ) = accessMat( surface, i+7, j-2 ) * ( 0.75 );
				if(i+7>=1 && j-1>=1 && i+7<rows-1 && j-1<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+7, j-1 ) = accessMat( surface, i+7, j-1 ) * ( 0.75 );
				if(i+7>=1 && j+0>=1 && i+7<rows-1 && j+0<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+7, j+0 ) = accessMat( surface, i+7, j+0 ) * ( 0.75 );
				if(i+7>=1 && j+1>=1 && i+7<rows-1 && j+1<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+7, j+1 ) = accessMat( surface, i+7, j+1 ) * ( 0.75 );
				if(i+7>=1 && j+2>=1 && i+7<rows-1 && j+2<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+7, j+2 ) = accessMat( surface, i+7, j+2 ) * ( 0.75 );
				if(i+7>=1 && j+3>=1 && i+7<rows-1 && j+3<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+7, j+3 ) = accessMat( surface, i+7, j+3 ) * ( 0.75 );
				if(i+7>=1 && j+4>=1 && i+7<rows-1 && j+4<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+7, j+4 ) = accessMat( surface, i+7, j+4 ) * ( 0.75 );
				if(i+7>=1 && j+5>=1 && i+7<rows-1 && j+5<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+7, j+5 ) = accessMat( surface, i+7, j+5 ) * ( 0.75 );
				if(i+8>=1 && j-4>=1 && i+8<rows-1 && j-4<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+8, j-4 ) = accessMat( surface, i+8, j-4 ) * ( 0.75 );
				if(i+8>=1 && j-3>=1 && i+8<rows-1 && j-3<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+8, j-3 ) = accessMat( surface, i+8, j-3 ) * ( 0.75 );
				if(i+8>=1 && j-2>=1 && i+8<rows-1 && j-2<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+8, j-2 ) = accessMat( surface, i+8, j-2 ) * ( 0.75 );
				if(i+8>=1 && j-1>=1 && i+8<rows-1 && j-1<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+8, j-1 ) = accessMat( surface, i+8, j-1 ) * ( 0.75 );
				if(i+8>=1 && j+0>=1 && i+8<rows-1 && j+0<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+8, j+0 ) = accessMat( surface, i+8, j+0 ) * ( 0.75 );
				if(i+8>=1 && j+1>=1 && i+8<rows-1 && j+1<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+8, j+1 ) = accessMat( surface, i+8, j+1 ) * ( 0.75 );
				if(i+8>=1 && j+2>=1 && i+8<rows-1 && j+2<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+8, j+2 ) = accessMat( surface, i+8, j+2 ) * ( 0.75 );
				if(i+8>=1 && j+3>=1 && i+8<rows-1 && j+3<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+8, j+3 ) = accessMat( surface, i+8, j+3 ) * ( 0.75 );
				if(i+8>=1 && j+4>=1 && i+8<rows-1 && j+4<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+8, j+4 ) = accessMat( surface, i+8, j+4 ) * ( 0.75 );
				if(i+9>=1 && j+0>=1 && i+9<rows-1 && j+0<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+9, j+0 ) = accessMat( surface, i+9, j+0 ) * ( 0.75 );
			}
			/*for( i=teams[t].x-radius; i<=teams[t].x+radius; i++ ) {
				for( j=teams[t].y-radius; j<=teams[t].y+radius; j++ ) {
					if ( i<1 || i>=rows-1 || j<1 || j>=columns-1 ) continue; // Out of the heated surface
					float dx = teams[t].x - i;
					float dy = teams[t].y - j;
					float distance = sqrtf( dx*dx + dy*dy );
					if ( distance <= radius ) {
						accessMat( surface, i, j ) = accessMat( surface, i, j ) * ( 0.75 ); // Team efficiency factor
					} //Puede dar problemas al paralelizar 
				}
			}*/
		
		
		}


			

		}else{
		
			int step;
			float global_residual = 0.0f;
			
			
					/* 4.2.2. Copy values of the surface in ancillary structure (Skip borders) */
					/*#pragma omp parallel for
					for( i=1; i<rows-1; i++ )
						for( j=1; j<columns-1; j++ )
							accessMat( surfaceCopy, i, j ) = accessMat( surface, i, j ); */

					/* 4.2.3. Update surface values (skip borders) */
			
					//1
					//printf("Inicial");
					//print_status( iter, rows, columns, surface, num_teams, teams, num_focal, focal, global_residual );
				#pragma omp parallel
				{
					
					#pragma omp  for private(j) 
					for( i=1; i<rows-1; i++ )
						for( j=1; j<columns-1; j++ ){
						

							accessMat( surfaceCopy, i, j ) = ( 
								accessMat( surface, i-1, j ) +
								accessMat( surface, i+1, j ) +
								accessMat( surface, i, j-1 ) +
								accessMat( surface, i, j+1 ) ) / 4;
								
								
						}

					
					#pragma omp for private(j) reduction(max:global_residual) nowait
					for( i=1; i<rows-1; i++ )
						for( j=1; j<columns-1; j++ )	
						if (accessMat( surfaceCopy, i, j )-accessMat( surface, i, j )> global_residual )  {
								global_residual = accessMat( surfaceCopy, i, j ) - accessMat( surface, i, j ); 
							}		
					
					//printf("Acabado 1");
					//print_status( iter, rows, columns, surfaceCopy, num_teams, teams, num_focal, focal, global_residual );
					//2
					#pragma omp   for private(j) 
					for( i=1; i<rows-1; i++ )
						for( j=1; j<columns-1; j++ ){
						

							accessMat( surfaceAuxiliar, i, j ) = ( 
								accessMat( surfaceCopy, i-1, j ) +
								accessMat( surfaceCopy, i+1, j ) +
								accessMat( surfaceCopy, i, j-1 ) +
								accessMat( surfaceCopy, i, j+1 ) ) / 4;
								
								
						}


		

					//printf("Acabado 2");
					//print_status( iter, rows, columns, surfaceAuxiliar, num_teams, teams, num_focal, focal, global_residual );
					//3
					#pragma omp for private(j) 
					for( i=1; i<rows-1; i++ )
						for( j=1; j<columns-1; j++ ){
						

							accessMat( surface, i, j ) = ( 
								accessMat( surfaceAuxiliar, i-1, j ) +
								accessMat( surfaceAuxiliar, i+1, j ) +
								accessMat( surfaceAuxiliar, i, j-1 ) +
								accessMat( surfaceAuxiliar, i, j+1 ) ) / 4;
								
								
						}


				

					//printf("Acabado 3");
					//print_status( iter, rows, columns, surface, num_teams, teams, num_focal, focal, global_residual );
					//4
					#pragma omp for private(j) 
					for( i=1; i<rows-1; i++ )
						for( j=1; j<columns-1; j++ ){
						

							accessMat( surfaceCopy, i, j ) = ( 
								accessMat( surface, i-1, j ) +
								accessMat( surface, i+1, j ) +
								accessMat( surface, i, j-1 ) +
								accessMat( surface, i, j+1 ) ) / 4;
								
								
						}

					
					//printf("Acabado 4");
					//print_status( iter, rows, columns, surfaceCopy, num_teams, teams, num_focal, focal, global_residual );
					//5
					#pragma omp for private(j) 
					for( i=1; i<rows-1; i++ )
						for( j=1; j<columns-1; j++ ){
						

							accessMat( surfaceAuxiliar, i, j ) = ( 
								accessMat( surfaceCopy, i-1, j ) +
								accessMat( surfaceCopy, i+1, j ) +
								accessMat( surfaceCopy, i, j-1 ) +
								accessMat( surfaceCopy, i, j+1 ) ) / 4;
								
								
						}

					
		

					//printf("Acabado 5");
					//print_status( iter, rows, columns, surfaceAuxiliar, num_teams, teams, num_focal, focal, global_residual );
					//6
					#pragma omp  for private(j) 
					for( i=1; i<rows-1; i++ )
						for( j=1; j<columns-1; j++ ){
						

							accessMat( surface, i, j ) = ( 
								accessMat( surfaceAuxiliar, i-1, j ) +
								accessMat( surfaceAuxiliar, i+1, j ) +
								accessMat( surfaceAuxiliar, i, j-1 ) +
								accessMat( surfaceAuxiliar, i, j+1 ) ) / 4;
								
								
						}


				

					//printf("Acabado 6");
					//print_status( iter, rows, columns, surface, num_teams, teams, num_focal, focal, global_residual );
					//7
					#pragma omp for private(j) 
					for( i=1; i<rows-1; i++ )
						for( j=1; j<columns-1; j++ ){
						

							accessMat( surfaceCopy, i, j ) = ( 
								accessMat( surface, i-1, j ) +
								accessMat( surface, i+1, j ) +
								accessMat( surface, i, j-1 ) +
								accessMat( surface, i, j+1 ) ) / 4;
								
								
						}


					//printf("Acabado 7");
					//print_status( iter, rows, columns, surfaceCopy, num_teams, teams, num_focal, focal, global_residual );
					//8
					#pragma omp  for private(j) 
					for( i=1; i<rows-1; i++ )
						for( j=1; j<columns-1; j++ ){
						

							accessMat( surfaceAuxiliar, i, j ) = ( 
								accessMat( surfaceCopy, i-1, j ) +
								accessMat( surfaceCopy, i+1, j ) +
								accessMat( surfaceCopy, i, j-1 ) +
								accessMat( surfaceCopy, i, j+1 ) ) / 4;
								
								
						}


	
					//printf("Acabado 8");
					//print_status( iter, rows, columns, surfaceAuxiliar, num_teams, teams, num_focal, focal, global_residual );
					//9
					#pragma omp  for private(j) 
					for( i=1; i<rows-1; i++ )
						for( j=1; j<columns-1; j++ ){
						

							accessMat( surface, i, j ) = ( 
								accessMat( surfaceAuxiliar, i-1, j ) +
								accessMat( surfaceAuxiliar, i+1, j ) +
								accessMat( surfaceAuxiliar, i, j-1 ) +
								accessMat( surfaceAuxiliar, i, j+1 ) ) / 4;
								
								
						}



					//printf("Acabado 9");
					//print_status( iter, rows, columns, surface, num_teams, teams, num_focal, focal, global_residual );
					//10
					#pragma omp for private(j) 
					for( i=1; i<rows-1; i++ )
						for( j=1; j<columns-1; j++ ){
						

							accessMat( surfaceCopy, i, j ) = ( 
								accessMat( surface, i-1, j ) +
								accessMat( surface, i+1, j ) +
								accessMat( surface, i, j-1 ) +
								accessMat( surface, i, j+1 ) ) / 4;
								
								
						}


					
					//printf("Acabado 10");
					//print_status( iter, rows, columns, surface, num_teams, teams, num_focal, focal, global_residual );
				
		}
				surfaceIntercambio = surfaceCopy;
					surfaceCopy = surfaceAuxiliar;
					surfaceAuxiliar = surface;
					surface = surfaceIntercambio;
			if(global_residual < THRESHOLD ) flag_stability = 1;
			
			
				
			/* 4.4. Team actions */
		#pragma omp parallel for private(i,j)
		for( t=0; t<num_teams; t++ ) {
			/* 4.4.1. Deactivate the target focal point when it is reached */

			/* 4.4.2. Reduce heat in a circle around the team */
			int radius;
			i=teams[t].x;
			j=teams[t].y;
			// Influence area of fixed radius depending on type
			if ( teams[t].type == 1 ) {
			//radius = RADIUS_TYPE_1;
			//else radius = RADIUS_TYPE_2_3;
			/** Podemos cargarnos los bucles para cargarnos las comprobaciones */
			
				if(i-3>=1 && j+0>=1 && i-3<rows-1 && j+0<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-3, j+0 ) = accessMat( surface, i-3, j+0 ) * ( 0.75 );
				if(i-2>=1 && j-2>=1 && i-2<rows-1 && j-2<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-2, j-2 ) = accessMat( surface, i-2, j-2 ) * ( 0.75 );
				if(i-2>=1 && j-1>=1 && i-2<rows-1 && j-1<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-2, j-1 ) = accessMat( surface, i-2, j-1 ) * ( 0.75 );
				if(i-2>=1 && j+0>=1 && i-2<rows-1 && j+0<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-2, j+0 ) = accessMat( surface, i-2, j+0 ) * ( 0.75 );
				if(i-2>=1 && j+1>=1 && i-2<rows-1 && j+1<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-2, j+1 ) = accessMat( surface, i-2, j+1 ) * ( 0.75 );
				if(i-2>=1 && j+2>=1 && i-2<rows-1 && j+2<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-2, j+2 ) = accessMat( surface, i-2, j+2 ) * ( 0.75 );
				if(i-1>=1 && j-2>=1 && i-1<rows-1 && j-2<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-1, j-2 ) = accessMat( surface, i-1, j-2 ) * ( 0.75 );
				if(i-1>=1 && j-1>=1 && i-1<rows-1 && j-1<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-1, j-1 ) = accessMat( surface, i-1, j-1 ) * ( 0.75 );
				if(i-1>=1 && j+0>=1 && i-1<rows-1 && j+0<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-1, j+0 ) = accessMat( surface, i-1, j+0 ) * ( 0.75 );
				if(i-1>=1 && j+1>=1 && i-1<rows-1 && j+1<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-1, j+1 ) = accessMat( surface, i-1, j+1 ) * ( 0.75 );
				if(i-1>=1 && j+2>=1 && i-1<rows-1 && j+2<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-1, j+2 ) = accessMat( surface, i-1, j+2 ) * ( 0.75 );
				if(i+0>=1 && j-3>=1 && i+0<rows-1 && j-3<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+0, j-3 ) = accessMat( surface, i+0, j-3 ) * ( 0.75 );
				if(i+0>=1 && j-2>=1 && i+0<rows-1 && j-2<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+0, j-2 ) = accessMat( surface, i+0, j-2 ) * ( 0.75 );
				if(i+0>=1 && j-1>=1 && i+0<rows-1 && j-1<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+0, j-1 ) = accessMat( surface, i+0, j-1 ) * ( 0.75 );
				if(i+0>=1 && j+0>=1 && i+0<rows-1 && j+0<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+0, j+0 ) = accessMat( surface, i+0, j+0 ) * ( 0.75 );
				if(i+0>=1 && j+1>=1 && i+0<rows-1 && j+1<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+0, j+1 ) = accessMat( surface, i+0, j+1 ) * ( 0.75 );
				if(i+0>=1 && j+2>=1 && i+0<rows-1 && j+2<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+0, j+2 ) = accessMat( surface, i+0, j+2 ) * ( 0.75 );
				if(i+0>=1 && j+3>=1 && i+0<rows-1 && j+3<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+0, j+3 ) = accessMat( surface, i+0, j+3 ) * ( 0.75 );
				if(i+1>=1 && j-2>=1 && i+1<rows-1 && j-2<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+1, j-2 ) = accessMat( surface, i+1, j-2 ) * ( 0.75 );
				if(i+1>=1 && j-1>=1 && i+1<rows-1 && j-1<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+1, j-1 ) = accessMat( surface, i+1, j-1 ) * ( 0.75 );
				if(i+1>=1 && j+0>=1 && i+1<rows-1 && j+0<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+1, j+0 ) = accessMat( surface, i+1, j+0 ) * ( 0.75 );
				if(i+1>=1 && j+1>=1 && i+1<rows-1 && j+1<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+1, j+1 ) = accessMat( surface, i+1, j+1 ) * ( 0.75 );
				if(i+1>=1 && j+2>=1 && i+1<rows-1 && j+2<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+1, j+2 ) = accessMat( surface, i+1, j+2 ) * ( 0.75 );
				if(i+2>=1 && j-2>=1 && i+2<rows-1 && j-2<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+2, j-2 ) = accessMat( surface, i+2, j-2 ) * ( 0.75 );
				if(i+2>=1 && j-1>=1 && i+2<rows-1 && j-1<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+2, j-1 ) = accessMat( surface, i+2, j-1 ) * ( 0.75 );
				if(i+2>=1 && j+0>=1 && i+2<rows-1 && j+0<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+2, j+0 ) = accessMat( surface, i+2, j+0 ) * ( 0.75 );
				if(i+2>=1 && j+1>=1 && i+2<rows-1 && j+1<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+2, j+1 ) = accessMat( surface, i+2, j+1 ) * ( 0.75 );
				if(i+2>=1 && j+2>=1 && i+2<rows-1 && j+2<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+2, j+2 ) = accessMat( surface, i+2, j+2 ) * ( 0.75 );
				if(i+3>=1 && j+0>=1 && i+3<rows-1 && j+0<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+3, j+0 ) = accessMat( surface, i+3, j+0 ) * ( 0.75 );
			
			}
			else{
				if(i-9>=1 && j+0>=1 && i-9<rows-1 && j+0<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-9, j+0 ) = accessMat( surface, i-9, j+0 ) * ( 0.75 );
				if(i-8>=1 && j-4>=1 && i-8<rows-1 && j-4<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-8, j-4 ) = accessMat( surface, i-8, j-4 ) * ( 0.75 );
				if(i-8>=1 && j-3>=1 && i-8<rows-1 && j-3<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-8, j-3 ) = accessMat( surface, i-8, j-3 ) * ( 0.75 );
				if(i-8>=1 && j-2>=1 && i-8<rows-1 && j-2<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-8, j-2 ) = accessMat( surface, i-8, j-2 ) * ( 0.75 );
				if(i-8>=1 && j-1>=1 && i-8<rows-1 && j-1<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-8, j-1 ) = accessMat( surface, i-8, j-1 ) * ( 0.75 );
				if(i-8>=1 && j+0>=1 && i-8<rows-1 && j+0<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-8, j+0 ) = accessMat( surface, i-8, j+0 ) * ( 0.75 );
				if(i-8>=1 && j+1>=1 && i-8<rows-1 && j+1<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-8, j+1 ) = accessMat( surface, i-8, j+1 ) * ( 0.75 );
				if(i-8>=1 && j+2>=1 && i-8<rows-1 && j+2<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-8, j+2 ) = accessMat( surface, i-8, j+2 ) * ( 0.75 );
				if(i-8>=1 && j+3>=1 && i-8<rows-1 && j+3<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-8, j+3 ) = accessMat( surface, i-8, j+3 ) * ( 0.75 );
				if(i-8>=1 && j+4>=1 && i-8<rows-1 && j+4<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-8, j+4 ) = accessMat( surface, i-8, j+4 ) * ( 0.75 );
				if(i-7>=1 && j-5>=1 && i-7<rows-1 && j-5<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-7, j-5 ) = accessMat( surface, i-7, j-5 ) * ( 0.75 );
				if(i-7>=1 && j-4>=1 && i-7<rows-1 && j-4<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-7, j-4 ) = accessMat( surface, i-7, j-4 ) * ( 0.75 );
				if(i-7>=1 && j-3>=1 && i-7<rows-1 && j-3<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-7, j-3 ) = accessMat( surface, i-7, j-3 ) * ( 0.75 );
				if(i-7>=1 && j-2>=1 && i-7<rows-1 && j-2<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-7, j-2 ) = accessMat( surface, i-7, j-2 ) * ( 0.75 );
				if(i-7>=1 && j-1>=1 && i-7<rows-1 && j-1<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-7, j-1 ) = accessMat( surface, i-7, j-1 ) * ( 0.75 );
				if(i-7>=1 && j+0>=1 && i-7<rows-1 && j+0<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-7, j+0 ) = accessMat( surface, i-7, j+0 ) * ( 0.75 );
				if(i-7>=1 && j+1>=1 && i-7<rows-1 && j+1<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-7, j+1 ) = accessMat( surface, i-7, j+1 ) * ( 0.75 );
				if(i-7>=1 && j+2>=1 && i-7<rows-1 && j+2<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-7, j+2 ) = accessMat( surface, i-7, j+2 ) * ( 0.75 );
				if(i-7>=1 && j+3>=1 && i-7<rows-1 && j+3<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-7, j+3 ) = accessMat( surface, i-7, j+3 ) * ( 0.75 );
				if(i-7>=1 && j+4>=1 && i-7<rows-1 && j+4<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-7, j+4 ) = accessMat( surface, i-7, j+4 ) * ( 0.75 );
				if(i-7>=1 && j+5>=1 && i-7<rows-1 && j+5<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-7, j+5 ) = accessMat( surface, i-7, j+5 ) * ( 0.75 );
				if(i-6>=1 && j-6>=1 && i-6<rows-1 && j-6<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-6, j-6 ) = accessMat( surface, i-6, j-6 ) * ( 0.75 );
				if(i-6>=1 && j-5>=1 && i-6<rows-1 && j-5<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-6, j-5 ) = accessMat( surface, i-6, j-5 ) * ( 0.75 );
				if(i-6>=1 && j-4>=1 && i-6<rows-1 && j-4<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-6, j-4 ) = accessMat( surface, i-6, j-4 ) * ( 0.75 );
				if(i-6>=1 && j-3>=1 && i-6<rows-1 && j-3<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-6, j-3 ) = accessMat( surface, i-6, j-3 ) * ( 0.75 );
				if(i-6>=1 && j-2>=1 && i-6<rows-1 && j-2<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-6, j-2 ) = accessMat( surface, i-6, j-2 ) * ( 0.75 );
				if(i-6>=1 && j-1>=1 && i-6<rows-1 && j-1<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-6, j-1 ) = accessMat( surface, i-6, j-1 ) * ( 0.75 );
				if(i-6>=1 && j+0>=1 && i-6<rows-1 && j+0<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-6, j+0 ) = accessMat( surface, i-6, j+0 ) * ( 0.75 );
				if(i-6>=1 && j+1>=1 && i-6<rows-1 && j+1<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-6, j+1 ) = accessMat( surface, i-6, j+1 ) * ( 0.75 );
				if(i-6>=1 && j+2>=1 && i-6<rows-1 && j+2<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-6, j+2 ) = accessMat( surface, i-6, j+2 ) * ( 0.75 );
				if(i-6>=1 && j+3>=1 && i-6<rows-1 && j+3<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-6, j+3 ) = accessMat( surface, i-6, j+3 ) * ( 0.75 );
				if(i-6>=1 && j+4>=1 && i-6<rows-1 && j+4<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-6, j+4 ) = accessMat( surface, i-6, j+4 ) * ( 0.75 );
				if(i-6>=1 && j+5>=1 && i-6<rows-1 && j+5<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-6, j+5 ) = accessMat( surface, i-6, j+5 ) * ( 0.75 );
				if(i-6>=1 && j+6>=1 && i-6<rows-1 && j+6<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-6, j+6 ) = accessMat( surface, i-6, j+6 ) * ( 0.75 );
				if(i-5>=1 && j-7>=1 && i-5<rows-1 && j-7<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-5, j-7 ) = accessMat( surface, i-5, j-7 ) * ( 0.75 );
				if(i-5>=1 && j-6>=1 && i-5<rows-1 && j-6<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-5, j-6 ) = accessMat( surface, i-5, j-6 ) * ( 0.75 );
				if(i-5>=1 && j-5>=1 && i-5<rows-1 && j-5<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-5, j-5 ) = accessMat( surface, i-5, j-5 ) * ( 0.75 );
				if(i-5>=1 && j-4>=1 && i-5<rows-1 && j-4<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-5, j-4 ) = accessMat( surface, i-5, j-4 ) * ( 0.75 );
				if(i-5>=1 && j-3>=1 && i-5<rows-1 && j-3<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-5, j-3 ) = accessMat( surface, i-5, j-3 ) * ( 0.75 );
				if(i-5>=1 && j-2>=1 && i-5<rows-1 && j-2<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-5, j-2 ) = accessMat( surface, i-5, j-2 ) * ( 0.75 );
				if(i-5>=1 && j-1>=1 && i-5<rows-1 && j-1<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-5, j-1 ) = accessMat( surface, i-5, j-1 ) * ( 0.75 );
				if(i-5>=1 && j+0>=1 && i-5<rows-1 && j+0<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-5, j+0 ) = accessMat( surface, i-5, j+0 ) * ( 0.75 );
				if(i-5>=1 && j+1>=1 && i-5<rows-1 && j+1<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-5, j+1 ) = accessMat( surface, i-5, j+1 ) * ( 0.75 );
				if(i-5>=1 && j+2>=1 && i-5<rows-1 && j+2<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-5, j+2 ) = accessMat( surface, i-5, j+2 ) * ( 0.75 );
				if(i-5>=1 && j+3>=1 && i-5<rows-1 && j+3<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-5, j+3 ) = accessMat( surface, i-5, j+3 ) * ( 0.75 );
				if(i-5>=1 && j+4>=1 && i-5<rows-1 && j+4<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-5, j+4 ) = accessMat( surface, i-5, j+4 ) * ( 0.75 );
				if(i-5>=1 && j+5>=1 && i-5<rows-1 && j+5<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-5, j+5 ) = accessMat( surface, i-5, j+5 ) * ( 0.75 );
				if(i-5>=1 && j+6>=1 && i-5<rows-1 && j+6<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-5, j+6 ) = accessMat( surface, i-5, j+6 ) * ( 0.75 );
				if(i-5>=1 && j+7>=1 && i-5<rows-1 && j+7<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-5, j+7 ) = accessMat( surface, i-5, j+7 ) * ( 0.75 );
				if(i-4>=1 && j-8>=1 && i-4<rows-1 && j-8<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-4, j-8 ) = accessMat( surface, i-4, j-8 ) * ( 0.75 );
				if(i-4>=1 && j-7>=1 && i-4<rows-1 && j-7<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-4, j-7 ) = accessMat( surface, i-4, j-7 ) * ( 0.75 );
				if(i-4>=1 && j-6>=1 && i-4<rows-1 && j-6<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-4, j-6 ) = accessMat( surface, i-4, j-6 ) * ( 0.75 );
				if(i-4>=1 && j-5>=1 && i-4<rows-1 && j-5<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-4, j-5 ) = accessMat( surface, i-4, j-5 ) * ( 0.75 );
				if(i-4>=1 && j-4>=1 && i-4<rows-1 && j-4<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-4, j-4 ) = accessMat( surface, i-4, j-4 ) * ( 0.75 );
				if(i-4>=1 && j-3>=1 && i-4<rows-1 && j-3<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-4, j-3 ) = accessMat( surface, i-4, j-3 ) * ( 0.75 );
				if(i-4>=1 && j-2>=1 && i-4<rows-1 && j-2<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-4, j-2 ) = accessMat( surface, i-4, j-2 ) * ( 0.75 );
				if(i-4>=1 && j-1>=1 && i-4<rows-1 && j-1<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-4, j-1 ) = accessMat( surface, i-4, j-1 ) * ( 0.75 );
				if(i-4>=1 && j+0>=1 && i-4<rows-1 && j+0<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-4, j+0 ) = accessMat( surface, i-4, j+0 ) * ( 0.75 );
				if(i-4>=1 && j+1>=1 && i-4<rows-1 && j+1<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-4, j+1 ) = accessMat( surface, i-4, j+1 ) * ( 0.75 );
				if(i-4>=1 && j+2>=1 && i-4<rows-1 && j+2<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-4, j+2 ) = accessMat( surface, i-4, j+2 ) * ( 0.75 );
				if(i-4>=1 && j+3>=1 && i-4<rows-1 && j+3<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-4, j+3 ) = accessMat( surface, i-4, j+3 ) * ( 0.75 );
				if(i-4>=1 && j+4>=1 && i-4<rows-1 && j+4<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-4, j+4 ) = accessMat( surface, i-4, j+4 ) * ( 0.75 );
				if(i-4>=1 && j+5>=1 && i-4<rows-1 && j+5<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-4, j+5 ) = accessMat( surface, i-4, j+5 ) * ( 0.75 );
				if(i-4>=1 && j+6>=1 && i-4<rows-1 && j+6<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-4, j+6 ) = accessMat( surface, i-4, j+6 ) * ( 0.75 );
				if(i-4>=1 && j+7>=1 && i-4<rows-1 && j+7<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-4, j+7 ) = accessMat( surface, i-4, j+7 ) * ( 0.75 );
				if(i-4>=1 && j+8>=1 && i-4<rows-1 && j+8<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-4, j+8 ) = accessMat( surface, i-4, j+8 ) * ( 0.75 );
				if(i-3>=1 && j-8>=1 && i-3<rows-1 && j-8<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-3, j-8 ) = accessMat( surface, i-3, j-8 ) * ( 0.75 );
				if(i-3>=1 && j-7>=1 && i-3<rows-1 && j-7<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-3, j-7 ) = accessMat( surface, i-3, j-7 ) * ( 0.75 );
				if(i-3>=1 && j-6>=1 && i-3<rows-1 && j-6<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-3, j-6 ) = accessMat( surface, i-3, j-6 ) * ( 0.75 );
				if(i-3>=1 && j-5>=1 && i-3<rows-1 && j-5<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-3, j-5 ) = accessMat( surface, i-3, j-5 ) * ( 0.75 );
				if(i-3>=1 && j-4>=1 && i-3<rows-1 && j-4<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-3, j-4 ) = accessMat( surface, i-3, j-4 ) * ( 0.75 );
				if(i-3>=1 && j-3>=1 && i-3<rows-1 && j-3<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-3, j-3 ) = accessMat( surface, i-3, j-3 ) * ( 0.75 );
				if(i-3>=1 && j-2>=1 && i-3<rows-1 && j-2<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-3, j-2 ) = accessMat( surface, i-3, j-2 ) * ( 0.75 );
				if(i-3>=1 && j-1>=1 && i-3<rows-1 && j-1<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-3, j-1 ) = accessMat( surface, i-3, j-1 ) * ( 0.75 );
				if(i-3>=1 && j+0>=1 && i-3<rows-1 && j+0<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-3, j+0 ) = accessMat( surface, i-3, j+0 ) * ( 0.75 );
				if(i-3>=1 && j+1>=1 && i-3<rows-1 && j+1<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-3, j+1 ) = accessMat( surface, i-3, j+1 ) * ( 0.75 );
				if(i-3>=1 && j+2>=1 && i-3<rows-1 && j+2<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-3, j+2 ) = accessMat( surface, i-3, j+2 ) * ( 0.75 );
				if(i-3>=1 && j+3>=1 && i-3<rows-1 && j+3<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-3, j+3 ) = accessMat( surface, i-3, j+3 ) * ( 0.75 );
				if(i-3>=1 && j+4>=1 && i-3<rows-1 && j+4<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-3, j+4 ) = accessMat( surface, i-3, j+4 ) * ( 0.75 );
				if(i-3>=1 && j+5>=1 && i-3<rows-1 && j+5<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-3, j+5 ) = accessMat( surface, i-3, j+5 ) * ( 0.75 );
				if(i-3>=1 && j+6>=1 && i-3<rows-1 && j+6<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-3, j+6 ) = accessMat( surface, i-3, j+6 ) * ( 0.75 );
				if(i-3>=1 && j+7>=1 && i-3<rows-1 && j+7<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-3, j+7 ) = accessMat( surface, i-3, j+7 ) * ( 0.75 );
				if(i-3>=1 && j+8>=1 && i-3<rows-1 && j+8<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-3, j+8 ) = accessMat( surface, i-3, j+8 ) * ( 0.75 );
				if(i-2>=1 && j-8>=1 && i-2<rows-1 && j-8<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-2, j-8 ) = accessMat( surface, i-2, j-8 ) * ( 0.75 );
				if(i-2>=1 && j-7>=1 && i-2<rows-1 && j-7<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-2, j-7 ) = accessMat( surface, i-2, j-7 ) * ( 0.75 );
				if(i-2>=1 && j-6>=1 && i-2<rows-1 && j-6<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-2, j-6 ) = accessMat( surface, i-2, j-6 ) * ( 0.75 );
				if(i-2>=1 && j-5>=1 && i-2<rows-1 && j-5<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-2, j-5 ) = accessMat( surface, i-2, j-5 ) * ( 0.75 );
				if(i-2>=1 && j-4>=1 && i-2<rows-1 && j-4<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-2, j-4 ) = accessMat( surface, i-2, j-4 ) * ( 0.75 );
				if(i-2>=1 && j-3>=1 && i-2<rows-1 && j-3<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-2, j-3 ) = accessMat( surface, i-2, j-3 ) * ( 0.75 );
				if(i-2>=1 && j-2>=1 && i-2<rows-1 && j-2<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-2, j-2 ) = accessMat( surface, i-2, j-2 ) * ( 0.75 );
				if(i-2>=1 && j-1>=1 && i-2<rows-1 && j-1<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-2, j-1 ) = accessMat( surface, i-2, j-1 ) * ( 0.75 );
				if(i-2>=1 && j+0>=1 && i-2<rows-1 && j+0<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-2, j+0 ) = accessMat( surface, i-2, j+0 ) * ( 0.75 );
				if(i-2>=1 && j+1>=1 && i-2<rows-1 && j+1<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-2, j+1 ) = accessMat( surface, i-2, j+1 ) * ( 0.75 );
				if(i-2>=1 && j+2>=1 && i-2<rows-1 && j+2<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-2, j+2 ) = accessMat( surface, i-2, j+2 ) * ( 0.75 );
				if(i-2>=1 && j+3>=1 && i-2<rows-1 && j+3<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-2, j+3 ) = accessMat( surface, i-2, j+3 ) * ( 0.75 );
				if(i-2>=1 && j+4>=1 && i-2<rows-1 && j+4<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-2, j+4 ) = accessMat( surface, i-2, j+4 ) * ( 0.75 );
				if(i-2>=1 && j+5>=1 && i-2<rows-1 && j+5<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-2, j+5 ) = accessMat( surface, i-2, j+5 ) * ( 0.75 );
				if(i-2>=1 && j+6>=1 && i-2<rows-1 && j+6<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-2, j+6 ) = accessMat( surface, i-2, j+6 ) * ( 0.75 );
				if(i-2>=1 && j+7>=1 && i-2<rows-1 && j+7<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-2, j+7 ) = accessMat( surface, i-2, j+7 ) * ( 0.75 );
				if(i-2>=1 && j+8>=1 && i-2<rows-1 && j+8<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-2, j+8 ) = accessMat( surface, i-2, j+8 ) * ( 0.75 );
				if(i-1>=1 && j-8>=1 && i-1<rows-1 && j-8<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-1, j-8 ) = accessMat( surface, i-1, j-8 ) * ( 0.75 );
				if(i-1>=1 && j-7>=1 && i-1<rows-1 && j-7<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-1, j-7 ) = accessMat( surface, i-1, j-7 ) * ( 0.75 );
				if(i-1>=1 && j-6>=1 && i-1<rows-1 && j-6<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-1, j-6 ) = accessMat( surface, i-1, j-6 ) * ( 0.75 );
				if(i-1>=1 && j-5>=1 && i-1<rows-1 && j-5<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-1, j-5 ) = accessMat( surface, i-1, j-5 ) * ( 0.75 );
				if(i-1>=1 && j-4>=1 && i-1<rows-1 && j-4<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-1, j-4 ) = accessMat( surface, i-1, j-4 ) * ( 0.75 );
				if(i-1>=1 && j-3>=1 && i-1<rows-1 && j-3<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-1, j-3 ) = accessMat( surface, i-1, j-3 ) * ( 0.75 );
				if(i-1>=1 && j-2>=1 && i-1<rows-1 && j-2<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-1, j-2 ) = accessMat( surface, i-1, j-2 ) * ( 0.75 );
				if(i-1>=1 && j-1>=1 && i-1<rows-1 && j-1<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-1, j-1 ) = accessMat( surface, i-1, j-1 ) * ( 0.75 );
				if(i-1>=1 && j+0>=1 && i-1<rows-1 && j+0<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-1, j+0 ) = accessMat( surface, i-1, j+0 ) * ( 0.75 );
				if(i-1>=1 && j+1>=1 && i-1<rows-1 && j+1<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-1, j+1 ) = accessMat( surface, i-1, j+1 ) * ( 0.75 );
				if(i-1>=1 && j+2>=1 && i-1<rows-1 && j+2<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-1, j+2 ) = accessMat( surface, i-1, j+2 ) * ( 0.75 );
				if(i-1>=1 && j+3>=1 && i-1<rows-1 && j+3<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-1, j+3 ) = accessMat( surface, i-1, j+3 ) * ( 0.75 );
				if(i-1>=1 && j+4>=1 && i-1<rows-1 && j+4<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-1, j+4 ) = accessMat( surface, i-1, j+4 ) * ( 0.75 );
				if(i-1>=1 && j+5>=1 && i-1<rows-1 && j+5<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-1, j+5 ) = accessMat( surface, i-1, j+5 ) * ( 0.75 );
				if(i-1>=1 && j+6>=1 && i-1<rows-1 && j+6<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-1, j+6 ) = accessMat( surface, i-1, j+6 ) * ( 0.75 );
				if(i-1>=1 && j+7>=1 && i-1<rows-1 && j+7<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-1, j+7 ) = accessMat( surface, i-1, j+7 ) * ( 0.75 );
				if(i-1>=1 && j+8>=1 && i-1<rows-1 && j+8<columns-1) 
 #pragma omp atomic
  accessMat( surface, i-1, j+8 ) = accessMat( surface, i-1, j+8 ) * ( 0.75 );
				if(i+0>=1 && j-9>=1 && i+0<rows-1 && j-9<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+0, j-9 ) = accessMat( surface, i+0, j-9 ) * ( 0.75 );
				if(i+0>=1 && j-8>=1 && i+0<rows-1 && j-8<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+0, j-8 ) = accessMat( surface, i+0, j-8 ) * ( 0.75 );
				if(i+0>=1 && j-7>=1 && i+0<rows-1 && j-7<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+0, j-7 ) = accessMat( surface, i+0, j-7 ) * ( 0.75 );
				if(i+0>=1 && j-6>=1 && i+0<rows-1 && j-6<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+0, j-6 ) = accessMat( surface, i+0, j-6 ) * ( 0.75 );
				if(i+0>=1 && j-5>=1 && i+0<rows-1 && j-5<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+0, j-5 ) = accessMat( surface, i+0, j-5 ) * ( 0.75 );
				if(i+0>=1 && j-4>=1 && i+0<rows-1 && j-4<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+0, j-4 ) = accessMat( surface, i+0, j-4 ) * ( 0.75 );
				if(i+0>=1 && j-3>=1 && i+0<rows-1 && j-3<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+0, j-3 ) = accessMat( surface, i+0, j-3 ) * ( 0.75 );
				if(i+0>=1 && j-2>=1 && i+0<rows-1 && j-2<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+0, j-2 ) = accessMat( surface, i+0, j-2 ) * ( 0.75 );
				if(i+0>=1 && j-1>=1 && i+0<rows-1 && j-1<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+0, j-1 ) = accessMat( surface, i+0, j-1 ) * ( 0.75 );
				if(i+0>=1 && j+0>=1 && i+0<rows-1 && j+0<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+0, j+0 ) = accessMat( surface, i+0, j+0 ) * ( 0.75 );
				if(i+0>=1 && j+1>=1 && i+0<rows-1 && j+1<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+0, j+1 ) = accessMat( surface, i+0, j+1 ) * ( 0.75 );
				if(i+0>=1 && j+2>=1 && i+0<rows-1 && j+2<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+0, j+2 ) = accessMat( surface, i+0, j+2 ) * ( 0.75 );
				if(i+0>=1 && j+3>=1 && i+0<rows-1 && j+3<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+0, j+3 ) = accessMat( surface, i+0, j+3 ) * ( 0.75 );
				if(i+0>=1 && j+4>=1 && i+0<rows-1 && j+4<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+0, j+4 ) = accessMat( surface, i+0, j+4 ) * ( 0.75 );
				if(i+0>=1 && j+5>=1 && i+0<rows-1 && j+5<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+0, j+5 ) = accessMat( surface, i+0, j+5 ) * ( 0.75 );
				if(i+0>=1 && j+6>=1 && i+0<rows-1 && j+6<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+0, j+6 ) = accessMat( surface, i+0, j+6 ) * ( 0.75 );
				if(i+0>=1 && j+7>=1 && i+0<rows-1 && j+7<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+0, j+7 ) = accessMat( surface, i+0, j+7 ) * ( 0.75 );
				if(i+0>=1 && j+8>=1 && i+0<rows-1 && j+8<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+0, j+8 ) = accessMat( surface, i+0, j+8 ) * ( 0.75 );
				if(i+0>=1 && j+9>=1 && i+0<rows-1 && j+9<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+0, j+9 ) = accessMat( surface, i+0, j+9 ) * ( 0.75 );
				if(i+1>=1 && j-8>=1 && i+1<rows-1 && j-8<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+1, j-8 ) = accessMat( surface, i+1, j-8 ) * ( 0.75 );
				if(i+1>=1 && j-7>=1 && i+1<rows-1 && j-7<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+1, j-7 ) = accessMat( surface, i+1, j-7 ) * ( 0.75 );
				if(i+1>=1 && j-6>=1 && i+1<rows-1 && j-6<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+1, j-6 ) = accessMat( surface, i+1, j-6 ) * ( 0.75 );
				if(i+1>=1 && j-5>=1 && i+1<rows-1 && j-5<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+1, j-5 ) = accessMat( surface, i+1, j-5 ) * ( 0.75 );
				if(i+1>=1 && j-4>=1 && i+1<rows-1 && j-4<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+1, j-4 ) = accessMat( surface, i+1, j-4 ) * ( 0.75 );
				if(i+1>=1 && j-3>=1 && i+1<rows-1 && j-3<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+1, j-3 ) = accessMat( surface, i+1, j-3 ) * ( 0.75 );
				if(i+1>=1 && j-2>=1 && i+1<rows-1 && j-2<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+1, j-2 ) = accessMat( surface, i+1, j-2 ) * ( 0.75 );
				if(i+1>=1 && j-1>=1 && i+1<rows-1 && j-1<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+1, j-1 ) = accessMat( surface, i+1, j-1 ) * ( 0.75 );
				if(i+1>=1 && j+0>=1 && i+1<rows-1 && j+0<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+1, j+0 ) = accessMat( surface, i+1, j+0 ) * ( 0.75 );
				if(i+1>=1 && j+1>=1 && i+1<rows-1 && j+1<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+1, j+1 ) = accessMat( surface, i+1, j+1 ) * ( 0.75 );
				if(i+1>=1 && j+2>=1 && i+1<rows-1 && j+2<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+1, j+2 ) = accessMat( surface, i+1, j+2 ) * ( 0.75 );
				if(i+1>=1 && j+3>=1 && i+1<rows-1 && j+3<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+1, j+3 ) = accessMat( surface, i+1, j+3 ) * ( 0.75 );
				if(i+1>=1 && j+4>=1 && i+1<rows-1 && j+4<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+1, j+4 ) = accessMat( surface, i+1, j+4 ) * ( 0.75 );
				if(i+1>=1 && j+5>=1 && i+1<rows-1 && j+5<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+1, j+5 ) = accessMat( surface, i+1, j+5 ) * ( 0.75 );
				if(i+1>=1 && j+6>=1 && i+1<rows-1 && j+6<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+1, j+6 ) = accessMat( surface, i+1, j+6 ) * ( 0.75 );
				if(i+1>=1 && j+7>=1 && i+1<rows-1 && j+7<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+1, j+7 ) = accessMat( surface, i+1, j+7 ) * ( 0.75 );
				if(i+1>=1 && j+8>=1 && i+1<rows-1 && j+8<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+1, j+8 ) = accessMat( surface, i+1, j+8 ) * ( 0.75 );
				if(i+2>=1 && j-8>=1 && i+2<rows-1 && j-8<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+2, j-8 ) = accessMat( surface, i+2, j-8 ) * ( 0.75 );
				if(i+2>=1 && j-7>=1 && i+2<rows-1 && j-7<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+2, j-7 ) = accessMat( surface, i+2, j-7 ) * ( 0.75 );
				if(i+2>=1 && j-6>=1 && i+2<rows-1 && j-6<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+2, j-6 ) = accessMat( surface, i+2, j-6 ) * ( 0.75 );
				if(i+2>=1 && j-5>=1 && i+2<rows-1 && j-5<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+2, j-5 ) = accessMat( surface, i+2, j-5 ) * ( 0.75 );
				if(i+2>=1 && j-4>=1 && i+2<rows-1 && j-4<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+2, j-4 ) = accessMat( surface, i+2, j-4 ) * ( 0.75 );
				if(i+2>=1 && j-3>=1 && i+2<rows-1 && j-3<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+2, j-3 ) = accessMat( surface, i+2, j-3 ) * ( 0.75 );
				if(i+2>=1 && j-2>=1 && i+2<rows-1 && j-2<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+2, j-2 ) = accessMat( surface, i+2, j-2 ) * ( 0.75 );
				if(i+2>=1 && j-1>=1 && i+2<rows-1 && j-1<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+2, j-1 ) = accessMat( surface, i+2, j-1 ) * ( 0.75 );
				if(i+2>=1 && j+0>=1 && i+2<rows-1 && j+0<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+2, j+0 ) = accessMat( surface, i+2, j+0 ) * ( 0.75 );
				if(i+2>=1 && j+1>=1 && i+2<rows-1 && j+1<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+2, j+1 ) = accessMat( surface, i+2, j+1 ) * ( 0.75 );
				if(i+2>=1 && j+2>=1 && i+2<rows-1 && j+2<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+2, j+2 ) = accessMat( surface, i+2, j+2 ) * ( 0.75 );
				if(i+2>=1 && j+3>=1 && i+2<rows-1 && j+3<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+2, j+3 ) = accessMat( surface, i+2, j+3 ) * ( 0.75 );
				if(i+2>=1 && j+4>=1 && i+2<rows-1 && j+4<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+2, j+4 ) = accessMat( surface, i+2, j+4 ) * ( 0.75 );
				if(i+2>=1 && j+5>=1 && i+2<rows-1 && j+5<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+2, j+5 ) = accessMat( surface, i+2, j+5 ) * ( 0.75 );
				if(i+2>=1 && j+6>=1 && i+2<rows-1 && j+6<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+2, j+6 ) = accessMat( surface, i+2, j+6 ) * ( 0.75 );
				if(i+2>=1 && j+7>=1 && i+2<rows-1 && j+7<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+2, j+7 ) = accessMat( surface, i+2, j+7 ) * ( 0.75 );
				if(i+2>=1 && j+8>=1 && i+2<rows-1 && j+8<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+2, j+8 ) = accessMat( surface, i+2, j+8 ) * ( 0.75 );
				if(i+3>=1 && j-8>=1 && i+3<rows-1 && j-8<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+3, j-8 ) = accessMat( surface, i+3, j-8 ) * ( 0.75 );
				if(i+3>=1 && j-7>=1 && i+3<rows-1 && j-7<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+3, j-7 ) = accessMat( surface, i+3, j-7 ) * ( 0.75 );
				if(i+3>=1 && j-6>=1 && i+3<rows-1 && j-6<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+3, j-6 ) = accessMat( surface, i+3, j-6 ) * ( 0.75 );
				if(i+3>=1 && j-5>=1 && i+3<rows-1 && j-5<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+3, j-5 ) = accessMat( surface, i+3, j-5 ) * ( 0.75 );
				if(i+3>=1 && j-4>=1 && i+3<rows-1 && j-4<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+3, j-4 ) = accessMat( surface, i+3, j-4 ) * ( 0.75 );
				if(i+3>=1 && j-3>=1 && i+3<rows-1 && j-3<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+3, j-3 ) = accessMat( surface, i+3, j-3 ) * ( 0.75 );
				if(i+3>=1 && j-2>=1 && i+3<rows-1 && j-2<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+3, j-2 ) = accessMat( surface, i+3, j-2 ) * ( 0.75 );
				if(i+3>=1 && j-1>=1 && i+3<rows-1 && j-1<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+3, j-1 ) = accessMat( surface, i+3, j-1 ) * ( 0.75 );
				if(i+3>=1 && j+0>=1 && i+3<rows-1 && j+0<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+3, j+0 ) = accessMat( surface, i+3, j+0 ) * ( 0.75 );
				if(i+3>=1 && j+1>=1 && i+3<rows-1 && j+1<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+3, j+1 ) = accessMat( surface, i+3, j+1 ) * ( 0.75 );
				if(i+3>=1 && j+2>=1 && i+3<rows-1 && j+2<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+3, j+2 ) = accessMat( surface, i+3, j+2 ) * ( 0.75 );
				if(i+3>=1 && j+3>=1 && i+3<rows-1 && j+3<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+3, j+3 ) = accessMat( surface, i+3, j+3 ) * ( 0.75 );
				if(i+3>=1 && j+4>=1 && i+3<rows-1 && j+4<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+3, j+4 ) = accessMat( surface, i+3, j+4 ) * ( 0.75 );
				if(i+3>=1 && j+5>=1 && i+3<rows-1 && j+5<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+3, j+5 ) = accessMat( surface, i+3, j+5 ) * ( 0.75 );
				if(i+3>=1 && j+6>=1 && i+3<rows-1 && j+6<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+3, j+6 ) = accessMat( surface, i+3, j+6 ) * ( 0.75 );
				if(i+3>=1 && j+7>=1 && i+3<rows-1 && j+7<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+3, j+7 ) = accessMat( surface, i+3, j+7 ) * ( 0.75 );
				if(i+3>=1 && j+8>=1 && i+3<rows-1 && j+8<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+3, j+8 ) = accessMat( surface, i+3, j+8 ) * ( 0.75 );
				if(i+4>=1 && j-8>=1 && i+4<rows-1 && j-8<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+4, j-8 ) = accessMat( surface, i+4, j-8 ) * ( 0.75 );
				if(i+4>=1 && j-7>=1 && i+4<rows-1 && j-7<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+4, j-7 ) = accessMat( surface, i+4, j-7 ) * ( 0.75 );
				if(i+4>=1 && j-6>=1 && i+4<rows-1 && j-6<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+4, j-6 ) = accessMat( surface, i+4, j-6 ) * ( 0.75 );
				if(i+4>=1 && j-5>=1 && i+4<rows-1 && j-5<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+4, j-5 ) = accessMat( surface, i+4, j-5 ) * ( 0.75 );
				if(i+4>=1 && j-4>=1 && i+4<rows-1 && j-4<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+4, j-4 ) = accessMat( surface, i+4, j-4 ) * ( 0.75 );
				if(i+4>=1 && j-3>=1 && i+4<rows-1 && j-3<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+4, j-3 ) = accessMat( surface, i+4, j-3 ) * ( 0.75 );
				if(i+4>=1 && j-2>=1 && i+4<rows-1 && j-2<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+4, j-2 ) = accessMat( surface, i+4, j-2 ) * ( 0.75 );
				if(i+4>=1 && j-1>=1 && i+4<rows-1 && j-1<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+4, j-1 ) = accessMat( surface, i+4, j-1 ) * ( 0.75 );
				if(i+4>=1 && j+0>=1 && i+4<rows-1 && j+0<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+4, j+0 ) = accessMat( surface, i+4, j+0 ) * ( 0.75 );
				if(i+4>=1 && j+1>=1 && i+4<rows-1 && j+1<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+4, j+1 ) = accessMat( surface, i+4, j+1 ) * ( 0.75 );
				if(i+4>=1 && j+2>=1 && i+4<rows-1 && j+2<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+4, j+2 ) = accessMat( surface, i+4, j+2 ) * ( 0.75 );
				if(i+4>=1 && j+3>=1 && i+4<rows-1 && j+3<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+4, j+3 ) = accessMat( surface, i+4, j+3 ) * ( 0.75 );
				if(i+4>=1 && j+4>=1 && i+4<rows-1 && j+4<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+4, j+4 ) = accessMat( surface, i+4, j+4 ) * ( 0.75 );
				if(i+4>=1 && j+5>=1 && i+4<rows-1 && j+5<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+4, j+5 ) = accessMat( surface, i+4, j+5 ) * ( 0.75 );
				if(i+4>=1 && j+6>=1 && i+4<rows-1 && j+6<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+4, j+6 ) = accessMat( surface, i+4, j+6 ) * ( 0.75 );
				if(i+4>=1 && j+7>=1 && i+4<rows-1 && j+7<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+4, j+7 ) = accessMat( surface, i+4, j+7 ) * ( 0.75 );
				if(i+4>=1 && j+8>=1 && i+4<rows-1 && j+8<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+4, j+8 ) = accessMat( surface, i+4, j+8 ) * ( 0.75 );
				if(i+5>=1 && j-7>=1 && i+5<rows-1 && j-7<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+5, j-7 ) = accessMat( surface, i+5, j-7 ) * ( 0.75 );
				if(i+5>=1 && j-6>=1 && i+5<rows-1 && j-6<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+5, j-6 ) = accessMat( surface, i+5, j-6 ) * ( 0.75 );
				if(i+5>=1 && j-5>=1 && i+5<rows-1 && j-5<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+5, j-5 ) = accessMat( surface, i+5, j-5 ) * ( 0.75 );
				if(i+5>=1 && j-4>=1 && i+5<rows-1 && j-4<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+5, j-4 ) = accessMat( surface, i+5, j-4 ) * ( 0.75 );
				if(i+5>=1 && j-3>=1 && i+5<rows-1 && j-3<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+5, j-3 ) = accessMat( surface, i+5, j-3 ) * ( 0.75 );
				if(i+5>=1 && j-2>=1 && i+5<rows-1 && j-2<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+5, j-2 ) = accessMat( surface, i+5, j-2 ) * ( 0.75 );
				if(i+5>=1 && j-1>=1 && i+5<rows-1 && j-1<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+5, j-1 ) = accessMat( surface, i+5, j-1 ) * ( 0.75 );
				if(i+5>=1 && j+0>=1 && i+5<rows-1 && j+0<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+5, j+0 ) = accessMat( surface, i+5, j+0 ) * ( 0.75 );
				if(i+5>=1 && j+1>=1 && i+5<rows-1 && j+1<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+5, j+1 ) = accessMat( surface, i+5, j+1 ) * ( 0.75 );
				if(i+5>=1 && j+2>=1 && i+5<rows-1 && j+2<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+5, j+2 ) = accessMat( surface, i+5, j+2 ) * ( 0.75 );
				if(i+5>=1 && j+3>=1 && i+5<rows-1 && j+3<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+5, j+3 ) = accessMat( surface, i+5, j+3 ) * ( 0.75 );
				if(i+5>=1 && j+4>=1 && i+5<rows-1 && j+4<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+5, j+4 ) = accessMat( surface, i+5, j+4 ) * ( 0.75 );
				if(i+5>=1 && j+5>=1 && i+5<rows-1 && j+5<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+5, j+5 ) = accessMat( surface, i+5, j+5 ) * ( 0.75 );
				if(i+5>=1 && j+6>=1 && i+5<rows-1 && j+6<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+5, j+6 ) = accessMat( surface, i+5, j+6 ) * ( 0.75 );
				if(i+5>=1 && j+7>=1 && i+5<rows-1 && j+7<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+5, j+7 ) = accessMat( surface, i+5, j+7 ) * ( 0.75 );
				if(i+6>=1 && j-6>=1 && i+6<rows-1 && j-6<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+6, j-6 ) = accessMat( surface, i+6, j-6 ) * ( 0.75 );
				if(i+6>=1 && j-5>=1 && i+6<rows-1 && j-5<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+6, j-5 ) = accessMat( surface, i+6, j-5 ) * ( 0.75 );
				if(i+6>=1 && j-4>=1 && i+6<rows-1 && j-4<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+6, j-4 ) = accessMat( surface, i+6, j-4 ) * ( 0.75 );
				if(i+6>=1 && j-3>=1 && i+6<rows-1 && j-3<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+6, j-3 ) = accessMat( surface, i+6, j-3 ) * ( 0.75 );
				if(i+6>=1 && j-2>=1 && i+6<rows-1 && j-2<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+6, j-2 ) = accessMat( surface, i+6, j-2 ) * ( 0.75 );
				if(i+6>=1 && j-1>=1 && i+6<rows-1 && j-1<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+6, j-1 ) = accessMat( surface, i+6, j-1 ) * ( 0.75 );
				if(i+6>=1 && j+0>=1 && i+6<rows-1 && j+0<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+6, j+0 ) = accessMat( surface, i+6, j+0 ) * ( 0.75 );
				if(i+6>=1 && j+1>=1 && i+6<rows-1 && j+1<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+6, j+1 ) = accessMat( surface, i+6, j+1 ) * ( 0.75 );
				if(i+6>=1 && j+2>=1 && i+6<rows-1 && j+2<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+6, j+2 ) = accessMat( surface, i+6, j+2 ) * ( 0.75 );
				if(i+6>=1 && j+3>=1 && i+6<rows-1 && j+3<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+6, j+3 ) = accessMat( surface, i+6, j+3 ) * ( 0.75 );
				if(i+6>=1 && j+4>=1 && i+6<rows-1 && j+4<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+6, j+4 ) = accessMat( surface, i+6, j+4 ) * ( 0.75 );
				if(i+6>=1 && j+5>=1 && i+6<rows-1 && j+5<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+6, j+5 ) = accessMat( surface, i+6, j+5 ) * ( 0.75 );
				if(i+6>=1 && j+6>=1 && i+6<rows-1 && j+6<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+6, j+6 ) = accessMat( surface, i+6, j+6 ) * ( 0.75 );
				if(i+7>=1 && j-5>=1 && i+7<rows-1 && j-5<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+7, j-5 ) = accessMat( surface, i+7, j-5 ) * ( 0.75 );
				if(i+7>=1 && j-4>=1 && i+7<rows-1 && j-4<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+7, j-4 ) = accessMat( surface, i+7, j-4 ) * ( 0.75 );
				if(i+7>=1 && j-3>=1 && i+7<rows-1 && j-3<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+7, j-3 ) = accessMat( surface, i+7, j-3 ) * ( 0.75 );
				if(i+7>=1 && j-2>=1 && i+7<rows-1 && j-2<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+7, j-2 ) = accessMat( surface, i+7, j-2 ) * ( 0.75 );
				if(i+7>=1 && j-1>=1 && i+7<rows-1 && j-1<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+7, j-1 ) = accessMat( surface, i+7, j-1 ) * ( 0.75 );
				if(i+7>=1 && j+0>=1 && i+7<rows-1 && j+0<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+7, j+0 ) = accessMat( surface, i+7, j+0 ) * ( 0.75 );
				if(i+7>=1 && j+1>=1 && i+7<rows-1 && j+1<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+7, j+1 ) = accessMat( surface, i+7, j+1 ) * ( 0.75 );
				if(i+7>=1 && j+2>=1 && i+7<rows-1 && j+2<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+7, j+2 ) = accessMat( surface, i+7, j+2 ) * ( 0.75 );
				if(i+7>=1 && j+3>=1 && i+7<rows-1 && j+3<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+7, j+3 ) = accessMat( surface, i+7, j+3 ) * ( 0.75 );
				if(i+7>=1 && j+4>=1 && i+7<rows-1 && j+4<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+7, j+4 ) = accessMat( surface, i+7, j+4 ) * ( 0.75 );
				if(i+7>=1 && j+5>=1 && i+7<rows-1 && j+5<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+7, j+5 ) = accessMat( surface, i+7, j+5 ) * ( 0.75 );
				if(i+8>=1 && j-4>=1 && i+8<rows-1 && j-4<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+8, j-4 ) = accessMat( surface, i+8, j-4 ) * ( 0.75 );
				if(i+8>=1 && j-3>=1 && i+8<rows-1 && j-3<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+8, j-3 ) = accessMat( surface, i+8, j-3 ) * ( 0.75 );
				if(i+8>=1 && j-2>=1 && i+8<rows-1 && j-2<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+8, j-2 ) = accessMat( surface, i+8, j-2 ) * ( 0.75 );
				if(i+8>=1 && j-1>=1 && i+8<rows-1 && j-1<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+8, j-1 ) = accessMat( surface, i+8, j-1 ) * ( 0.75 );
				if(i+8>=1 && j+0>=1 && i+8<rows-1 && j+0<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+8, j+0 ) = accessMat( surface, i+8, j+0 ) * ( 0.75 );
				if(i+8>=1 && j+1>=1 && i+8<rows-1 && j+1<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+8, j+1 ) = accessMat( surface, i+8, j+1 ) * ( 0.75 );
				if(i+8>=1 && j+2>=1 && i+8<rows-1 && j+2<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+8, j+2 ) = accessMat( surface, i+8, j+2 ) * ( 0.75 );
				if(i+8>=1 && j+3>=1 && i+8<rows-1 && j+3<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+8, j+3 ) = accessMat( surface, i+8, j+3 ) * ( 0.75 );
				if(i+8>=1 && j+4>=1 && i+8<rows-1 && j+4<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+8, j+4 ) = accessMat( surface, i+8, j+4 ) * ( 0.75 );
				if(i+9>=1 && j+0>=1 && i+9<rows-1 && j+0<columns-1) 
 #pragma omp atomic
  accessMat( surface, i+9, j+0 ) = accessMat( surface, i+9, j+0 ) * ( 0.75 );
			}
			/*for( i=teams[t].x-radius; i<=teams[t].x+radius; i++ ) {
				for( j=teams[t].y-radius; j<=teams[t].y+radius; j++ ) {
					if ( i<1 || i>=rows-1 || j<1 || j>=columns-1 ) continue; // Out of the heated surface
					float dx = teams[t].x - i;
					float dy = teams[t].y - j;
					float distance = sqrtf( dx*dx + dy*dy );
					if ( distance <= radius ) {
						accessMat( surface, i, j ) = accessMat( surface, i, j ) * ( 0.75 ); // Team efficiency factor
					} //Puede dar problemas al paralelizar 
				}
			}*/
		
		
		}
		

		}

			#ifdef DEBUG
			/* 4.5. DEBUG: Print the current state of the simulation at the end of each iteration */
			//print_status( iter, rows, columns, surface, num_teams, teams, num_focal, focal, global_residual );
			#endif
	}

		

		
	
	
/*
 *
 * STOP HERE: DO NOT CHANGE THE CODE BELOW THIS POINT
 *
 */

	/* 5. Stop global time */
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