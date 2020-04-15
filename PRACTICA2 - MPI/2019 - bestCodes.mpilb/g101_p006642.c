// User: g101@90.173.167.29 
// ExecutionRequest[P:'RobaPole3.c',P:1,T:1,args:'',q:'mpilb'] 
// Apr 04 2019 15:36:16
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
#include <stddef.h>
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

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))
/*
 * Function: Print usage line in stderr
 */
void show_usage( char *program_name ) {
	//fprintf(stderr,"Usage: %s <config_file> | <command_line_args>\n", program_name );
	//fprintf(stderr,"\t<config_file> ::= -f <file_name>\n");
	//fprintf(stderr,"\t<command_line_args> ::= <rows> <columns> <maxIter> <numTeams> [ <teamX> <teamY> <teamType> ... ] <numFocalPoints> [ <focalX> <focalY> <focalStart> <focalTemperature> ... ]\n");
	//fprintf(stderr,"\n");
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
		//fprintf(stderr,"-- Error in arguments: No arguments\n");
		show_usage( argv[0] );
		MPI_Abort( MPI_COMM_WORLD,  EXIT_FAILURE );
	}

	int read_from_file = ! strcmp( argv[1], "-f" );
	/* 1.2. Read configuration from file */
	if ( read_from_file ) {
		/* 1.2.1. Open file */
		if (argc<3) {
			//fprintf(stderr,"-- Error in arguments: file-name argument missing\n");
			show_usage( argv[0] );
			MPI_Abort( MPI_COMM_WORLD,  EXIT_FAILURE );
		}
		FILE *args = cp_abrir_fichero( argv[2] );
		if ( args == NULL ) {
			//fprintf(stderr,"-- Error in file: not found: %s\n", argv[1]);
			MPI_Abort( MPI_COMM_WORLD,  EXIT_FAILURE );
		}

		/* 1.2.2. Read surface size and maximum number of iterations */
		int ok;
		ok = fscanf(args, "%d %d %d", &rows, &columns, &max_iter);
		if ( ok != 3 ) {
			//fprintf(stderr,"-- Error in file: reading rows, columns, max_iter from file: %s\n", argv[1]);
			MPI_Abort( MPI_COMM_WORLD,  EXIT_FAILURE );
		}

		/* 1.2.3. Teams information */
		ok = fscanf(args, "%d", &num_teams );
		if ( ok != 1 ) {
			//fprintf(stderr,"-- Error file, reading num_teams from file: %s\n", argv[1]);
			MPI_Abort( MPI_COMM_WORLD,  EXIT_FAILURE );
		}
		teams = (Team *)malloc( sizeof(Team) * (size_t)num_teams );
		if ( teams == NULL ) {
			//fprintf(stderr,"-- Error allocating: %d teams\n", num_teams );
			MPI_Abort( MPI_COMM_WORLD,  EXIT_FAILURE );
		}
		for( i=0; i<num_teams; i++ ) {
			ok = fscanf(args, "%d %d %d", &teams[i].x, &teams[i].y, &teams[i].type);
			if ( ok != 3 ) {
				//fprintf(stderr,"-- Error in file: reading team %d from file: %s\n", i, argv[1]);
				MPI_Abort( MPI_COMM_WORLD,  EXIT_FAILURE );
			}
		}

		/* 1.2.4. Focal points information */
		ok = fscanf(args, "%d", &num_focal );
		if ( ok != 1 ) {
			//fprintf(stderr,"-- Error in file: reading num_focal from file: %s\n", argv[1]);
			MPI_Abort( MPI_COMM_WORLD,  EXIT_FAILURE );
		}
		focal = (FocalPoint *)malloc( sizeof(FocalPoint) * (size_t)num_focal );
		if ( focal == NULL ) {
			//fprintf(stderr,"-- Error allocating: %d focal points\n", num_focal );
			MPI_Abort( MPI_COMM_WORLD,  EXIT_FAILURE );
		}
		for( i=0; i<num_focal; i++ ) {
			ok = fscanf(args, "%d %d %d %d", &focal[i].x, &focal[i].y, &focal[i].start, &focal[i].heat);
			if ( ok != 4 ) {
				//fprintf(stderr,"-- Error in file: reading focal point %d from file: %s\n", i, argv[1]);
				MPI_Abort( MPI_COMM_WORLD,  EXIT_FAILURE );
			}
			focal[i].active = 0;
		}
	}
	/* 1.3. Read configuration from arguments */
	else {
		/* 1.3.1. Check minimum number of arguments */
		if (argc<6) {
			//fprintf(stderr, "-- Error in arguments: not enough arguments when reading configuration from the command line\n");
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
			//fprintf(stderr,"-- Error allocating: %d teams\n", num_teams );
			MPI_Abort( MPI_COMM_WORLD,  EXIT_FAILURE );
		}
		if ( argc < num_teams*3 + 5 ) {
			//fprintf(stderr,"-- Error in arguments: not enough arguments for %d teams\n", num_teams );
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
			//fprintf(stderr,"-- Error in arguments: not enough arguments for the number of focal points\n");
			show_usage( argv[0] );
			MPI_Abort( MPI_COMM_WORLD,  EXIT_FAILURE );
		}
		num_focal = atoi( argv[focal_args] );
		focal = (FocalPoint *)malloc( sizeof(FocalPoint) * (size_t)num_focal );
		if ( teams == NULL ) {
			//fprintf(stderr,"-- Error allocating: %d focal points\n", num_focal );
			MPI_Abort( MPI_COMM_WORLD,  EXIT_FAILURE );
		}
		if ( argc < focal_args + 1 + num_focal*4 ) {
			//fprintf(stderr,"-- Error in arguments: not enough arguments for %d focal points\n", num_focal );
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
			//fprintf(stderr,"-- Error in arguments: extra arguments at the end of the command line\n");
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
		//fprintf(stderr,"-- Error allocating: residualHeat\n");
		MPI_Abort( MPI_COMM_WORLD,  EXIT_FAILURE );
	}

/*
 *
 * START HERE: DO NOT CHANGE THE CODE ABOVE THIS POINT
 *
 */

 	MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
	float *copiaResidual = (float*)malloc( sizeof(float) * (size_t)num_focal );
	if ( copiaResidual == NULL ) {
		//fprintf(stderr,"-- Error allocating: residualHeat\n");
		MPI_Abort( MPI_COMM_WORLD,  EXIT_FAILURE );
	}

	//int cerr;
	int iter;
	int my_rows;
	int nprocs;
	int my_first_row, my_last_row;
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
	int tag = 1000;


	//Si el numero de filas es divisible entre el numero de procs a cada proc se le asignan rows/size filas + 2 bordes
	//Si no, se hace lo mismo pero a la última se le asigna, ademas, el resto (modulo)
	
	if (rows%nprocs==0) {
		my_rows = rows/nprocs;
	} else {
		if (rank < nprocs-1 ) {
			my_rows = rows/nprocs;
		} else {
			my_rows = rows/nprocs+rows%nprocs;
		}
	}

	// el tamano del margen se define aqui. usar constante? (TODO) Primera y ultima reciben solo 1
	if(rank==0 || rank == nprocs-1){
		my_rows = my_rows + 1;
	}
	else{
		my_rows = my_rows +2;
	}
	//double section_timer=0.0;
	//my_rows_t = my_rows +2;
	int cnst = rows/nprocs;
	my_first_row= rank*cnst-1;
	//QUITAR
	if(rank==0) {
		my_first_row = 0;
		//printf("Filas %d Columnas %d\n", my_rows, columns);
	}

	my_last_row= my_first_row + my_rows - 1;
	//printf("%d Mi primera columna: %d La ultima: %d\n",rank, my_first_row, my_last_row);
	//printf("%d Mi num columnas: %d\n",rank, my_rows);
	//printf("Rank %d -> %d\n",rank,my_rows);

	float *surface = (float *)malloc( sizeof(float) * (size_t)my_rows * (size_t)columns );
	float *surfaceCopy = (float *)malloc( sizeof(float) * (size_t)my_rows * (size_t)columns );

	if ( surface == NULL || surfaceCopy == NULL ) {
		//fprintf(stderr,"-- Error allocating: surface structures\n");
		MPI_Abort( MPI_COMM_WORLD,  EXIT_FAILURE );
	}
	//MPI_Finalize();
	//QUITAR
	//printf("Variables declaradas\n");

	/* 3. Initialize surface */
	for( i=0; i<my_rows; i++ )
		for( j=0; j<columns; j++ ){
			accessMat( surface, i, j ) = 0.0f;
			accessMat( surfaceCopy, i, j ) = 0.0f;
		}

	//QUITAR
	//printf("Surface inicializada\n");

	/* 4. Simulation */
	int flag_stability = 0;
	int first_activation = 0;

/*

    int blocklengths[4] = {1,1,1,1};
    MPI_Datatype types[4] = {MPI_INT, MPI_INT,MPI_INT, MPI_INT};
    MPI_Datatype mpi_team;
    MPI_Aint    offsets[4];

    offsets[0] = offsetof(Team, x);
    offsets[1] = offsetof(Team, y);
    offsets[2] = offsetof(Team, type);
    offsets[3] = offsetof(Team, target);

    MPI_Type_create_struct(4, blocklengths, offsets, types, &mpi_team);
    MPI_Type_commit(&mpi_team);*/

	int init_iter = 9999999;
	for( i=0; i<num_focal; i++ ) {
			if ( focal[i].start < init_iter ) {
				init_iter = focal[i].start;
			}
	}
	int ganamos = 0;
	int step;
	for( iter=init_iter; ! flag_stability && iter<max_iter ; iter++ ) {
		//section_timer = cp_Wtime();
				
		/* 4.1. Activate focal points */
		int num_deactivated = 0;
		for( i=0; i<num_focal; i++ ) {
			if ( focal[i].start == iter ) {
				ganamos = 1;
				focal[i].active = 1;
				if ( ! first_activation ) first_activation = 1;
			}
			if ( focal[i].active == 2 ) num_deactivated++;
		}
		if(ganamos==0) continue;
		/* 4.2. Propagate heat (10 steps per each team movement) */
		float global_residual = 0.0f;
		float parcial_residual = 0.0f;
		
		for( step=0;step<10; step++ )	{
			/* 4.2.1. Update heat on active focal points */
			for( i=0; i<num_focal; i++ ) {
				if ( focal[i].active != 1 ) continue;
				// ver si hay que cambiar un foco en la matriz local
				int x = focal[i].x;
				if (x-my_first_row  >= 0 && x-my_first_row < my_rows) {
					int y = focal[i].y;
					//QUITAR
					//printf("%d Actualizado calor en punto focal %d Coordenadas: %d %d \n",rank, i, x, y);
					accessMat( surface, x-my_first_row, y ) = focal[i].heat;
				}

			}
			//MPI_Barrier(MPI_COMM_WORLD);
			
			float *copia = surface;
			surface = surfaceCopy;
			surfaceCopy = copia;

			/* 4.2.2. Copy values of the surface in ancillary structure (Skip borders?)
			for( i=0; i<my_rows; i++ )
				for( j=0; j<columns; j++ )
					accessMat( surfaceCopy, i, j ) = accessMat( surface, i, j ); */

			//ENVIAR Y RECIBIR INFO DE LAS BANDAS CONTIGUAS
			

			/* 4.2.3. Update surface values (skip borders) */
			// NECESARIA INFORMACIÓN DE LAS BANDAS CONTIGUAS: MARGENES (TODO)
			for( i=1; i<my_rows-1; i++ ){
				for( j=1; j<columns-1; j++ ){
					accessMat( surface, i, j ) = (
						accessMat( surfaceCopy, i-1, j ) +
						accessMat( surfaceCopy, i+1, j ) +
						accessMat( surfaceCopy, i, j-1 ) +
						accessMat( surfaceCopy, i, j+1 ) )*0.25;

						if (step==0 && num_deactivated==num_focal ){
							if ( fabs( accessMat( surface, i, j ) - accessMat( surfaceCopy, i, j ) > global_residual ) ) {
							global_residual = fabs( accessMat( surface, i, j ) - accessMat( surfaceCopy, i, j ) );
							}
						}
				}
			}	
				
				if (rank>0 && rank < nprocs-1) {
					MPI_Request request,request2,request3,request4;

					MPI_Isend( &accessMat(surface,1,0),columns, MPI_FLOAT, rank-1, tag, MPI_COMM_WORLD,&request2);							
					MPI_Isend( &accessMat(surface,my_rows-2,0),columns, MPI_FLOAT, rank+1, tag, MPI_COMM_WORLD,&request3);
				
					MPI_Irecv( &accessMat(surface,0,0),columns, MPI_FLOAT, rank-1, tag, MPI_COMM_WORLD, &request);		
					//MPI_Wait(&request, MPI_STATUS_IGNORE);			
					MPI_Irecv( &accessMat(surface,my_rows-1,0),columns, MPI_FLOAT, rank+1, tag, MPI_COMM_WORLD, &request4);
					MPI_Wait(&request4, MPI_STATUS_IGNORE);
					MPI_Wait(&request, MPI_STATUS_IGNORE);			

				}
				else if (rank==nprocs-1) {
					MPI_Request request3,request4;

					MPI_Isend( &accessMat(surface,1,0),columns, MPI_FLOAT, rank-1, tag, MPI_COMM_WORLD ,&request4);
					MPI_Irecv( &accessMat(surface,0,0),columns, MPI_FLOAT, rank-1, tag, MPI_COMM_WORLD, &request3);
					MPI_Wait(&request3, MPI_STATUS_IGNORE);

				}
				else  {
					MPI_Request request,request2;

					MPI_Isend( &accessMat(surface,my_rows-2,0),columns, MPI_FLOAT, rank+1, tag, MPI_COMM_WORLD ,&request);
					MPI_Irecv( &accessMat(surface,my_rows-1,0),columns, MPI_FLOAT, rank+1, tag, MPI_COMM_WORLD, &request2);
					MPI_Wait(&request2, MPI_STATUS_IGNORE);

				}  
				

			/* 4.2.4. Compute the maximum residual difference (absolute value) */
			//NECESITA INFO DE TODO. REDUCE??? (TODO)
			
		//Se viene un Reduce. (Se ha calculado el global residual maximo para este proceso, no para todos)
		
		//QUITAR
		//printf("%f, %f\n",parcial_residual, global_residual);
		}

		
		
		MPI_Allreduce(&global_residual, &parcial_residual, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
		global_residual = parcial_residual;
		//MPI_Barrier(MPI_COMM_WORLD);
		//if (rank==0) 			printf("%f\n",global_residual);

		/* If the global residual is lower than THRESHOLD, we have reached enough stability, stop simulation at the end of this iteration */
		if( num_deactivated == num_focal && global_residual < 0.1f ) {
			flag_stability = 1;
		}

		/* 4.3. Move teams */



		if((num_deactivated!=num_focal || !first_activation))
		for(t=0; t<num_teams ; t++ ) {
			/* 4.3.1. Choose nearest focal point */
			float distance = FLT_MAX;
			int target = -1;
			for( j=0; j<num_focal; j++ ) {
				if ( focal[j].active != 1 ) continue; // Skip non-active focal points
				//Son absolutas, luego habria que mirar donde caen solo para modificar los fuegos,
				//que es lo que almacena surface
			/*	float dx = focal[j].x - teams[t].x;
				float dy = focal[j].y - teams[t].y;
				float local_distance = sqrtf( dx*dx + dy*dy );
			*/	float local_distance  = ( (focal[j].x - teams[t].x)*(focal[j].x - teams[t].x) + (focal[j].y - teams[t].y)*(focal[j].y - teams[t].y) );
	
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
		//MPI_Bcast(&teams[0],num_teams,mpi_team,0,MPI_COMM_WORLD);
		//MPI_Barrier(MPI_COMM_WORLD);
		
		/* 4.4. Team actions */
		for( t=0; t<num_teams; t++ ) {
			/* 4.4.1. Deactivate the target focal point when it is reached */
			int target = teams[t].target;
			if ( target != -1 && focal[target].x == teams[t].x && focal[target].y == teams[t].y
				&& focal[target].active == 1 )
				focal[target].active = 2;

			/* 4.4.2. Reduce heat in a circle around the team */
			int radius = 9;
			if ( teams[t].type == 1 ) radius = 3;
			//int cosa1 = MIN(teams[t].x+radius,my_last_row);
			int cosa1 = (teams[t].x+radius < my_last_row) ? teams[t].x+radius : my_last_row; 
			//printf("MIN %d %d --> %d\n",teams[t].x+radius,my_last_row,cosa1);
			//int cosa2 = MAX(teams[t].x-radius,my_first_row);
			int cosa2 = (teams[t].x-radius > my_first_row) ? teams[t].x-radius : my_first_row; 

			//printf("MAX %d %d --> %d [%d ]\n",teams[t].x-radius,my_first_row,cosa2, cosa1-cosa2);
			for( i=cosa2; i<=cosa1; i++ ) {
				for( j=teams[t].y-radius; j<=teams[t].y+radius; j++ ) {
					if ( i<1 || i>=rows-1 || j<1 || j>=columns-1 ) continue; // Out of the heated surface
					//if (i-my_first_row>=0 && i- my_first_row<my_rows){
					/*	float dx = teams[t].x - i;
						float dy = teams[t].y - j;
						float distance = sqrtf( dx*dx + dy*dy );
					*/	float distance = (teams[t].x - i)*(teams[t].x - i) + (teams[t].y - j)*(teams[t].y - j) ;
		
						if ( distance <= radius*radius ) {
							//printf("%d \n",i-my_first_row+1);
							accessMat( surface, i-my_first_row, j ) = accessMat( surface, i-my_first_row, j ) * ( 0.75f ); // Team efficiency factor
						}
					//}
				}
			}
		}






#ifdef DEBUG
		/* 4.5. DEBUG: Print the current state of the simulation at the end of each iteration */
		print_status( iter, rows, columns, surface, num_teams, teams, num_focal, focal, global_residual );
#endif // DEBUG

					/*section_timer = cp_Wtime() - section_timer;
					printf("Initialization time: %lf\n", section_timer);*/
	}


	// MPI Version: Store the results of residual heat on focal points
	for (i=0; i<num_focal; i++) {
		//TODO CAMBIARsimilares
		residualHeat[i] = 0.0f;
		if(focal[i].x - my_first_row >= 0 && focal[i].x - my_first_row < my_rows)
		residualHeat[i] = accessMat( surface, focal[i].x- my_first_row, focal[i].y );
	}

	int red;
	MPI_Request redu;
	red = MPI_Ireduce(residualHeat,copiaResidual , num_focal , MPI_FLOAT , MPI_MAX ,0, MPI_COMM_WORLD,&redu);

	if (red!= MPI_SUCCESS){
		MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
	}
	residualHeat = copiaResidual;
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

	/* 6. Output for leaderbEl usuario reserva espacio para el buffer de mensajesoard */
	if ( rank == 0 ) {
		printf("\n");
		/* 6.1. Total computation time */
		printf("Time: %lf\n", ttotal );
		/* 6.2. Results: Number of iterations, residual heat on the focal points */
		printf("Result: %d", iter);
		//int i = 0;
		for (int i=0; i<num_focal; i++)
			printf(" %.6f", residualHeat[i] );
		printf("\n");
	}

	/* 7. Free resources */
	free( teams );
	free( focal );

	/* 8. End */
//    MPI_Type_free(&mpi_team);
	MPI_Finalize();
	return 0;
}
