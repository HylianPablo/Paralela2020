// User: g106@157.88.139.133 
// ExecutionRequest[P:'pipo.c',P:1,T:1,args:'',q:'mpilb'] 
// Apr 04 2019 18:44:57
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

	int iter,num_procs;
	 MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
	 MPI_Request p1,p2,p3,p4,p5,p6;	
	
	int count = (rows-2) / num_procs;
	int resto = (rows-2) % num_procs;
	int start, stop;//inicio y fin de nuestra seccion sin contar el halo

	start = 0;
	stop = 0;


	for (int w=0;w<num_procs;w++){
		start = stop+1;
    stop = start+count-1;
		if (w<resto)
        stop+=1;

		if(w==rank)
			break;
	} 

	int my_rows_nh = (stop-start+1);
	int my_rows_halo = my_rows_nh + 2;
// MPI Version: Eliminate this conditional to start doing the work in parallel

		float *surface = (float *)malloc( sizeof(float) * (size_t)my_rows_halo * (size_t)columns );
	float *surfaceCopy = (float *)malloc( sizeof(float) * (size_t)my_rows_halo * (size_t)columns );
	if ( surface == NULL || surfaceCopy == NULL ) {
		fprintf(stderr,"-- Error allocating: surface structures\n");
		MPI_Abort( MPI_COMM_WORLD,  EXIT_FAILURE );
	}


		/* 3. Initialize surface */
	for( i=0; i<my_rows_halo; i++ )
		for( j=0; j<columns; j++ ){
			accessMat( surface, i, j ) = 0.0;
			accessMat( surfaceCopy, i, j ) = 0.0;
		}

	/* 4. Simulation */
	int flag_stability = 0;
	int first_activation = 0;
	int flag_deactivate = 0;
	
		int step;
	for( iter=0; iter<max_iter; iter++ ) {

	

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
	
		int step;
			int lower_halo_changed = 1;
		int upper_halo_changed =1;
	
		for( step=0; step<10; step++ )	{
				/* 4.2.1. Update heat on active focal points */
			for( i=0; i<num_focal; i++ ) {
				if ( focal[i].active != 1 ) continue;
				int x = focal[i].x;
				int y = focal[i].y;
				if(x>=start && stop>=x)
					accessMat( surface, x-start+1, y ) = focal[i].heat;
			}
			
				
			float *surfaceAux;
			surfaceAux = surfaceCopy;
			surfaceCopy = surface;
			surface = surfaceAux;


		

					int receive_lower,receive_higher;
				
			  if(rank!=0){
			
				MPI_Irecv(&receive_higher,1,MPI_INT,rank-1,3,MPI_COMM_WORLD,&p5);
				MPI_Send(&upper_halo_changed,1,MPI_INT,rank-1,3,MPI_COMM_WORLD);
				
				if (upper_halo_changed){
					 float *pointer = &surfaceCopy[columns+1]; 
					MPI_Isend(pointer,columns-2,MPI_FLOAT,rank-1,0,MPI_COMM_WORLD,&p3);
				}
      }
		
      if (rank!=num_procs -1){
					MPI_Irecv(&receive_lower,1,MPI_INT,rank+1,3,MPI_COMM_WORLD,&p6);
				MPI_Send(&lower_halo_changed,1,MPI_INT,rank+1,3,MPI_COMM_WORLD);

				if(lower_halo_changed){
        float *pointer = &surfaceCopy[columns*(my_rows_nh)+1];  
			//	 printf("Rank %d envia a %d\n",rank, rank+1);
        MPI_Isend(pointer,columns-2,MPI_FLOAT,rank+1,1,MPI_COMM_WORLD,&p4);
				}
			}

			if(rank!=0){
				MPI_Wait(&p5,MPI_STATUS_IGNORE);
      
			// printf("Rank %d envia a %d",rank, rank-1);
		
				if(receive_higher)
			 		MPI_Irecv(&surfaceCopy[1],columns,MPI_FLOAT,rank-1,1,MPI_COMM_WORLD, &p1);
			}

			if(rank!=num_procs-1){
				MPI_Wait(&p6,MPI_STATUS_IGNORE);
					if(receive_lower){
					float *pointerRecv = &surfaceCopy[columns*(my_rows_nh+1)+1];
					MPI_Irecv(pointerRecv,columns-1,MPI_FLOAT,rank+1,0,MPI_COMM_WORLD, &p2);
					}
			}




				for( i=2; i<my_rows_halo-2; i++ )
				for( j=1; j<columns-1; j++ )
				accessMat( surface, i, j ) = ( 
					accessMat( surfaceCopy, i-1, j ) +
					accessMat( surfaceCopy, i+1, j ) +
					accessMat( surfaceCopy, i, j-1 ) +
					accessMat( surfaceCopy, i, j+1 ) ) / 4;

		
			
			 if(rank!=0){
				 MPI_Wait(&p1,MPI_STATUS_IGNORE);
			 }

			lower_halo_changed = 0;
		 upper_halo_changed =0;
				for( j=1; j<columns-1; j++ ){
				accessMat( surface, 1, j ) = ( 
					accessMat( surfaceCopy, 0, j ) +
					accessMat( surfaceCopy, 2, j ) +
					accessMat( surfaceCopy, 1, j-1 ) +
					accessMat( surfaceCopy, 1, j+1 ) ) / 4;
				}

				for( j=1; j<columns-1; j++ ){
					if(	accessMat( surface, 1, j ) != 	accessMat( surfaceCopy, 1, j ) ){
						upper_halo_changed = 1;
						break;
					}
				}
				
				if(rank!=num_procs-1)
					MPI_Wait(&p2,MPI_STATUS_IGNORE);

				for( j=1; j<columns-1; j++ ){
	
				accessMat( surface, my_rows_halo-2, j ) = ( 
					accessMat( surfaceCopy, my_rows_halo-3, j ) +
					accessMat( surfaceCopy, my_rows_halo-1, j ) +
					accessMat( surfaceCopy, my_rows_halo-2, j-1 ) +
					accessMat( surfaceCopy, my_rows_halo-2, j+1 ) ) / 4;
			
				}

				for( j=1; j<columns-1; j++ ){
					if(accessMat( surface, my_rows_halo-2, j )!=accessMat( surfaceCopy, my_rows_halo-2, j )){
						lower_halo_changed=1;
						break;
					}
				}

	

		} //Aqui acaba el bucle step
		
	  	if(num_deactivated == num_focal) flag_deactivate=1;	
	
		for( t=0; t<num_teams; t++ ) {
			
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
		
			teams[t].target = target;

		
			if ( target == -1 ) continue; 

			if ( teams[t].type == 1 ) { 
				// Type 1: Can move in diagonal
				if ( focal[target].x < teams[t].x ) teams[t].x--;
				else if ( focal[target].x > teams[t].x ) teams[t].x++;
				if ( focal[target].y < teams[t].y ) teams[t].y--;
				else if ( focal[target].y > teams[t].y ) teams[t].y++;
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
		for( t=0; t<num_teams; t++ ) {
			/* 4.4.1. Deactivate the target focal point when it is reached */
			int target = teams[t].target;
			if ( target != -1 && focal[target].x == teams[t].x && focal[target].y == teams[t].y 
				&& focal[target].active == 1 )
				focal[target].active = 2;

			/* 4.4.2. Reduce heat in a circle around the team */
		i=teams[t].x-start+1;
j=teams[t].y;
// Influence area of fixed radius depending on type
if ( teams[t].type == 1 ) {
//radius = RADIUS_TYPE_1;
//else radius = RADIUS_TYPE_2_3;
				if(teams[t].x+RADIUS_TYPE_1<start || teams[t].x-RADIUS_TYPE_1>stop)
					continue;
			
				if(i-3>=1 && j+0>=1 && i-3<my_rows_halo-1 && j+0<columns-1) 
 
  accessMat( surface, i-3, j+0 ) = accessMat( surface, i-3, j+0 ) * ( 0.75 );
				if(i-2>=1 && j-2>=1 && i-2<my_rows_halo-1 && j-2<columns-1) 
 
  accessMat( surface, i-2, j-2 ) = accessMat( surface, i-2, j-2 ) * ( 0.75 );
				if(i-2>=1 && j-1>=1 && i-2<my_rows_halo-1 && j-1<columns-1) 
 
  accessMat( surface, i-2, j-1 ) = accessMat( surface, i-2, j-1 ) * ( 0.75 );
				if(i-2>=1 && j+0>=1 && i-2<my_rows_halo-1 && j+0<columns-1) 
 
  accessMat( surface, i-2, j+0 ) = accessMat( surface, i-2, j+0 ) * ( 0.75 );
				if(i-2>=1 && j+1>=1 && i-2<my_rows_halo-1 && j+1<columns-1) 
 
  accessMat( surface, i-2, j+1 ) = accessMat( surface, i-2, j+1 ) * ( 0.75 );
				if(i-2>=1 && j+2>=1 && i-2<my_rows_halo-1 && j+2<columns-1) 
 
  accessMat( surface, i-2, j+2 ) = accessMat( surface, i-2, j+2 ) * ( 0.75 );
				if(i-1>=1 && j-2>=1 && i-1<my_rows_halo-1 && j-2<columns-1) 
 
  accessMat( surface, i-1, j-2 ) = accessMat( surface, i-1, j-2 ) * ( 0.75 );
				if(i-1>=1 && j-1>=1 && i-1<my_rows_halo-1 && j-1<columns-1) 
 
  accessMat( surface, i-1, j-1 ) = accessMat( surface, i-1, j-1 ) * ( 0.75 );
				if(i-1>=1 && j+0>=1 && i-1<my_rows_halo-1 && j+0<columns-1) 
 
  accessMat( surface, i-1, j+0 ) = accessMat( surface, i-1, j+0 ) * ( 0.75 );
				if(i-1>=1 && j+1>=1 && i-1<my_rows_halo-1 && j+1<columns-1) 
 
  accessMat( surface, i-1, j+1 ) = accessMat( surface, i-1, j+1 ) * ( 0.75 );
				if(i-1>=1 && j+2>=1 && i-1<my_rows_halo-1 && j+2<columns-1) 
 
  accessMat( surface, i-1, j+2 ) = accessMat( surface, i-1, j+2 ) * ( 0.75 );
				if(i+0>=1 && j-3>=1 && i+0<my_rows_halo-1 && j-3<columns-1) 
 
  accessMat( surface, i+0, j-3 ) = accessMat( surface, i+0, j-3 ) * ( 0.75 );
				if(i+0>=1 && j-2>=1 && i+0<my_rows_halo-1 && j-2<columns-1) 
 
  accessMat( surface, i+0, j-2 ) = accessMat( surface, i+0, j-2 ) * ( 0.75 );
				if(i+0>=1 && j-1>=1 && i+0<my_rows_halo-1 && j-1<columns-1) 
 
  accessMat( surface, i+0, j-1 ) = accessMat( surface, i+0, j-1 ) * ( 0.75 );
				if(i+0>=1 && j+0>=1 && i+0<my_rows_halo-1 && j+0<columns-1) 
 
  accessMat( surface, i+0, j+0 ) = accessMat( surface, i+0, j+0 ) * ( 0.75 );
				if(i+0>=1 && j+1>=1 && i+0<my_rows_halo-1 && j+1<columns-1) 
 
  accessMat( surface, i+0, j+1 ) = accessMat( surface, i+0, j+1 ) * ( 0.75 );
				if(i+0>=1 && j+2>=1 && i+0<my_rows_halo-1 && j+2<columns-1) 
 
  accessMat( surface, i+0, j+2 ) = accessMat( surface, i+0, j+2 ) * ( 0.75 );
				if(i+0>=1 && j+3>=1 && i+0<my_rows_halo-1 && j+3<columns-1) 
 
  accessMat( surface, i+0, j+3 ) = accessMat( surface, i+0, j+3 ) * ( 0.75 );
				if(i+1>=1 && j-2>=1 && i+1<my_rows_halo-1 && j-2<columns-1) 
 
  accessMat( surface, i+1, j-2 ) = accessMat( surface, i+1, j-2 ) * ( 0.75 );
				if(i+1>=1 && j-1>=1 && i+1<my_rows_halo-1 && j-1<columns-1) 
 
  accessMat( surface, i+1, j-1 ) = accessMat( surface, i+1, j-1 ) * ( 0.75 );
				if(i+1>=1 && j+0>=1 && i+1<my_rows_halo-1 && j+0<columns-1) 
 
  accessMat( surface, i+1, j+0 ) = accessMat( surface, i+1, j+0 ) * ( 0.75 );
				if(i+1>=1 && j+1>=1 && i+1<my_rows_halo-1 && j+1<columns-1) 
 
  accessMat( surface, i+1, j+1 ) = accessMat( surface, i+1, j+1 ) * ( 0.75 );
				if(i+1>=1 && j+2>=1 && i+1<my_rows_halo-1 && j+2<columns-1) 
 
  accessMat( surface, i+1, j+2 ) = accessMat( surface, i+1, j+2 ) * ( 0.75 );
				if(i+2>=1 && j-2>=1 && i+2<my_rows_halo-1 && j-2<columns-1) 
 
  accessMat( surface, i+2, j-2 ) = accessMat( surface, i+2, j-2 ) * ( 0.75 );
				if(i+2>=1 && j-1>=1 && i+2<my_rows_halo-1 && j-1<columns-1) 
 
  accessMat( surface, i+2, j-1 ) = accessMat( surface, i+2, j-1 ) * ( 0.75 );
				if(i+2>=1 && j+0>=1 && i+2<my_rows_halo-1 && j+0<columns-1) 
 
  accessMat( surface, i+2, j+0 ) = accessMat( surface, i+2, j+0 ) * ( 0.75 );
				if(i+2>=1 && j+1>=1 && i+2<my_rows_halo-1 && j+1<columns-1) 
 
  accessMat( surface, i+2, j+1 ) = accessMat( surface, i+2, j+1 ) * ( 0.75 );
				if(i+2>=1 && j+2>=1 && i+2<my_rows_halo-1 && j+2<columns-1) 
 
  accessMat( surface, i+2, j+2 ) = accessMat( surface, i+2, j+2 ) * ( 0.75 );
				if(i+3>=1 && j+0>=1 && i+3<my_rows_halo-1 && j+0<columns-1) 
 
  accessMat( surface, i+3, j+0 ) = accessMat( surface, i+3, j+0 ) * ( 0.75 );
			
			}
			else{
					if(teams[t].x+RADIUS_TYPE_2_3<start || teams[t].x-RADIUS_TYPE_2_3>stop)
					continue;
			

				if(i-9>=1 && j+0>=1 && i-9<my_rows_halo-1 && j+0<columns-1) 
 
  accessMat( surface, i-9, j+0 ) = accessMat( surface, i-9, j+0 ) * ( 0.75 );
				if(i-8>=1 && j-4>=1 && i-8<my_rows_halo-1 && j-4<columns-1) 
 
  accessMat( surface, i-8, j-4 ) = accessMat( surface, i-8, j-4 ) * ( 0.75 );
				if(i-8>=1 && j-3>=1 && i-8<my_rows_halo-1 && j-3<columns-1) 
 
  accessMat( surface, i-8, j-3 ) = accessMat( surface, i-8, j-3 ) * ( 0.75 );
				if(i-8>=1 && j-2>=1 && i-8<my_rows_halo-1 && j-2<columns-1) 
 
  accessMat( surface, i-8, j-2 ) = accessMat( surface, i-8, j-2 ) * ( 0.75 );
				if(i-8>=1 && j-1>=1 && i-8<my_rows_halo-1 && j-1<columns-1) 
 
  accessMat( surface, i-8, j-1 ) = accessMat( surface, i-8, j-1 ) * ( 0.75 );
				if(i-8>=1 && j+0>=1 && i-8<my_rows_halo-1 && j+0<columns-1) 
 
  accessMat( surface, i-8, j+0 ) = accessMat( surface, i-8, j+0 ) * ( 0.75 );
				if(i-8>=1 && j+1>=1 && i-8<my_rows_halo-1 && j+1<columns-1) 
 
  accessMat( surface, i-8, j+1 ) = accessMat( surface, i-8, j+1 ) * ( 0.75 );
				if(i-8>=1 && j+2>=1 && i-8<my_rows_halo-1 && j+2<columns-1) 
 
  accessMat( surface, i-8, j+2 ) = accessMat( surface, i-8, j+2 ) * ( 0.75 );
				if(i-8>=1 && j+3>=1 && i-8<my_rows_halo-1 && j+3<columns-1) 
 
  accessMat( surface, i-8, j+3 ) = accessMat( surface, i-8, j+3 ) * ( 0.75 );
				if(i-8>=1 && j+4>=1 && i-8<my_rows_halo-1 && j+4<columns-1) 
 
  accessMat( surface, i-8, j+4 ) = accessMat( surface, i-8, j+4 ) * ( 0.75 );
				if(i-7>=1 && j-5>=1 && i-7<my_rows_halo-1 && j-5<columns-1) 
 
  accessMat( surface, i-7, j-5 ) = accessMat( surface, i-7, j-5 ) * ( 0.75 );
				if(i-7>=1 && j-4>=1 && i-7<my_rows_halo-1 && j-4<columns-1) 
 
  accessMat( surface, i-7, j-4 ) = accessMat( surface, i-7, j-4 ) * ( 0.75 );
				if(i-7>=1 && j-3>=1 && i-7<my_rows_halo-1 && j-3<columns-1) 
 
  accessMat( surface, i-7, j-3 ) = accessMat( surface, i-7, j-3 ) * ( 0.75 );
				if(i-7>=1 && j-2>=1 && i-7<my_rows_halo-1 && j-2<columns-1) 
 
  accessMat( surface, i-7, j-2 ) = accessMat( surface, i-7, j-2 ) * ( 0.75 );
				if(i-7>=1 && j-1>=1 && i-7<my_rows_halo-1 && j-1<columns-1) 
 
  accessMat( surface, i-7, j-1 ) = accessMat( surface, i-7, j-1 ) * ( 0.75 );
				if(i-7>=1 && j+0>=1 && i-7<my_rows_halo-1 && j+0<columns-1) 
 
  accessMat( surface, i-7, j+0 ) = accessMat( surface, i-7, j+0 ) * ( 0.75 );
				if(i-7>=1 && j+1>=1 && i-7<my_rows_halo-1 && j+1<columns-1) 
 
  accessMat( surface, i-7, j+1 ) = accessMat( surface, i-7, j+1 ) * ( 0.75 );
				if(i-7>=1 && j+2>=1 && i-7<my_rows_halo-1 && j+2<columns-1) 
 
  accessMat( surface, i-7, j+2 ) = accessMat( surface, i-7, j+2 ) * ( 0.75 );
				if(i-7>=1 && j+3>=1 && i-7<my_rows_halo-1 && j+3<columns-1) 
 
  accessMat( surface, i-7, j+3 ) = accessMat( surface, i-7, j+3 ) * ( 0.75 );
				if(i-7>=1 && j+4>=1 && i-7<my_rows_halo-1 && j+4<columns-1) 
 
  accessMat( surface, i-7, j+4 ) = accessMat( surface, i-7, j+4 ) * ( 0.75 );
				if(i-7>=1 && j+5>=1 && i-7<my_rows_halo-1 && j+5<columns-1) 
 
  accessMat( surface, i-7, j+5 ) = accessMat( surface, i-7, j+5 ) * ( 0.75 );
				if(i-6>=1 && j-6>=1 && i-6<my_rows_halo-1 && j-6<columns-1) 
 
  accessMat( surface, i-6, j-6 ) = accessMat( surface, i-6, j-6 ) * ( 0.75 );
				if(i-6>=1 && j-5>=1 && i-6<my_rows_halo-1 && j-5<columns-1) 
 
  accessMat( surface, i-6, j-5 ) = accessMat( surface, i-6, j-5 ) * ( 0.75 );
				if(i-6>=1 && j-4>=1 && i-6<my_rows_halo-1 && j-4<columns-1) 
 
  accessMat( surface, i-6, j-4 ) = accessMat( surface, i-6, j-4 ) * ( 0.75 );
				if(i-6>=1 && j-3>=1 && i-6<my_rows_halo-1 && j-3<columns-1) 
 
  accessMat( surface, i-6, j-3 ) = accessMat( surface, i-6, j-3 ) * ( 0.75 );
				if(i-6>=1 && j-2>=1 && i-6<my_rows_halo-1 && j-2<columns-1) 
 
  accessMat( surface, i-6, j-2 ) = accessMat( surface, i-6, j-2 ) * ( 0.75 );
				if(i-6>=1 && j-1>=1 && i-6<my_rows_halo-1 && j-1<columns-1) 
 
  accessMat( surface, i-6, j-1 ) = accessMat( surface, i-6, j-1 ) * ( 0.75 );
				if(i-6>=1 && j+0>=1 && i-6<my_rows_halo-1 && j+0<columns-1) 
 
  accessMat( surface, i-6, j+0 ) = accessMat( surface, i-6, j+0 ) * ( 0.75 );
				if(i-6>=1 && j+1>=1 && i-6<my_rows_halo-1 && j+1<columns-1) 
 
  accessMat( surface, i-6, j+1 ) = accessMat( surface, i-6, j+1 ) * ( 0.75 );
				if(i-6>=1 && j+2>=1 && i-6<my_rows_halo-1 && j+2<columns-1) 
 
  accessMat( surface, i-6, j+2 ) = accessMat( surface, i-6, j+2 ) * ( 0.75 );
				if(i-6>=1 && j+3>=1 && i-6<my_rows_halo-1 && j+3<columns-1) 
 
  accessMat( surface, i-6, j+3 ) = accessMat( surface, i-6, j+3 ) * ( 0.75 );
				if(i-6>=1 && j+4>=1 && i-6<my_rows_halo-1 && j+4<columns-1) 
 
  accessMat( surface, i-6, j+4 ) = accessMat( surface, i-6, j+4 ) * ( 0.75 );
				if(i-6>=1 && j+5>=1 && i-6<my_rows_halo-1 && j+5<columns-1) 
 
  accessMat( surface, i-6, j+5 ) = accessMat( surface, i-6, j+5 ) * ( 0.75 );
				if(i-6>=1 && j+6>=1 && i-6<my_rows_halo-1 && j+6<columns-1) 
 
  accessMat( surface, i-6, j+6 ) = accessMat( surface, i-6, j+6 ) * ( 0.75 );
				if(i-5>=1 && j-7>=1 && i-5<my_rows_halo-1 && j-7<columns-1) 
 
  accessMat( surface, i-5, j-7 ) = accessMat( surface, i-5, j-7 ) * ( 0.75 );
				if(i-5>=1 && j-6>=1 && i-5<my_rows_halo-1 && j-6<columns-1) 
 
  accessMat( surface, i-5, j-6 ) = accessMat( surface, i-5, j-6 ) * ( 0.75 );
				if(i-5>=1 && j-5>=1 && i-5<my_rows_halo-1 && j-5<columns-1) 
 
  accessMat( surface, i-5, j-5 ) = accessMat( surface, i-5, j-5 ) * ( 0.75 );
				if(i-5>=1 && j-4>=1 && i-5<my_rows_halo-1 && j-4<columns-1) 
 
  accessMat( surface, i-5, j-4 ) = accessMat( surface, i-5, j-4 ) * ( 0.75 );
				if(i-5>=1 && j-3>=1 && i-5<my_rows_halo-1 && j-3<columns-1) 
 
  accessMat( surface, i-5, j-3 ) = accessMat( surface, i-5, j-3 ) * ( 0.75 );
				if(i-5>=1 && j-2>=1 && i-5<my_rows_halo-1 && j-2<columns-1) 
 
  accessMat( surface, i-5, j-2 ) = accessMat( surface, i-5, j-2 ) * ( 0.75 );
				if(i-5>=1 && j-1>=1 && i-5<my_rows_halo-1 && j-1<columns-1) 
 
  accessMat( surface, i-5, j-1 ) = accessMat( surface, i-5, j-1 ) * ( 0.75 );
				if(i-5>=1 && j+0>=1 && i-5<my_rows_halo-1 && j+0<columns-1) 
 
  accessMat( surface, i-5, j+0 ) = accessMat( surface, i-5, j+0 ) * ( 0.75 );
				if(i-5>=1 && j+1>=1 && i-5<my_rows_halo-1 && j+1<columns-1) 
 
  accessMat( surface, i-5, j+1 ) = accessMat( surface, i-5, j+1 ) * ( 0.75 );
				if(i-5>=1 && j+2>=1 && i-5<my_rows_halo-1 && j+2<columns-1) 
 
  accessMat( surface, i-5, j+2 ) = accessMat( surface, i-5, j+2 ) * ( 0.75 );
				if(i-5>=1 && j+3>=1 && i-5<my_rows_halo-1 && j+3<columns-1) 
 
  accessMat( surface, i-5, j+3 ) = accessMat( surface, i-5, j+3 ) * ( 0.75 );
				if(i-5>=1 && j+4>=1 && i-5<my_rows_halo-1 && j+4<columns-1) 
 
  accessMat( surface, i-5, j+4 ) = accessMat( surface, i-5, j+4 ) * ( 0.75 );
				if(i-5>=1 && j+5>=1 && i-5<my_rows_halo-1 && j+5<columns-1) 
 
  accessMat( surface, i-5, j+5 ) = accessMat( surface, i-5, j+5 ) * ( 0.75 );
				if(i-5>=1 && j+6>=1 && i-5<my_rows_halo-1 && j+6<columns-1) 
 
  accessMat( surface, i-5, j+6 ) = accessMat( surface, i-5, j+6 ) * ( 0.75 );
				if(i-5>=1 && j+7>=1 && i-5<my_rows_halo-1 && j+7<columns-1) 
 
  accessMat( surface, i-5, j+7 ) = accessMat( surface, i-5, j+7 ) * ( 0.75 );
				if(i-4>=1 && j-8>=1 && i-4<my_rows_halo-1 && j-8<columns-1) 
 
  accessMat( surface, i-4, j-8 ) = accessMat( surface, i-4, j-8 ) * ( 0.75 );
				if(i-4>=1 && j-7>=1 && i-4<my_rows_halo-1 && j-7<columns-1) 
 
  accessMat( surface, i-4, j-7 ) = accessMat( surface, i-4, j-7 ) * ( 0.75 );
				if(i-4>=1 && j-6>=1 && i-4<my_rows_halo-1 && j-6<columns-1) 
 
  accessMat( surface, i-4, j-6 ) = accessMat( surface, i-4, j-6 ) * ( 0.75 );
				if(i-4>=1 && j-5>=1 && i-4<my_rows_halo-1 && j-5<columns-1) 
 
  accessMat( surface, i-4, j-5 ) = accessMat( surface, i-4, j-5 ) * ( 0.75 );
				if(i-4>=1 && j-4>=1 && i-4<my_rows_halo-1 && j-4<columns-1) 
 
  accessMat( surface, i-4, j-4 ) = accessMat( surface, i-4, j-4 ) * ( 0.75 );
				if(i-4>=1 && j-3>=1 && i-4<my_rows_halo-1 && j-3<columns-1) 
 
  accessMat( surface, i-4, j-3 ) = accessMat( surface, i-4, j-3 ) * ( 0.75 );
				if(i-4>=1 && j-2>=1 && i-4<my_rows_halo-1 && j-2<columns-1) 
 
  accessMat( surface, i-4, j-2 ) = accessMat( surface, i-4, j-2 ) * ( 0.75 );
				if(i-4>=1 && j-1>=1 && i-4<my_rows_halo-1 && j-1<columns-1) 
 
  accessMat( surface, i-4, j-1 ) = accessMat( surface, i-4, j-1 ) * ( 0.75 );
				if(i-4>=1 && j+0>=1 && i-4<my_rows_halo-1 && j+0<columns-1) 
 
  accessMat( surface, i-4, j+0 ) = accessMat( surface, i-4, j+0 ) * ( 0.75 );
				if(i-4>=1 && j+1>=1 && i-4<my_rows_halo-1 && j+1<columns-1) 
 
  accessMat( surface, i-4, j+1 ) = accessMat( surface, i-4, j+1 ) * ( 0.75 );
				if(i-4>=1 && j+2>=1 && i-4<my_rows_halo-1 && j+2<columns-1) 
 
  accessMat( surface, i-4, j+2 ) = accessMat( surface, i-4, j+2 ) * ( 0.75 );
				if(i-4>=1 && j+3>=1 && i-4<my_rows_halo-1 && j+3<columns-1) 
 
  accessMat( surface, i-4, j+3 ) = accessMat( surface, i-4, j+3 ) * ( 0.75 );
				if(i-4>=1 && j+4>=1 && i-4<my_rows_halo-1 && j+4<columns-1) 
 
  accessMat( surface, i-4, j+4 ) = accessMat( surface, i-4, j+4 ) * ( 0.75 );
				if(i-4>=1 && j+5>=1 && i-4<my_rows_halo-1 && j+5<columns-1) 
 
  accessMat( surface, i-4, j+5 ) = accessMat( surface, i-4, j+5 ) * ( 0.75 );
				if(i-4>=1 && j+6>=1 && i-4<my_rows_halo-1 && j+6<columns-1) 
 
  accessMat( surface, i-4, j+6 ) = accessMat( surface, i-4, j+6 ) * ( 0.75 );
				if(i-4>=1 && j+7>=1 && i-4<my_rows_halo-1 && j+7<columns-1) 
 
  accessMat( surface, i-4, j+7 ) = accessMat( surface, i-4, j+7 ) * ( 0.75 );
				if(i-4>=1 && j+8>=1 && i-4<my_rows_halo-1 && j+8<columns-1) 
 
  accessMat( surface, i-4, j+8 ) = accessMat( surface, i-4, j+8 ) * ( 0.75 );
				if(i-3>=1 && j-8>=1 && i-3<my_rows_halo-1 && j-8<columns-1) 
 
  accessMat( surface, i-3, j-8 ) = accessMat( surface, i-3, j-8 ) * ( 0.75 );
				if(i-3>=1 && j-7>=1 && i-3<my_rows_halo-1 && j-7<columns-1) 
 
  accessMat( surface, i-3, j-7 ) = accessMat( surface, i-3, j-7 ) * ( 0.75 );
				if(i-3>=1 && j-6>=1 && i-3<my_rows_halo-1 && j-6<columns-1) 
 
  accessMat( surface, i-3, j-6 ) = accessMat( surface, i-3, j-6 ) * ( 0.75 );
				if(i-3>=1 && j-5>=1 && i-3<my_rows_halo-1 && j-5<columns-1) 
 
  accessMat( surface, i-3, j-5 ) = accessMat( surface, i-3, j-5 ) * ( 0.75 );
				if(i-3>=1 && j-4>=1 && i-3<my_rows_halo-1 && j-4<columns-1) 
 
  accessMat( surface, i-3, j-4 ) = accessMat( surface, i-3, j-4 ) * ( 0.75 );
				if(i-3>=1 && j-3>=1 && i-3<my_rows_halo-1 && j-3<columns-1) 
 
  accessMat( surface, i-3, j-3 ) = accessMat( surface, i-3, j-3 ) * ( 0.75 );
				if(i-3>=1 && j-2>=1 && i-3<my_rows_halo-1 && j-2<columns-1) 
 
  accessMat( surface, i-3, j-2 ) = accessMat( surface, i-3, j-2 ) * ( 0.75 );
				if(i-3>=1 && j-1>=1 && i-3<my_rows_halo-1 && j-1<columns-1) 
 
  accessMat( surface, i-3, j-1 ) = accessMat( surface, i-3, j-1 ) * ( 0.75 );
				if(i-3>=1 && j+0>=1 && i-3<my_rows_halo-1 && j+0<columns-1) 
 
  accessMat( surface, i-3, j+0 ) = accessMat( surface, i-3, j+0 ) * ( 0.75 );
				if(i-3>=1 && j+1>=1 && i-3<my_rows_halo-1 && j+1<columns-1) 
 
  accessMat( surface, i-3, j+1 ) = accessMat( surface, i-3, j+1 ) * ( 0.75 );
				if(i-3>=1 && j+2>=1 && i-3<my_rows_halo-1 && j+2<columns-1) 
 
  accessMat( surface, i-3, j+2 ) = accessMat( surface, i-3, j+2 ) * ( 0.75 );
				if(i-3>=1 && j+3>=1 && i-3<my_rows_halo-1 && j+3<columns-1) 
 
  accessMat( surface, i-3, j+3 ) = accessMat( surface, i-3, j+3 ) * ( 0.75 );
				if(i-3>=1 && j+4>=1 && i-3<my_rows_halo-1 && j+4<columns-1) 
 
  accessMat( surface, i-3, j+4 ) = accessMat( surface, i-3, j+4 ) * ( 0.75 );
				if(i-3>=1 && j+5>=1 && i-3<my_rows_halo-1 && j+5<columns-1) 
 
  accessMat( surface, i-3, j+5 ) = accessMat( surface, i-3, j+5 ) * ( 0.75 );
				if(i-3>=1 && j+6>=1 && i-3<my_rows_halo-1 && j+6<columns-1) 
 
  accessMat( surface, i-3, j+6 ) = accessMat( surface, i-3, j+6 ) * ( 0.75 );
				if(i-3>=1 && j+7>=1 && i-3<my_rows_halo-1 && j+7<columns-1) 
 
  accessMat( surface, i-3, j+7 ) = accessMat( surface, i-3, j+7 ) * ( 0.75 );
				if(i-3>=1 && j+8>=1 && i-3<my_rows_halo-1 && j+8<columns-1) 
 
  accessMat( surface, i-3, j+8 ) = accessMat( surface, i-3, j+8 ) * ( 0.75 );
				if(i-2>=1 && j-8>=1 && i-2<my_rows_halo-1 && j-8<columns-1) 
 
  accessMat( surface, i-2, j-8 ) = accessMat( surface, i-2, j-8 ) * ( 0.75 );
				if(i-2>=1 && j-7>=1 && i-2<my_rows_halo-1 && j-7<columns-1) 
 
  accessMat( surface, i-2, j-7 ) = accessMat( surface, i-2, j-7 ) * ( 0.75 );
				if(i-2>=1 && j-6>=1 && i-2<my_rows_halo-1 && j-6<columns-1) 
 
  accessMat( surface, i-2, j-6 ) = accessMat( surface, i-2, j-6 ) * ( 0.75 );
				if(i-2>=1 && j-5>=1 && i-2<my_rows_halo-1 && j-5<columns-1) 
 
  accessMat( surface, i-2, j-5 ) = accessMat( surface, i-2, j-5 ) * ( 0.75 );
				if(i-2>=1 && j-4>=1 && i-2<my_rows_halo-1 && j-4<columns-1) 
 
  accessMat( surface, i-2, j-4 ) = accessMat( surface, i-2, j-4 ) * ( 0.75 );
				if(i-2>=1 && j-3>=1 && i-2<my_rows_halo-1 && j-3<columns-1) 
 
  accessMat( surface, i-2, j-3 ) = accessMat( surface, i-2, j-3 ) * ( 0.75 );
				if(i-2>=1 && j-2>=1 && i-2<my_rows_halo-1 && j-2<columns-1) 
 
  accessMat( surface, i-2, j-2 ) = accessMat( surface, i-2, j-2 ) * ( 0.75 );
				if(i-2>=1 && j-1>=1 && i-2<my_rows_halo-1 && j-1<columns-1) 
 
  accessMat( surface, i-2, j-1 ) = accessMat( surface, i-2, j-1 ) * ( 0.75 );
				if(i-2>=1 && j+0>=1 && i-2<my_rows_halo-1 && j+0<columns-1) 
 
  accessMat( surface, i-2, j+0 ) = accessMat( surface, i-2, j+0 ) * ( 0.75 );
				if(i-2>=1 && j+1>=1 && i-2<my_rows_halo-1 && j+1<columns-1) 
 
  accessMat( surface, i-2, j+1 ) = accessMat( surface, i-2, j+1 ) * ( 0.75 );
				if(i-2>=1 && j+2>=1 && i-2<my_rows_halo-1 && j+2<columns-1) 
 
  accessMat( surface, i-2, j+2 ) = accessMat( surface, i-2, j+2 ) * ( 0.75 );
				if(i-2>=1 && j+3>=1 && i-2<my_rows_halo-1 && j+3<columns-1) 
 
  accessMat( surface, i-2, j+3 ) = accessMat( surface, i-2, j+3 ) * ( 0.75 );
				if(i-2>=1 && j+4>=1 && i-2<my_rows_halo-1 && j+4<columns-1) 
 
  accessMat( surface, i-2, j+4 ) = accessMat( surface, i-2, j+4 ) * ( 0.75 );
				if(i-2>=1 && j+5>=1 && i-2<my_rows_halo-1 && j+5<columns-1) 
 
  accessMat( surface, i-2, j+5 ) = accessMat( surface, i-2, j+5 ) * ( 0.75 );
				if(i-2>=1 && j+6>=1 && i-2<my_rows_halo-1 && j+6<columns-1) 
 
  accessMat( surface, i-2, j+6 ) = accessMat( surface, i-2, j+6 ) * ( 0.75 );
				if(i-2>=1 && j+7>=1 && i-2<my_rows_halo-1 && j+7<columns-1) 
 
  accessMat( surface, i-2, j+7 ) = accessMat( surface, i-2, j+7 ) * ( 0.75 );
				if(i-2>=1 && j+8>=1 && i-2<my_rows_halo-1 && j+8<columns-1) 
 
  accessMat( surface, i-2, j+8 ) = accessMat( surface, i-2, j+8 ) * ( 0.75 );
				if(i-1>=1 && j-8>=1 && i-1<my_rows_halo-1 && j-8<columns-1) 
 
  accessMat( surface, i-1, j-8 ) = accessMat( surface, i-1, j-8 ) * ( 0.75 );
				if(i-1>=1 && j-7>=1 && i-1<my_rows_halo-1 && j-7<columns-1) 
 
  accessMat( surface, i-1, j-7 ) = accessMat( surface, i-1, j-7 ) * ( 0.75 );
				if(i-1>=1 && j-6>=1 && i-1<my_rows_halo-1 && j-6<columns-1) 
 
  accessMat( surface, i-1, j-6 ) = accessMat( surface, i-1, j-6 ) * ( 0.75 );
				if(i-1>=1 && j-5>=1 && i-1<my_rows_halo-1 && j-5<columns-1) 
 
  accessMat( surface, i-1, j-5 ) = accessMat( surface, i-1, j-5 ) * ( 0.75 );
				if(i-1>=1 && j-4>=1 && i-1<my_rows_halo-1 && j-4<columns-1) 
 
  accessMat( surface, i-1, j-4 ) = accessMat( surface, i-1, j-4 ) * ( 0.75 );
				if(i-1>=1 && j-3>=1 && i-1<my_rows_halo-1 && j-3<columns-1) 
 
  accessMat( surface, i-1, j-3 ) = accessMat( surface, i-1, j-3 ) * ( 0.75 );
				if(i-1>=1 && j-2>=1 && i-1<my_rows_halo-1 && j-2<columns-1) 
 
  accessMat( surface, i-1, j-2 ) = accessMat( surface, i-1, j-2 ) * ( 0.75 );
				if(i-1>=1 && j-1>=1 && i-1<my_rows_halo-1 && j-1<columns-1) 
 
  accessMat( surface, i-1, j-1 ) = accessMat( surface, i-1, j-1 ) * ( 0.75 );
				if(i-1>=1 && j+0>=1 && i-1<my_rows_halo-1 && j+0<columns-1) 
 
  accessMat( surface, i-1, j+0 ) = accessMat( surface, i-1, j+0 ) * ( 0.75 );
				if(i-1>=1 && j+1>=1 && i-1<my_rows_halo-1 && j+1<columns-1) 
 
  accessMat( surface, i-1, j+1 ) = accessMat( surface, i-1, j+1 ) * ( 0.75 );
				if(i-1>=1 && j+2>=1 && i-1<my_rows_halo-1 && j+2<columns-1) 
 
  accessMat( surface, i-1, j+2 ) = accessMat( surface, i-1, j+2 ) * ( 0.75 );
				if(i-1>=1 && j+3>=1 && i-1<my_rows_halo-1 && j+3<columns-1) 
 
  accessMat( surface, i-1, j+3 ) = accessMat( surface, i-1, j+3 ) * ( 0.75 );
				if(i-1>=1 && j+4>=1 && i-1<my_rows_halo-1 && j+4<columns-1) 
 
  accessMat( surface, i-1, j+4 ) = accessMat( surface, i-1, j+4 ) * ( 0.75 );
				if(i-1>=1 && j+5>=1 && i-1<my_rows_halo-1 && j+5<columns-1) 
 
  accessMat( surface, i-1, j+5 ) = accessMat( surface, i-1, j+5 ) * ( 0.75 );
				if(i-1>=1 && j+6>=1 && i-1<my_rows_halo-1 && j+6<columns-1) 
 
  accessMat( surface, i-1, j+6 ) = accessMat( surface, i-1, j+6 ) * ( 0.75 );
				if(i-1>=1 && j+7>=1 && i-1<my_rows_halo-1 && j+7<columns-1) 
 
  accessMat( surface, i-1, j+7 ) = accessMat( surface, i-1, j+7 ) * ( 0.75 );
				if(i-1>=1 && j+8>=1 && i-1<my_rows_halo-1 && j+8<columns-1) 
 
  accessMat( surface, i-1, j+8 ) = accessMat( surface, i-1, j+8 ) * ( 0.75 );
				if(i+0>=1 && j-9>=1 && i+0<my_rows_halo-1 && j-9<columns-1) 
 
  accessMat( surface, i+0, j-9 ) = accessMat( surface, i+0, j-9 ) * ( 0.75 );
				if(i+0>=1 && j-8>=1 && i+0<my_rows_halo-1 && j-8<columns-1) 
 
  accessMat( surface, i+0, j-8 ) = accessMat( surface, i+0, j-8 ) * ( 0.75 );
				if(i+0>=1 && j-7>=1 && i+0<my_rows_halo-1 && j-7<columns-1) 
 
  accessMat( surface, i+0, j-7 ) = accessMat( surface, i+0, j-7 ) * ( 0.75 );
				if(i+0>=1 && j-6>=1 && i+0<my_rows_halo-1 && j-6<columns-1) 
 
  accessMat( surface, i+0, j-6 ) = accessMat( surface, i+0, j-6 ) * ( 0.75 );
				if(i+0>=1 && j-5>=1 && i+0<my_rows_halo-1 && j-5<columns-1) 
 
  accessMat( surface, i+0, j-5 ) = accessMat( surface, i+0, j-5 ) * ( 0.75 );
				if(i+0>=1 && j-4>=1 && i+0<my_rows_halo-1 && j-4<columns-1) 
 
  accessMat( surface, i+0, j-4 ) = accessMat( surface, i+0, j-4 ) * ( 0.75 );
				if(i+0>=1 && j-3>=1 && i+0<my_rows_halo-1 && j-3<columns-1) 
 
  accessMat( surface, i+0, j-3 ) = accessMat( surface, i+0, j-3 ) * ( 0.75 );
				if(i+0>=1 && j-2>=1 && i+0<my_rows_halo-1 && j-2<columns-1) 
 
  accessMat( surface, i+0, j-2 ) = accessMat( surface, i+0, j-2 ) * ( 0.75 );
				if(i+0>=1 && j-1>=1 && i+0<my_rows_halo-1 && j-1<columns-1) 
 
  accessMat( surface, i+0, j-1 ) = accessMat( surface, i+0, j-1 ) * ( 0.75 );
				if(i+0>=1 && j+0>=1 && i+0<my_rows_halo-1 && j+0<columns-1) 
 
  accessMat( surface, i+0, j+0 ) = accessMat( surface, i+0, j+0 ) * ( 0.75 );
				if(i+0>=1 && j+1>=1 && i+0<my_rows_halo-1 && j+1<columns-1) 
 
  accessMat( surface, i+0, j+1 ) = accessMat( surface, i+0, j+1 ) * ( 0.75 );
				if(i+0>=1 && j+2>=1 && i+0<my_rows_halo-1 && j+2<columns-1) 
 
  accessMat( surface, i+0, j+2 ) = accessMat( surface, i+0, j+2 ) * ( 0.75 );
				if(i+0>=1 && j+3>=1 && i+0<my_rows_halo-1 && j+3<columns-1) 
 
  accessMat( surface, i+0, j+3 ) = accessMat( surface, i+0, j+3 ) * ( 0.75 );
				if(i+0>=1 && j+4>=1 && i+0<my_rows_halo-1 && j+4<columns-1) 
 
  accessMat( surface, i+0, j+4 ) = accessMat( surface, i+0, j+4 ) * ( 0.75 );
				if(i+0>=1 && j+5>=1 && i+0<my_rows_halo-1 && j+5<columns-1) 
 
  accessMat( surface, i+0, j+5 ) = accessMat( surface, i+0, j+5 ) * ( 0.75 );
				if(i+0>=1 && j+6>=1 && i+0<my_rows_halo-1 && j+6<columns-1) 
 
  accessMat( surface, i+0, j+6 ) = accessMat( surface, i+0, j+6 ) * ( 0.75 );
				if(i+0>=1 && j+7>=1 && i+0<my_rows_halo-1 && j+7<columns-1) 
 
  accessMat( surface, i+0, j+7 ) = accessMat( surface, i+0, j+7 ) * ( 0.75 );
				if(i+0>=1 && j+8>=1 && i+0<my_rows_halo-1 && j+8<columns-1) 
 
  accessMat( surface, i+0, j+8 ) = accessMat( surface, i+0, j+8 ) * ( 0.75 );
				if(i+0>=1 && j+9>=1 && i+0<my_rows_halo-1 && j+9<columns-1) 
 
  accessMat( surface, i+0, j+9 ) = accessMat( surface, i+0, j+9 ) * ( 0.75 );
				if(i+1>=1 && j-8>=1 && i+1<my_rows_halo-1 && j-8<columns-1) 
 
  accessMat( surface, i+1, j-8 ) = accessMat( surface, i+1, j-8 ) * ( 0.75 );
				if(i+1>=1 && j-7>=1 && i+1<my_rows_halo-1 && j-7<columns-1) 
 
  accessMat( surface, i+1, j-7 ) = accessMat( surface, i+1, j-7 ) * ( 0.75 );
				if(i+1>=1 && j-6>=1 && i+1<my_rows_halo-1 && j-6<columns-1) 
 
  accessMat( surface, i+1, j-6 ) = accessMat( surface, i+1, j-6 ) * ( 0.75 );
				if(i+1>=1 && j-5>=1 && i+1<my_rows_halo-1 && j-5<columns-1) 
 
  accessMat( surface, i+1, j-5 ) = accessMat( surface, i+1, j-5 ) * ( 0.75 );
				if(i+1>=1 && j-4>=1 && i+1<my_rows_halo-1 && j-4<columns-1) 
 
  accessMat( surface, i+1, j-4 ) = accessMat( surface, i+1, j-4 ) * ( 0.75 );
				if(i+1>=1 && j-3>=1 && i+1<my_rows_halo-1 && j-3<columns-1) 
 
  accessMat( surface, i+1, j-3 ) = accessMat( surface, i+1, j-3 ) * ( 0.75 );
				if(i+1>=1 && j-2>=1 && i+1<my_rows_halo-1 && j-2<columns-1) 
 
  accessMat( surface, i+1, j-2 ) = accessMat( surface, i+1, j-2 ) * ( 0.75 );
				if(i+1>=1 && j-1>=1 && i+1<my_rows_halo-1 && j-1<columns-1) 
 
  accessMat( surface, i+1, j-1 ) = accessMat( surface, i+1, j-1 ) * ( 0.75 );
				if(i+1>=1 && j+0>=1 && i+1<my_rows_halo-1 && j+0<columns-1) 
 
  accessMat( surface, i+1, j+0 ) = accessMat( surface, i+1, j+0 ) * ( 0.75 );
				if(i+1>=1 && j+1>=1 && i+1<my_rows_halo-1 && j+1<columns-1) 
 
  accessMat( surface, i+1, j+1 ) = accessMat( surface, i+1, j+1 ) * ( 0.75 );
				if(i+1>=1 && j+2>=1 && i+1<my_rows_halo-1 && j+2<columns-1) 
 
  accessMat( surface, i+1, j+2 ) = accessMat( surface, i+1, j+2 ) * ( 0.75 );
				if(i+1>=1 && j+3>=1 && i+1<my_rows_halo-1 && j+3<columns-1) 
 
  accessMat( surface, i+1, j+3 ) = accessMat( surface, i+1, j+3 ) * ( 0.75 );
				if(i+1>=1 && j+4>=1 && i+1<my_rows_halo-1 && j+4<columns-1) 
 
  accessMat( surface, i+1, j+4 ) = accessMat( surface, i+1, j+4 ) * ( 0.75 );
				if(i+1>=1 && j+5>=1 && i+1<my_rows_halo-1 && j+5<columns-1) 
 
  accessMat( surface, i+1, j+5 ) = accessMat( surface, i+1, j+5 ) * ( 0.75 );
				if(i+1>=1 && j+6>=1 && i+1<my_rows_halo-1 && j+6<columns-1) 
 
  accessMat( surface, i+1, j+6 ) = accessMat( surface, i+1, j+6 ) * ( 0.75 );
				if(i+1>=1 && j+7>=1 && i+1<my_rows_halo-1 && j+7<columns-1) 
 
  accessMat( surface, i+1, j+7 ) = accessMat( surface, i+1, j+7 ) * ( 0.75 );
				if(i+1>=1 && j+8>=1 && i+1<my_rows_halo-1 && j+8<columns-1) 
 
  accessMat( surface, i+1, j+8 ) = accessMat( surface, i+1, j+8 ) * ( 0.75 );
				if(i+2>=1 && j-8>=1 && i+2<my_rows_halo-1 && j-8<columns-1) 
 
  accessMat( surface, i+2, j-8 ) = accessMat( surface, i+2, j-8 ) * ( 0.75 );
				if(i+2>=1 && j-7>=1 && i+2<my_rows_halo-1 && j-7<columns-1) 
 
  accessMat( surface, i+2, j-7 ) = accessMat( surface, i+2, j-7 ) * ( 0.75 );
				if(i+2>=1 && j-6>=1 && i+2<my_rows_halo-1 && j-6<columns-1) 
 
  accessMat( surface, i+2, j-6 ) = accessMat( surface, i+2, j-6 ) * ( 0.75 );
				if(i+2>=1 && j-5>=1 && i+2<my_rows_halo-1 && j-5<columns-1) 
 
  accessMat( surface, i+2, j-5 ) = accessMat( surface, i+2, j-5 ) * ( 0.75 );
				if(i+2>=1 && j-4>=1 && i+2<my_rows_halo-1 && j-4<columns-1) 
 
  accessMat( surface, i+2, j-4 ) = accessMat( surface, i+2, j-4 ) * ( 0.75 );
				if(i+2>=1 && j-3>=1 && i+2<my_rows_halo-1 && j-3<columns-1) 
 
  accessMat( surface, i+2, j-3 ) = accessMat( surface, i+2, j-3 ) * ( 0.75 );
				if(i+2>=1 && j-2>=1 && i+2<my_rows_halo-1 && j-2<columns-1) 
 
  accessMat( surface, i+2, j-2 ) = accessMat( surface, i+2, j-2 ) * ( 0.75 );
				if(i+2>=1 && j-1>=1 && i+2<my_rows_halo-1 && j-1<columns-1) 
 
  accessMat( surface, i+2, j-1 ) = accessMat( surface, i+2, j-1 ) * ( 0.75 );
				if(i+2>=1 && j+0>=1 && i+2<my_rows_halo-1 && j+0<columns-1) 
 
  accessMat( surface, i+2, j+0 ) = accessMat( surface, i+2, j+0 ) * ( 0.75 );
				if(i+2>=1 && j+1>=1 && i+2<my_rows_halo-1 && j+1<columns-1) 
 
  accessMat( surface, i+2, j+1 ) = accessMat( surface, i+2, j+1 ) * ( 0.75 );
				if(i+2>=1 && j+2>=1 && i+2<my_rows_halo-1 && j+2<columns-1) 
 
  accessMat( surface, i+2, j+2 ) = accessMat( surface, i+2, j+2 ) * ( 0.75 );
				if(i+2>=1 && j+3>=1 && i+2<my_rows_halo-1 && j+3<columns-1) 
 
  accessMat( surface, i+2, j+3 ) = accessMat( surface, i+2, j+3 ) * ( 0.75 );
				if(i+2>=1 && j+4>=1 && i+2<my_rows_halo-1 && j+4<columns-1) 
 
  accessMat( surface, i+2, j+4 ) = accessMat( surface, i+2, j+4 ) * ( 0.75 );
				if(i+2>=1 && j+5>=1 && i+2<my_rows_halo-1 && j+5<columns-1) 
 
  accessMat( surface, i+2, j+5 ) = accessMat( surface, i+2, j+5 ) * ( 0.75 );
				if(i+2>=1 && j+6>=1 && i+2<my_rows_halo-1 && j+6<columns-1) 
 
  accessMat( surface, i+2, j+6 ) = accessMat( surface, i+2, j+6 ) * ( 0.75 );
				if(i+2>=1 && j+7>=1 && i+2<my_rows_halo-1 && j+7<columns-1) 
 
  accessMat( surface, i+2, j+7 ) = accessMat( surface, i+2, j+7 ) * ( 0.75 );
				if(i+2>=1 && j+8>=1 && i+2<my_rows_halo-1 && j+8<columns-1) 
 
  accessMat( surface, i+2, j+8 ) = accessMat( surface, i+2, j+8 ) * ( 0.75 );
				if(i+3>=1 && j-8>=1 && i+3<my_rows_halo-1 && j-8<columns-1) 
 
  accessMat( surface, i+3, j-8 ) = accessMat( surface, i+3, j-8 ) * ( 0.75 );
				if(i+3>=1 && j-7>=1 && i+3<my_rows_halo-1 && j-7<columns-1) 
 
  accessMat( surface, i+3, j-7 ) = accessMat( surface, i+3, j-7 ) * ( 0.75 );
				if(i+3>=1 && j-6>=1 && i+3<my_rows_halo-1 && j-6<columns-1) 
 
  accessMat( surface, i+3, j-6 ) = accessMat( surface, i+3, j-6 ) * ( 0.75 );
				if(i+3>=1 && j-5>=1 && i+3<my_rows_halo-1 && j-5<columns-1) 
 
  accessMat( surface, i+3, j-5 ) = accessMat( surface, i+3, j-5 ) * ( 0.75 );
				if(i+3>=1 && j-4>=1 && i+3<my_rows_halo-1 && j-4<columns-1) 
 
  accessMat( surface, i+3, j-4 ) = accessMat( surface, i+3, j-4 ) * ( 0.75 );
				if(i+3>=1 && j-3>=1 && i+3<my_rows_halo-1 && j-3<columns-1) 
 
  accessMat( surface, i+3, j-3 ) = accessMat( surface, i+3, j-3 ) * ( 0.75 );
				if(i+3>=1 && j-2>=1 && i+3<my_rows_halo-1 && j-2<columns-1) 
 
  accessMat( surface, i+3, j-2 ) = accessMat( surface, i+3, j-2 ) * ( 0.75 );
				if(i+3>=1 && j-1>=1 && i+3<my_rows_halo-1 && j-1<columns-1) 
 
  accessMat( surface, i+3, j-1 ) = accessMat( surface, i+3, j-1 ) * ( 0.75 );
				if(i+3>=1 && j+0>=1 && i+3<my_rows_halo-1 && j+0<columns-1) 
 
  accessMat( surface, i+3, j+0 ) = accessMat( surface, i+3, j+0 ) * ( 0.75 );
				if(i+3>=1 && j+1>=1 && i+3<my_rows_halo-1 && j+1<columns-1) 
 
  accessMat( surface, i+3, j+1 ) = accessMat( surface, i+3, j+1 ) * ( 0.75 );
				if(i+3>=1 && j+2>=1 && i+3<my_rows_halo-1 && j+2<columns-1) 
 
  accessMat( surface, i+3, j+2 ) = accessMat( surface, i+3, j+2 ) * ( 0.75 );
				if(i+3>=1 && j+3>=1 && i+3<my_rows_halo-1 && j+3<columns-1) 
 
  accessMat( surface, i+3, j+3 ) = accessMat( surface, i+3, j+3 ) * ( 0.75 );
				if(i+3>=1 && j+4>=1 && i+3<my_rows_halo-1 && j+4<columns-1) 
 
  accessMat( surface, i+3, j+4 ) = accessMat( surface, i+3, j+4 ) * ( 0.75 );
				if(i+3>=1 && j+5>=1 && i+3<my_rows_halo-1 && j+5<columns-1) 
 
  accessMat( surface, i+3, j+5 ) = accessMat( surface, i+3, j+5 ) * ( 0.75 );
				if(i+3>=1 && j+6>=1 && i+3<my_rows_halo-1 && j+6<columns-1) 
 
  accessMat( surface, i+3, j+6 ) = accessMat( surface, i+3, j+6 ) * ( 0.75 );
				if(i+3>=1 && j+7>=1 && i+3<my_rows_halo-1 && j+7<columns-1) 
 
  accessMat( surface, i+3, j+7 ) = accessMat( surface, i+3, j+7 ) * ( 0.75 );
				if(i+3>=1 && j+8>=1 && i+3<my_rows_halo-1 && j+8<columns-1) 
 
  accessMat( surface, i+3, j+8 ) = accessMat( surface, i+3, j+8 ) * ( 0.75 );
				if(i+4>=1 && j-8>=1 && i+4<my_rows_halo-1 && j-8<columns-1) 
 
  accessMat( surface, i+4, j-8 ) = accessMat( surface, i+4, j-8 ) * ( 0.75 );
				if(i+4>=1 && j-7>=1 && i+4<my_rows_halo-1 && j-7<columns-1) 
 
  accessMat( surface, i+4, j-7 ) = accessMat( surface, i+4, j-7 ) * ( 0.75 );
				if(i+4>=1 && j-6>=1 && i+4<my_rows_halo-1 && j-6<columns-1) 
 
  accessMat( surface, i+4, j-6 ) = accessMat( surface, i+4, j-6 ) * ( 0.75 );
				if(i+4>=1 && j-5>=1 && i+4<my_rows_halo-1 && j-5<columns-1) 
 
  accessMat( surface, i+4, j-5 ) = accessMat( surface, i+4, j-5 ) * ( 0.75 );
				if(i+4>=1 && j-4>=1 && i+4<my_rows_halo-1 && j-4<columns-1) 
 
  accessMat( surface, i+4, j-4 ) = accessMat( surface, i+4, j-4 ) * ( 0.75 );
				if(i+4>=1 && j-3>=1 && i+4<my_rows_halo-1 && j-3<columns-1) 
 
  accessMat( surface, i+4, j-3 ) = accessMat( surface, i+4, j-3 ) * ( 0.75 );
				if(i+4>=1 && j-2>=1 && i+4<my_rows_halo-1 && j-2<columns-1) 
 
  accessMat( surface, i+4, j-2 ) = accessMat( surface, i+4, j-2 ) * ( 0.75 );
				if(i+4>=1 && j-1>=1 && i+4<my_rows_halo-1 && j-1<columns-1) 
 
  accessMat( surface, i+4, j-1 ) = accessMat( surface, i+4, j-1 ) * ( 0.75 );
				if(i+4>=1 && j+0>=1 && i+4<my_rows_halo-1 && j+0<columns-1) 
 
  accessMat( surface, i+4, j+0 ) = accessMat( surface, i+4, j+0 ) * ( 0.75 );
				if(i+4>=1 && j+1>=1 && i+4<my_rows_halo-1 && j+1<columns-1) 
 
  accessMat( surface, i+4, j+1 ) = accessMat( surface, i+4, j+1 ) * ( 0.75 );
				if(i+4>=1 && j+2>=1 && i+4<my_rows_halo-1 && j+2<columns-1) 
 
  accessMat( surface, i+4, j+2 ) = accessMat( surface, i+4, j+2 ) * ( 0.75 );
				if(i+4>=1 && j+3>=1 && i+4<my_rows_halo-1 && j+3<columns-1) 
 
  accessMat( surface, i+4, j+3 ) = accessMat( surface, i+4, j+3 ) * ( 0.75 );
				if(i+4>=1 && j+4>=1 && i+4<my_rows_halo-1 && j+4<columns-1) 
 
  accessMat( surface, i+4, j+4 ) = accessMat( surface, i+4, j+4 ) * ( 0.75 );
				if(i+4>=1 && j+5>=1 && i+4<my_rows_halo-1 && j+5<columns-1) 
 
  accessMat( surface, i+4, j+5 ) = accessMat( surface, i+4, j+5 ) * ( 0.75 );
				if(i+4>=1 && j+6>=1 && i+4<my_rows_halo-1 && j+6<columns-1) 
 
  accessMat( surface, i+4, j+6 ) = accessMat( surface, i+4, j+6 ) * ( 0.75 );
				if(i+4>=1 && j+7>=1 && i+4<my_rows_halo-1 && j+7<columns-1) 
 
  accessMat( surface, i+4, j+7 ) = accessMat( surface, i+4, j+7 ) * ( 0.75 );
				if(i+4>=1 && j+8>=1 && i+4<my_rows_halo-1 && j+8<columns-1) 
 
  accessMat( surface, i+4, j+8 ) = accessMat( surface, i+4, j+8 ) * ( 0.75 );
				if(i+5>=1 && j-7>=1 && i+5<my_rows_halo-1 && j-7<columns-1) 
 
  accessMat( surface, i+5, j-7 ) = accessMat( surface, i+5, j-7 ) * ( 0.75 );
				if(i+5>=1 && j-6>=1 && i+5<my_rows_halo-1 && j-6<columns-1) 
 
  accessMat( surface, i+5, j-6 ) = accessMat( surface, i+5, j-6 ) * ( 0.75 );
				if(i+5>=1 && j-5>=1 && i+5<my_rows_halo-1 && j-5<columns-1) 
 
  accessMat( surface, i+5, j-5 ) = accessMat( surface, i+5, j-5 ) * ( 0.75 );
				if(i+5>=1 && j-4>=1 && i+5<my_rows_halo-1 && j-4<columns-1) 
 
  accessMat( surface, i+5, j-4 ) = accessMat( surface, i+5, j-4 ) * ( 0.75 );
				if(i+5>=1 && j-3>=1 && i+5<my_rows_halo-1 && j-3<columns-1) 
 
  accessMat( surface, i+5, j-3 ) = accessMat( surface, i+5, j-3 ) * ( 0.75 );
				if(i+5>=1 && j-2>=1 && i+5<my_rows_halo-1 && j-2<columns-1) 
 
  accessMat( surface, i+5, j-2 ) = accessMat( surface, i+5, j-2 ) * ( 0.75 );
				if(i+5>=1 && j-1>=1 && i+5<my_rows_halo-1 && j-1<columns-1) 
 
  accessMat( surface, i+5, j-1 ) = accessMat( surface, i+5, j-1 ) * ( 0.75 );
				if(i+5>=1 && j+0>=1 && i+5<my_rows_halo-1 && j+0<columns-1) 
 
  accessMat( surface, i+5, j+0 ) = accessMat( surface, i+5, j+0 ) * ( 0.75 );
				if(i+5>=1 && j+1>=1 && i+5<my_rows_halo-1 && j+1<columns-1) 
 
  accessMat( surface, i+5, j+1 ) = accessMat( surface, i+5, j+1 ) * ( 0.75 );
				if(i+5>=1 && j+2>=1 && i+5<my_rows_halo-1 && j+2<columns-1) 
 
  accessMat( surface, i+5, j+2 ) = accessMat( surface, i+5, j+2 ) * ( 0.75 );
				if(i+5>=1 && j+3>=1 && i+5<my_rows_halo-1 && j+3<columns-1) 
 
  accessMat( surface, i+5, j+3 ) = accessMat( surface, i+5, j+3 ) * ( 0.75 );
				if(i+5>=1 && j+4>=1 && i+5<my_rows_halo-1 && j+4<columns-1) 
 
  accessMat( surface, i+5, j+4 ) = accessMat( surface, i+5, j+4 ) * ( 0.75 );
				if(i+5>=1 && j+5>=1 && i+5<my_rows_halo-1 && j+5<columns-1) 
 
  accessMat( surface, i+5, j+5 ) = accessMat( surface, i+5, j+5 ) * ( 0.75 );
				if(i+5>=1 && j+6>=1 && i+5<my_rows_halo-1 && j+6<columns-1) 
 
  accessMat( surface, i+5, j+6 ) = accessMat( surface, i+5, j+6 ) * ( 0.75 );
				if(i+5>=1 && j+7>=1 && i+5<my_rows_halo-1 && j+7<columns-1) 
 
  accessMat( surface, i+5, j+7 ) = accessMat( surface, i+5, j+7 ) * ( 0.75 );
				if(i+6>=1 && j-6>=1 && i+6<my_rows_halo-1 && j-6<columns-1) 
 
  accessMat( surface, i+6, j-6 ) = accessMat( surface, i+6, j-6 ) * ( 0.75 );
				if(i+6>=1 && j-5>=1 && i+6<my_rows_halo-1 && j-5<columns-1) 
 
  accessMat( surface, i+6, j-5 ) = accessMat( surface, i+6, j-5 ) * ( 0.75 );
				if(i+6>=1 && j-4>=1 && i+6<my_rows_halo-1 && j-4<columns-1) 
 
  accessMat( surface, i+6, j-4 ) = accessMat( surface, i+6, j-4 ) * ( 0.75 );
				if(i+6>=1 && j-3>=1 && i+6<my_rows_halo-1 && j-3<columns-1) 
 
  accessMat( surface, i+6, j-3 ) = accessMat( surface, i+6, j-3 ) * ( 0.75 );
				if(i+6>=1 && j-2>=1 && i+6<my_rows_halo-1 && j-2<columns-1) 
 
  accessMat( surface, i+6, j-2 ) = accessMat( surface, i+6, j-2 ) * ( 0.75 );
				if(i+6>=1 && j-1>=1 && i+6<my_rows_halo-1 && j-1<columns-1) 
 
  accessMat( surface, i+6, j-1 ) = accessMat( surface, i+6, j-1 ) * ( 0.75 );
				if(i+6>=1 && j+0>=1 && i+6<my_rows_halo-1 && j+0<columns-1) 
 
  accessMat( surface, i+6, j+0 ) = accessMat( surface, i+6, j+0 ) * ( 0.75 );
				if(i+6>=1 && j+1>=1 && i+6<my_rows_halo-1 && j+1<columns-1) 
 
  accessMat( surface, i+6, j+1 ) = accessMat( surface, i+6, j+1 ) * ( 0.75 );
				if(i+6>=1 && j+2>=1 && i+6<my_rows_halo-1 && j+2<columns-1) 
 
  accessMat( surface, i+6, j+2 ) = accessMat( surface, i+6, j+2 ) * ( 0.75 );
				if(i+6>=1 && j+3>=1 && i+6<my_rows_halo-1 && j+3<columns-1) 
 
  accessMat( surface, i+6, j+3 ) = accessMat( surface, i+6, j+3 ) * ( 0.75 );
				if(i+6>=1 && j+4>=1 && i+6<my_rows_halo-1 && j+4<columns-1) 
 
  accessMat( surface, i+6, j+4 ) = accessMat( surface, i+6, j+4 ) * ( 0.75 );
				if(i+6>=1 && j+5>=1 && i+6<my_rows_halo-1 && j+5<columns-1) 
 
  accessMat( surface, i+6, j+5 ) = accessMat( surface, i+6, j+5 ) * ( 0.75 );
				if(i+6>=1 && j+6>=1 && i+6<my_rows_halo-1 && j+6<columns-1) 
 
  accessMat( surface, i+6, j+6 ) = accessMat( surface, i+6, j+6 ) * ( 0.75 );
				if(i+7>=1 && j-5>=1 && i+7<my_rows_halo-1 && j-5<columns-1) 
 
  accessMat( surface, i+7, j-5 ) = accessMat( surface, i+7, j-5 ) * ( 0.75 );
				if(i+7>=1 && j-4>=1 && i+7<my_rows_halo-1 && j-4<columns-1) 
 
  accessMat( surface, i+7, j-4 ) = accessMat( surface, i+7, j-4 ) * ( 0.75 );
				if(i+7>=1 && j-3>=1 && i+7<my_rows_halo-1 && j-3<columns-1) 
 
  accessMat( surface, i+7, j-3 ) = accessMat( surface, i+7, j-3 ) * ( 0.75 );
				if(i+7>=1 && j-2>=1 && i+7<my_rows_halo-1 && j-2<columns-1) 
 
  accessMat( surface, i+7, j-2 ) = accessMat( surface, i+7, j-2 ) * ( 0.75 );
				if(i+7>=1 && j-1>=1 && i+7<my_rows_halo-1 && j-1<columns-1) 
 
  accessMat( surface, i+7, j-1 ) = accessMat( surface, i+7, j-1 ) * ( 0.75 );
				if(i+7>=1 && j+0>=1 && i+7<my_rows_halo-1 && j+0<columns-1) 
 
  accessMat( surface, i+7, j+0 ) = accessMat( surface, i+7, j+0 ) * ( 0.75 );
				if(i+7>=1 && j+1>=1 && i+7<my_rows_halo-1 && j+1<columns-1) 
 
  accessMat( surface, i+7, j+1 ) = accessMat( surface, i+7, j+1 ) * ( 0.75 );
				if(i+7>=1 && j+2>=1 && i+7<my_rows_halo-1 && j+2<columns-1) 
 
  accessMat( surface, i+7, j+2 ) = accessMat( surface, i+7, j+2 ) * ( 0.75 );
				if(i+7>=1 && j+3>=1 && i+7<my_rows_halo-1 && j+3<columns-1) 
 
  accessMat( surface, i+7, j+3 ) = accessMat( surface, i+7, j+3 ) * ( 0.75 );
				if(i+7>=1 && j+4>=1 && i+7<my_rows_halo-1 && j+4<columns-1) 
 
  accessMat( surface, i+7, j+4 ) = accessMat( surface, i+7, j+4 ) * ( 0.75 );
				if(i+7>=1 && j+5>=1 && i+7<my_rows_halo-1 && j+5<columns-1) 
 
  accessMat( surface, i+7, j+5 ) = accessMat( surface, i+7, j+5 ) * ( 0.75 );
				if(i+8>=1 && j-4>=1 && i+8<my_rows_halo-1 && j-4<columns-1) 
 
  accessMat( surface, i+8, j-4 ) = accessMat( surface, i+8, j-4 ) * ( 0.75 );
				if(i+8>=1 && j-3>=1 && i+8<my_rows_halo-1 && j-3<columns-1) 
 
  accessMat( surface, i+8, j-3 ) = accessMat( surface, i+8, j-3 ) * ( 0.75 );
				if(i+8>=1 && j-2>=1 && i+8<my_rows_halo-1 && j-2<columns-1) 
 
  accessMat( surface, i+8, j-2 ) = accessMat( surface, i+8, j-2 ) * ( 0.75 );
				if(i+8>=1 && j-1>=1 && i+8<my_rows_halo-1 && j-1<columns-1) 
 
  accessMat( surface, i+8, j-1 ) = accessMat( surface, i+8, j-1 ) * ( 0.75 );
				if(i+8>=1 && j+0>=1 && i+8<my_rows_halo-1 && j+0<columns-1) 
 
  accessMat( surface, i+8, j+0 ) = accessMat( surface, i+8, j+0 ) * ( 0.75 );
				if(i+8>=1 && j+1>=1 && i+8<my_rows_halo-1 && j+1<columns-1) 
 
  accessMat( surface, i+8, j+1 ) = accessMat( surface, i+8, j+1 ) * ( 0.75 );
				if(i+8>=1 && j+2>=1 && i+8<my_rows_halo-1 && j+2<columns-1) 
 
  accessMat( surface, i+8, j+2 ) = accessMat( surface, i+8, j+2 ) * ( 0.75 );
				if(i+8>=1 && j+3>=1 && i+8<my_rows_halo-1 && j+3<columns-1) 
 
  accessMat( surface, i+8, j+3 ) = accessMat( surface, i+8, j+3 ) * ( 0.75 );
				if(i+8>=1 && j+4>=1 && i+8<my_rows_halo-1 && j+4<columns-1) 
 
  accessMat( surface, i+8, j+4 ) = accessMat( surface, i+8, j+4 ) * ( 0.75 );
				if(i+9>=1 && j+0>=1 && i+9<my_rows_halo-1 && j+0<columns-1) 
 
  accessMat( surface, i+9, j+0 ) = accessMat( surface, i+9, j+0 ) * ( 0.75 );
			}
			
		}
		if(flag_deactivate==1){
			iter++;
			break;
		}
	}
	
	/*
	
         ,--"""",--.__,---[],-------._         
       ,"   __,'            \         \--""""""==;-
     ," _,-"  "/---.___     \       ___\   ,-'',"
    /,-'      / ;. ,.--'-.__\  _,-"" ,| `,'   /
   /``""""-._/,-|:\       []\,' ```-/:;-. `. /
             `  ;:::      ||       /:,;  `-.\
                =.,'__,---||-.____',.=
                =(:\_     ||__    ):)=
               ,"::::`----||::`--':::"._
             ,':::::::::::||::::::::::::'.
    .__     ;:::.-.:::::__||___:::::.-.:::\     __,
       """-;:::( O )::::>_|| _<::::( O )::::-"""
   =======;:::::`-`:::::::||':::::::`-`:::::\=======
    ,--"";:::_____________||______________::::""----.          , ,
         ; ::`._(    |    |||     |   )_,'::::\_,,,,,,,,,,____/,'_,
       ,;    :::`--._|____[]|_____|_.-'::::::::::::::::::::::::);_
      ;/ /      :::::::::,||,:::::::::::::::::::::::::::::::::::/
     /; ``''''----------/,'/,__,,,,,____:::::::::::::::::::::,"
     ;/                :);/|_;| ,--.. . ```-.:::::::::::::_,"
    /;                :::):__,'//""\\. ,--.. \:::,:::::_,"
   ;/              :::::/ . . . . . . //""\\. \::":__,"
   ;/          :::::::,' . . . . . . . . . . .:`::\
   ';      :::::::__,'. ,--.. . .,--. . . . . .:`::`
   ';   __,..--'''-. . //""\\. .//""\\ . ,--.. :`:::`
   ;    /  \\ .//""\\ . . . . . . . . . //""\\. :`::`
   ;   /       . . . . . . . . . . . . . . . . .:`::`
   ;   (          . . . . . . . . . . . . . . . ;:::`
   ,:  ;,            . . . . . . . . . . . . . ;':::`
   ,:  ;,             . . . . . . . . . . . . .;`:::
   ,:   ;,             . . . . . . . . . . . . ;`::;`
    ,:  ;             . . . . . . . . . . . . ;':::;`
     :   ;             . . . . . . . . . . . ,':::;
      :   '.          . . . . . . . .. . . .,':::;`
       :    `.       . . . . . . . . . . . ;::::;`
        '.    `-.   . . . . . . . . . . ,-'::::;
          `:_    ``--..___________..--'':::::;'`
             `._::,.:,.:,:_ctr_:,:,.::,.:_;'`
________________`"\/"\/\/'""""`\/"\/""\/"____________________________


*/
	// Los focos están apagados
	for(;iter<max_iter && ! flag_stability;iter++){
	
		// 1a iteracion del bucle step
		float global_residual = 0.0f;
					/* 4.2.1. Update heat on active focal points */
			for( i=0; i<num_focal; i++ ) {
				if ( focal[i].active != 1 ) continue;
				int x = focal[i].x;
				int y = focal[i].y;
				if(x>=start && stop>=x)
					accessMat( surface, x-start+1, y ) = focal[i].heat;
			}
			
				int lower_halo_changed = 1;
		int upper_halo_changed =1;

			float *surfaceAux;
			surfaceAux = surfaceCopy;
			surfaceCopy = surface;
			surface = surfaceAux;


		

					int receive_lower,receive_higher;
					MPI_Request p1,p2,p3,p4,p5,p6;	
			  if(rank!=0){
			
				MPI_Irecv(&receive_higher,1,MPI_INT,rank-1,3,MPI_COMM_WORLD,&p5);
				MPI_Send(&upper_halo_changed,1,MPI_INT,rank-1,3,MPI_COMM_WORLD);
				
				if (upper_halo_changed){
					 float *pointer = &surfaceCopy[columns+1]; 
					MPI_Isend(pointer,columns-2,MPI_FLOAT,rank-1,0,MPI_COMM_WORLD,&p3);
				}
      }
		
      if (rank!=num_procs -1){
					MPI_Irecv(&receive_lower,1,MPI_INT,rank+1,3,MPI_COMM_WORLD,&p6);
				MPI_Send(&lower_halo_changed,1,MPI_INT,rank+1,3,MPI_COMM_WORLD);

				if(lower_halo_changed){
        float *pointer = &surfaceCopy[columns*(my_rows_nh)+1];  
			//	 printf("Rank %d envia a %d\n",rank, rank+1);
        MPI_Isend(pointer,columns-2,MPI_FLOAT,rank+1,1,MPI_COMM_WORLD,&p4);
				}
			}

			if(rank!=0){
				MPI_Wait(&p5,MPI_STATUS_IGNORE);
      
			// printf("Rank %d envia a %d",rank, rank-1);
		
				if(receive_higher)
			 		MPI_Irecv(&surfaceCopy[1],columns-2,MPI_FLOAT,rank-1,1,MPI_COMM_WORLD, &p1);
			}

			if(rank!=num_procs-1){
				MPI_Wait(&p6,MPI_STATUS_IGNORE);
					if(receive_lower){
					float *pointerRecv = &surfaceCopy[columns*(my_rows_nh+1)+1];
					MPI_Irecv(pointerRecv,columns-2,MPI_FLOAT,rank+1,0,MPI_COMM_WORLD, &p2);
					}
			}




				for( i=2; i<my_rows_halo-2; i++ )
				for( j=1; j<columns-1; j++ )
				accessMat( surface, i, j ) = ( 
					accessMat( surfaceCopy, i-1, j ) +
					accessMat( surfaceCopy, i+1, j ) +
					accessMat( surfaceCopy, i, j-1 ) +
					accessMat( surfaceCopy, i, j+1 ) ) / 4;

		
			
			 if(rank!=0){
				 MPI_Wait(&p1,MPI_STATUS_IGNORE);
			 }

			lower_halo_changed = 0;
		 upper_halo_changed =0;
				for( j=1; j<columns-1; j++ ){
				accessMat( surface, 1, j ) = ( 
					accessMat( surfaceCopy, 0, j ) +
					accessMat( surfaceCopy, 2, j ) +
					accessMat( surfaceCopy, 1, j-1 ) +
					accessMat( surfaceCopy, 1, j+1 ) ) / 4;
				}

				for( j=1; j<columns-1; j++ ){
					if(	accessMat( surface, 1, j ) != 	accessMat( surfaceCopy, 1, j ) ){
						upper_halo_changed = 1;
						break;
					}
				}
				
				if(rank!=num_procs-1)
					MPI_Wait(&p2,MPI_STATUS_IGNORE);

				for( j=1; j<columns-1; j++ ){
	
				accessMat( surface, my_rows_halo-2, j ) = ( 
					accessMat( surfaceCopy, my_rows_halo-3, j ) +
					accessMat( surfaceCopy, my_rows_halo-1, j ) +
					accessMat( surfaceCopy, my_rows_halo-2, j-1 ) +
					accessMat( surfaceCopy, my_rows_halo-2, j+1 ) ) / 4;
			
				}

				for( j=1; j<columns-1; j++ ){
					if(accessMat( surface, my_rows_halo-2, j )!=accessMat( surfaceCopy, my_rows_halo-2, j )){
						lower_halo_changed=1;
						break;
					}
				}

	
				//Segunda iteracion del bucle step
	lower_halo_changed = 1;
	upper_halo_changed =1;

			/* 4.2.1. Update heat on active focal points */
			for( i=0; i<num_focal; i++ ) {
				if ( focal[i].active != 1 ) continue;
				int x = focal[i].x;
				int y = focal[i].y;
				if(x>=start && stop>=x)
					accessMat( surface, x-start+1, y ) = focal[i].heat;
			}
			
				
		
			surfaceAux = surfaceCopy;
			surfaceCopy = surface;
			surface = surfaceAux;


		

				
				
			  if(rank!=0){
			
				MPI_Irecv(&receive_higher,1,MPI_INT,rank-1,3,MPI_COMM_WORLD,&p5);
				MPI_Send(&upper_halo_changed,1,MPI_INT,rank-1,3,MPI_COMM_WORLD);
				
				if (upper_halo_changed){
					 float *pointer = &surfaceCopy[columns+1]; 
					MPI_Isend(pointer,columns-2,MPI_FLOAT,rank-1,0,MPI_COMM_WORLD,&p3);
				}
      }
		
      if (rank!=num_procs -1){
					MPI_Irecv(&receive_lower,1,MPI_INT,rank+1,3,MPI_COMM_WORLD,&p6);
				MPI_Send(&lower_halo_changed,1,MPI_INT,rank+1,3,MPI_COMM_WORLD);

				if(lower_halo_changed){
        float *pointer = &surfaceCopy[columns*(my_rows_nh)+1];  
			//	 printf("Rank %d envia a %d\n",rank, rank+1);
        MPI_Isend(pointer,columns-2,MPI_FLOAT,rank+1,1,MPI_COMM_WORLD,&p4);
				}
			}

			if(rank!=0){
				MPI_Wait(&p5,MPI_STATUS_IGNORE);
      
			// printf("Rank %d envia a %d",rank, rank-1);
		
				if(receive_higher)
			 		MPI_Irecv(&surfaceCopy[1],columns-2,MPI_FLOAT,rank-1,1,MPI_COMM_WORLD, &p1);
			}

			if(rank!=num_procs-1){
				MPI_Wait(&p6,MPI_STATUS_IGNORE);
					if(receive_lower){
					float *pointerRecv = &surfaceCopy[columns*(my_rows_nh+1)+1];
					MPI_Irecv(pointerRecv,columns-2,MPI_FLOAT,rank+1,0,MPI_COMM_WORLD, &p2);
					}
			}


                	//Calculode global residual de la primera iteración de step
					for( i=1; i<my_rows_halo-1; i++ )
				for( j=1; j<columns-1; j++ ){
					float calc = accessMat( surface, i, j ) - accessMat( surfaceCopy, i, j );
					if(calc>THRESHOLD){
						global_residual = calc;
						break;
					}
					if ( fabs( calc) > global_residual  ) {
						global_residual = fabs(calc );
					}
				}
					float max_global_residual;
      MPI_Allreduce(&global_residual, &max_global_residual,1,MPI_FLOAT,MPI_MAX,MPI_COMM_WORLD);
      global_residual = max_global_residual;
	

    
				for( i=2; i<my_rows_halo-2; i++ )
				for( j=1; j<columns-1; j++ )
				accessMat( surface, i, j ) = ( 
					accessMat( surfaceCopy, i-1, j ) +
					accessMat( surfaceCopy, i+1, j ) +
					accessMat( surfaceCopy, i, j-1 ) +
					accessMat( surfaceCopy, i, j+1 ) ) / 4;

		
			
			 if(rank!=0){
				 MPI_Wait(&p1,MPI_STATUS_IGNORE);
			 }

			lower_halo_changed = 0;
		 upper_halo_changed =0;
				for( j=1; j<columns-1; j++ ){
				accessMat( surface, 1, j ) = ( 
					accessMat( surfaceCopy, 0, j ) +
					accessMat( surfaceCopy, 2, j ) +
					accessMat( surfaceCopy, 1, j-1 ) +
					accessMat( surfaceCopy, 1, j+1 ) ) / 4;
				}

				for( j=1; j<columns-1; j++ ){
					if(	accessMat( surface, 1, j ) != 	accessMat( surfaceCopy, 1, j ) ){
						upper_halo_changed = 1;
						break;
					}
				}
				
				if(rank!=num_procs-1)
					MPI_Wait(&p2,MPI_STATUS_IGNORE);

				for( j=1; j<columns-1; j++ ){
	
				accessMat( surface, my_rows_halo-2, j ) = ( 
					accessMat( surfaceCopy, my_rows_halo-3, j ) +
					accessMat( surfaceCopy, my_rows_halo-1, j ) +
					accessMat( surfaceCopy, my_rows_halo-2, j-1 ) +
					accessMat( surfaceCopy, my_rows_halo-2, j+1 ) ) / 4;
			
				}

				for( j=1; j<columns-1; j++ ){
					if(accessMat( surface, my_rows_halo-2, j )!=accessMat( surfaceCopy, my_rows_halo-2, j )){
						lower_halo_changed=1;
						break;
					}
				}


		lower_halo_changed = 1;
		upper_halo_changed =1;

		for( step=0; step<8; step++ )	{
			/* 4.2.1. Update heat on active focal points */
			for( i=0; i<num_focal; i++ ) {
				if ( focal[i].active != 1 ) continue;
				int x = focal[i].x;
				int y = focal[i].y;
				if(x>=start && stop>=x)
					accessMat( surface, x-start+1, y ) = focal[i].heat;
			}
			
				
			float *surfaceAux;
			surfaceAux = surfaceCopy;
			surfaceCopy = surface;
			surface = surfaceAux;


		

					int receive_lower,receive_higher;
				
				
			  if(rank!=0){
			
				MPI_Irecv(&receive_higher,1,MPI_INT,rank-1,3,MPI_COMM_WORLD,&p5);
				MPI_Send(&upper_halo_changed,1,MPI_INT,rank-1,3,MPI_COMM_WORLD);
				
				if (upper_halo_changed){
					 float *pointer = &surfaceCopy[columns+1]; 
					MPI_Isend(pointer,columns-2,MPI_FLOAT,rank-1,0,MPI_COMM_WORLD,&p3);
				}
      }
		
      if (rank!=num_procs -1){
					MPI_Irecv(&receive_lower,1,MPI_INT,rank+1,3,MPI_COMM_WORLD,&p6);
				MPI_Send(&lower_halo_changed,1,MPI_INT,rank+1,3,MPI_COMM_WORLD);

				if(lower_halo_changed){
        float *pointer = &surfaceCopy[columns*(my_rows_nh)+1];  
        MPI_Isend(pointer,columns-2,MPI_FLOAT,rank+1,1,MPI_COMM_WORLD,&p4);
				}
			}

			if(rank!=0){
				MPI_Wait(&p5,MPI_STATUS_IGNORE);
		
				if(receive_higher)
			 		MPI_Irecv(&surfaceCopy[1],columns-2,MPI_FLOAT,rank-1,1,MPI_COMM_WORLD, &p1);
			}

			if(rank!=num_procs-1){
				MPI_Wait(&p6,MPI_STATUS_IGNORE);
					if(receive_lower){
					float *pointerRecv = &surfaceCopy[columns*(my_rows_nh+1)+1];
					MPI_Irecv(pointerRecv,columns-2,MPI_FLOAT,rank+1,0,MPI_COMM_WORLD, &p2);
					}
			}


				for( i=2; i<my_rows_halo-2; i++ )
				for( j=1; j<columns-1; j++ )
				accessMat( surface, i, j ) = ( 
					accessMat( surfaceCopy, i-1, j ) +
					accessMat( surfaceCopy, i+1, j ) +
					accessMat( surfaceCopy, i, j-1 ) +
					accessMat( surfaceCopy, i, j+1 ) ) / 4;

		
			
			 if(rank!=0){
				 MPI_Wait(&p1,MPI_STATUS_IGNORE);
			 }

			lower_halo_changed = 0;
		 upper_halo_changed =0;
				for( j=1; j<columns-1; j++ ){
				accessMat( surface, 1, j ) = ( 
					accessMat( surfaceCopy, 0, j ) +
					accessMat( surfaceCopy, 2, j ) +
					accessMat( surfaceCopy, 1, j-1 ) +
					accessMat( surfaceCopy, 1, j+1 ) ) / 4;
				}

				for( j=1; j<columns-1; j++ ){
					if(	accessMat( surface, 1, j ) != 	accessMat( surfaceCopy, 1, j ) ){
						upper_halo_changed = 1;
						break;
					}
				}
				
				if(rank!=num_procs-1)
					MPI_Wait(&p2,MPI_STATUS_IGNORE);

				for( j=1; j<columns-1; j++ ){
	
				accessMat( surface, my_rows_halo-2, j ) = ( 
					accessMat( surfaceCopy, my_rows_halo-3, j ) +
					accessMat( surfaceCopy, my_rows_halo-1, j ) +
					accessMat( surfaceCopy, my_rows_halo-2, j-1 ) +
					accessMat( surfaceCopy, my_rows_halo-2, j+1 ) ) / 4;
			
				}

				for( j=1; j<columns-1; j++ ){
					if(accessMat( surface, my_rows_halo-2, j )!=accessMat( surfaceCopy, my_rows_halo-2, j )){
						lower_halo_changed=1;
						break;
					}
				}



		} //Acaba bucle step

		
		if(  global_residual < THRESHOLD ) {flag_stability = 1;} 

	
	
		/* 4.4. Team actions */
		for( t=0; t<num_teams; t++ ) {
			
i=teams[t].x-start+1;
j=teams[t].y;
// Influence area of fixed radius depending on type
if ( teams[t].type == 1 ) {
//radius = RADIUS_TYPE_1;
//else radius = RADIUS_TYPE_2_3;

					if(teams[t].x+RADIUS_TYPE_1<start || teams[t].x-RADIUS_TYPE_1>stop)
					continue;
			
				if(i-3>=1 && j+0>=1 && i-3<my_rows_halo-1 && j+0<columns-1) 
 
  accessMat( surface, i-3, j+0 ) = accessMat( surface, i-3, j+0 ) * ( 0.75 );
				if(i-2>=1 && j-2>=1 && i-2<my_rows_halo-1 && j-2<columns-1) 
 
  accessMat( surface, i-2, j-2 ) = accessMat( surface, i-2, j-2 ) * ( 0.75 );
				if(i-2>=1 && j-1>=1 && i-2<my_rows_halo-1 && j-1<columns-1) 
 
  accessMat( surface, i-2, j-1 ) = accessMat( surface, i-2, j-1 ) * ( 0.75 );
				if(i-2>=1 && j+0>=1 && i-2<my_rows_halo-1 && j+0<columns-1) 
 
  accessMat( surface, i-2, j+0 ) = accessMat( surface, i-2, j+0 ) * ( 0.75 );
				if(i-2>=1 && j+1>=1 && i-2<my_rows_halo-1 && j+1<columns-1) 
 
  accessMat( surface, i-2, j+1 ) = accessMat( surface, i-2, j+1 ) * ( 0.75 );
				if(i-2>=1 && j+2>=1 && i-2<my_rows_halo-1 && j+2<columns-1) 
 
  accessMat( surface, i-2, j+2 ) = accessMat( surface, i-2, j+2 ) * ( 0.75 );
				if(i-1>=1 && j-2>=1 && i-1<my_rows_halo-1 && j-2<columns-1) 
 
  accessMat( surface, i-1, j-2 ) = accessMat( surface, i-1, j-2 ) * ( 0.75 );
				if(i-1>=1 && j-1>=1 && i-1<my_rows_halo-1 && j-1<columns-1) 
 
  accessMat( surface, i-1, j-1 ) = accessMat( surface, i-1, j-1 ) * ( 0.75 );
				if(i-1>=1 && j+0>=1 && i-1<my_rows_halo-1 && j+0<columns-1) 
 
  accessMat( surface, i-1, j+0 ) = accessMat( surface, i-1, j+0 ) * ( 0.75 );
				if(i-1>=1 && j+1>=1 && i-1<my_rows_halo-1 && j+1<columns-1) 
 
  accessMat( surface, i-1, j+1 ) = accessMat( surface, i-1, j+1 ) * ( 0.75 );
				if(i-1>=1 && j+2>=1 && i-1<my_rows_halo-1 && j+2<columns-1) 
 
  accessMat( surface, i-1, j+2 ) = accessMat( surface, i-1, j+2 ) * ( 0.75 );
				if(i+0>=1 && j-3>=1 && i+0<my_rows_halo-1 && j-3<columns-1) 
 
  accessMat( surface, i+0, j-3 ) = accessMat( surface, i+0, j-3 ) * ( 0.75 );
				if(i+0>=1 && j-2>=1 && i+0<my_rows_halo-1 && j-2<columns-1) 
 
  accessMat( surface, i+0, j-2 ) = accessMat( surface, i+0, j-2 ) * ( 0.75 );
				if(i+0>=1 && j-1>=1 && i+0<my_rows_halo-1 && j-1<columns-1) 
 
  accessMat( surface, i+0, j-1 ) = accessMat( surface, i+0, j-1 ) * ( 0.75 );
				if(i+0>=1 && j+0>=1 && i+0<my_rows_halo-1 && j+0<columns-1) 
 
  accessMat( surface, i+0, j+0 ) = accessMat( surface, i+0, j+0 ) * ( 0.75 );
				if(i+0>=1 && j+1>=1 && i+0<my_rows_halo-1 && j+1<columns-1) 
 
  accessMat( surface, i+0, j+1 ) = accessMat( surface, i+0, j+1 ) * ( 0.75 );
				if(i+0>=1 && j+2>=1 && i+0<my_rows_halo-1 && j+2<columns-1) 
 
  accessMat( surface, i+0, j+2 ) = accessMat( surface, i+0, j+2 ) * ( 0.75 );
				if(i+0>=1 && j+3>=1 && i+0<my_rows_halo-1 && j+3<columns-1) 
 
  accessMat( surface, i+0, j+3 ) = accessMat( surface, i+0, j+3 ) * ( 0.75 );
				if(i+1>=1 && j-2>=1 && i+1<my_rows_halo-1 && j-2<columns-1) 
 
  accessMat( surface, i+1, j-2 ) = accessMat( surface, i+1, j-2 ) * ( 0.75 );
				if(i+1>=1 && j-1>=1 && i+1<my_rows_halo-1 && j-1<columns-1) 
 
  accessMat( surface, i+1, j-1 ) = accessMat( surface, i+1, j-1 ) * ( 0.75 );
				if(i+1>=1 && j+0>=1 && i+1<my_rows_halo-1 && j+0<columns-1) 
 
  accessMat( surface, i+1, j+0 ) = accessMat( surface, i+1, j+0 ) * ( 0.75 );
				if(i+1>=1 && j+1>=1 && i+1<my_rows_halo-1 && j+1<columns-1) 
 
  accessMat( surface, i+1, j+1 ) = accessMat( surface, i+1, j+1 ) * ( 0.75 );
				if(i+1>=1 && j+2>=1 && i+1<my_rows_halo-1 && j+2<columns-1) 
 
  accessMat( surface, i+1, j+2 ) = accessMat( surface, i+1, j+2 ) * ( 0.75 );
				if(i+2>=1 && j-2>=1 && i+2<my_rows_halo-1 && j-2<columns-1) 
 
  accessMat( surface, i+2, j-2 ) = accessMat( surface, i+2, j-2 ) * ( 0.75 );
				if(i+2>=1 && j-1>=1 && i+2<my_rows_halo-1 && j-1<columns-1) 
 
  accessMat( surface, i+2, j-1 ) = accessMat( surface, i+2, j-1 ) * ( 0.75 );
				if(i+2>=1 && j+0>=1 && i+2<my_rows_halo-1 && j+0<columns-1) 
 
  accessMat( surface, i+2, j+0 ) = accessMat( surface, i+2, j+0 ) * ( 0.75 );
				if(i+2>=1 && j+1>=1 && i+2<my_rows_halo-1 && j+1<columns-1) 
 
  accessMat( surface, i+2, j+1 ) = accessMat( surface, i+2, j+1 ) * ( 0.75 );
				if(i+2>=1 && j+2>=1 && i+2<my_rows_halo-1 && j+2<columns-1) 
 
  accessMat( surface, i+2, j+2 ) = accessMat( surface, i+2, j+2 ) * ( 0.75 );
				if(i+3>=1 && j+0>=1 && i+3<my_rows_halo-1 && j+0<columns-1) 
 
  accessMat( surface, i+3, j+0 ) = accessMat( surface, i+3, j+0 ) * ( 0.75 );
			
			}
			else{
					if(teams[t].x+RADIUS_TYPE_2_3<start || teams[t].x-RADIUS_TYPE_2_3>stop)
					continue;
			
				if(i-9>=1 && j+0>=1 && i-9<my_rows_halo-1 && j+0<columns-1) 
 
  accessMat( surface, i-9, j+0 ) = accessMat( surface, i-9, j+0 ) * ( 0.75 );
				if(i-8>=1 && j-4>=1 && i-8<my_rows_halo-1 && j-4<columns-1) 
 
  accessMat( surface, i-8, j-4 ) = accessMat( surface, i-8, j-4 ) * ( 0.75 );
				if(i-8>=1 && j-3>=1 && i-8<my_rows_halo-1 && j-3<columns-1) 
 
  accessMat( surface, i-8, j-3 ) = accessMat( surface, i-8, j-3 ) * ( 0.75 );
				if(i-8>=1 && j-2>=1 && i-8<my_rows_halo-1 && j-2<columns-1) 
 
  accessMat( surface, i-8, j-2 ) = accessMat( surface, i-8, j-2 ) * ( 0.75 );
				if(i-8>=1 && j-1>=1 && i-8<my_rows_halo-1 && j-1<columns-1) 
 
  accessMat( surface, i-8, j-1 ) = accessMat( surface, i-8, j-1 ) * ( 0.75 );
				if(i-8>=1 && j+0>=1 && i-8<my_rows_halo-1 && j+0<columns-1) 
 
  accessMat( surface, i-8, j+0 ) = accessMat( surface, i-8, j+0 ) * ( 0.75 );
				if(i-8>=1 && j+1>=1 && i-8<my_rows_halo-1 && j+1<columns-1) 
 
  accessMat( surface, i-8, j+1 ) = accessMat( surface, i-8, j+1 ) * ( 0.75 );
				if(i-8>=1 && j+2>=1 && i-8<my_rows_halo-1 && j+2<columns-1) 
 
  accessMat( surface, i-8, j+2 ) = accessMat( surface, i-8, j+2 ) * ( 0.75 );
				if(i-8>=1 && j+3>=1 && i-8<my_rows_halo-1 && j+3<columns-1) 
 
  accessMat( surface, i-8, j+3 ) = accessMat( surface, i-8, j+3 ) * ( 0.75 );
				if(i-8>=1 && j+4>=1 && i-8<my_rows_halo-1 && j+4<columns-1) 
 
  accessMat( surface, i-8, j+4 ) = accessMat( surface, i-8, j+4 ) * ( 0.75 );
				if(i-7>=1 && j-5>=1 && i-7<my_rows_halo-1 && j-5<columns-1) 
 
  accessMat( surface, i-7, j-5 ) = accessMat( surface, i-7, j-5 ) * ( 0.75 );
				if(i-7>=1 && j-4>=1 && i-7<my_rows_halo-1 && j-4<columns-1) 
 
  accessMat( surface, i-7, j-4 ) = accessMat( surface, i-7, j-4 ) * ( 0.75 );
				if(i-7>=1 && j-3>=1 && i-7<my_rows_halo-1 && j-3<columns-1) 
 
  accessMat( surface, i-7, j-3 ) = accessMat( surface, i-7, j-3 ) * ( 0.75 );
				if(i-7>=1 && j-2>=1 && i-7<my_rows_halo-1 && j-2<columns-1) 
 
  accessMat( surface, i-7, j-2 ) = accessMat( surface, i-7, j-2 ) * ( 0.75 );
				if(i-7>=1 && j-1>=1 && i-7<my_rows_halo-1 && j-1<columns-1) 
 
  accessMat( surface, i-7, j-1 ) = accessMat( surface, i-7, j-1 ) * ( 0.75 );
				if(i-7>=1 && j+0>=1 && i-7<my_rows_halo-1 && j+0<columns-1) 
 
  accessMat( surface, i-7, j+0 ) = accessMat( surface, i-7, j+0 ) * ( 0.75 );
				if(i-7>=1 && j+1>=1 && i-7<my_rows_halo-1 && j+1<columns-1) 
 
  accessMat( surface, i-7, j+1 ) = accessMat( surface, i-7, j+1 ) * ( 0.75 );
				if(i-7>=1 && j+2>=1 && i-7<my_rows_halo-1 && j+2<columns-1) 
 
  accessMat( surface, i-7, j+2 ) = accessMat( surface, i-7, j+2 ) * ( 0.75 );
				if(i-7>=1 && j+3>=1 && i-7<my_rows_halo-1 && j+3<columns-1) 
 
  accessMat( surface, i-7, j+3 ) = accessMat( surface, i-7, j+3 ) * ( 0.75 );
				if(i-7>=1 && j+4>=1 && i-7<my_rows_halo-1 && j+4<columns-1) 
 
  accessMat( surface, i-7, j+4 ) = accessMat( surface, i-7, j+4 ) * ( 0.75 );
				if(i-7>=1 && j+5>=1 && i-7<my_rows_halo-1 && j+5<columns-1) 
 
  accessMat( surface, i-7, j+5 ) = accessMat( surface, i-7, j+5 ) * ( 0.75 );
				if(i-6>=1 && j-6>=1 && i-6<my_rows_halo-1 && j-6<columns-1) 
 
  accessMat( surface, i-6, j-6 ) = accessMat( surface, i-6, j-6 ) * ( 0.75 );
				if(i-6>=1 && j-5>=1 && i-6<my_rows_halo-1 && j-5<columns-1) 
 
  accessMat( surface, i-6, j-5 ) = accessMat( surface, i-6, j-5 ) * ( 0.75 );
				if(i-6>=1 && j-4>=1 && i-6<my_rows_halo-1 && j-4<columns-1) 
 
  accessMat( surface, i-6, j-4 ) = accessMat( surface, i-6, j-4 ) * ( 0.75 );
				if(i-6>=1 && j-3>=1 && i-6<my_rows_halo-1 && j-3<columns-1) 
 
  accessMat( surface, i-6, j-3 ) = accessMat( surface, i-6, j-3 ) * ( 0.75 );
				if(i-6>=1 && j-2>=1 && i-6<my_rows_halo-1 && j-2<columns-1) 
 
  accessMat( surface, i-6, j-2 ) = accessMat( surface, i-6, j-2 ) * ( 0.75 );
				if(i-6>=1 && j-1>=1 && i-6<my_rows_halo-1 && j-1<columns-1) 
 
  accessMat( surface, i-6, j-1 ) = accessMat( surface, i-6, j-1 ) * ( 0.75 );
				if(i-6>=1 && j+0>=1 && i-6<my_rows_halo-1 && j+0<columns-1) 
 
  accessMat( surface, i-6, j+0 ) = accessMat( surface, i-6, j+0 ) * ( 0.75 );
				if(i-6>=1 && j+1>=1 && i-6<my_rows_halo-1 && j+1<columns-1) 
 
  accessMat( surface, i-6, j+1 ) = accessMat( surface, i-6, j+1 ) * ( 0.75 );
				if(i-6>=1 && j+2>=1 && i-6<my_rows_halo-1 && j+2<columns-1) 
 
  accessMat( surface, i-6, j+2 ) = accessMat( surface, i-6, j+2 ) * ( 0.75 );
				if(i-6>=1 && j+3>=1 && i-6<my_rows_halo-1 && j+3<columns-1) 
 
  accessMat( surface, i-6, j+3 ) = accessMat( surface, i-6, j+3 ) * ( 0.75 );
				if(i-6>=1 && j+4>=1 && i-6<my_rows_halo-1 && j+4<columns-1) 
 
  accessMat( surface, i-6, j+4 ) = accessMat( surface, i-6, j+4 ) * ( 0.75 );
				if(i-6>=1 && j+5>=1 && i-6<my_rows_halo-1 && j+5<columns-1) 
 
  accessMat( surface, i-6, j+5 ) = accessMat( surface, i-6, j+5 ) * ( 0.75 );
				if(i-6>=1 && j+6>=1 && i-6<my_rows_halo-1 && j+6<columns-1) 
 
  accessMat( surface, i-6, j+6 ) = accessMat( surface, i-6, j+6 ) * ( 0.75 );
				if(i-5>=1 && j-7>=1 && i-5<my_rows_halo-1 && j-7<columns-1) 
 
  accessMat( surface, i-5, j-7 ) = accessMat( surface, i-5, j-7 ) * ( 0.75 );
				if(i-5>=1 && j-6>=1 && i-5<my_rows_halo-1 && j-6<columns-1) 
 
  accessMat( surface, i-5, j-6 ) = accessMat( surface, i-5, j-6 ) * ( 0.75 );
				if(i-5>=1 && j-5>=1 && i-5<my_rows_halo-1 && j-5<columns-1) 
 
  accessMat( surface, i-5, j-5 ) = accessMat( surface, i-5, j-5 ) * ( 0.75 );
				if(i-5>=1 && j-4>=1 && i-5<my_rows_halo-1 && j-4<columns-1) 
 
  accessMat( surface, i-5, j-4 ) = accessMat( surface, i-5, j-4 ) * ( 0.75 );
				if(i-5>=1 && j-3>=1 && i-5<my_rows_halo-1 && j-3<columns-1) 
 
  accessMat( surface, i-5, j-3 ) = accessMat( surface, i-5, j-3 ) * ( 0.75 );
				if(i-5>=1 && j-2>=1 && i-5<my_rows_halo-1 && j-2<columns-1) 
 
  accessMat( surface, i-5, j-2 ) = accessMat( surface, i-5, j-2 ) * ( 0.75 );
				if(i-5>=1 && j-1>=1 && i-5<my_rows_halo-1 && j-1<columns-1) 
 
  accessMat( surface, i-5, j-1 ) = accessMat( surface, i-5, j-1 ) * ( 0.75 );
				if(i-5>=1 && j+0>=1 && i-5<my_rows_halo-1 && j+0<columns-1) 
 
  accessMat( surface, i-5, j+0 ) = accessMat( surface, i-5, j+0 ) * ( 0.75 );
				if(i-5>=1 && j+1>=1 && i-5<my_rows_halo-1 && j+1<columns-1) 
 
  accessMat( surface, i-5, j+1 ) = accessMat( surface, i-5, j+1 ) * ( 0.75 );
				if(i-5>=1 && j+2>=1 && i-5<my_rows_halo-1 && j+2<columns-1) 
 
  accessMat( surface, i-5, j+2 ) = accessMat( surface, i-5, j+2 ) * ( 0.75 );
				if(i-5>=1 && j+3>=1 && i-5<my_rows_halo-1 && j+3<columns-1) 
 
  accessMat( surface, i-5, j+3 ) = accessMat( surface, i-5, j+3 ) * ( 0.75 );
				if(i-5>=1 && j+4>=1 && i-5<my_rows_halo-1 && j+4<columns-1) 
 
  accessMat( surface, i-5, j+4 ) = accessMat( surface, i-5, j+4 ) * ( 0.75 );
				if(i-5>=1 && j+5>=1 && i-5<my_rows_halo-1 && j+5<columns-1) 
 
  accessMat( surface, i-5, j+5 ) = accessMat( surface, i-5, j+5 ) * ( 0.75 );
				if(i-5>=1 && j+6>=1 && i-5<my_rows_halo-1 && j+6<columns-1) 
 
  accessMat( surface, i-5, j+6 ) = accessMat( surface, i-5, j+6 ) * ( 0.75 );
				if(i-5>=1 && j+7>=1 && i-5<my_rows_halo-1 && j+7<columns-1) 
 
  accessMat( surface, i-5, j+7 ) = accessMat( surface, i-5, j+7 ) * ( 0.75 );
				if(i-4>=1 && j-8>=1 && i-4<my_rows_halo-1 && j-8<columns-1) 
 
  accessMat( surface, i-4, j-8 ) = accessMat( surface, i-4, j-8 ) * ( 0.75 );
				if(i-4>=1 && j-7>=1 && i-4<my_rows_halo-1 && j-7<columns-1) 
 
  accessMat( surface, i-4, j-7 ) = accessMat( surface, i-4, j-7 ) * ( 0.75 );
				if(i-4>=1 && j-6>=1 && i-4<my_rows_halo-1 && j-6<columns-1) 
 
  accessMat( surface, i-4, j-6 ) = accessMat( surface, i-4, j-6 ) * ( 0.75 );
				if(i-4>=1 && j-5>=1 && i-4<my_rows_halo-1 && j-5<columns-1) 
 
  accessMat( surface, i-4, j-5 ) = accessMat( surface, i-4, j-5 ) * ( 0.75 );
				if(i-4>=1 && j-4>=1 && i-4<my_rows_halo-1 && j-4<columns-1) 
 
  accessMat( surface, i-4, j-4 ) = accessMat( surface, i-4, j-4 ) * ( 0.75 );
				if(i-4>=1 && j-3>=1 && i-4<my_rows_halo-1 && j-3<columns-1) 
 
  accessMat( surface, i-4, j-3 ) = accessMat( surface, i-4, j-3 ) * ( 0.75 );
				if(i-4>=1 && j-2>=1 && i-4<my_rows_halo-1 && j-2<columns-1) 
 
  accessMat( surface, i-4, j-2 ) = accessMat( surface, i-4, j-2 ) * ( 0.75 );
				if(i-4>=1 && j-1>=1 && i-4<my_rows_halo-1 && j-1<columns-1) 
 
  accessMat( surface, i-4, j-1 ) = accessMat( surface, i-4, j-1 ) * ( 0.75 );
				if(i-4>=1 && j+0>=1 && i-4<my_rows_halo-1 && j+0<columns-1) 
 
  accessMat( surface, i-4, j+0 ) = accessMat( surface, i-4, j+0 ) * ( 0.75 );
				if(i-4>=1 && j+1>=1 && i-4<my_rows_halo-1 && j+1<columns-1) 
 
  accessMat( surface, i-4, j+1 ) = accessMat( surface, i-4, j+1 ) * ( 0.75 );
				if(i-4>=1 && j+2>=1 && i-4<my_rows_halo-1 && j+2<columns-1) 
 
  accessMat( surface, i-4, j+2 ) = accessMat( surface, i-4, j+2 ) * ( 0.75 );
				if(i-4>=1 && j+3>=1 && i-4<my_rows_halo-1 && j+3<columns-1) 
 
  accessMat( surface, i-4, j+3 ) = accessMat( surface, i-4, j+3 ) * ( 0.75 );
				if(i-4>=1 && j+4>=1 && i-4<my_rows_halo-1 && j+4<columns-1) 
 
  accessMat( surface, i-4, j+4 ) = accessMat( surface, i-4, j+4 ) * ( 0.75 );
				if(i-4>=1 && j+5>=1 && i-4<my_rows_halo-1 && j+5<columns-1) 
 
  accessMat( surface, i-4, j+5 ) = accessMat( surface, i-4, j+5 ) * ( 0.75 );
				if(i-4>=1 && j+6>=1 && i-4<my_rows_halo-1 && j+6<columns-1) 
 
  accessMat( surface, i-4, j+6 ) = accessMat( surface, i-4, j+6 ) * ( 0.75 );
				if(i-4>=1 && j+7>=1 && i-4<my_rows_halo-1 && j+7<columns-1) 
 
  accessMat( surface, i-4, j+7 ) = accessMat( surface, i-4, j+7 ) * ( 0.75 );
				if(i-4>=1 && j+8>=1 && i-4<my_rows_halo-1 && j+8<columns-1) 
 
  accessMat( surface, i-4, j+8 ) = accessMat( surface, i-4, j+8 ) * ( 0.75 );
				if(i-3>=1 && j-8>=1 && i-3<my_rows_halo-1 && j-8<columns-1) 
 
  accessMat( surface, i-3, j-8 ) = accessMat( surface, i-3, j-8 ) * ( 0.75 );
				if(i-3>=1 && j-7>=1 && i-3<my_rows_halo-1 && j-7<columns-1) 
 
  accessMat( surface, i-3, j-7 ) = accessMat( surface, i-3, j-7 ) * ( 0.75 );
				if(i-3>=1 && j-6>=1 && i-3<my_rows_halo-1 && j-6<columns-1) 
 
  accessMat( surface, i-3, j-6 ) = accessMat( surface, i-3, j-6 ) * ( 0.75 );
				if(i-3>=1 && j-5>=1 && i-3<my_rows_halo-1 && j-5<columns-1) 
 
  accessMat( surface, i-3, j-5 ) = accessMat( surface, i-3, j-5 ) * ( 0.75 );
				if(i-3>=1 && j-4>=1 && i-3<my_rows_halo-1 && j-4<columns-1) 
 
  accessMat( surface, i-3, j-4 ) = accessMat( surface, i-3, j-4 ) * ( 0.75 );
				if(i-3>=1 && j-3>=1 && i-3<my_rows_halo-1 && j-3<columns-1) 
 
  accessMat( surface, i-3, j-3 ) = accessMat( surface, i-3, j-3 ) * ( 0.75 );
				if(i-3>=1 && j-2>=1 && i-3<my_rows_halo-1 && j-2<columns-1) 
 
  accessMat( surface, i-3, j-2 ) = accessMat( surface, i-3, j-2 ) * ( 0.75 );
				if(i-3>=1 && j-1>=1 && i-3<my_rows_halo-1 && j-1<columns-1) 
 
  accessMat( surface, i-3, j-1 ) = accessMat( surface, i-3, j-1 ) * ( 0.75 );
				if(i-3>=1 && j+0>=1 && i-3<my_rows_halo-1 && j+0<columns-1) 
 
  accessMat( surface, i-3, j+0 ) = accessMat( surface, i-3, j+0 ) * ( 0.75 );
				if(i-3>=1 && j+1>=1 && i-3<my_rows_halo-1 && j+1<columns-1) 
 
  accessMat( surface, i-3, j+1 ) = accessMat( surface, i-3, j+1 ) * ( 0.75 );
				if(i-3>=1 && j+2>=1 && i-3<my_rows_halo-1 && j+2<columns-1) 
 
  accessMat( surface, i-3, j+2 ) = accessMat( surface, i-3, j+2 ) * ( 0.75 );
				if(i-3>=1 && j+3>=1 && i-3<my_rows_halo-1 && j+3<columns-1) 
 
  accessMat( surface, i-3, j+3 ) = accessMat( surface, i-3, j+3 ) * ( 0.75 );
				if(i-3>=1 && j+4>=1 && i-3<my_rows_halo-1 && j+4<columns-1) 
 
  accessMat( surface, i-3, j+4 ) = accessMat( surface, i-3, j+4 ) * ( 0.75 );
				if(i-3>=1 && j+5>=1 && i-3<my_rows_halo-1 && j+5<columns-1) 
 
  accessMat( surface, i-3, j+5 ) = accessMat( surface, i-3, j+5 ) * ( 0.75 );
				if(i-3>=1 && j+6>=1 && i-3<my_rows_halo-1 && j+6<columns-1) 
 
  accessMat( surface, i-3, j+6 ) = accessMat( surface, i-3, j+6 ) * ( 0.75 );
				if(i-3>=1 && j+7>=1 && i-3<my_rows_halo-1 && j+7<columns-1) 
 
  accessMat( surface, i-3, j+7 ) = accessMat( surface, i-3, j+7 ) * ( 0.75 );
				if(i-3>=1 && j+8>=1 && i-3<my_rows_halo-1 && j+8<columns-1) 
 
  accessMat( surface, i-3, j+8 ) = accessMat( surface, i-3, j+8 ) * ( 0.75 );
				if(i-2>=1 && j-8>=1 && i-2<my_rows_halo-1 && j-8<columns-1) 
 
  accessMat( surface, i-2, j-8 ) = accessMat( surface, i-2, j-8 ) * ( 0.75 );
				if(i-2>=1 && j-7>=1 && i-2<my_rows_halo-1 && j-7<columns-1) 
 
  accessMat( surface, i-2, j-7 ) = accessMat( surface, i-2, j-7 ) * ( 0.75 );
				if(i-2>=1 && j-6>=1 && i-2<my_rows_halo-1 && j-6<columns-1) 
 
  accessMat( surface, i-2, j-6 ) = accessMat( surface, i-2, j-6 ) * ( 0.75 );
				if(i-2>=1 && j-5>=1 && i-2<my_rows_halo-1 && j-5<columns-1) 
 
  accessMat( surface, i-2, j-5 ) = accessMat( surface, i-2, j-5 ) * ( 0.75 );
				if(i-2>=1 && j-4>=1 && i-2<my_rows_halo-1 && j-4<columns-1) 
 
  accessMat( surface, i-2, j-4 ) = accessMat( surface, i-2, j-4 ) * ( 0.75 );
				if(i-2>=1 && j-3>=1 && i-2<my_rows_halo-1 && j-3<columns-1) 
 
  accessMat( surface, i-2, j-3 ) = accessMat( surface, i-2, j-3 ) * ( 0.75 );
				if(i-2>=1 && j-2>=1 && i-2<my_rows_halo-1 && j-2<columns-1) 
 
  accessMat( surface, i-2, j-2 ) = accessMat( surface, i-2, j-2 ) * ( 0.75 );
				if(i-2>=1 && j-1>=1 && i-2<my_rows_halo-1 && j-1<columns-1) 
 
  accessMat( surface, i-2, j-1 ) = accessMat( surface, i-2, j-1 ) * ( 0.75 );
				if(i-2>=1 && j+0>=1 && i-2<my_rows_halo-1 && j+0<columns-1) 
 
  accessMat( surface, i-2, j+0 ) = accessMat( surface, i-2, j+0 ) * ( 0.75 );
				if(i-2>=1 && j+1>=1 && i-2<my_rows_halo-1 && j+1<columns-1) 
 
  accessMat( surface, i-2, j+1 ) = accessMat( surface, i-2, j+1 ) * ( 0.75 );
				if(i-2>=1 && j+2>=1 && i-2<my_rows_halo-1 && j+2<columns-1) 
 
  accessMat( surface, i-2, j+2 ) = accessMat( surface, i-2, j+2 ) * ( 0.75 );
				if(i-2>=1 && j+3>=1 && i-2<my_rows_halo-1 && j+3<columns-1) 
 
  accessMat( surface, i-2, j+3 ) = accessMat( surface, i-2, j+3 ) * ( 0.75 );
				if(i-2>=1 && j+4>=1 && i-2<my_rows_halo-1 && j+4<columns-1) 
 
  accessMat( surface, i-2, j+4 ) = accessMat( surface, i-2, j+4 ) * ( 0.75 );
				if(i-2>=1 && j+5>=1 && i-2<my_rows_halo-1 && j+5<columns-1) 
 
  accessMat( surface, i-2, j+5 ) = accessMat( surface, i-2, j+5 ) * ( 0.75 );
				if(i-2>=1 && j+6>=1 && i-2<my_rows_halo-1 && j+6<columns-1) 
 
  accessMat( surface, i-2, j+6 ) = accessMat( surface, i-2, j+6 ) * ( 0.75 );
				if(i-2>=1 && j+7>=1 && i-2<my_rows_halo-1 && j+7<columns-1) 
 
  accessMat( surface, i-2, j+7 ) = accessMat( surface, i-2, j+7 ) * ( 0.75 );
				if(i-2>=1 && j+8>=1 && i-2<my_rows_halo-1 && j+8<columns-1) 
 
  accessMat( surface, i-2, j+8 ) = accessMat( surface, i-2, j+8 ) * ( 0.75 );
				if(i-1>=1 && j-8>=1 && i-1<my_rows_halo-1 && j-8<columns-1) 
 
  accessMat( surface, i-1, j-8 ) = accessMat( surface, i-1, j-8 ) * ( 0.75 );
				if(i-1>=1 && j-7>=1 && i-1<my_rows_halo-1 && j-7<columns-1) 
 
  accessMat( surface, i-1, j-7 ) = accessMat( surface, i-1, j-7 ) * ( 0.75 );
				if(i-1>=1 && j-6>=1 && i-1<my_rows_halo-1 && j-6<columns-1) 
 
  accessMat( surface, i-1, j-6 ) = accessMat( surface, i-1, j-6 ) * ( 0.75 );
				if(i-1>=1 && j-5>=1 && i-1<my_rows_halo-1 && j-5<columns-1) 
 
  accessMat( surface, i-1, j-5 ) = accessMat( surface, i-1, j-5 ) * ( 0.75 );
				if(i-1>=1 && j-4>=1 && i-1<my_rows_halo-1 && j-4<columns-1) 
 
  accessMat( surface, i-1, j-4 ) = accessMat( surface, i-1, j-4 ) * ( 0.75 );
				if(i-1>=1 && j-3>=1 && i-1<my_rows_halo-1 && j-3<columns-1) 
 
  accessMat( surface, i-1, j-3 ) = accessMat( surface, i-1, j-3 ) * ( 0.75 );
				if(i-1>=1 && j-2>=1 && i-1<my_rows_halo-1 && j-2<columns-1) 
 
  accessMat( surface, i-1, j-2 ) = accessMat( surface, i-1, j-2 ) * ( 0.75 );
				if(i-1>=1 && j-1>=1 && i-1<my_rows_halo-1 && j-1<columns-1) 
 
  accessMat( surface, i-1, j-1 ) = accessMat( surface, i-1, j-1 ) * ( 0.75 );
				if(i-1>=1 && j+0>=1 && i-1<my_rows_halo-1 && j+0<columns-1) 
 
  accessMat( surface, i-1, j+0 ) = accessMat( surface, i-1, j+0 ) * ( 0.75 );
				if(i-1>=1 && j+1>=1 && i-1<my_rows_halo-1 && j+1<columns-1) 
 
  accessMat( surface, i-1, j+1 ) = accessMat( surface, i-1, j+1 ) * ( 0.75 );
				if(i-1>=1 && j+2>=1 && i-1<my_rows_halo-1 && j+2<columns-1) 
 
  accessMat( surface, i-1, j+2 ) = accessMat( surface, i-1, j+2 ) * ( 0.75 );
				if(i-1>=1 && j+3>=1 && i-1<my_rows_halo-1 && j+3<columns-1) 
 
  accessMat( surface, i-1, j+3 ) = accessMat( surface, i-1, j+3 ) * ( 0.75 );
				if(i-1>=1 && j+4>=1 && i-1<my_rows_halo-1 && j+4<columns-1) 
 
  accessMat( surface, i-1, j+4 ) = accessMat( surface, i-1, j+4 ) * ( 0.75 );
				if(i-1>=1 && j+5>=1 && i-1<my_rows_halo-1 && j+5<columns-1) 
 
  accessMat( surface, i-1, j+5 ) = accessMat( surface, i-1, j+5 ) * ( 0.75 );
				if(i-1>=1 && j+6>=1 && i-1<my_rows_halo-1 && j+6<columns-1) 
 
  accessMat( surface, i-1, j+6 ) = accessMat( surface, i-1, j+6 ) * ( 0.75 );
				if(i-1>=1 && j+7>=1 && i-1<my_rows_halo-1 && j+7<columns-1) 
 
  accessMat( surface, i-1, j+7 ) = accessMat( surface, i-1, j+7 ) * ( 0.75 );
				if(i-1>=1 && j+8>=1 && i-1<my_rows_halo-1 && j+8<columns-1) 
 
  accessMat( surface, i-1, j+8 ) = accessMat( surface, i-1, j+8 ) * ( 0.75 );
				if(i+0>=1 && j-9>=1 && i+0<my_rows_halo-1 && j-9<columns-1) 
 
  accessMat( surface, i+0, j-9 ) = accessMat( surface, i+0, j-9 ) * ( 0.75 );
				if(i+0>=1 && j-8>=1 && i+0<my_rows_halo-1 && j-8<columns-1) 
 
  accessMat( surface, i+0, j-8 ) = accessMat( surface, i+0, j-8 ) * ( 0.75 );
				if(i+0>=1 && j-7>=1 && i+0<my_rows_halo-1 && j-7<columns-1) 
 
  accessMat( surface, i+0, j-7 ) = accessMat( surface, i+0, j-7 ) * ( 0.75 );
				if(i+0>=1 && j-6>=1 && i+0<my_rows_halo-1 && j-6<columns-1) 
 
  accessMat( surface, i+0, j-6 ) = accessMat( surface, i+0, j-6 ) * ( 0.75 );
				if(i+0>=1 && j-5>=1 && i+0<my_rows_halo-1 && j-5<columns-1) 
 
  accessMat( surface, i+0, j-5 ) = accessMat( surface, i+0, j-5 ) * ( 0.75 );
				if(i+0>=1 && j-4>=1 && i+0<my_rows_halo-1 && j-4<columns-1) 
 
  accessMat( surface, i+0, j-4 ) = accessMat( surface, i+0, j-4 ) * ( 0.75 );
				if(i+0>=1 && j-3>=1 && i+0<my_rows_halo-1 && j-3<columns-1) 
 
  accessMat( surface, i+0, j-3 ) = accessMat( surface, i+0, j-3 ) * ( 0.75 );
				if(i+0>=1 && j-2>=1 && i+0<my_rows_halo-1 && j-2<columns-1) 
 
  accessMat( surface, i+0, j-2 ) = accessMat( surface, i+0, j-2 ) * ( 0.75 );
				if(i+0>=1 && j-1>=1 && i+0<my_rows_halo-1 && j-1<columns-1) 
 
  accessMat( surface, i+0, j-1 ) = accessMat( surface, i+0, j-1 ) * ( 0.75 );
				if(i+0>=1 && j+0>=1 && i+0<my_rows_halo-1 && j+0<columns-1) 
 
  accessMat( surface, i+0, j+0 ) = accessMat( surface, i+0, j+0 ) * ( 0.75 );
				if(i+0>=1 && j+1>=1 && i+0<my_rows_halo-1 && j+1<columns-1) 
 
  accessMat( surface, i+0, j+1 ) = accessMat( surface, i+0, j+1 ) * ( 0.75 );
				if(i+0>=1 && j+2>=1 && i+0<my_rows_halo-1 && j+2<columns-1) 
 
  accessMat( surface, i+0, j+2 ) = accessMat( surface, i+0, j+2 ) * ( 0.75 );
				if(i+0>=1 && j+3>=1 && i+0<my_rows_halo-1 && j+3<columns-1) 
 
  accessMat( surface, i+0, j+3 ) = accessMat( surface, i+0, j+3 ) * ( 0.75 );
				if(i+0>=1 && j+4>=1 && i+0<my_rows_halo-1 && j+4<columns-1) 
 
  accessMat( surface, i+0, j+4 ) = accessMat( surface, i+0, j+4 ) * ( 0.75 );
				if(i+0>=1 && j+5>=1 && i+0<my_rows_halo-1 && j+5<columns-1) 
 
  accessMat( surface, i+0, j+5 ) = accessMat( surface, i+0, j+5 ) * ( 0.75 );
				if(i+0>=1 && j+6>=1 && i+0<my_rows_halo-1 && j+6<columns-1) 
 
  accessMat( surface, i+0, j+6 ) = accessMat( surface, i+0, j+6 ) * ( 0.75 );
				if(i+0>=1 && j+7>=1 && i+0<my_rows_halo-1 && j+7<columns-1) 
 
  accessMat( surface, i+0, j+7 ) = accessMat( surface, i+0, j+7 ) * ( 0.75 );
				if(i+0>=1 && j+8>=1 && i+0<my_rows_halo-1 && j+8<columns-1) 
 
  accessMat( surface, i+0, j+8 ) = accessMat( surface, i+0, j+8 ) * ( 0.75 );
				if(i+0>=1 && j+9>=1 && i+0<my_rows_halo-1 && j+9<columns-1) 
 
  accessMat( surface, i+0, j+9 ) = accessMat( surface, i+0, j+9 ) * ( 0.75 );
				if(i+1>=1 && j-8>=1 && i+1<my_rows_halo-1 && j-8<columns-1) 
 
  accessMat( surface, i+1, j-8 ) = accessMat( surface, i+1, j-8 ) * ( 0.75 );
				if(i+1>=1 && j-7>=1 && i+1<my_rows_halo-1 && j-7<columns-1) 
 
  accessMat( surface, i+1, j-7 ) = accessMat( surface, i+1, j-7 ) * ( 0.75 );
				if(i+1>=1 && j-6>=1 && i+1<my_rows_halo-1 && j-6<columns-1) 
 
  accessMat( surface, i+1, j-6 ) = accessMat( surface, i+1, j-6 ) * ( 0.75 );
				if(i+1>=1 && j-5>=1 && i+1<my_rows_halo-1 && j-5<columns-1) 
 
  accessMat( surface, i+1, j-5 ) = accessMat( surface, i+1, j-5 ) * ( 0.75 );
				if(i+1>=1 && j-4>=1 && i+1<my_rows_halo-1 && j-4<columns-1) 
 
  accessMat( surface, i+1, j-4 ) = accessMat( surface, i+1, j-4 ) * ( 0.75 );
				if(i+1>=1 && j-3>=1 && i+1<my_rows_halo-1 && j-3<columns-1) 
 
  accessMat( surface, i+1, j-3 ) = accessMat( surface, i+1, j-3 ) * ( 0.75 );
				if(i+1>=1 && j-2>=1 && i+1<my_rows_halo-1 && j-2<columns-1) 
 
  accessMat( surface, i+1, j-2 ) = accessMat( surface, i+1, j-2 ) * ( 0.75 );
				if(i+1>=1 && j-1>=1 && i+1<my_rows_halo-1 && j-1<columns-1) 
 
  accessMat( surface, i+1, j-1 ) = accessMat( surface, i+1, j-1 ) * ( 0.75 );
				if(i+1>=1 && j+0>=1 && i+1<my_rows_halo-1 && j+0<columns-1) 
 
  accessMat( surface, i+1, j+0 ) = accessMat( surface, i+1, j+0 ) * ( 0.75 );
				if(i+1>=1 && j+1>=1 && i+1<my_rows_halo-1 && j+1<columns-1) 
 
  accessMat( surface, i+1, j+1 ) = accessMat( surface, i+1, j+1 ) * ( 0.75 );
				if(i+1>=1 && j+2>=1 && i+1<my_rows_halo-1 && j+2<columns-1) 
 
  accessMat( surface, i+1, j+2 ) = accessMat( surface, i+1, j+2 ) * ( 0.75 );
				if(i+1>=1 && j+3>=1 && i+1<my_rows_halo-1 && j+3<columns-1) 
 
  accessMat( surface, i+1, j+3 ) = accessMat( surface, i+1, j+3 ) * ( 0.75 );
				if(i+1>=1 && j+4>=1 && i+1<my_rows_halo-1 && j+4<columns-1) 
 
  accessMat( surface, i+1, j+4 ) = accessMat( surface, i+1, j+4 ) * ( 0.75 );
				if(i+1>=1 && j+5>=1 && i+1<my_rows_halo-1 && j+5<columns-1) 
 
  accessMat( surface, i+1, j+5 ) = accessMat( surface, i+1, j+5 ) * ( 0.75 );
				if(i+1>=1 && j+6>=1 && i+1<my_rows_halo-1 && j+6<columns-1) 
 
  accessMat( surface, i+1, j+6 ) = accessMat( surface, i+1, j+6 ) * ( 0.75 );
				if(i+1>=1 && j+7>=1 && i+1<my_rows_halo-1 && j+7<columns-1) 
 
  accessMat( surface, i+1, j+7 ) = accessMat( surface, i+1, j+7 ) * ( 0.75 );
				if(i+1>=1 && j+8>=1 && i+1<my_rows_halo-1 && j+8<columns-1) 
 
  accessMat( surface, i+1, j+8 ) = accessMat( surface, i+1, j+8 ) * ( 0.75 );
				if(i+2>=1 && j-8>=1 && i+2<my_rows_halo-1 && j-8<columns-1) 
 
  accessMat( surface, i+2, j-8 ) = accessMat( surface, i+2, j-8 ) * ( 0.75 );
				if(i+2>=1 && j-7>=1 && i+2<my_rows_halo-1 && j-7<columns-1) 
 
  accessMat( surface, i+2, j-7 ) = accessMat( surface, i+2, j-7 ) * ( 0.75 );
				if(i+2>=1 && j-6>=1 && i+2<my_rows_halo-1 && j-6<columns-1) 
 
  accessMat( surface, i+2, j-6 ) = accessMat( surface, i+2, j-6 ) * ( 0.75 );
				if(i+2>=1 && j-5>=1 && i+2<my_rows_halo-1 && j-5<columns-1) 
 
  accessMat( surface, i+2, j-5 ) = accessMat( surface, i+2, j-5 ) * ( 0.75 );
				if(i+2>=1 && j-4>=1 && i+2<my_rows_halo-1 && j-4<columns-1) 
 
  accessMat( surface, i+2, j-4 ) = accessMat( surface, i+2, j-4 ) * ( 0.75 );
				if(i+2>=1 && j-3>=1 && i+2<my_rows_halo-1 && j-3<columns-1) 
 
  accessMat( surface, i+2, j-3 ) = accessMat( surface, i+2, j-3 ) * ( 0.75 );
				if(i+2>=1 && j-2>=1 && i+2<my_rows_halo-1 && j-2<columns-1) 
 
  accessMat( surface, i+2, j-2 ) = accessMat( surface, i+2, j-2 ) * ( 0.75 );
				if(i+2>=1 && j-1>=1 && i+2<my_rows_halo-1 && j-1<columns-1) 
 
  accessMat( surface, i+2, j-1 ) = accessMat( surface, i+2, j-1 ) * ( 0.75 );
				if(i+2>=1 && j+0>=1 && i+2<my_rows_halo-1 && j+0<columns-1) 
 
  accessMat( surface, i+2, j+0 ) = accessMat( surface, i+2, j+0 ) * ( 0.75 );
				if(i+2>=1 && j+1>=1 && i+2<my_rows_halo-1 && j+1<columns-1) 
 
  accessMat( surface, i+2, j+1 ) = accessMat( surface, i+2, j+1 ) * ( 0.75 );
				if(i+2>=1 && j+2>=1 && i+2<my_rows_halo-1 && j+2<columns-1) 
 
  accessMat( surface, i+2, j+2 ) = accessMat( surface, i+2, j+2 ) * ( 0.75 );
				if(i+2>=1 && j+3>=1 && i+2<my_rows_halo-1 && j+3<columns-1) 
 
  accessMat( surface, i+2, j+3 ) = accessMat( surface, i+2, j+3 ) * ( 0.75 );
				if(i+2>=1 && j+4>=1 && i+2<my_rows_halo-1 && j+4<columns-1) 
 
  accessMat( surface, i+2, j+4 ) = accessMat( surface, i+2, j+4 ) * ( 0.75 );
				if(i+2>=1 && j+5>=1 && i+2<my_rows_halo-1 && j+5<columns-1) 
 
  accessMat( surface, i+2, j+5 ) = accessMat( surface, i+2, j+5 ) * ( 0.75 );
				if(i+2>=1 && j+6>=1 && i+2<my_rows_halo-1 && j+6<columns-1) 
 
  accessMat( surface, i+2, j+6 ) = accessMat( surface, i+2, j+6 ) * ( 0.75 );
				if(i+2>=1 && j+7>=1 && i+2<my_rows_halo-1 && j+7<columns-1) 
 
  accessMat( surface, i+2, j+7 ) = accessMat( surface, i+2, j+7 ) * ( 0.75 );
				if(i+2>=1 && j+8>=1 && i+2<my_rows_halo-1 && j+8<columns-1) 
 
  accessMat( surface, i+2, j+8 ) = accessMat( surface, i+2, j+8 ) * ( 0.75 );
				if(i+3>=1 && j-8>=1 && i+3<my_rows_halo-1 && j-8<columns-1) 
 
  accessMat( surface, i+3, j-8 ) = accessMat( surface, i+3, j-8 ) * ( 0.75 );
				if(i+3>=1 && j-7>=1 && i+3<my_rows_halo-1 && j-7<columns-1) 
 
  accessMat( surface, i+3, j-7 ) = accessMat( surface, i+3, j-7 ) * ( 0.75 );
				if(i+3>=1 && j-6>=1 && i+3<my_rows_halo-1 && j-6<columns-1) 
 
  accessMat( surface, i+3, j-6 ) = accessMat( surface, i+3, j-6 ) * ( 0.75 );
				if(i+3>=1 && j-5>=1 && i+3<my_rows_halo-1 && j-5<columns-1) 
 
  accessMat( surface, i+3, j-5 ) = accessMat( surface, i+3, j-5 ) * ( 0.75 );
				if(i+3>=1 && j-4>=1 && i+3<my_rows_halo-1 && j-4<columns-1) 
 
  accessMat( surface, i+3, j-4 ) = accessMat( surface, i+3, j-4 ) * ( 0.75 );
				if(i+3>=1 && j-3>=1 && i+3<my_rows_halo-1 && j-3<columns-1) 
 
  accessMat( surface, i+3, j-3 ) = accessMat( surface, i+3, j-3 ) * ( 0.75 );
				if(i+3>=1 && j-2>=1 && i+3<my_rows_halo-1 && j-2<columns-1) 
 
  accessMat( surface, i+3, j-2 ) = accessMat( surface, i+3, j-2 ) * ( 0.75 );
				if(i+3>=1 && j-1>=1 && i+3<my_rows_halo-1 && j-1<columns-1) 
 
  accessMat( surface, i+3, j-1 ) = accessMat( surface, i+3, j-1 ) * ( 0.75 );
				if(i+3>=1 && j+0>=1 && i+3<my_rows_halo-1 && j+0<columns-1) 
 
  accessMat( surface, i+3, j+0 ) = accessMat( surface, i+3, j+0 ) * ( 0.75 );
				if(i+3>=1 && j+1>=1 && i+3<my_rows_halo-1 && j+1<columns-1) 
 
  accessMat( surface, i+3, j+1 ) = accessMat( surface, i+3, j+1 ) * ( 0.75 );
				if(i+3>=1 && j+2>=1 && i+3<my_rows_halo-1 && j+2<columns-1) 
 
  accessMat( surface, i+3, j+2 ) = accessMat( surface, i+3, j+2 ) * ( 0.75 );
				if(i+3>=1 && j+3>=1 && i+3<my_rows_halo-1 && j+3<columns-1) 
 
  accessMat( surface, i+3, j+3 ) = accessMat( surface, i+3, j+3 ) * ( 0.75 );
				if(i+3>=1 && j+4>=1 && i+3<my_rows_halo-1 && j+4<columns-1) 
 
  accessMat( surface, i+3, j+4 ) = accessMat( surface, i+3, j+4 ) * ( 0.75 );
				if(i+3>=1 && j+5>=1 && i+3<my_rows_halo-1 && j+5<columns-1) 
 
  accessMat( surface, i+3, j+5 ) = accessMat( surface, i+3, j+5 ) * ( 0.75 );
				if(i+3>=1 && j+6>=1 && i+3<my_rows_halo-1 && j+6<columns-1) 
 
  accessMat( surface, i+3, j+6 ) = accessMat( surface, i+3, j+6 ) * ( 0.75 );
				if(i+3>=1 && j+7>=1 && i+3<my_rows_halo-1 && j+7<columns-1) 
 
  accessMat( surface, i+3, j+7 ) = accessMat( surface, i+3, j+7 ) * ( 0.75 );
				if(i+3>=1 && j+8>=1 && i+3<my_rows_halo-1 && j+8<columns-1) 
 
  accessMat( surface, i+3, j+8 ) = accessMat( surface, i+3, j+8 ) * ( 0.75 );
				if(i+4>=1 && j-8>=1 && i+4<my_rows_halo-1 && j-8<columns-1) 
 
  accessMat( surface, i+4, j-8 ) = accessMat( surface, i+4, j-8 ) * ( 0.75 );
				if(i+4>=1 && j-7>=1 && i+4<my_rows_halo-1 && j-7<columns-1) 
 
  accessMat( surface, i+4, j-7 ) = accessMat( surface, i+4, j-7 ) * ( 0.75 );
				if(i+4>=1 && j-6>=1 && i+4<my_rows_halo-1 && j-6<columns-1) 
 
  accessMat( surface, i+4, j-6 ) = accessMat( surface, i+4, j-6 ) * ( 0.75 );
				if(i+4>=1 && j-5>=1 && i+4<my_rows_halo-1 && j-5<columns-1) 
 
  accessMat( surface, i+4, j-5 ) = accessMat( surface, i+4, j-5 ) * ( 0.75 );
				if(i+4>=1 && j-4>=1 && i+4<my_rows_halo-1 && j-4<columns-1) 
 
  accessMat( surface, i+4, j-4 ) = accessMat( surface, i+4, j-4 ) * ( 0.75 );
				if(i+4>=1 && j-3>=1 && i+4<my_rows_halo-1 && j-3<columns-1) 
 
  accessMat( surface, i+4, j-3 ) = accessMat( surface, i+4, j-3 ) * ( 0.75 );
				if(i+4>=1 && j-2>=1 && i+4<my_rows_halo-1 && j-2<columns-1) 
 
  accessMat( surface, i+4, j-2 ) = accessMat( surface, i+4, j-2 ) * ( 0.75 );
				if(i+4>=1 && j-1>=1 && i+4<my_rows_halo-1 && j-1<columns-1) 
 
  accessMat( surface, i+4, j-1 ) = accessMat( surface, i+4, j-1 ) * ( 0.75 );
				if(i+4>=1 && j+0>=1 && i+4<my_rows_halo-1 && j+0<columns-1) 
 
  accessMat( surface, i+4, j+0 ) = accessMat( surface, i+4, j+0 ) * ( 0.75 );
				if(i+4>=1 && j+1>=1 && i+4<my_rows_halo-1 && j+1<columns-1) 
 
  accessMat( surface, i+4, j+1 ) = accessMat( surface, i+4, j+1 ) * ( 0.75 );
				if(i+4>=1 && j+2>=1 && i+4<my_rows_halo-1 && j+2<columns-1) 
 
  accessMat( surface, i+4, j+2 ) = accessMat( surface, i+4, j+2 ) * ( 0.75 );
				if(i+4>=1 && j+3>=1 && i+4<my_rows_halo-1 && j+3<columns-1) 
 
  accessMat( surface, i+4, j+3 ) = accessMat( surface, i+4, j+3 ) * ( 0.75 );
				if(i+4>=1 && j+4>=1 && i+4<my_rows_halo-1 && j+4<columns-1) 
 
  accessMat( surface, i+4, j+4 ) = accessMat( surface, i+4, j+4 ) * ( 0.75 );
				if(i+4>=1 && j+5>=1 && i+4<my_rows_halo-1 && j+5<columns-1) 
 
  accessMat( surface, i+4, j+5 ) = accessMat( surface, i+4, j+5 ) * ( 0.75 );
				if(i+4>=1 && j+6>=1 && i+4<my_rows_halo-1 && j+6<columns-1) 
 
  accessMat( surface, i+4, j+6 ) = accessMat( surface, i+4, j+6 ) * ( 0.75 );
				if(i+4>=1 && j+7>=1 && i+4<my_rows_halo-1 && j+7<columns-1) 
 
  accessMat( surface, i+4, j+7 ) = accessMat( surface, i+4, j+7 ) * ( 0.75 );
				if(i+4>=1 && j+8>=1 && i+4<my_rows_halo-1 && j+8<columns-1) 
 
  accessMat( surface, i+4, j+8 ) = accessMat( surface, i+4, j+8 ) * ( 0.75 );
				if(i+5>=1 && j-7>=1 && i+5<my_rows_halo-1 && j-7<columns-1) 
 
  accessMat( surface, i+5, j-7 ) = accessMat( surface, i+5, j-7 ) * ( 0.75 );
				if(i+5>=1 && j-6>=1 && i+5<my_rows_halo-1 && j-6<columns-1) 
 
  accessMat( surface, i+5, j-6 ) = accessMat( surface, i+5, j-6 ) * ( 0.75 );
				if(i+5>=1 && j-5>=1 && i+5<my_rows_halo-1 && j-5<columns-1) 
 
  accessMat( surface, i+5, j-5 ) = accessMat( surface, i+5, j-5 ) * ( 0.75 );
				if(i+5>=1 && j-4>=1 && i+5<my_rows_halo-1 && j-4<columns-1) 
 
  accessMat( surface, i+5, j-4 ) = accessMat( surface, i+5, j-4 ) * ( 0.75 );
				if(i+5>=1 && j-3>=1 && i+5<my_rows_halo-1 && j-3<columns-1) 
 
  accessMat( surface, i+5, j-3 ) = accessMat( surface, i+5, j-3 ) * ( 0.75 );
				if(i+5>=1 && j-2>=1 && i+5<my_rows_halo-1 && j-2<columns-1) 
 
  accessMat( surface, i+5, j-2 ) = accessMat( surface, i+5, j-2 ) * ( 0.75 );
				if(i+5>=1 && j-1>=1 && i+5<my_rows_halo-1 && j-1<columns-1) 
 
  accessMat( surface, i+5, j-1 ) = accessMat( surface, i+5, j-1 ) * ( 0.75 );
				if(i+5>=1 && j+0>=1 && i+5<my_rows_halo-1 && j+0<columns-1) 
 
  accessMat( surface, i+5, j+0 ) = accessMat( surface, i+5, j+0 ) * ( 0.75 );
				if(i+5>=1 && j+1>=1 && i+5<my_rows_halo-1 && j+1<columns-1) 
 
  accessMat( surface, i+5, j+1 ) = accessMat( surface, i+5, j+1 ) * ( 0.75 );
				if(i+5>=1 && j+2>=1 && i+5<my_rows_halo-1 && j+2<columns-1) 
 
  accessMat( surface, i+5, j+2 ) = accessMat( surface, i+5, j+2 ) * ( 0.75 );
				if(i+5>=1 && j+3>=1 && i+5<my_rows_halo-1 && j+3<columns-1) 
 
  accessMat( surface, i+5, j+3 ) = accessMat( surface, i+5, j+3 ) * ( 0.75 );
				if(i+5>=1 && j+4>=1 && i+5<my_rows_halo-1 && j+4<columns-1) 
 
  accessMat( surface, i+5, j+4 ) = accessMat( surface, i+5, j+4 ) * ( 0.75 );
				if(i+5>=1 && j+5>=1 && i+5<my_rows_halo-1 && j+5<columns-1) 
 
  accessMat( surface, i+5, j+5 ) = accessMat( surface, i+5, j+5 ) * ( 0.75 );
				if(i+5>=1 && j+6>=1 && i+5<my_rows_halo-1 && j+6<columns-1) 
 
  accessMat( surface, i+5, j+6 ) = accessMat( surface, i+5, j+6 ) * ( 0.75 );
				if(i+5>=1 && j+7>=1 && i+5<my_rows_halo-1 && j+7<columns-1) 
 
  accessMat( surface, i+5, j+7 ) = accessMat( surface, i+5, j+7 ) * ( 0.75 );
				if(i+6>=1 && j-6>=1 && i+6<my_rows_halo-1 && j-6<columns-1) 
 
  accessMat( surface, i+6, j-6 ) = accessMat( surface, i+6, j-6 ) * ( 0.75 );
				if(i+6>=1 && j-5>=1 && i+6<my_rows_halo-1 && j-5<columns-1) 
 
  accessMat( surface, i+6, j-5 ) = accessMat( surface, i+6, j-5 ) * ( 0.75 );
				if(i+6>=1 && j-4>=1 && i+6<my_rows_halo-1 && j-4<columns-1) 
 
  accessMat( surface, i+6, j-4 ) = accessMat( surface, i+6, j-4 ) * ( 0.75 );
				if(i+6>=1 && j-3>=1 && i+6<my_rows_halo-1 && j-3<columns-1) 
 
  accessMat( surface, i+6, j-3 ) = accessMat( surface, i+6, j-3 ) * ( 0.75 );
				if(i+6>=1 && j-2>=1 && i+6<my_rows_halo-1 && j-2<columns-1) 
 
  accessMat( surface, i+6, j-2 ) = accessMat( surface, i+6, j-2 ) * ( 0.75 );
				if(i+6>=1 && j-1>=1 && i+6<my_rows_halo-1 && j-1<columns-1) 
 
  accessMat( surface, i+6, j-1 ) = accessMat( surface, i+6, j-1 ) * ( 0.75 );
				if(i+6>=1 && j+0>=1 && i+6<my_rows_halo-1 && j+0<columns-1) 
 
  accessMat( surface, i+6, j+0 ) = accessMat( surface, i+6, j+0 ) * ( 0.75 );
				if(i+6>=1 && j+1>=1 && i+6<my_rows_halo-1 && j+1<columns-1) 
 
  accessMat( surface, i+6, j+1 ) = accessMat( surface, i+6, j+1 ) * ( 0.75 );
				if(i+6>=1 && j+2>=1 && i+6<my_rows_halo-1 && j+2<columns-1) 
 
  accessMat( surface, i+6, j+2 ) = accessMat( surface, i+6, j+2 ) * ( 0.75 );
				if(i+6>=1 && j+3>=1 && i+6<my_rows_halo-1 && j+3<columns-1) 
 
  accessMat( surface, i+6, j+3 ) = accessMat( surface, i+6, j+3 ) * ( 0.75 );
				if(i+6>=1 && j+4>=1 && i+6<my_rows_halo-1 && j+4<columns-1) 
 
  accessMat( surface, i+6, j+4 ) = accessMat( surface, i+6, j+4 ) * ( 0.75 );
				if(i+6>=1 && j+5>=1 && i+6<my_rows_halo-1 && j+5<columns-1) 
 
  accessMat( surface, i+6, j+5 ) = accessMat( surface, i+6, j+5 ) * ( 0.75 );
				if(i+6>=1 && j+6>=1 && i+6<my_rows_halo-1 && j+6<columns-1) 
 
  accessMat( surface, i+6, j+6 ) = accessMat( surface, i+6, j+6 ) * ( 0.75 );
				if(i+7>=1 && j-5>=1 && i+7<my_rows_halo-1 && j-5<columns-1) 
 
  accessMat( surface, i+7, j-5 ) = accessMat( surface, i+7, j-5 ) * ( 0.75 );
				if(i+7>=1 && j-4>=1 && i+7<my_rows_halo-1 && j-4<columns-1) 
 
  accessMat( surface, i+7, j-4 ) = accessMat( surface, i+7, j-4 ) * ( 0.75 );
				if(i+7>=1 && j-3>=1 && i+7<my_rows_halo-1 && j-3<columns-1) 
 
  accessMat( surface, i+7, j-3 ) = accessMat( surface, i+7, j-3 ) * ( 0.75 );
				if(i+7>=1 && j-2>=1 && i+7<my_rows_halo-1 && j-2<columns-1) 
 
  accessMat( surface, i+7, j-2 ) = accessMat( surface, i+7, j-2 ) * ( 0.75 );
				if(i+7>=1 && j-1>=1 && i+7<my_rows_halo-1 && j-1<columns-1) 
 
  accessMat( surface, i+7, j-1 ) = accessMat( surface, i+7, j-1 ) * ( 0.75 );
				if(i+7>=1 && j+0>=1 && i+7<my_rows_halo-1 && j+0<columns-1) 
 
  accessMat( surface, i+7, j+0 ) = accessMat( surface, i+7, j+0 ) * ( 0.75 );
				if(i+7>=1 && j+1>=1 && i+7<my_rows_halo-1 && j+1<columns-1) 
 
  accessMat( surface, i+7, j+1 ) = accessMat( surface, i+7, j+1 ) * ( 0.75 );
				if(i+7>=1 && j+2>=1 && i+7<my_rows_halo-1 && j+2<columns-1) 
 
  accessMat( surface, i+7, j+2 ) = accessMat( surface, i+7, j+2 ) * ( 0.75 );
				if(i+7>=1 && j+3>=1 && i+7<my_rows_halo-1 && j+3<columns-1) 
 
  accessMat( surface, i+7, j+3 ) = accessMat( surface, i+7, j+3 ) * ( 0.75 );
				if(i+7>=1 && j+4>=1 && i+7<my_rows_halo-1 && j+4<columns-1) 
 
  accessMat( surface, i+7, j+4 ) = accessMat( surface, i+7, j+4 ) * ( 0.75 );
				if(i+7>=1 && j+5>=1 && i+7<my_rows_halo-1 && j+5<columns-1) 
 
  accessMat( surface, i+7, j+5 ) = accessMat( surface, i+7, j+5 ) * ( 0.75 );
				if(i+8>=1 && j-4>=1 && i+8<my_rows_halo-1 && j-4<columns-1) 
 
  accessMat( surface, i+8, j-4 ) = accessMat( surface, i+8, j-4 ) * ( 0.75 );
				if(i+8>=1 && j-3>=1 && i+8<my_rows_halo-1 && j-3<columns-1) 
 
  accessMat( surface, i+8, j-3 ) = accessMat( surface, i+8, j-3 ) * ( 0.75 );
				if(i+8>=1 && j-2>=1 && i+8<my_rows_halo-1 && j-2<columns-1) 
 
  accessMat( surface, i+8, j-2 ) = accessMat( surface, i+8, j-2 ) * ( 0.75 );
				if(i+8>=1 && j-1>=1 && i+8<my_rows_halo-1 && j-1<columns-1) 
 
  accessMat( surface, i+8, j-1 ) = accessMat( surface, i+8, j-1 ) * ( 0.75 );
				if(i+8>=1 && j+0>=1 && i+8<my_rows_halo-1 && j+0<columns-1) 
 
  accessMat( surface, i+8, j+0 ) = accessMat( surface, i+8, j+0 ) * ( 0.75 );
				if(i+8>=1 && j+1>=1 && i+8<my_rows_halo-1 && j+1<columns-1) 
 
  accessMat( surface, i+8, j+1 ) = accessMat( surface, i+8, j+1 ) * ( 0.75 );
				if(i+8>=1 && j+2>=1 && i+8<my_rows_halo-1 && j+2<columns-1) 
 
  accessMat( surface, i+8, j+2 ) = accessMat( surface, i+8, j+2 ) * ( 0.75 );
				if(i+8>=1 && j+3>=1 && i+8<my_rows_halo-1 && j+3<columns-1) 
 
  accessMat( surface, i+8, j+3 ) = accessMat( surface, i+8, j+3 ) * ( 0.75 );
				if(i+8>=1 && j+4>=1 && i+8<my_rows_halo-1 && j+4<columns-1) 
 
  accessMat( surface, i+8, j+4 ) = accessMat( surface, i+8, j+4 ) * ( 0.75 );
				if(i+9>=1 && j+0>=1 && i+9<my_rows_halo-1 && j+0<columns-1) 
 
  accessMat( surface, i+9, j+0 ) = accessMat( surface, i+9, j+0 ) * ( 0.75 );
			}
			}
	
#ifdef DEBUG
		/* 4.5. DEBUG: Print the current state of the simulation at the end of each iteration */
		print_status( iter, rows, columns, surface, num_teams, teams, num_focal, focal, global_residual );
#endif // DEBUG
	}

		float *buffer_residualHeat = (float*)malloc( sizeof(float) * (size_t)num_focal );
	for(i=0;i<num_focal;i++){
		buffer_residualHeat[i]=0.0;
	}
	
	// MPI Version: Store the results of residual heat on focal points
	for (i=0; i<num_focal; i++){
			if(focal[i].x<=stop && focal[i].x>=start)
			buffer_residualHeat[i]= accessMat( surface, focal[i].x-start+1, focal[i].y );
		}



	MPI_Reduce(buffer_residualHeat,residualHeat,num_focal,MPI_FLOAT,MPI_MAX,0,MPI_COMM_WORLD);
	free(buffer_residualHeat);




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
