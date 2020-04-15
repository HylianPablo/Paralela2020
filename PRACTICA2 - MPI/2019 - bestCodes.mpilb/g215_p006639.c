// User: g215@157.88.139.133 
// ExecutionRequest[P:'inicial-05.c',P:1,T:1,args:'',q:'mpilb'] 
// Apr 04 2019 15:32:04
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
 	MPI_Status stat;
	MPI_Request send_reqs, send_reqs2, recv_reqs;
	int error=0;
	for(i=0;i<num_focal;i++)
		if(focal[i].x==0 || focal[i].y==0 || focal[i].y==columns-1 || focal[i].x==rows-1)
			error=1;
	int empiece;
	int num_proc;
	MPI_Comm_size(MPI_COMM_WORLD,&num_proc);
	int size;
	int iter;
	size=rows/num_proc+(rank< rows % num_proc ? 1 : 0);
	empiece= rank*(rows/num_proc)+(rank < rows % num_proc ? rank : rows % num_proc);
// MPI Version: Eliminate this conditional to start doing the work in parallel
/*int error=0;
	for(i=0;i<num_focal;i++)
		if(focal[i].x==0 || focal[i].y==0 || focal[i].x>=999 || focal[i].y>=999)
			error++;
			printf("error  %d \n",error);
*/
	float *surface = (float *)malloc( sizeof(float) * (size_t)(size+2) * (size_t)columns );
	float *surfaceCopy = (float *)malloc( sizeof(float) * (size_t)(size+2) * (size_t)columns );
	if ( surface == NULL || surfaceCopy == NULL ) {
		fprintf(stderr,"-- Error allocating: surface structures\n");
		MPI_Abort( MPI_COMM_WORLD,  EXIT_FAILURE );
	}
	//printf("size %d empiece %d empiece+size %d\n",size,empiece,empiece+size);
	/* 3. Initialize surface */
	MPI_Barrier(MPI_COMM_WORLD);
	for( i=0; i<size+2; i++ )
		for( j=0; j<columns; j++ ){
			accessMat( surface, i, j ) = 0.0;
			accessMat( surfaceCopy, i, j ) = 0.0;
		}
	/* 4. Simulation */
	int flag_stability = 0;
	int first_activation = 0;
	for( iter=0; iter<max_iter && ! flag_stability; iter++ ) {
		if(error)
			break;
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
		if(!first_activation) continue;
		//if (!first_activation) continue;
		/* 4.2. Propagate heat (10 steps per each team movement) */
		float global_residual = 0.0f;
		int step;

			//printf("rank %d, size %d, empiece %d empiece+size %d \n",rank,size,empiece,empiece+size-1);
		/* 4.2.1. Update heat on active focal points */

			for( step=0; step<10; step++ )	{
				for( i=0; i<num_focal; i++ ) {
					if ( focal[i].active != 1 ) continue;
					int x = focal[i].x;
					int y = focal[i].y;
					if(empiece-1<=x && x<=empiece+size){
					accessMat( surface, x-empiece+1, y ) = focal[i].heat;
					//printf("rank %d focal %d posicion donde se guarda %d\n",rank,focal[i].x,x-empiece+1);

					}
				}
				float *aux=surface;
				surface=surfaceCopy;
				surfaceCopy=aux;

			/* 4.2.2. Copy values of the surface in ancillary structure (Skip borders) */
			/*if(rank==0)
			for( i=1; i<size+2; i++ )
				for( j=1; j<columns-1; j++ )
					accessMat( surfaceCopy, i, j ) = accessMat( surface, i, j );
			else if (rank==num_proc-1)
				for( i=0; i<size+1; i++ )
					for( j=1; j<columns-1; j++ )
						accessMat( surfaceCopy, i, j ) = accessMat( surface, i, j );
			else
				for( i=0; i<size+2; i++ )
					for( j=1; j<columns-1; j++ )
						accessMat( surfaceCopy, i, j ) = accessMat( surface, i, j );


			/* 4.2.3. Update surface values (skip borders) */
			if(rank==0)
				for( i=2; i<size+1; i++ )
					for( j=1; j<columns-1; j++ )
							accessMat( surface, i, j ) = (
							accessMat( surfaceCopy, i-1, j ) +
							accessMat( surfaceCopy, i+1, j ) +
							accessMat( surfaceCopy, i, j-1 ) +
							accessMat( surfaceCopy, i, j+1 ) ) / 4;
			else if(rank==num_proc-1)
				for( i=1; i<size; i++ )
					for( j=1; j<columns-1; j++ )
							accessMat( surface, i, j ) = (
							accessMat( surfaceCopy, i-1, j ) +
							accessMat( surfaceCopy, i+1, j ) +
							accessMat( surfaceCopy, i, j-1 ) +
							accessMat( surfaceCopy, i, j+1 ) ) / 4;
			else
				for( i=1; i<size+1; i++ )
					for( j=1; j<columns-1; j++ )
							accessMat( surface, i, j ) = (
								accessMat( surfaceCopy, i-1, j ) +
								accessMat( surfaceCopy, i+1, j ) +
								accessMat( surfaceCopy, i, j-1 ) +
								accessMat( surfaceCopy, i, j+1 ) ) / 4;

			/* 4.2.4. Compute the maximum residual difference (absolute value) */
			if( num_deactivated == num_focal && step==0)
			for( i=1; i<size+1; i++ )
				for( j=1; j<columns-1; j++ )
					if ( accessMat( surface, i, j ) - accessMat( surfaceCopy, i, j ) > global_residual ) {
						global_residual = accessMat( surface, i, j ) - accessMat( surfaceCopy, i, j ) ;
					}

		//	for(j=0;j<columns;j++){
		if(rank!=num_proc-1){
				MPI_Isend(&accessMat( surface, size, 0 ), columns, MPI_FLOAT, rank+1, 1, MPI_COMM_WORLD,&send_reqs2);
				MPI_Recv(&accessMat( surface, size+1, 0 ), columns, MPI_FLOAT, rank+1, 1, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
		}
		if(rank!=0){
			MPI_Isend(&accessMat( surface, 1, 0 ), columns, MPI_FLOAT, rank-1, 1, MPI_COMM_WORLD,&send_reqs);
			MPI_Irecv(&accessMat( surface, 0, 0 ), columns, MPI_FLOAT, rank-1, 1, MPI_COMM_WORLD,&recv_reqs);
		}
		
		if( rank == 0){//comunico con +1
			MPI_Wait( &send_reqs2, MPI_STATUS_IGNORE);
		}else if( rank == (num_proc-1)){//comunico con -1
			MPI_Wait( &send_reqs, MPI_STATUS_IGNORE);
			MPI_Wait( &recv_reqs, MPI_STATUS_IGNORE);
		}else{//comunico con +-1
			MPI_Wait( &send_reqs, MPI_STATUS_IGNORE);
			MPI_Wait( &send_reqs2, MPI_STATUS_IGNORE);
			MPI_Wait( &recv_reqs, MPI_STATUS_IGNORE);
		}
		//	}

		}


			float g_residual=0;
			//MPI_Barrier(MPI_COMM_WORLD);
			//printf("global_residual %f num %d iter %d rank %d\n",global_residual,num_deactivated,iter,rank);
	//printf("iter %d rank %d\n",iter,rank);
		/* If the global residual is lower than THRESHOLD, we have reached enough stability, stop simulation at the end of this iteration */
		MPI_Reduce(&global_residual, &g_residual, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
		/*for(int m=0;m<num_proc;m++){
			if(rank==m){
				printf("iter %d rank %d \n",iter,rank );
				for (i=0; i <size+2;  i++) {
					for(j=0;j<columns;j++)
						printf("%.2f ", accessMat( surface, i, j ));
					printf("\n" );
				}
			}
			MPI_Barrier(MPI_COMM_WORLD);
		}*/
		//if(rank==0) printf("reduce global %f num %d iter %d\n",g_residual,num_deactivated,iter);

		//	MPI_Reduce(&num_deactivated, &aux2 ,1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
		if(rank==0){
			if( num_deactivated == num_focal && g_residual < THRESHOLD ) {
				flag_stability=1;

			}
		}
		MPI_Bcast(&flag_stability, 1,MPI_INT, 0,MPI_COMM_WORLD);
		/* 4.3. Move teams */
		if(num_deactivated!=num_focal){
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
					float dx = teams[t].x - i;
					float dy = teams[t].y - j;
					float distance = sqrtf( dx*dx + dy*dy );
					if ( distance <= radius ) {
						if(empiece<=i && i<=empiece+size)
							accessMat( surface, i-empiece+1, j ) = accessMat( surface, i-empiece+1, j ) * ( 1 - 0.25 );
					}
				}
			}
		}
		//for(j=0;j<columns;j++){
		if(rank!=num_proc-1){
			MPI_Isend(&accessMat( surface, size, 0 ), columns, MPI_FLOAT, rank+1, 1, MPI_COMM_WORLD,&send_reqs2);
			MPI_Recv(&accessMat( surface, size+1, 0 ), columns, MPI_FLOAT, rank+1, 1, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
		}
		if(rank!=0){
			MPI_Isend(&accessMat( surface, 1, 0 ), columns, MPI_FLOAT, rank-1, 1, MPI_COMM_WORLD,&send_reqs);
			MPI_Irecv(&accessMat( surface, 0, 0 ), columns, MPI_FLOAT, rank-1, 1, MPI_COMM_WORLD,&recv_reqs);
		}
		
		if( rank == 0){//comunico con +1
			MPI_Wait( &send_reqs2, MPI_STATUS_IGNORE);
		}else if( rank == (num_proc-1)){//comunico con -1
			MPI_Wait( &send_reqs, MPI_STATUS_IGNORE);
			MPI_Wait( &recv_reqs, MPI_STATUS_IGNORE);
		}else{//comunico con +-1
			MPI_Wait( &send_reqs, MPI_STATUS_IGNORE);
			MPI_Wait( &send_reqs2, MPI_STATUS_IGNORE);
			MPI_Wait( &recv_reqs, MPI_STATUS_IGNORE);
		}
		//}

		//printf("flag %d rank %d\n",flag_stability,rank );
#ifdef DEBUG
		/* 4.5. DEBUG: Print the current state of the simulation at the end of each iteration */
		print_status( iter, rows, columns, surface, num_teams, teams, num_focal, focal, global_residual );
#endif // DEBUG
	}


	// MPI Version: Store the results of residual heat on focal points
	int envio=0;
	int recibe=0;
		//	printf("empiece %d empiece+size %d\n",empiece,empiece+size );
	for (i=0; i<num_focal; i++){
		if(empiece<=focal[i].x && focal[i].x<empiece+size && rank==0){
			residualHeat[i] = accessMat( surface, focal[i].x-empiece+1, focal[i].y );
			//envio++;
		}
		else if(empiece<=focal[i].x && focal[i].x<empiece+size && rank!=0){
			MPI_Send(&accessMat( surface, focal[i].x-empiece+1, focal[i].y ) , 1, MPI_FLOAT, 0, i, MPI_COMM_WORLD);
			//if(focal[i].x==0 || focal[i].x==100)
			//printf("focal.y %d \n",focal[i].x );
			//envio++;
		}
		else if(rank==0){
			//recibe++;
			//if(focal[i].x==0 || focal[i].x==100)
			//printf("focal.x %d \n",focal[i].x );
			MPI_Recv(&residualHeat[i], 1, MPI_FLOAT, MPI_ANY_SOURCE, i, MPI_COMM_WORLD,&stat);

			//printf("recibe %d \n",i );
		}
		//MPI_Barrier(MPI_COMM_WORLD);
	}
	//MPI_Gather(void *sendbuf, int sendcnt,MPI_Datatype stype, void *recvbuf, int recvcnt,MPI_Datatype recvtype, int root, MPI_Comm comm);
	free( surface );
	free( surfaceCopy );
	//printf("rank %d envia %d recibe %d \n", rank,envio,recibe );

// MPI Version: Eliminate this conditional-end to start doing the work in parallel



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
