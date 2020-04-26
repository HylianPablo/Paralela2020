/*
 * Simplified simulation of life evolution
 *
 * Computacion Paralela, Grado en Informatica (Universidad de Valladolid)
 * 2019/2020
 *
 * v1.2
 *
 * (c) 2020 Arturo Gonzalez Escribano
 */
#include<stdio.h>
#include<stddef.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>
#include<float.h>
#include<stdbool.h>
#include<cputils.h>
#include<mpi.h>

/* Structure to store data of a cell */
typedef struct {
	float pos_row, pos_col;		// Position
	float mov_row, mov_col;		// Direction of movement
	float choose_mov[3];		// Genes: Probabilities of 0 turning-left; 1 advance; 2 turning-right
	float storage;			// Food/Energy stored
	int age;			// Number of steps that the cell has been alive
	unsigned short random_seq[3];	// Status value of its particular random sequence
	bool alive;			// Flag indicating if the cell is still alive
} Cell;


/* Structure for simulation statistics */
typedef struct {
	int history_total_cells;	// Accumulated number of cells created
	int history_dead_cells;		// Accumulated number of dead cells
	int history_max_alive_cells;	// Maximum number of cells alive in a step
	int history_max_new_cells;	// Maximum number of cells created in a step
	int history_max_dead_cells;	// Maximum number of cells died in a step
	int history_max_age;		// Maximum age achieved by a cell
	float history_max_food;		// Maximum food level in a position of the culture
} Statistics;

/* Estructura para mover las celulas de procesador */
typedef struct{
    int origen;
    int destino;
    Cell celula_movida;
} Transporte_celula;
/* 
 * Macro function to simplify accessing with two coordinates to a flattened array
 * 	This macro-function can be changed and/or optimized by the students
 *
 */
#define accessMat( arr, exp1, exp2 )	arr[ ((int)(exp1) * columns + (int)(exp2)) - (int)(my_begin) ]


/*
 * Macro function to check if a point in a matrix belongs to this process' section. (by g110.)
 *
 */
#define mine(exp1, exp2) ((int)exp1 * columns + (int)exp2) >= my_begin && ((int)exp1 * columns + (int)exp2) < my_begin + my_size

/*
 * Function: Choose a new direction of movement for a cell
 * 	This function can be changed and/or optimized by the students
 */
void cell_new_direction( Cell *cell ) {
	float angle = (float)(2 * M_PI * erand48( cell->random_seq ));
	cell->mov_row = sinf( angle );
	cell->mov_col = cosf( angle );
}

/*
 * Function: Mutation of the movement genes on a new cell
 * 	This function can be changed and/or optimized by the students
 */
void cell_mutation( Cell *cell ) {
	/* 1. Select which genes change:
	 	0 Left grows taking part of the Advance part
	 	1 Advance grows taking part of the Left part
	 	2 Advance grows taking part of the Right part
	 	3 Right grows taking part of the Advance part
	*/
	int mutation_type = (int)(4 * erand48( cell->random_seq ));
	/* 2. Select the amount of mutation (up to 50%) */
	float mutation_percentage = (float)(0.5 * erand48( cell->random_seq ));
	/* 3. Apply the mutation */
	float mutation_value;
	switch( mutation_type ) {
		case 0:
			mutation_value = cell->choose_mov[1] * mutation_percentage;
			cell->choose_mov[1] -= mutation_value;
			cell->choose_mov[0] += mutation_value;
			break;
		case 1:
			mutation_value = cell->choose_mov[0] * mutation_percentage;
			cell->choose_mov[0] -= mutation_value;
			cell->choose_mov[1] += mutation_value;
			break;
		case 2:
			mutation_value = cell->choose_mov[2] * mutation_percentage;
			cell->choose_mov[2] -= mutation_value;
			cell->choose_mov[1] += mutation_value;
			break;
		default:
			mutation_value = cell->choose_mov[1] * mutation_percentage;
			cell->choose_mov[1] -= mutation_value;
			cell->choose_mov[2] += mutation_value;
	} 

	/* <-------------------------------------------------------------------------------------------------------------------------------------
		default:
			fprintf(stderr,"Error: Imposible type of mutation\n");
			MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
	}
	*/

	/* 4. Correct potential precision problems */
	cell->choose_mov[2] = 1.0f - cell->choose_mov[1] - cell->choose_mov[0];
}

#ifdef DEBUG
/* 
 * Function: Print the current state of the simulation 
 */
void print_status( int iteration, int rows, int columns, float *culture, int num_cells, Cell *cells, int num_cells_alive, Statistics sim_stat ) {
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
			if ( accessMat( culture, i, j ) >= 20 ) symbol = '+';
			else if ( accessMat( culture, i, j ) >= 10 ) symbol = '*';
			else if ( accessMat( culture, i, j ) >= 5 ) symbol = '.';
			else symbol = ' ';

			int t;
			int counter = 0;
			for( t=0; t<num_cells; t++ ) {
				int row = (int)(cells[t].pos_row);
				int col = (int)(cells[t].pos_col);
				if ( cells[t].alive && row == i && col == j ) {
					counter ++;
				}
			}
			if ( counter > 9 ) printf("(M)" );
			else if ( counter > 0 ) printf("(%1d)", counter );
			else printf(" %c ", symbol );
		}
		printf("|\n");
	}
	printf("+");
	for( j=0; j<columns; j++ ) printf("---");
	printf("+\n");
	printf("Num_cells_alive: %04d\nHistory( Cells: %04d, Dead: %04d, Max.alive: %04d, Max.new: %04d, Max.dead: %04d, Max.age: %04d, Max.food: %6f )\n\n", 
		num_cells_alive, 
		sim_stat.history_total_cells, 
		sim_stat.history_dead_cells, 
		sim_stat.history_max_alive_cells, 
		sim_stat.history_max_new_cells, 
		sim_stat.history_max_dead_cells, 
		sim_stat.history_max_age,
		sim_stat.history_max_food
	);
}
#endif

/*
 * Function: Print usage line in stderr
 */
void show_usage( char *program_name ) {
	fprintf(stderr,"Usage: %s ", program_name );
	fprintf(stderr,"<rows> <columns> <maxIter> <max_food> <food_density> <food_level> <short_rnd1> <short_rnd2> <short_rnd3> <num_cells>\n");
	fprintf(stderr,"\tOptional arguments for special food spot: [ <row> <col> <size_rows> <size_cols> <density> <level> ]\n");
	fprintf(stderr,"\n");
}


/*
 * MAIN PROGRAM
 */
int main(int argc, char *argv[]) {
	int i,j;

	// Simulation data
	int max_iter;			// Maximum number of simulation steps
	int rows, columns;		// Cultivation area sizes
	float *culture;			// Cultivation area values
	short *culture_cells;		// Ancillary structure to count the number of cells in a culture space

	float max_food;			// Maximum level of food on any position
	float food_density;		// Number of food sources introduced per step
	float food_level;		// Maximum number of food level in a new source

	bool food_spot_active = false;	// Special food spot: Active
	int food_spot_row = 0;		// Special food spot: Initial row
	int food_spot_col = 0;		// Special food spot: Initial row
	int food_spot_size_rows = 0;	// Special food spot: Rows size
	int food_spot_size_cols = 0;	// Special food spot: Cols size
	float food_spot_density = 0.0f;	// Special food spot: Food density
	float food_spot_level = 0.0f;	// Special food spot: Food level

	unsigned short init_random_seq[3];	// Status of the init random sequence
	unsigned short food_random_seq[3];	// Status of the food random sequence
	unsigned short food_spot_random_seq[3];	// Status of the special food spot random sequence

	int	num_cells;		// Number of cells currently stored in the list
	Cell	*cells;			// List to store cells information

	// Statistics
	Statistics sim_stat;	
	sim_stat.history_total_cells = 0;
	sim_stat.history_dead_cells = 0;
	sim_stat.history_max_alive_cells = 0;
	sim_stat.history_max_new_cells = 0;
	sim_stat.history_max_dead_cells = 0;
	sim_stat.history_max_age = 0;
	sim_stat.history_max_food = 0.0f;

/*
 * Funcion de debuggeo
 * 	
 *
 */

int iter=0;
int num_cells_alive = num_cells;

void debuggeo(int flag, int rank){

			printf("Rank; %d, Prints de control: %d\nResult: %d, %d, %d, %d, %d, %d, %d, %d, %f\n ------------------------------------------------------------------------------\n \n \n", 
			rank,
			flag,
			iter,
			num_cells_alive, 
			sim_stat.history_total_cells, 
			sim_stat.history_dead_cells, 
			sim_stat.history_max_alive_cells, 
			sim_stat.history_max_new_cells, 
			sim_stat.history_max_dead_cells, 
			sim_stat.history_max_age,
			sim_stat.history_max_food
		);
}






	/* 0. Initialize MPI */
	int rank;
	MPI_Init( &argc, &argv );
	MPI_Comm_rank( MPI_COMM_WORLD, &rank );

	/* 1. Read simulation arguments */
	/* 1.1. Check minimum number of arguments */
	if (argc < 11) {
		fprintf(stderr, "-- Error: Not enough arguments when reading configuration from the command line\n\n");
		show_usage( argv[0] );
		MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
	}

	/* 1.2. Read culture sizes, maximum number of iterations */
	rows = atoi( argv[1] );
	columns = atoi( argv[2] );
	max_iter = atoi( argv[3] );

	/* 1.3. Food data */
	max_food = atof( argv[4] );
	food_density = atof( argv[5] );
	food_level = atof( argv[6] );

	/* 1.4. Read random sequences initializer */
	for( i=0; i<3; i++ ) {
		init_random_seq[i] = (unsigned short)atoi( argv[7+i] );
	}

	/* 1.5. Read number of cells */
	num_cells = atoi( argv[10] );

	/* 1.6. Read special food spot */
	if (argc > 11 ) {
		if ( argc < 17 ) {
			fprintf(stderr, "-- Error in number of special-food-spot arguments in the command line\n\n");
			show_usage( argv[0] );
			MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
		}
		else {
			food_spot_active = true;
			food_spot_row = atoi( argv[11] );
			food_spot_col = atoi( argv[12] );
			food_spot_size_rows = atoi( argv[13] );
			food_spot_size_cols = atoi( argv[14] );
			food_spot_density = atof( argv[15] );
			food_spot_level = atof( argv[16] );

			// Check non-used trailing arguments
			if ( argc > 17 ) {
				fprintf(stderr, "-- Error: too many arguments in the command line\n\n");
				show_usage( argv[0] );
				MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
			}
		}
	}

#ifdef DEBUG
	/* 1.7. Print arguments */
	printf("Arguments, Rows: %d, Columns: %d, max_iter: %d\n", rows, columns, max_iter);
	printf("Arguments, Max.food: %f, Food density: %f, Food level: %f\n", max_food, food_density, food_level);
	printf("Arguments, Init Random Sequence: %hu,%hu,%hu\n", init_random_seq[0], init_random_seq[1], init_random_seq[2]);
	if ( food_spot_active ) {
		printf("Arguments, Food_spot, pos(%d,%d), size(%d,%d), Density: %f, Level: %f\n",
			food_spot_row, food_spot_col, food_spot_size_rows, food_spot_size_cols, food_spot_density, food_spot_level );
	}
	printf("Initial cells: %d\n", num_cells );
#endif // DEBUG


	/* 1.8. Initialize random sequences for food dropping */
	for( i=0; i<3; i++ ) {
		food_random_seq[i] = (unsigned short)nrand48( init_random_seq );
		food_spot_random_seq[i] = (unsigned short)nrand48( init_random_seq );
	}

	/* 1.9. Initialize random sequences of cells */
	cells = (Cell *)malloc( sizeof(Cell) * (size_t)num_cells );
	if ( cells == NULL ) {
		fprintf(stderr,"-- Error allocating: %d cells\n", num_cells );
		MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
	}
	for( i=0; i<num_cells; i++ ) {
		// Initialize the cell ramdom sequences
		for( j=0; j<3; j++ ) 
			cells[i].random_seq[j] = (unsigned short)nrand48( init_random_seq );
	}


#ifdef DEBUG
	/* 1.10. Print random seed of the initial cells */
	/*
	printf("Initial cells random seeds: %d\n", num_cells );
	for( i=0; i<num_cells; i++ )
		printf("\tCell %d, Random seq: %hu,%hu,%hu\n", i, cells[i].random_seq[0], cells[i].random_seq[1], cells[i].random_seq[2] );
	*/
#endif // DEBUG

	/* 2. Start global timer */
	MPI_Barrier( MPI_COMM_WORLD );
	double ttotal = cp_Wtime();

/*
 *
 * START HERE: DO NOT CHANGE THE CODE ABOVE THIS POINT
 *
 */

	//Inicializacion de variables MPI


	int num_procs;

	MPI_Comm_size( MPI_COMM_WORLD, &num_procs );

	int my_size = rows*columns/num_procs ; ////////
	int resto = rows*columns%num_procs;
	int my_begin ;

	if(rank < resto ){
		my_size=my_size+1;
		my_begin = my_size*rank ;
	}else{
		my_begin = resto + my_size*rank;
	}
    
	
	/* 3. Initialize culture surface and initial cells */
	culture = (float *)malloc( sizeof(float) * (size_t)my_size );
	culture_cells = (short *)malloc( sizeof(short) * (size_t)my_size);

    
    for(i=0; i<my_size; i++){ //<------------------------------------------------ Creo que así lo mejora
        culture[i] = 0.0;
        culture_cells[i] = 0; //<------------------------------------------------------------------------------------Podria ser mejor meterlo aqui
    }

	// 4. PODEMOS BORRAR CELLS PARA LIBERAR MEMORIA

    
	for( i=0; i<num_cells; i++ ) {
        
		cells[i].alive = true;
		// Initial age: Between 1 and 20 
		cells[i].age = 1 + (int)(19 * erand48( cells[i].random_seq ));
		// Initial storage: Between 10 and 20 units
		cells[i].storage = (float)(10 + 10 * erand48( cells[i].random_seq ));
		// Initial position: Anywhere in the culture arena
		cells[i].pos_row = (float)(rows * erand48( cells[i].random_seq ));
		cells[i].pos_col = (float)(columns * erand48( cells[i].random_seq ));
		// Movement direction: Unity vector in a random direction
		float angle = (float)(2 * M_PI * erand48( cells[i].random_seq )); //<------------------------------------------------------------------------------------------------------
		cells[i].mov_row = sinf( angle );
		cells[i].mov_col = cosf( angle );

		// Movement genes: Probabilities of advancing or changing direction: The sum should be 1.00
		cells[i].choose_mov[0] = 0.33f;
		cells[i].choose_mov[1] = 0.34f;
		cells[i].choose_mov[2] = 0.33f;
	}

	/* VAMOS A DISTRIBUIR CELLS*/

    // 1. SABER CUANTOS CELULAS TIENE CADA MINI MATRIZ PARA HACER MALLOC
    int my_num_cells=0;

    for(i=0; i<num_cells; i++){
    	if(mine(cells[i].pos_row, cells[i].pos_col)){		
    		my_num_cells+=1;
    	}
    }
    // 2. HACER MALLOC

	Cell *cells_chiquito = (Cell *)malloc( sizeof(Cell) * (size_t)my_num_cells );    

    // 3. RELLENAR LAS MATRICES CHIQUITAS

    j=0;

	for(i=0; i<num_cells; i++){

		if(mine(cells[i].pos_row, cells[i].pos_col)){

			cells_chiquito[j] = cells[i];
			j++;

		}

	}
	
	num_cells_alive = my_num_cells;
    
	//printf("LLEGA 2 \n");
	
	//Liberamos memoria--------------------------------------
    //printf("LLEGA 3 \n");
	free(cells);

	//------------------------------------------------------

	// Statistics: Initialize total number of cells, and max. alive
	sim_stat.history_total_cells = my_num_cells;
	sim_stat.history_max_alive_cells = num_cells;


/* <----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#ifdef DEBUG
	printf("Initial cells data: %d\n", num_cells );
	for( i=0; i<num_cells; i++ ) {
		printf("\tCell %d, Pos(%f,%f), Mov(%f,%f), Choose_mov(%f,%f,%f), Storage: %f, Age: %d\n",
				i, 
				cells[i].pos_row, 
				cells[i].pos_col, 
				cells[i].mov_row, 
				cells[i].mov_col, 
				cells[i].choose_mov[0], 
				cells[i].choose_mov[1], 
				cells[i].choose_mov[2], 
				cells[i].storage,
				cells[i].age );
	}
#endif // DEBUG
*/

	//debuggeo(rank, 1);

	/* 4. Simulation */
	float current_max_food = 0.0f;
	
	int step_new_cells; //<---------------------------------------------------------------------------------------------------------------------not sure de mejoria
	int step_dead_cells;
	int num_new_sources;
	int row, col;
	float food;	
	short count;
	float my_food;
	int free_position;
	int alive_in_main_list;
    
    int celulas_vivas; //<--------- Se suman las de todos los procesos y se le resta a num_cells
    int celulas_muertas;
    int maximo_viva=0;
    int auxiliar_step_new_cells;
    int celulas_vivas_historico = 0;
    int num_cells_movidas;
    int num_cells_alive_root = num_cells; //<------------ USAR ESTO COMO INDICE DEL BUCLE GRANDE
    int step_dead_cells_root;
    int vivas_root;
    int vivas_historico_root;
    float current_max_food_root = 0.0f;
    Cell *celulas_a_mover = (Cell *) malloc(sizeof(Cell)); //<--------- Vector de 3 posiciones para guardar el rank, el rank de destino y el indice.
    
    MPI_Datatype MPI_CELL;
    int lengths[9] = { 1, 1, 1, 1, 3, 1, 1, 3, 1};
    
    MPI_Aint disp[9] ={
        offsetof(Cell, pos_row),
        offsetof(Cell, pos_col),
        offsetof(Cell, mov_row),
        offsetof(Cell, mov_col),
        offsetof(Cell, choose_mov),
        offsetof(Cell, storage),
        offsetof(Cell, age),
        offsetof(Cell, random_seq),
        offsetof(Cell, alive),
    };
    
    MPI_Datatype types[9] = { MPI_FLOAT, MPI_FLOAT, MPI_FLOAT, MPI_FLOAT, MPI_FLOAT, MPI_FLOAT, MPI_INT, MPI_UNSIGNED_SHORT, MPI_C_BOOL};
    
    MPI_Type_create_struct(9, lengths, disp, types, &MPI_CELL);
    MPI_Aint lb, ext;
    MPI_Datatype MPI_CELL_EXT;
    MPI_Type_get_extent(MPI_CELL, &lb, &ext);
    MPI_Type_create_resized(MPI_CELL, lb, ext, &MPI_CELL_EXT);
    MPI_Type_commit(&MPI_CELL_EXT);
    
    //-------------------------------------------------------------------------------------//
   
	for( iter=0; iter<max_iter && current_max_food_root <= max_food && num_cells_alive_root > 0; iter++ ) {
		//if(iter==8) printf("Mi rank: %d, Llego hasta 0\n", rank);
		/*if(iter == 283 ){
			printf("1- My rank: %d\n", rank);
			fflush(stdout);
		}*/
		step_new_cells = 0;
		step_dead_cells = 0;
        auxiliar_step_new_cells = 0;
        celulas_vivas = 0;
        celulas_muertas = 0;
        step_dead_cells_root = 0;
		/* 4.1. Spreading new food */
		// Across the whole culture
		num_new_sources = (int)(rows * columns * food_density);
		
        //if(iter==240) printf("Rank %d entra en 240\n", rank);
        
        //double aleatorio[num_new_sources][3];
        float posiciones[num_new_sources][3];
/*
        int my_size_num_new_sources = num_new_sources/num_procs; 
		resto = num_new_sources%num_procs;
		int my_begin_num_new_sources ;
		if(rank < resto ){
			my_size_num_new_sources = my_size_num_new_sources+1;
			my_begin_num_new_sources = my_size_num_new_sources*rank;
		}else{
			my_begin_num_new_sources = resto + my_size_num_new_sources*rank;
		}
        */
        //<----------------------- CON ESTO CREO QUE RULA ----------------------------------------------------REPASR ESTA COSA, PROBABLEMENTE ESTE MAL
        for(j=0; j<num_new_sources; j++){ //igual toca dividir por tema de memoria
            /*aleatorio[z][0]=erand48( food_random_seq );
            aleatorio[z][1]=erand48( food_random_seq );
            aleatorio[z][2]=erand48( food_random_seq );*/
            posiciones[j][0] = (int)(rows * erand48( food_random_seq )); //row
			posiciones[j][1] = (int)(columns * erand48( food_random_seq )); //col
			posiciones[j][2] = (float)( food_level * erand48( food_random_seq )); //food
            
        }

		for (i=0; i<num_new_sources; i++) {
            
			if(mine(posiciones[i][0],posiciones[i][1])){
            	accessMat( culture, posiciones[i][0],posiciones[i][1] ) += posiciones[i][2];
        	}
		}
        //printf("LLEGA 5 \n");
        // In the special food spot
		if ( food_spot_active ) {

			num_new_sources = (int)(food_spot_size_rows * food_spot_size_cols * food_spot_density);
            //posiciones[num_new_sources][3];
            /*
	        my_size_num_new_sources = num_new_sources/num_procs; 
			resto = num_new_sources%num_procs;
			if(rank < resto ){
				my_size_num_new_sources = my_size_num_new_sources+1;
				my_begin_num_new_sources = my_size_num_new_sources*rank ;
			}else{
				my_begin_num_new_sources = resto + my_size_num_new_sources*rank;
			}
			
*/
            for(j=0; j<num_new_sources; j++){
                
                posiciones[j][0] = (int)(food_spot_row + (food_spot_size_rows * erand48( food_spot_random_seq ))); //row
				posiciones[j][1] = (int)(food_spot_col + (food_spot_size_cols * erand48( food_spot_random_seq ))); // col
				posiciones[j][2] = (float)( food_spot_level * erand48( food_spot_random_seq )); // food
                
            }

            
			for (i=0; i<num_new_sources; i++) {
				if(mine(posiciones[i][0],posiciones[i][1])){
                    accessMat( culture, posiciones[i][0],posiciones[i][1] ) += posiciones[i][2];
                }
			}
		}
		/* 4.2. Prepare ancillary data structures */
		/* 4.2.1. Clear ancillary structure of the culture to account alive cells in a position after movement */


 		/* 4.2.2. Allocate ancillary structure to store the food level to be shared by cells in the same culture place */
		float *food_to_share = (float *)malloc( sizeof(float) * (size_t)my_num_cells );


        /* 4.3. Cell movements */
       	//if(iter==8) printf("Mi rank: %d, Llego hasta 1\n", rank);
        num_cells_movidas=0;
        /*if(iter == 283 ){
			printf("2- My rank: %d\n", rank);
			fflush(stdout);
		}*/
                  
		for (i=0; i<my_num_cells; i++) {
          
			if ( cells_chiquito[i].alive ) {
				cells_chiquito[i].age ++;
				// Statistics: Max age of a cell in the simulation history
				if ( cells_chiquito[i].age > sim_stat.history_max_age ) sim_stat.history_max_age = cells_chiquito[i].age;
                
                
				/* 4.3.1. Check if the cell has the needed energy to move or keep alive */
				if ( cells_chiquito[i].storage < 0.1f ) {
					// Cell has died
					cells_chiquito[i].alive = false;
                    //<--------------------------------------- Variables auxiliares
					num_cells_alive --;
					step_dead_cells ++;
                 
                    continue;
				}
				if ( cells_chiquito[i].storage < 1.0f ) {
					// Almost dying cell, it cannot move, only if enough food is dropped here it will survive
					cells_chiquito[i].storage -= 0.2f;
				}
				else {
					// Consume energy to move
					cells_chiquito[i].storage -= 1.0f;
						
					/* 4.3.2. Choose movement direction */
					float prob = (float)erand48( cells_chiquito[i].random_seq );
					if ( prob < cells_chiquito[i].choose_mov[0] ) {
						// Turn left (90 degrees)
						float tmp = cells_chiquito[i].mov_col;
						cells_chiquito[i].mov_col = cells_chiquito[i].mov_row;
						cells_chiquito[i].mov_row = -tmp;
					}
					else if ( prob >= cells_chiquito[i].choose_mov[0] + cells_chiquito[i].choose_mov[1] ) {
						// Turn right (90 degrees)
						float tmp = cells_chiquito[i].mov_row;
						cells_chiquito[i].mov_row = cells_chiquito[i].mov_col;
						cells_chiquito[i].mov_col = -tmp;
					}
					// else do not change the direction
					
					/* 4.3.3. Update position moving in the choosen direction*/

					cells_chiquito[i].pos_row += cells_chiquito[i].mov_row;
					cells_chiquito[i].pos_col += cells_chiquito[i].mov_col;


					// Periodic arena: Left/Rigth edges are connected, Top/Bottom edges are connected
					if ( cells_chiquito[i].pos_row < 0 ) cells_chiquito[i].pos_row += rows;
					if ( cells_chiquito[i].pos_row >= rows ) cells_chiquito[i].pos_row -= rows;
					if ( cells_chiquito[i].pos_col < 0 ) cells_chiquito[i].pos_col += columns;
					if ( cells_chiquito[i].pos_col >= columns ) cells_chiquito[i].pos_col -= columns;
				}
				
				/* 4.3.4. Annotate that there is one more cell in this culture position */
            
                if(mine(cells_chiquito[i].pos_row, cells_chiquito[i].pos_col)){
					accessMat( culture_cells, cells_chiquito[i].pos_row, cells_chiquito[i].pos_col ) += 1;
					/* 4.3.5. Annotate the amount of food to be shared in this culture position */
					food_to_share[i] = accessMat( culture, cells_chiquito[i].pos_row, cells_chiquito[i].pos_col );
				} else{
                    num_cells_movidas++;
                    /*if(iter==242){
                        printf("1 realloc entra en la iter %d rank %d\n", iter, rank);
                    }*/
                    celulas_a_mover = (Cell *)realloc(celulas_a_mover, sizeof(Cell)* (size_t)num_cells_movidas);
                    /*if(iter==242){
                        printf("1 realloc va bien en la iter %d rank %d\n", iter, rank);
                    }*/
                    
                    celulas_a_mover[num_cells_movidas-1] = cells_chiquito[i];
                    cells_chiquito[i].alive = false;
                }
			}
		}


        int *celulas_movidas_por_proceso = (int *)malloc(sizeof(int) * (size_t)num_procs);
        //if(iter==8) printf("Mi rank: %d, Llego hasta 2\n", rank);
        MPI_Allgather(&num_cells_movidas, 1, MPI_INT, celulas_movidas_por_proceso, 1, MPI_INT, MPI_COMM_WORLD);

        /*if(iter == 283){
        	printf("Rank %d - ", rank);
        	for(j=0; j< num_procs; j++){
        		printf("%d ", celulas_movidas_por_proceso[j]);
        	}
        	printf(" \n");
        }*/
        
        int cantidad_celulas_movidas_total = 0;
        int offset = 0;
        int disp[num_procs];

        for(j=0; j<num_procs; j++){
            disp[j]=offset;
            cantidad_celulas_movidas_total += celulas_movidas_por_proceso[j];
            offset += celulas_movidas_por_proceso[j];
        }

        
        Cell *celulas_movidas_totales = (Cell *)malloc(sizeof(Cell)* (size_t)cantidad_celulas_movidas_total);  
        
        if(iter==240){
            //printf("Malloc va bien en la iter %d rank %d\n", iter, rank);
        }
        //if(iter==283) printf("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n");

        /*if(iter == 283){
        	printf("Rank %d - he movido: %d - movidas total - %d - desplaz: ", rank, num_cells_movidas, cantidad_celulas_movidas_total);
        	for(j=0; j< num_procs; j++){
        		printf("%d ", disp[j]);
        	}
        	printf("- Elementos por proceso: ");
        	for(j=0;j<num_procs;j++){
        	printf("%d ", disp[j]);
        	}
        	printf(" \n");
        }*/

        
        MPI_Allgatherv(celulas_a_mover, num_cells_movidas, MPI_CELL_EXT, celulas_movidas_totales, celulas_movidas_por_proceso, disp, MPI_CELL_EXT, MPI_COMM_WORLD);
    	//if(iter==283 ) printf("Mi rank: %d\n", rank);
    	//if(iter==283) printf("eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee\n");
        //debuggeo(rank, 3);
        // BUCLE PARA QUE CADA CELULA AÑADA LA NUEVA CELULA
        for(j=0; j<cantidad_celulas_movidas_total; j++){
            
            if(mine(celulas_movidas_totales[j].pos_row, celulas_movidas_totales[j].pos_col)){
                my_num_cells += 1;
                //step_dead_cells--; //<----------------
                /*if(iter==242){
                    printf("2 realloc entra en la iter %d rank %d, numero de celulas %d\n", iter, rank, my_num_cells);
                }*/
                cells_chiquito = (Cell *)realloc(cells_chiquito, sizeof(Cell)* (size_t)my_num_cells);
                food_to_share = (float *)realloc(food_to_share, sizeof(float) * (size_t)my_num_cells);
               
                /*if(iter==242){
                    printf("2 realloc va bien en la iter %d rank %d\n", iter, rank);
                }*/
                cells_chiquito[my_num_cells-1] = celulas_movidas_totales[j];
                
                accessMat( culture_cells, cells_chiquito[my_num_cells-1].pos_row, cells_chiquito[my_num_cells-1].pos_col ) += 1;
                /* 4.3.5. Annotate the amount of food to be shared in this culture position */
                food_to_share[my_num_cells-1] = accessMat( culture, cells_chiquito[my_num_cells-1].pos_row, cells_chiquito[my_num_cells-1].pos_col);
            }
            
        }
        
        //free(celulas_a_mover);
        //if(iter==8) printf("Mi rank: %d, Llego hasta 3\n", rank);
        int muertas;
        MPI_Reduce( &step_dead_cells, &muertas, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD); 
        
        int maximo_viva_total;
        MPI_Reduce(&sim_stat.history_max_age, &maximo_viva_total, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
        
        if(rank == 0){
            //num_cells_alive_root -= no_vivas;
            step_dead_cells_root = muertas;
            sim_stat.history_max_age = maximo_viva_total;
            //printf("Celulas movidas total: %d, iteracion %d\n", cantidad_celulas_movidas_total, iter);//<-----------------
        }
        
		/* 4.4. Cell actions */
		// Space for the list of new cells (maximum number of new cells is num_cells)
		Cell *new_cells = (Cell *)malloc( sizeof(Cell) * (size_t)my_num_cells );
        free_position = 0;
		alive_in_main_list = 0;
	
		for (i=0; i<my_num_cells; i++) {
			if ( cells_chiquito[i].alive ) {
				/* 4.4.1. Food harvesting */
				food = food_to_share[i];
				if(mine(cells_chiquito[i].pos_row, cells_chiquito[i].pos_col)){
					count = accessMat( culture_cells, cells_chiquito[i].pos_row, cells_chiquito[i].pos_col );
				}
				my_food = food / count;
				cells_chiquito[i].storage += my_food;

				/* 4.4.2. Split cell if the conditions are met: Enough maturity and energy */
				if ( cells_chiquito[i].age > 30 && cells_chiquito[i].storage > 20 ) {
					// Split: Create new cell
                    //<--------------------------- VARIABLES AUXILIARES
					num_cells_alive++;
					sim_stat.history_total_cells ++; //<--------------------------------- CAMBIARLA Y QUE SEA LOCAL DE CADA PROC
                    //celulas_vivas_historico ++;
					auxiliar_step_new_cells ++;

					// New cell is a copy of parent cell
					new_cells[ auxiliar_step_new_cells-1 ] = cells_chiquito[i];

					// Split energy stored and update age in both cells
					cells_chiquito[i].storage /= 2.0f;
					new_cells[ auxiliar_step_new_cells-1 ].storage /= 2.0f;
					cells_chiquito[i].age = 1;
					new_cells[ auxiliar_step_new_cells-1 ].age = 1;

					// Random seed for the new cell, obtained using the parent random sequence
					new_cells[ auxiliar_step_new_cells-1 ].random_seq[0] = (unsigned short)nrand48( cells_chiquito[i].random_seq );
					new_cells[ auxiliar_step_new_cells-1 ].random_seq[1] = (unsigned short)nrand48( cells_chiquito[i].random_seq );
					new_cells[ auxiliar_step_new_cells-1 ].random_seq[2] = (unsigned short)nrand48( cells_chiquito[i].random_seq );

					// Both cells start in random directions
					cell_new_direction( &cells_chiquito[i] );
					cell_new_direction( &new_cells[ auxiliar_step_new_cells-1 ] );
				
					// Mutations of the movement genes in both cells
					cell_mutation( &cells_chiquito[i] );
					cell_mutation( &new_cells[ auxiliar_step_new_cells-1 ] );
				}
				if(mine(cells_chiquito[i].pos_row, cells_chiquito[i].pos_col))
                    accessMat( culture, cells_chiquito[i].pos_row, cells_chiquito[i].pos_col ) = 0.0f; // 4.5
				
				// 4.6
				
				alive_in_main_list ++;
				if ( free_position != i ) {
					cells_chiquito[free_position] = cells_chiquito[i];
				}
				free_position ++;
                
			}
		} 
        //if(iter==8) printf("Mi rank: %d, Llego hasta 4\n", rank);
        MPI_Allreduce( &num_cells_alive, &num_cells_alive_root, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        
        MPI_Reduce(&sim_stat.history_total_cells, &vivas_historico_root,  1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        
        //printf("LLEGA 10.1 nuevas %d vivas %d\n", auxiliar_step_new_cells, celulas_vivas);
        
        int new_cells_root;
        //MPI_Reduce(&new_cells_root, &auxiliar_step_new_cells, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&auxiliar_step_new_cells, &new_cells_root, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
       
        if(rank == 0){
            step_new_cells += new_cells_root;
        }
		free( food_to_share );

				
		// 4.6.2. Reduce the storage space of the list to the current number of cells
		my_num_cells = alive_in_main_list;
        

		cells_chiquito = (Cell *)realloc( cells_chiquito, sizeof(Cell) * (size_t)my_num_cells );
        

//printf("LLEGA 11.1 \n");
		/* 4.7. Join cell lists: Old and new cells list */
		if ( auxiliar_step_new_cells > 0 ) {
			cells_chiquito = (Cell *)realloc( cells_chiquito, sizeof(Cell) * (size_t)( my_num_cells + auxiliar_step_new_cells ) );
            
            //printf("LLEGA 11.2, numero celulas: %d, nuevas celulas: %d\n", my_num_cells, auxiliar_step_new_cells);
            
			for (j=0; j<auxiliar_step_new_cells; j++)
				cells_chiquito[ my_num_cells + j ] = new_cells[ j ];
            
            //printf("LLEGA 11.3 \n");
            my_num_cells += auxiliar_step_new_cells;
            
		}
		//printf("LLEGA 11.3 \n");
		free( new_cells );
        //if(iter==8 || iter ==7) printf("Mi rank: %d, Llego hasta 5\n", rank);
        //printf("LLEGA 12 \n");
		/* 4.8. Decrease non-harvested food */


		current_max_food = 0.0f;
		current_max_food_root = 0.0f;


		for (i=0; i < my_size; i++){

			culture_cells[i] = 0.0f;
			culture[i] *= 0.95f;

			if(culture[i] > current_max_food){
				current_max_food = culture[i];
			}

		}
        //debuggeo(rank, 3);
		//printf("%f", current_max_food);

		MPI_Allreduce(&current_max_food, &current_max_food_root, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
		//if(iter==8 || iter == 7) printf("Mi rank: %d, Llego hasta 6\n", rank);

		if(rank == 0){
            if ( current_max_food_root > sim_stat.history_max_food ) sim_stat.history_max_food = current_max_food_root;
            // Statistics: Max new cells per step
            if ( step_new_cells > sim_stat.history_max_new_cells ) sim_stat.history_max_new_cells = step_new_cells;
            // Statistics: Accumulated dead and Max dead cells per step
            
            sim_stat.history_dead_cells += step_dead_cells_root;
            
            if ( step_dead_cells_root > sim_stat.history_max_dead_cells ) sim_stat.history_max_dead_cells = step_dead_cells_root;
            // Statistics: Max alive cells per step
            if ( num_cells_alive_root > sim_stat.history_max_alive_cells ) sim_stat.history_max_alive_cells = num_cells_alive_root;
            //printf("HISTORY DEAD CELLS %d STEP DEAD CELLS %d\n", sim_stat.history_dead_cells, step_dead_cells_root);

            //printf("Iter: %d\n", iter);

        }

        //free(food_to_share);
#ifdef DEBUG
		/* 4.10. DEBUG: Print the current state of the simulation at the end of each iteration */
		print_status( iter, rows, columns, culture, num_cells, cells, num_cells_alive, sim_stat );
#endif // DEBUG
	}
	
	if(rank==0){
		num_cells_alive = num_cells_alive_root;
        sim_stat.history_total_cells = vivas_historico_root;
	}

	//free(cells_chiquito);
	//free(culture_cells);
	//free(culture);


/*
 *
 * STOP HERE: DO NOT CHANGE THE CODE BELOW THIS POINT
 *
 */

	/* 5. Stop global time */
	MPI_Barrier( MPI_COMM_WORLD );
	ttotal = cp_Wtime() - ttotal;

#ifdef DEBUG
	printf("List of cells at the end of the simulation: %d\n\n", num_cells );
	for( i=0; i<num_cells; i++ ) {
		printf("Cell %d, Alive: %d, Pos(%f,%f), Mov(%f,%f), Choose_mov(%f,%f,%f), Storage: %f, Age: %d\n",
				i,
				cells[i].alive,
				cells[i].pos_row, 
				cells[i].pos_col, 
				cells[i].mov_row, 
				cells[i].mov_col, 
				cells[i].choose_mov[0], 
				cells[i].choose_mov[1], 
				cells[i].choose_mov[2], 
				cells[i].storage,
				cells[i].age );
	}
#endif // DEBUG

	/* 6. Output for leaderboard */
	if ( rank == 0 ) {
		printf("\n");
		/* 6.1. Total computation time */
		printf("Time: %lf\n", ttotal );

		/* 6.2. Results: Number of iterations and other statistics */
		printf("Result: %d, ", iter);
		printf("%d, %d, %d, %d, %d, %d, %d, %f\n", 
			num_cells_alive, 
			sim_stat.history_total_cells, 
			sim_stat.history_dead_cells, 
			sim_stat.history_max_alive_cells, 
			sim_stat.history_max_new_cells, 
			sim_stat.history_max_dead_cells, 
			sim_stat.history_max_age,
			sim_stat.history_max_food
		);
	}

	/* 7. Free resources */	
	free( culture );
	free( culture_cells );
	free( cells );

	/* 8. End */
	MPI_Finalize();
	return 0;
}