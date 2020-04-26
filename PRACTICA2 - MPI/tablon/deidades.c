#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

/*
 * MAIN PROGRAM
 */
int main(int argc, char *argv[])
{
	/* 0. Initialize MPI */
	int rank;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	int longitud;
	char nombre[MPI_MAX_PROCESSOR_NAME];

	MPI_Get_processor_name( nombre, &longitud );
	printf("%s ejecuta el rank %d\n", nombre, rank);

	fflush(stdout);
	MPI_Barrier(MPI_COMM_WORLD);
	if (rank == 0) printf(":woozy_face:\n");

	/* 8. End */
	MPI_Finalize();
	return 0;
}
