// User: g201@79.109.79.14 
// ExecutionRequest[P:'erCho.cu',P:1,T:1,args:'',q:'cudalb'] 
// May 16 2019 18:03:51
#include "cputils.h" // Added by tablon
/*30 30 100 2 9 18 2 29 26 3 2 3 6 4 800 25 20 2 900
 * Simplified simulation of fire extinguishing
 *
 * Computacion Paralela, Grado en Informatica (Universidad de Valladolid)
 * 2018/2019
 *
 * v1.4
 *
 * (c) 2019 Arturo Gonzalez Escribano
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <cputils.h>

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

__global__ void init(float *surface,int rows,int columns){

	int idX=threadIdx.y+blockDim.y*blockIdx.y;
	int idY=threadIdx.x+blockDim.x*blockIdx.x;

	if (idX >= rows || idY>= columns) return;

	surface[idX*columns+idY]=0;


}

__global__ void initInt(int *surface, int rows, int columns){

	int idX=threadIdx.y+blockDim.y*blockIdx.y;
	int idY=threadIdx.x+blockDim.x*blockIdx.x;

	if (idX >= rows || idY>= columns) return;

	surface[idX*columns+idY]=0;
}

__global__ void get_first_activation(FocalPoint *focal, int num_focal,int *salida){

	__shared__ int first_activation;

	int id=threadIdx.x+blockDim.x*blockIdx.x;
	if(id>=num_focal) return;
	first_activation=0;


	atomicMin(&first_activation,focal[id].start);

	__syncthreads();
	if(id==0)
		salida[0]=first_activation;
}

__global__ void activate_focal(FocalPoint *focal,int num_focal,int *salida,int iter){

	__shared__ int num_deactivated;
	int id=threadIdx.x+blockDim.x*blockIdx.x;


	if(id>=num_focal) return;
	num_deactivated=0;


//printf("iter hilo %d num_ %d\n",iter,num_deactivated );
	if ( focal[id].active == 2 ) {
		atomicAdd(&num_deactivated,1);


	}
		if ( focal[id].start == iter ) {
			focal[id].active = 1;

		}
			__syncthreads();
		if(id==0)
		salida[0]=num_deactivated;
		// Count focal points already deactivated by a team



}
__global__ void update_heat(float *surface,FocalPoint *focal, int columns , int num_focal){

		int id=threadIdx.x+blockDim.x*blockIdx.x;
		if(id>=num_focal || focal[id].active!=1) return;

	surface[focal[id].x*columns+focal[id].y]=focal[id].heat;
}

__global__ void copy_surface(float *surface, float *surfaceCopy,int rows,int columns){

	int idX=threadIdx.y+blockDim.y*blockIdx.y;
	int idY=threadIdx.x+blockDim.x*blockIdx.x;

	if (idX >= rows-1 || idX==0 || idY>= columns-1 || idY==0) return;

	surfaceCopy[idX*columns+idY]=surface[idX*columns+idY];

}

__global__ void update_surface(float *surface, float *surfaceCopy,int rows, int columns){
	int idX=threadIdx.y+blockDim.y*blockIdx.y;
	int idY=threadIdx.x+blockDim.x*blockIdx.x;
  //printf("hola\n" );
	if (idX >= rows-1 || idX==0 || idY>= columns-1 || idY==0) return;

	surface[idX*columns+idY]=(
		surfaceCopy[(idX-1)*columns+idY]+
		surfaceCopy[(idX+1)*columns+idY]+
		surfaceCopy[idX*columns+idY-1]+
		surfaceCopy[idX*columns+idY+1])/4;
	//printf("%f",surface[idX*columns+idY]);
	/*int i, j;
	for( i=1; i<rows-1; i++ )
		for( j=1; j<columns-1; j++ )
			accessMat( surface, i, j ) = (
				accessMat( surfaceCopy, i-1, j ) +
				accessMat( surfaceCopy, i+1, j ) +
				accessMat( surfaceCopy, i, j-1 ) +
				accessMat( surfaceCopy, i, j+1 ) ) / 4;*/
}

__global__ void compute_residual(float *surface, float *surfaceCopy,int rows,int columns,float *residuals){

	int idX=threadIdx.y+blockDim.y*blockIdx.y;
	int idY=threadIdx.x+blockDim.x*blockIdx.x;
	//printf("hola\n" );
	//if (idX >= rows-1 || idX==0 || idY>= columns-1 || idY==0) return;
	if(idX>=rows || idY>=columns) return;
	residuals[idX*columns+idY]=surface[idX*columns+idY]-surfaceCopy[idX*columns+idY];
}



__global__ void move_teams(Team *teams,FocalPoint *focal, int num_teams,int num_focal){

		int id=threadIdx.x+blockDim.x*blockIdx.x;

		if(id>=num_teams) return;
		unsigned int j;
		int distance = INT_MAX;
		int target = -1;
		int teamX = teams[id].x;
		int teamY = teams[id].y;
		#pragma unroll
		for( j=0; j<num_focal; j++ ) {
			if ( focal[j].active != 1 ) continue; // Skip non-active focal points

			int local_distance =  (focal[j].x - teamX)*(focal[j].x - teamX) + (focal[j].y - teamY)*(focal[j].y - teamY) ;
			if ( local_distance < distance ) {
				distance = local_distance;
				target = j;
			}
		}
		/* 4.3.2. Annotate target for the next stage */
		teams[id].target = target;

		/* 4.3.3. No active focal point to choose, no movement */
		if ( target == -1 ) return;
		//__syncthreads();
		/* 4.3.4. Move in the focal point direction */

		int focalX = focal[target].x;
		int focalY = focal[target].y;
		if ( teams[id].type == 1 ) {
			// Type 1: Can move in diagonal
			if ( focalX < teams[id].x ) teams[id].x--;
			if ( focalX > teams[id].x ) teams[id].x++;
			if ( focalY < teams[id].y ) teams[id].y--;
			if ( focalY > teams[id].y) teams[id].y++;
		}
		else if ( teams[id].type == 2 ) {
			// Type 2: First in horizontal direction, then in vertical direction
			if ( focalY < teamY ) teams[id].y--;
			else if ( focalY > teamY ) teams[id].y++;
			else if ( focalX < teamX ) teams[id].x--;
			else if ( focalX > teamX ) teams[id].x++;
		}
		else {
			// Type 3: First in vertical direction, then in horizontal direction
			if ( focalX < teamX ) teams[id].x--;
			else if ( focalX > teamX ) teams[id].x++;
			else if ( focalY < teamY ) teams[id].y--;
			else if ( focalY > teamY ) teams[id].y++;
		}

		//printf("x %d y %d id %d\n", teams[id].x,teams[id].y,id);
		if ( target != -1 && focalX == teams[id].x && focalY == teams[id].y
			&& focal[target].active == 1 ){
			focal[target].active = 2;
			//printf("id %d\n",id);
		}
}

__global__ void compute_heat_reduction(Team *teams,int *gpuAux,int num_teams,int rows,int columns){

	int id=threadIdx.x+blockDim.x*blockIdx.x;
	if(id>=num_teams) return;
	//int radius;

	// Influence area of fixed radius depending on type
	//if ( teams[id].type == 1 ) radius = 3;
	//else radius = 9;
	int teamX=teams[id].x;
	int teamY=teams[id].y;
	//#pragma unroll
	//for( i=teams[id].x-radius; i<=teams[id].x+radius; i++ ) {
		//#pragma unroll
		//for( j=teams[id].y-radius; j<=teams[id].y+radius; j++ ) {
		if (teams[id].type!=1){

			if ( (teamX-9)>0 && (teamY)>0 && teamY<columns-1 )
			atomicAdd(&gpuAux[(teamX-9)*columns+teamY],1);

			if ( (teamX-8)>0 && (teamY-4)>0 )
			atomicAdd(&gpuAux[(teamX-8)*columns+teamY-4],1);
			if ( (teamX-8)>0 && (teamY-3)>0 )
			atomicAdd(&gpuAux[(teamX-8)*columns+teamY-3],1);
			if ( (teamX-8)>0 && (teamY-2)>0 )
			atomicAdd(&gpuAux[(teamX-8)*columns+teamY-2],1);
			if ( (teamX-8)>0 && (teamY-1)>0 )
			atomicAdd(&gpuAux[(teamX-8)*columns+teamY-1],1);
			if ( (teamX-8)>0 && (teamY)>0 && teamY<columns-1 )
			atomicAdd(&gpuAux[(teamX-8)*columns+teamY],1);

			if ( (teamX-8)>0 && (teamY+1)<columns-1 )
			atomicAdd(&gpuAux[(teamX-8)*columns+teamY+1],1);
			if ( (teamX-8)>0 && (teamY+2)<columns-1 )
			atomicAdd(&gpuAux[(teamX-8)*columns+teamY+2],1);
			if ( (teamX-8)>0 && (teamY+3)<columns-1 )
			atomicAdd(&gpuAux[(teamX-8)*columns+teamY+3],1);
			if ( (teamX-8)>0 && (teamY+4)<columns-1 )
			atomicAdd(&gpuAux[(teamX-8)*columns+teamY+4],1);

			if ( (teamX-7)>0 && (teamY-5)>0 )
			atomicAdd(&gpuAux[(teamX-7)*columns+teamY-5],1);
			if ( (teamX-7)>0 && (teamY-4)>0 )
			atomicAdd(&gpuAux[(teamX-7)*columns+teamY-4],1);
			if ( (teamX-7)>0 && (teamY-3)>0 )
			atomicAdd(&gpuAux[(teamX-7)*columns+teamY-3],1);
			if ( (teamX-7)>0 && (teamY-2)>0 )
			atomicAdd(&gpuAux[(teamX-7)*columns+teamY-2],1);
			if ( (teamX-7)>0 && (teamY-1)>0 )
			atomicAdd(&gpuAux[(teamX-7)*columns+teamY-1],1);
			if ( (teamX-7)>0 && (teamY)>0 && teamY<columns-1 )
			atomicAdd(&gpuAux[(teamX-7)*columns+teamY],1);

			if ( (teamX-7)>0 && (teamY+1)<columns-1 )
			atomicAdd(&gpuAux[(teamX-7)*columns+teamY+1],1);
			if ( (teamX-7)>0 && (teamY+2)<columns-1 )
			atomicAdd(&gpuAux[(teamX-7)*columns+teamY+2],1);
			if ( (teamX-7)>0 && (teamY+3)<columns-1 )
			atomicAdd(&gpuAux[(teamX-7)*columns+teamY+3],1);
			if ( (teamX-7)>0 && (teamY+4)<columns-1 )
			atomicAdd(&gpuAux[(teamX-7)*columns+teamY+4],1);
			if ( (teamX-7)>0 && (teamY+5)<columns-1 )
			atomicAdd(&gpuAux[(teamX-7)*columns+teamY+5],1);

			if ( (teamX-6)>0 && (teamY-6)>0 )
			atomicAdd(&gpuAux[(teamX-6)*columns+teamY-6],1);
			if ( (teamX-6)>0 && (teamY-5)>0 )
			atomicAdd(&gpuAux[(teamX-6)*columns+teamY-5],1);
			if ( (teamX-6)>0 && (teamY-4)>0 )
			atomicAdd(&gpuAux[(teamX-6)*columns+teamY-4],1);
			if ( (teamX-6)>0 && (teamY-3)>0 )
			atomicAdd(&gpuAux[(teamX-6)*columns+teamY-3],1);
			if ( (teamX-6)>0 && (teamY-2)>0 )
			atomicAdd(&gpuAux[(teamX-6)*columns+teamY-2],1);
			if ( (teamX-6)>0 && (teamY-1)>0 )
			atomicAdd(&gpuAux[(teamX-6)*columns+teamY-1],1);
			if ( (teamX-6)>0 && (teamY)>0 && teamY<columns-1 )
			atomicAdd(&gpuAux[(teamX-6)*columns+teamY],1);

			if ( (teamX-6)>0 && (teamY+1)<columns-1 )
			atomicAdd(&gpuAux[(teamX-6)*columns+teamY+1],1);
			if ( (teamX-6)>0 && (teamY+2)<columns-1 )
			atomicAdd(&gpuAux[(teamX-6)*columns+teamY+2],1);
			if ( (teamX-6)>0 && (teamY+3)<columns-1 )
			atomicAdd(&gpuAux[(teamX-6)*columns+teamY+3],1);
			if ( (teamX-6)>0 && (teamY+4)<columns-1 )
			atomicAdd(&gpuAux[(teamX-6)*columns+teamY+4],1);
			if ( (teamX-6)>0 && (teamY+5)<columns-1 )
			atomicAdd(&gpuAux[(teamX-6)*columns+teamY+5],1);
			if ( (teamX-6)>0 && (teamY+6)<columns-1 )
			atomicAdd(&gpuAux[(teamX-6)*columns+teamY+6],1);

			if ( (teamX-5)>0 && (teamY-7)>0 )
			atomicAdd(&gpuAux[(teamX-5)*columns+teamY-7],1);
			if ( (teamX-5)>0 && (teamY-6)>0 )
			atomicAdd(&gpuAux[(teamX-5)*columns+teamY-6],1);
			if ( (teamX-5)>0 && (teamY-5)>0 )
			atomicAdd(&gpuAux[(teamX-5)*columns+teamY-5],1);
			if ( (teamX-5)>0 && (teamY-4)>0 )
			atomicAdd(&gpuAux[(teamX-5)*columns+teamY-4],1);
			if ( (teamX-5)>0 && (teamY-3)>0 )
			atomicAdd(&gpuAux[(teamX-5)*columns+teamY-3],1);
			if ( (teamX-5)>0 && (teamY-2)>0 )
			atomicAdd(&gpuAux[(teamX-5)*columns+teamY-2],1);
			if ( (teamX-5)>0 && (teamY-1)>0 )
			atomicAdd(&gpuAux[(teamX-5)*columns+teamY-1],1);
			if ( (teamX-5)>0 && (teamY)>0 && teamY<columns-1 )
			atomicAdd(&gpuAux[(teamX-5)*columns+teamY],1);

			if ( (teamX-5)>0 && (teamY+1)<columns-1 )
			atomicAdd(&gpuAux[(teamX-5)*columns+teamY+1],1);
			if ( (teamX-5)>0 && (teamY+2)<columns-1 )
			atomicAdd(&gpuAux[(teamX-5)*columns+teamY+2],1);
			if ( (teamX-5)>0 && (teamY+3)<columns-1 )
			atomicAdd(&gpuAux[(teamX-5)*columns+teamY+3],1);
			if ( (teamX-5)>0 && (teamY+4)<columns-1 )
			atomicAdd(&gpuAux[(teamX-5)*columns+teamY+4],1);
			if ( (teamX-5)>0 && (teamY+5)<columns-1 )
			atomicAdd(&gpuAux[(teamX-5)*columns+teamY+5],1);
			if ( (teamX-5)>0 && (teamY+6)<columns-1 )
			atomicAdd(&gpuAux[(teamX-5)*columns+teamY+6],1);
			if ( (teamX-5)>0 && (teamY+7)<columns-1 )
			atomicAdd(&gpuAux[(teamX-5)*columns+teamY+7],1);

			if ( (teamX-4)>0 && (teamY-8)>0 )
			atomicAdd(&gpuAux[(teamX-4)*columns+teamY-8],1);
			if ( (teamX-4)>0 && (teamY-7)>0 )
			atomicAdd(&gpuAux[(teamX-4)*columns+teamY-7],1);
			if ( (teamX-4)>0 && (teamY-6)>0 )
			atomicAdd(&gpuAux[(teamX-4)*columns+teamY-6],1);
			if ( (teamX-4)>0 && (teamY-5)>0 )
			atomicAdd(&gpuAux[(teamX-4)*columns+teamY-5],1);
			if ( (teamX-4)>0 && (teamY-4)>0 )
			atomicAdd(&gpuAux[(teamX-4)*columns+teamY-4],1);
			if ( (teamX-4)>0 && (teamY-3)>0 )
			atomicAdd(&gpuAux[(teamX-4)*columns+teamY-3],1);
			if ( (teamX-4)>0 && (teamY-2)>0 )
			atomicAdd(&gpuAux[(teamX-4)*columns+teamY-2],1);
			if ( (teamX-4)>0 && (teamY-1)>0 )
			atomicAdd(&gpuAux[(teamX-4)*columns+teamY-1],1);
			if ( (teamX-4)>0 && (teamY)>0 && teamY<columns-1 )
			atomicAdd(&gpuAux[(teamX-4)*columns+teamY],1);

			if ( (teamX-4)>0 && (teamY+1)<columns-1 )
			atomicAdd(&gpuAux[(teamX-4)*columns+teamY+1],1);
			if ( (teamX-4)>0 && (teamY+2)<columns-1 )
			atomicAdd(&gpuAux[(teamX-4)*columns+teamY+2],1);
			if ( (teamX-4)>0 && (teamY+3)<columns-1 )
			atomicAdd(&gpuAux[(teamX-4)*columns+teamY+3],1);
			if ( (teamX-4)>0 && (teamY+4)<columns-1 )
			atomicAdd(&gpuAux[(teamX-4)*columns+teamY+4],1);
			if ( (teamX-4)>0 && (teamY+5)<columns-1 )
			atomicAdd(&gpuAux[(teamX-4)*columns+teamY+5],1);
			if ( (teamX-4)>0 && (teamY+6)<columns-1 )
			atomicAdd(&gpuAux[(teamX-4)*columns+teamY+6],1);
			if ( (teamX-4)>0 && (teamY+7)<columns-1 )
			atomicAdd(&gpuAux[(teamX-4)*columns+teamY+7],1);
			if ( (teamX-4)>0 && (teamY+8)<columns-1 )
			atomicAdd(&gpuAux[(teamX-4)*columns+teamY+8],1);

			if ( (teamX-3)>0 && (teamY-8)>0 )
			atomicAdd(&gpuAux[(teamX-3)*columns+teamY-8],1);
			if ( (teamX-3)>0 && (teamY-7)>0 )
			atomicAdd(&gpuAux[(teamX-3)*columns+teamY-7],1);
			if ( (teamX-3)>0 && (teamY-6)>0 )
			atomicAdd(&gpuAux[(teamX-3)*columns+teamY-6],1);
			if ( (teamX-3)>0 && (teamY-5)>0 )
			atomicAdd(&gpuAux[(teamX-3)*columns+teamY-5],1);
			if ( (teamX-3)>0 && (teamY-4)>0 )
			atomicAdd(&gpuAux[(teamX-3)*columns+teamY-4],1);
			if ( (teamX-3)>0 && (teamY-3)>0 )
			atomicAdd(&gpuAux[(teamX-3)*columns+teamY-3],1);
			if ( (teamX-3)>0 && (teamY-2)>0 )
			atomicAdd(&gpuAux[(teamX-3)*columns+teamY-2],1);
			if ( (teamX-3)>0 && (teamY-1)>0 )
			atomicAdd(&gpuAux[(teamX-3)*columns+teamY-1],1);
			if ( (teamX-3)>0 && (teamY)>0 && teamY<columns-1 )
			atomicAdd(&gpuAux[(teamX-3)*columns+teamY],1);

			if ( (teamX-3)>0 && (teamY+1)<columns-1 )
			atomicAdd(&gpuAux[(teamX-3)*columns+teamY+1],1);
			if ( (teamX-3)>0 && (teamY+2)<columns-1 )
			atomicAdd(&gpuAux[(teamX-3)*columns+teamY+2],1);
			if ( (teamX-3)>0 && (teamY+3)<columns-1 )
			atomicAdd(&gpuAux[(teamX-3)*columns+teamY+3],1);
			if ( (teamX-3)>0 && (teamY+4)<columns-1 )
			atomicAdd(&gpuAux[(teamX-3)*columns+teamY+4],1);
			if ( (teamX-3)>0 && (teamY+5)<columns-1 )
			atomicAdd(&gpuAux[(teamX-3)*columns+teamY+5],1);
			if ( (teamX-3)>0 && (teamY+6)<columns-1 )
			atomicAdd(&gpuAux[(teamX-3)*columns+teamY+6],1);
			if ( (teamX-3)>0 && (teamY+7)<columns-1 )
			atomicAdd(&gpuAux[(teamX-3)*columns+teamY+7],1);
			if ( (teamX-3)>0 && (teamY+8)<columns-1 )
			atomicAdd(&gpuAux[(teamX-3)*columns+teamY+8],1);

			if ( (teamX-2)>0 && (teamY-8)>0 )
			atomicAdd(&gpuAux[(teamX-2)*columns+teamY-8],1);
			if ( (teamX-2)>0 && (teamY-7)>0 )
			atomicAdd(&gpuAux[(teamX-2)*columns+teamY-7],1);
			if ( (teamX-2)>0 && (teamY-6)>0 )
			atomicAdd(&gpuAux[(teamX-2)*columns+teamY-6],1);
			if ( (teamX-2)>0 && (teamY-5)>0 )
			atomicAdd(&gpuAux[(teamX-2)*columns+teamY-5],1);
			if ( (teamX-2)>0 && (teamY-4)>0 )
			atomicAdd(&gpuAux[(teamX-2)*columns+teamY-4],1);
			if ( (teamX-2)>0 && (teamY-3)>0 )
			atomicAdd(&gpuAux[(teamX-2)*columns+teamY-3],1);
			if ( (teamX-2)>0 && (teamY-2)>0 )
			atomicAdd(&gpuAux[(teamX-2)*columns+teamY-2],1);
			if ( (teamX-2)>0 && (teamY-1)>0 )
			atomicAdd(&gpuAux[(teamX-2)*columns+teamY-1],1);
			if ( (teamX-2)>0 && (teamY)>0 && teamY<columns-1 )
			atomicAdd(&gpuAux[(teamX-2)*columns+teamY],1);

			if ( (teamX-2)>0 && (teamY+1)<columns-1 )
			atomicAdd(&gpuAux[(teamX-2)*columns+teamY+1],1);
			if ( (teamX-2)>0 && (teamY+2)<columns-1 )
			atomicAdd(&gpuAux[(teamX-2)*columns+teamY+2],1);
			if ( (teamX-2)>0 && (teamY+3)<columns-1 )
			atomicAdd(&gpuAux[(teamX-2)*columns+teamY+3],1);
			if ( (teamX-2)>0 && (teamY+4)<columns-1 )
			atomicAdd(&gpuAux[(teamX-2)*columns+teamY+4],1);
			if ( (teamX-2)>0 && (teamY+5)<columns-1 )
			atomicAdd(&gpuAux[(teamX-2)*columns+teamY+5],1);
			if ( (teamX-2)>0 && (teamY+6)<columns-1 )
			atomicAdd(&gpuAux[(teamX-2)*columns+teamY+6],1);
			if ( (teamX-2)>0 && (teamY+7)<columns-1 )
			atomicAdd(&gpuAux[(teamX-2)*columns+teamY+7],1);
			if ( (teamX-2)>0 && (teamY+8)<columns-1 )
			atomicAdd(&gpuAux[(teamX-2)*columns+teamY+8],1);
	    if ( (teamX-1)>0 && (teamY-8)>0 )
	    atomicAdd(&gpuAux[(teamX-1)*columns+teamY-8],1);
	    if ( (teamX-1)>0 && (teamY-7)>0 )
	    atomicAdd(&gpuAux[(teamX-1)*columns+teamY-7],1);
	    if ( (teamX-1)>0 && (teamY-6)>0 )
	    atomicAdd(&gpuAux[(teamX-1)*columns+teamY-6],1);
	    if ( (teamX-1)>0 && (teamY-5)>0 )
	    atomicAdd(&gpuAux[(teamX-1)*columns+teamY-5],1);
	    if ( (teamX-1)>0 && (teamY-4)>0 )
	    atomicAdd(&gpuAux[(teamX-1)*columns+teamY-4],1);
	    if ( (teamX-1)>0 && (teamY-3)>0 )
	    atomicAdd(&gpuAux[(teamX-1)*columns+teamY-3],1);
	    if ( (teamX-1)>0 && (teamY-2)>0 )
	    atomicAdd(&gpuAux[(teamX-1)*columns+teamY-2],1);
	    if ( (teamX-1)>0 && (teamY-1)>0 )
	    atomicAdd(&gpuAux[(teamX-1)*columns+teamY-1],1);
	    if ( (teamX-1)>0 && (teamY)>0 && teamY<columns-1 )
	    atomicAdd(&gpuAux[(teamX-1)*columns+teamY],1);

	    if ( (teamX-1)>0 && (teamY+1)<columns-1 )
	    atomicAdd(&gpuAux[(teamX-1)*columns+teamY+1],1);
	    if ( (teamX-1)>0 && (teamY+2)<columns-1 )
	    atomicAdd(&gpuAux[(teamX-1)*columns+teamY+2],1);
	    if ( (teamX-1)>0 && (teamY+3)<columns-1 )
	    atomicAdd(&gpuAux[(teamX-1)*columns+teamY+3],1);
	    if ( (teamX-1)>0 && (teamY+4)<columns-1 )
	    atomicAdd(&gpuAux[(teamX-1)*columns+teamY+4],1);
	    if ( (teamX-1)>0 && (teamY+5)<columns-1 )
	    atomicAdd(&gpuAux[(teamX-1)*columns+teamY+5],1);
	    if ( (teamX-1)>0 && (teamY+6)<columns-1 )
	    atomicAdd(&gpuAux[(teamX-1)*columns+teamY+6],1);
	    if ( (teamX-1)>0 && (teamY+7)<columns-1 )
	    atomicAdd(&gpuAux[(teamX-1)*columns+teamY+7],1);
	    if ( (teamX-1)>0 && (teamY+8)<columns-1 )
	    atomicAdd(&gpuAux[(teamX-1)*columns+teamY+8],1);


			if ( (teamX)>0 && (teamY-9)>0 )
			atomicAdd(&gpuAux[teamX*columns+teamY-9],1);
			if ( (teamX)>0 && (teamY-8)>0 )
			atomicAdd(&gpuAux[teamX*columns+teamY-8],1);
			if ( (teamX)>0 && (teamY-7)>0 )
			atomicAdd(&gpuAux[teamX*columns+teamY-7],1);
			if ( (teamX)>0 && (teamY-6)>0 )
			atomicAdd(&gpuAux[teamX*columns+teamY-6],1);
			if ( (teamX)>0 && (teamY-5)>0 )
			atomicAdd(&gpuAux[teamX*columns+teamY-5],1);
			if ( (teamX)>0 && (teamY-4)>0 )
			atomicAdd(&gpuAux[teamX*columns+teamY-4],1);
			if ( (teamX)>0 && (teamY-3)>0 )
			atomicAdd(&gpuAux[teamX*columns+teamY-3],1);
			if ( (teamX)>0 && (teamY-2)>0 )
			atomicAdd(&gpuAux[teamX*columns+teamY-2],1);
			if ( (teamX)>0 && (teamY-1)>0 )
			atomicAdd(&gpuAux[teamX*columns+teamY-1],1);
			if ( (teamX)>0 && (teamY)>0 && teamY<columns-1 )
			atomicAdd(&gpuAux[teamX*columns+teamY],1);

			if ( (teamX)>0 && (teamY+1)<columns-1 )
			atomicAdd(&gpuAux[teamX*columns+teamY+1],1);
			if ( (teamX)>0 && (teamY+2)<columns-1 )
			atomicAdd(&gpuAux[teamX*columns+teamY+2],1);
			if ( (teamX)>0 && (teamY+3)<columns-1 )
			atomicAdd(&gpuAux[teamX*columns+teamY+3],1);
			if ( (teamX)>0 && (teamY+4)<columns-1 )
			atomicAdd(&gpuAux[teamX*columns+teamY+4],1);
			if ( (teamX)>0 && (teamY+5)<columns-1 )
			atomicAdd(&gpuAux[teamX*columns+teamY+5],1);
			if ( (teamX)>0 && (teamY+6)<columns-1 )
			atomicAdd(&gpuAux[teamX*columns+teamY+6],1);
			if ( (teamX)>0 && (teamY+7)<columns-1 )
			atomicAdd(&gpuAux[teamX*columns+teamY+7],1);
			if ( (teamX)>0 && (teamY+8)<columns-1 )
			atomicAdd(&gpuAux[teamX*columns+teamY+8],1);
			if ( (teamX)>0 && (teamY+9)<columns-1 )
			atomicAdd(&gpuAux[teamX*columns+teamY+9],1);


	    if ( (teamX+1)<rows-1 && (teamY-8)>0 )
	    atomicAdd(&gpuAux[(teamX+1)*columns+teamY-8],1);
	    if ( (teamX+1)<rows-1 && (teamY-7)>0 )
	    atomicAdd(&gpuAux[(teamX+1)*columns+teamY-7],1);
	    if ( (teamX+1)<rows-1 && (teamY-6)>0 )
	    atomicAdd(&gpuAux[(teamX+1)*columns+teamY-6],1);
	    if ( (teamX+1)<rows-1 && (teamY-5)>0 )
	    atomicAdd(&gpuAux[(teamX+1)*columns+teamY-5],1);
	    if ( (teamX+1)<rows-1 && (teamY-4)>0 )
	    atomicAdd(&gpuAux[(teamX+1)*columns+teamY-4],1);
	    if ( (teamX+1)<rows-1 && (teamY-3)>0 )
	    atomicAdd(&gpuAux[(teamX+1)*columns+teamY-3],1);
	    if ( (teamX+1)<rows-1 && (teamY-2)>0 )
	    atomicAdd(&gpuAux[(teamX+1)*columns+teamY-2],1);
	    if ( (teamX+1)<rows-1 && (teamY-1)>0 )
	    atomicAdd(&gpuAux[(teamX+1)*columns+teamY-1],1);
	    if ( (teamX+1)<rows-1 && (teamY)>0 && teamY<columns-1 )
	    atomicAdd(&gpuAux[(teamX+1)*columns+teamY],1);

	    if ( (teamX+1)<rows-1 && (teamY+1)<columns-1 )
	    atomicAdd(&gpuAux[(teamX+1)*columns+teamY+1],1);
	    if ( (teamX+1)<rows-1 && (teamY+2)<columns-1 )
	    atomicAdd(&gpuAux[(teamX+1)*columns+teamY+2],1);
	    if ( (teamX+1)<rows-1 && (teamY+3)<columns-1 )
	    atomicAdd(&gpuAux[(teamX+1)*columns+teamY+3],1);
	    if ( (teamX+1)<rows-1 && (teamY+4)<columns-1 )
	    atomicAdd(&gpuAux[(teamX+1)*columns+teamY+4],1);
	    if ( (teamX+1)<rows-1 && (teamY+5)<columns-1 )
	    atomicAdd(&gpuAux[(teamX+1)*columns+teamY+5],1);
	    if ( (teamX+1)<rows-1 && (teamY+6)<columns-1 )
	    atomicAdd(&gpuAux[(teamX+1)*columns+teamY+6],1);
	    if ( (teamX+1)<rows-1 && (teamY+7)<columns-1 )
	    atomicAdd(&gpuAux[(teamX+1)*columns+teamY+7],1);
	    if ( (teamX+1)<rows-1 && (teamY+8)<columns-1 )
	    atomicAdd(&gpuAux[(teamX+1)*columns+teamY+8],1);



			if ( (teamX+2)<rows-1 && (teamY-8)>0 )
			atomicAdd(&gpuAux[(teamX+2)*columns+teamY-8],1);
			if ( (teamX+2)<rows-1 && (teamY-7)>0 )
			atomicAdd(&gpuAux[(teamX+2)*columns+teamY-7],1);
			if ( (teamX+2)<rows-1 && (teamY-6)>0 )
			atomicAdd(&gpuAux[(teamX+2)*columns+teamY-6],1);
			if ( (teamX+2)<rows-1 && (teamY-5)>0 )
			atomicAdd(&gpuAux[(teamX+2)*columns+teamY-5],1);
			if ( (teamX+2)<rows-1 && (teamY-4)>0 )
			atomicAdd(&gpuAux[(teamX+2)*columns+teamY-4],1);
			if ( (teamX+2)<rows-1 && (teamY-3)>0 )
			atomicAdd(&gpuAux[(teamX+2)*columns+teamY-3],1);
			if ( (teamX+2)<rows-1 && (teamY-2)>0 )
			atomicAdd(&gpuAux[(teamX+2)*columns+teamY-2],1);
			if ( (teamX+2)<rows-1 && (teamY-1)>0 )
			atomicAdd(&gpuAux[(teamX+2)*columns+teamY-1],1);
			if ( (teamX+2)<rows-1 && (teamY)>0 && teamY<columns-1 )
			atomicAdd(&gpuAux[(teamX+2)*columns+teamY],1);

			if ( (teamX+2)<rows-1 && (teamY+1)<columns-1 )
			atomicAdd(&gpuAux[(teamX+2)*columns+teamY+1],1);
			if ( (teamX+2)<rows-1 && (teamY+2)<columns-1 )
			atomicAdd(&gpuAux[(teamX+2)*columns+teamY+2],1);
			if ( (teamX+2)<rows-1 && (teamY+3)<columns-1 )
			atomicAdd(&gpuAux[(teamX+2)*columns+teamY+3],1);
			if ( (teamX+2)<rows-1 && (teamY+4)<columns-1 )
			atomicAdd(&gpuAux[(teamX+2)*columns+teamY+4],1);
			if ( (teamX+2)<rows-1 && (teamY+5)<columns-1 )
			atomicAdd(&gpuAux[(teamX+2)*columns+teamY+5],1);
			if ( (teamX+2)<rows-1 && (teamY+6)<columns-1 )
			atomicAdd(&gpuAux[(teamX+2)*columns+teamY+6],1);
			if ( (teamX+2)<rows-1 && (teamY+7)<columns-1 )
			atomicAdd(&gpuAux[(teamX+2)*columns+teamY+7],1);
			if ( (teamX+2)<rows-1 && (teamY+8)<columns-1 )
			atomicAdd(&gpuAux[(teamX+2)*columns+teamY+8],1);



			if ( (teamX+3)<rows-1 && (teamY-8)>0 )
			atomicAdd(&gpuAux[(teamX+3)*columns+teamY-8],1);
			if ( (teamX+3)<rows-1 && (teamY-7)>0 )
			atomicAdd(&gpuAux[(teamX+3)*columns+teamY-7],1);
			if ( (teamX+3)<rows-1 && (teamY-6)>0 )
			atomicAdd(&gpuAux[(teamX+3)*columns+teamY-6],1);
			if ( (teamX+3)<rows-1 && (teamY-5)>0 )
			atomicAdd(&gpuAux[(teamX+3)*columns+teamY-5],1);
			if ( (teamX+3)<rows-1 && (teamY-4)>0 )
			atomicAdd(&gpuAux[(teamX+3)*columns+teamY-4],1);
			if ( (teamX+3)<rows-1 && (teamY-3)>0 )
			atomicAdd(&gpuAux[(teamX+3)*columns+teamY-3],1);
			if ( (teamX+3)<rows-1 && (teamY-2)>0 )
			atomicAdd(&gpuAux[(teamX+3)*columns+teamY-2],1);
			if ( (teamX+3)<rows-1 && (teamY-1)>0 )
			atomicAdd(&gpuAux[(teamX+3)*columns+teamY-1],1);
			if ( (teamX+3)<rows-1 && (teamY)>0 && teamY<columns-1 )
			atomicAdd(&gpuAux[(teamX+3)*columns+teamY],1);

			if ( (teamX+3)<rows-1 && (teamY+1)<columns-1 )
			atomicAdd(&gpuAux[(teamX+3)*columns+teamY+1],1);
			if ( (teamX+3)<rows-1 && (teamY+2)<columns-1 )
			atomicAdd(&gpuAux[(teamX+3)*columns+teamY+2],1);
			if ( (teamX+3)<rows-1 && (teamY+3)<columns-1 )
			atomicAdd(&gpuAux[(teamX+3)*columns+teamY+3],1);
			if ( (teamX+3)<rows-1 && (teamY+4)<columns-1 )
			atomicAdd(&gpuAux[(teamX+3)*columns+teamY+4],1);
			if ( (teamX+3)<rows-1 && (teamY+5)<columns-1 )
			atomicAdd(&gpuAux[(teamX+3)*columns+teamY+5],1);
			if ( (teamX+3)<rows-1 && (teamY+6)<columns-1 )
			atomicAdd(&gpuAux[(teamX+3)*columns+teamY+6],1);
			if ( (teamX+3)<rows-1 && (teamY+7)<columns-1 )
			atomicAdd(&gpuAux[(teamX+3)*columns+teamY+7],1);
			if ( (teamX+3)<rows-1 && (teamY+8)<columns-1 )
			atomicAdd(&gpuAux[(teamX+3)*columns+teamY+8],1);



			if ( (teamX+4)<rows-1 && (teamY-8)>0 )
			atomicAdd(&gpuAux[(teamX+4)*columns+teamY-8],1);
			if ( (teamX+4)<rows-1 && (teamY-7)>0 )
			atomicAdd(&gpuAux[(teamX+4)*columns+teamY-7],1);
			if ( (teamX+4)<rows-1 && (teamY-6)>0 )
			atomicAdd(&gpuAux[(teamX+4)*columns+teamY-6],1);
			if ( (teamX+4)<rows-1 && (teamY-5)>0 )
			atomicAdd(&gpuAux[(teamX+4)*columns+teamY-5],1);
			if ( (teamX+4)<rows-1 && (teamY-4)>0 )
			atomicAdd(&gpuAux[(teamX+4)*columns+teamY-4],1);
			if ( (teamX+4)<rows-1 && (teamY-3)>0 )
			atomicAdd(&gpuAux[(teamX+4)*columns+teamY-3],1);
			if ( (teamX+4)<rows-1 && (teamY-2)>0 )
			atomicAdd(&gpuAux[(teamX+4)*columns+teamY-2],1);
			if ( (teamX+4)<rows-1 && (teamY-1)>0 )
			atomicAdd(&gpuAux[(teamX+4)*columns+teamY-1],1);
			if ( (teamX+4)<rows-1 && (teamY)>0 && teamY<columns-1 )
			atomicAdd(&gpuAux[(teamX+4)*columns+teamY],1);

			if ( (teamX+4)<rows-1 && (teamY+1)<columns-1 )
			atomicAdd(&gpuAux[(teamX+4)*columns+teamY+1],1);
			if ( (teamX+4)<rows-1 && (teamY+2)<columns-1 )
			atomicAdd(&gpuAux[(teamX+4)*columns+teamY+2],1);
			if ( (teamX+4)<rows-1 && (teamY+3)<columns-1 )
			atomicAdd(&gpuAux[(teamX+4)*columns+teamY+3],1);
			if ( (teamX+4)<rows-1 && (teamY+4)<columns-1 )
			atomicAdd(&gpuAux[(teamX+4)*columns+teamY+4],1);
			if ( (teamX+4)<rows-1 && (teamY+5)<columns-1 )
			atomicAdd(&gpuAux[(teamX+4)*columns+teamY+5],1);
			if ( (teamX+4)<rows-1 && (teamY+6)<columns-1 )
			atomicAdd(&gpuAux[(teamX+4)*columns+teamY+6],1);
			if ( (teamX+4)<rows-1 && (teamY+7)<columns-1 )
			atomicAdd(&gpuAux[(teamX+4)*columns+teamY+7],1);
			if ( (teamX+4)<rows-1 && (teamY+8)<columns-1 )
			atomicAdd(&gpuAux[(teamX+4)*columns+teamY+8],1);


			if ( (teamX+5)<rows-1 && (teamY-7)>0 )
			atomicAdd(&gpuAux[(teamX+5)*columns+teamY-7],1);
			if ( (teamX+5)<rows-1 && (teamY-6)>0 )
			atomicAdd(&gpuAux[(teamX+5)*columns+teamY-6],1);
			if ( (teamX+5)<rows-1 && (teamY-5)>0 )
			atomicAdd(&gpuAux[(teamX+5)*columns+teamY-5],1);
			if ( (teamX+5)<rows-1 && (teamY-4)>0 )
			atomicAdd(&gpuAux[(teamX+5)*columns+teamY-4],1);
			if ( (teamX+5)<rows-1 && (teamY-3)>0 )
			atomicAdd(&gpuAux[(teamX+5)*columns+teamY-3],1);
			if ( (teamX+5)<rows-1 && (teamY-2)>0 )
			atomicAdd(&gpuAux[(teamX+5)*columns+teamY-2],1);
			if ( (teamX+5)<rows-1 && (teamY-1)>0 )
			atomicAdd(&gpuAux[(teamX+5)*columns+teamY-1],1);
			if ( (teamX+5)<rows-1 && (teamY)>0 && teamY<columns-1 )
			atomicAdd(&gpuAux[(teamX+5)*columns+teamY],1);

			if ( (teamX+5)<rows-1 && (teamY+1)<columns-1 )
			atomicAdd(&gpuAux[(teamX+5)*columns+teamY+1],1);
			if ( (teamX+5)<rows-1 && (teamY+2)<columns-1 )
			atomicAdd(&gpuAux[(teamX+5)*columns+teamY+2],1);
			if ( (teamX+5)<rows-1 && (teamY+3)<columns-1 )
			atomicAdd(&gpuAux[(teamX+5)*columns+teamY+3],1);
			if ( (teamX+5)<rows-1 && (teamY+4)<columns-1 )
			atomicAdd(&gpuAux[(teamX+5)*columns+teamY+4],1);
			if ( (teamX+5)<rows-1 && (teamY+5)<columns-1 )
			atomicAdd(&gpuAux[(teamX+5)*columns+teamY+5],1);
			if ( (teamX+5)<rows-1 && (teamY+6)<columns-1 )
			atomicAdd(&gpuAux[(teamX+5)*columns+teamY+6],1);
			if ( (teamX+5)<rows-1 && (teamY+7)<columns-1 )
			atomicAdd(&gpuAux[(teamX+5)*columns+teamY+7],1);



			if ( (teamX+6)<rows-1 && (teamY-6)>0 )
			atomicAdd(&gpuAux[(teamX+6)*columns+teamY-6],1);
			if ( (teamX+6)<rows-1 && (teamY-5)>0 )
			atomicAdd(&gpuAux[(teamX+6)*columns+teamY-5],1);
			if ( (teamX+6)<rows-1 && (teamY-4)>0 )
			atomicAdd(&gpuAux[(teamX+6)*columns+teamY-4],1);
			if ( (teamX+6)<rows-1 && (teamY-3)>0 )
			atomicAdd(&gpuAux[(teamX+6)*columns+teamY-3],1);
			if ( (teamX+6)<rows-1 && (teamY-2)>0 )
			atomicAdd(&gpuAux[(teamX+6)*columns+teamY-2],1);
			if ( (teamX+6)<rows-1 && (teamY-1)>0 )
			atomicAdd(&gpuAux[(teamX+6)*columns+teamY-1],1);
			if ( (teamX+6)<rows-1 && (teamY)>0 && teamY<columns-1 )
			atomicAdd(&gpuAux[(teamX+6)*columns+teamY],1);

			if ( (teamX+6)<rows-1 && (teamY+1)<columns-1 )
			atomicAdd(&gpuAux[(teamX+6)*columns+teamY+1],1);
			if ( (teamX+6)<rows-1 && (teamY+2)<columns-1 )
			atomicAdd(&gpuAux[(teamX+6)*columns+teamY+2],1);
			if ( (teamX+6)<rows-1 && (teamY+3)<columns-1 )
			atomicAdd(&gpuAux[(teamX+6)*columns+teamY+3],1);
			if ( (teamX+6)<rows-1 && (teamY+4)<columns-1 )
			atomicAdd(&gpuAux[(teamX+6)*columns+teamY+4],1);
			if ( (teamX+6)<rows-1 && (teamY+5)<columns-1 )
			atomicAdd(&gpuAux[(teamX+6)*columns+teamY+5],1);
			if ( (teamX+6)<rows-1 && (teamY+6)<columns-1 )
			atomicAdd(&gpuAux[(teamX+6)*columns+teamY+6],1);


	    if ( (teamX+7)<rows-1 && (teamY-5)>0 )
	    atomicAdd(&gpuAux[(teamX+7)*columns+teamY-5],1);
	    if ( (teamX+7)<rows-1 && (teamY-4)>0 )
	    atomicAdd(&gpuAux[(teamX+7)*columns+teamY-4],1);
	    if ( (teamX+7)<rows-1 && (teamY-3)>0 )
	    atomicAdd(&gpuAux[(teamX+7)*columns+teamY-3],1);
	    if ( (teamX+7)<rows-1 && (teamY-2)>0 )
	    atomicAdd(&gpuAux[(teamX+7)*columns+teamY-2],1);
	    if ( (teamX+7)<rows-1 && (teamY-1)>0 )
	    atomicAdd(&gpuAux[(teamX+7)*columns+teamY-1],1);
	    if ( (teamX+7)<rows-1 && (teamY)>0 && teamY<columns-1 )
	    atomicAdd(&gpuAux[(teamX+7)*columns+teamY],1);

	    if ( (teamX+7)<rows-1 && (teamY+1)<columns-1 )
	    atomicAdd(&gpuAux[(teamX+7)*columns+teamY+1],1);
	    if ( (teamX+7)<rows-1 && (teamY+2)<columns-1 )
	    atomicAdd(&gpuAux[(teamX+7)*columns+teamY+2],1);
	    if ( (teamX+7)<rows-1 && (teamY+3)<columns-1 )
	    atomicAdd(&gpuAux[(teamX+7)*columns+teamY+3],1);
	    if ( (teamX+7)<rows-1 && (teamY+4)<columns-1 )
	    atomicAdd(&gpuAux[(teamX+7)*columns+teamY+4],1);
	    if ( (teamX+7)<rows-1 && (teamY+5)<columns-1 )
	    atomicAdd(&gpuAux[(teamX+7)*columns+teamY+5],1);


	    if ( (teamX+8)<rows-1 && (teamY-4)>0 )
	    atomicAdd(&gpuAux[(teamX+8)*columns+teamY-4],1);
	    if ( (teamX+8)<rows-1 && (teamY-3)>0 )
	    atomicAdd(&gpuAux[(teamX+8)*columns+teamY-3],1);
	    if ( (teamX+8)<rows-1 && (teamY-2)>0 )
	    atomicAdd(&gpuAux[(teamX+8)*columns+teamY-2],1);
	    if ( (teamX+8)<rows-1 && (teamY-1)>0 )
	    atomicAdd(&gpuAux[(teamX+8)*columns+teamY-1],1);
	    if ( (teamX+8)<rows-1 && (teamY)>0 && teamY<columns-1 )
	    atomicAdd(&gpuAux[(teamX+8)*columns+teamY],1);

	    if ( (teamX+8)<rows-1 && (teamY+1)<columns-1 )
	    atomicAdd(&gpuAux[(teamX+8)*columns+teamY+1],1);
	    if ( (teamX+8)<rows-1 && (teamY+2)<columns-1 )
	    atomicAdd(&gpuAux[(teamX+8)*columns+teamY+2],1);
	    if ( (teamX+8)<rows-1 && (teamY+3)<columns-1 )
	    atomicAdd(&gpuAux[(teamX+8)*columns+teamY+3],1);
	    if ( (teamX+8)<rows-1 && (teamY+4)<columns-1 )
	    atomicAdd(&gpuAux[(teamX+8)*columns+teamY+4],1);


	    if ( (teamX+9)<rows-1 && (teamY)>0 && teamY<columns-1 )
	    atomicAdd(&gpuAux[(teamX+9)*columns+teamY],1);





		}

		else{


			if ( (teamX-3)>0 && (teamY)>0 && teamY<columns-1 )
			atomicAdd(&gpuAux[(teamX-3)*columns+teamY],1);




			if ( (teamX-2)>0 && (teamY-2)>0 )
			atomicAdd(&gpuAux[(teamX-2)*columns+teamY-2],1);
			if ( (teamX-2)>0 && (teamY-1)>0 )
			atomicAdd(&gpuAux[(teamX-2)*columns+teamY-1],1);
			if ( (teamX-2)>0 && (teamY)>0 && teamY<columns-1 )
			atomicAdd(&gpuAux[(teamX-2)*columns+teamY],1);

			if ( (teamX-2)>0 && (teamY+1)<columns-1 )
			atomicAdd(&gpuAux[(teamX-2)*columns+teamY+1],1);
			if ( (teamX-2)>0 && (teamY+2)<columns-1 )
			atomicAdd(&gpuAux[(teamX-2)*columns+teamY+2],1);




			if ( (teamX-1)>0 && (teamY-2)>0 )
			atomicAdd(&gpuAux[(teamX-1)*columns+teamY-2],1);
			if ( (teamX-1)>0 && (teamY-1)>0 )
			atomicAdd(&gpuAux[(teamX-1)*columns+teamY-1],1);
			if ( (teamX-1)>0 && (teamY)>0 && teamY<columns-1 )
			atomicAdd(&gpuAux[(teamX-1)*columns+teamY],1);

			if ( (teamX-1)>0 && (teamY+1)<columns-1 )
			atomicAdd(&gpuAux[(teamX-1)*columns+teamY+1],1);
			if ( (teamX-1)>0 && (teamY+2)<columns-1 )
			atomicAdd(&gpuAux[(teamX-1)*columns+teamY+2],1);




			if ( (teamX)>0 && (teamY-3)>0 )
			atomicAdd(&gpuAux[teamX*columns+teamY-3],1);
			if ( (teamX)>0 && (teamY-2)>0 )
			atomicAdd(&gpuAux[teamX*columns+teamY-2],1);
			if ( (teamX)>0 && (teamY-1)>0 )
			atomicAdd(&gpuAux[teamX*columns+teamY-1],1);
			if ( (teamX)>0 && (teamY)>0 && teamY<columns-1 )
			atomicAdd(&gpuAux[teamX*columns+teamY],1);

			if ( (teamX)>0 && (teamY+1)<columns-1 )
			atomicAdd(&gpuAux[teamX*columns+teamY+1],1);
			if ( (teamX)>0 && (teamY+2)<columns-1 )
			atomicAdd(&gpuAux[teamX*columns+teamY+2],1);
			if ( (teamX)>0 && (teamY+3)<columns-1 )
			atomicAdd(&gpuAux[teamX*columns+teamY+3],1);





			if ( (teamX+1)<rows-1 && (teamY-2)>0 )
			atomicAdd(&gpuAux[(teamX+1)*columns+teamY-2],1);
			if ( (teamX+1)<rows-1 && (teamY-1)>0 )
			atomicAdd(&gpuAux[(teamX+1)*columns+teamY-1],1);
			if ( (teamX+1)<rows-1 && (teamY)>0 && teamY<columns-1 )
			atomicAdd(&gpuAux[(teamX+1)*columns+teamY],1);

			if ( (teamX+1)<rows-1 && (teamY+1)<columns-1 )
			atomicAdd(&gpuAux[(teamX+1)*columns+teamY+1],1);
			if ( (teamX+1)<rows-1 && (teamY+2)<columns-1 )
			atomicAdd(&gpuAux[(teamX+1)*columns+teamY+2],1);






			if ( (teamX+2)<rows-1 && (teamY-2)>0 )
			atomicAdd(&gpuAux[(teamX+2)*columns+teamY-2],1);
			if ( (teamX+2)<rows-1 && (teamY-1)>0 )
			atomicAdd(&gpuAux[(teamX+2)*columns+teamY-1],1);
			if ( (teamX+2)<rows-1 && (teamY)>0 && teamY<columns-1 )
			atomicAdd(&gpuAux[(teamX+2)*columns+teamY],1);

			if ( (teamX+2)<rows-1 && (teamY+1)<columns-1 )
			atomicAdd(&gpuAux[(teamX+2)*columns+teamY+1],1);
			if ( (teamX+2)<rows-1 && (teamY+2)<columns-1 )
			atomicAdd(&gpuAux[(teamX+2)*columns+teamY+2],1);





			if ( (teamX+3)<rows-1 && (teamY)>0 && teamY<columns-1 )
			atomicAdd(&gpuAux[(teamX+3)*columns+teamY],1);



		}

}

__global__ void reduce_heat3(float *surface, int *aux,int rows,int columns){

	int idX=threadIdx.y+blockDim.y*blockIdx.y;
	int idY=threadIdx.x+blockDim.x*blockIdx.x;
  //printf("hola\n" );
	if (idX >= rows-1 || idX==0 || idY>= columns-1 || idY==0) return;
	#pragma unroll
	for(unsigned int i=aux[idX*columns+idY];i>0;i--)
		surface[idX*columns+idY]*=0.75;

	aux[idX*columns+idY]=0;
}



__global__ void reduce_kernel(const float* g_idata, float* g_odata, int size)
{
	// Memoria shared
	extern __shared__ float tmp[];

	// Desactivar hilos que excedan los límites del array de entrada
	int gid = threadIdx.x+blockDim.x*blockIdx.x;
   if ( gid >= size ) return;

	// Cargar dato en memoria shared
	int tid = threadIdx.x;
	tmp[ tid ] = g_idata[ gid ];
//printf("entrada  %f glob red %f\n",g_idata[gid],tmp[tid]);
	// Asegurarse que todos los warps del bloque han cargado los datos
	__syncthreads();

	// Generalización: El único bloque del último nivel puede tener menos datos para reducir
	int mysize = blockDim.x;
	if ( gridDim.x==1 )
		mysize = size;

	// Hacemos la reducción en memoria shared
	#pragma unroll
	for(unsigned int s = mysize/2; s >0; s /= 2) {
		// Comprobamos si el hilo actual es activo para esta iteración

		if (tid<s) {
			// Hacemos la reducción sumando los dos elementos que le tocan a este hilo
			if(tmp[tid+s]>tmp[tid])
			tmp[tid]  =tmp[tid+s];
		}
		__syncthreads();
	}

	// El hilo 0 de cada bloque escribe el resultado final de la reducción
	// en la memoria global del dispositivo pasada por parámetro (g_odata[])
	if (tid == 0){
		g_odata[blockIdx.x] = tmp[tid];

	}
}
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

	/* 2. Select GPU and start global timer */
	cudaSetDevice(0);
	cudaDeviceSynchronize();
	double ttotal = cp_Wtime();

/*
 *
 * START HERE: DO NOT CHANGE THE CODE ABOVE THIS POINT
 *
 */


	float *gpuSurface, *gpuSurfaceCopy, *gpuResiduals;
	int *gpuAux;
	FocalPoint *gpuFocal;
	Team *gpuTeam;
//	double time1,time2;
	int nearestUpperPow2 = pow(2,ceil(log2((double) rows*columns)));

	cudaMalloc((void **)&gpuSurface,sizeof(float)*rows*columns);

	cudaMalloc((void **)&gpuAux,sizeof(int)*rows*columns);


	cudaMalloc((void **) &gpuSurfaceCopy,sizeof(float)*rows*columns);

	cudaMalloc((void **) &gpuResiduals,sizeof(float)*nearestUpperPow2);


cudaMalloc((void **) &gpuTeam,sizeof(Team)*num_teams);

	cudaMemcpy(gpuTeam,teams,sizeof(Team)*num_teams,cudaMemcpyHostToDevice);

	cudaMalloc((void **) &gpuFocal,sizeof(FocalPoint)*num_focal);
	cudaMemcpy(gpuFocal,focal,sizeof(FocalPoint)*num_focal,cudaMemcpyHostToDevice);

	int tamBlockX= 128;
	int tamBlockY= 1;
	int tamGridX, tamGridY;
	int tamBlockTeams=224;
	int tamGridTeams;
	int tamBlockFocal=224;
	int tamGridFocal;

	tamGridTeams= num_teams/tamBlockTeams;
	if (num_teams%tamBlockTeams!=0) tamGridTeams++;

	tamGridFocal= num_focal/tamBlockFocal;
	if (num_focal%tamBlockFocal!=0) tamGridFocal++;

	tamGridX= columns/tamBlockX;
	if (columns%tamBlockX!=0) tamGridX++;
	tamGridY= rows/tamBlockY;
	if (rows%tamBlockY!=0) tamGridY++;

	dim3 blockSize(tamBlockX,tamBlockY);
	dim3 gridSize(tamGridX,tamGridY);
	#ifdef DEBUG
	printf("tamGx %d tamGy %d\n",tamGridX,tamGridY);
	#endif

	init<<<blockSize,gridSize>>>(gpuSurface,rows,columns);

	//CUDA_CHECK();
	init<<<blockSize,gridSize>>>(gpuSurfaceCopy,rows,columns);

	//CUDA_CHECK();
	/* 3. Initialize surface */
	/*for( i=0; i<rows; i++ )
		for( j=0; j<columns; j++ )
			accessMat( surface, i, j ) = 0.0;

	/* 4. Simulation */
	int *gpuNum_deactivated;
	//gpuNum_deactivated[0]=0;
	cudaMallocHost((void**) &gpuNum_deactivated,sizeof(int));
	int iter;
	int flag_stability = 0;
	//int first_activation = 0;
	//int *gpuFirstActivation;

	//cudaMallocHost((void**) &gpuFirstActivation,sizeof(int));
	//check_first_activation<<<tamGridFocal,tamBlockFocal>>>(gpuFocal,num_focal); hace falta reduccion


	//get_first_activation<<<tamGridFocal,tamBlockFocal>>>(gpuFocal,num_focal,gpuFirstActivation);
	#pragma unroll
	for( iter=0; iter<max_iter && ! flag_stability; iter++ ) {
		//printf("iter %d\n",iter);
		/* 4.1. Activate focal points */
		//printf("num %d\n",gpuNum_deactivated[0] );
		//cudaMemcpy(gpuNum_deactivated,&num_deactivated,sizeof(int),cudaMemcpyHostToDevice);
		//printf("num %d\n",num_deactivated);
		if(gpuNum_deactivated[0]<num_focal){
		activate_focal<<<tamGridFocal,tamBlockFocal>>>(gpuFocal,num_focal,gpuNum_deactivated,iter);
			cudaDeviceSynchronize();
			//cudaMemcpyAsync(&num_deactivated,gpuNum_deactivated,sizeof(int),cudaMemcpyDeviceToHost,0);
		}
		//printf("num %d",num_deactivated);
		//if(!first_activation) continue;
		/* 4.2. Propagate heat (10 steps per each team movement) */
		float global_residual;
		int step;

		//cudaMemcpy(surfaceCopy,gpuSurfaceCopy,sizeof(float)*rows*columns,cudaMemcpyDeviceToHost);
		#pragma unroll
		for( step=0; step<10; step++ )	{
			/* 4.2.1. Update heat on active focal points */
				//if(gpuNum_deactivated[0]<num_focal)
				update_heat<<<tamGridFocal,tamBlockFocal>>>(gpuSurface,gpuFocal,columns,num_focal);

				//CUDA_CHECK();
				//accessMat( surface, x, y ) = focal[i].heat;



			/* 4.2.2. Copy values of the surface in ancillary structure (Skip borders) */
			//copy_surface<<<gridSize,blockSize>>>(gpuSurface,gpuSurfaceCopy,rows,columns);
		//	error=cudaGetLastError();
		//	if(error!= cudaSuccess)
		//		printf("%s\n",cudaGetErrorString(error));
			float *aux=gpuSurface;
			gpuSurface=gpuSurfaceCopy;
			gpuSurfaceCopy=aux;
		//CUDA_CHECK();
			/*for( i=1; i<rows-1; i++ )
				for( j=1; j<columns-1; j++ )
					accessMat( surfaceCopy, i, j ) = accessMat( surface, i, j );

			/* 4.2.3. Update surface values (skip borders) */
			update_surface<<<gridSize,blockSize>>>(gpuSurface,gpuSurfaceCopy,rows,columns);

			//CUDA_CHECK();
			/*for( i=1; i<rows-1; i++ )
				for( j=1; j<columns-1; j++ )
					accessMat( surface, i, j ) = (
						accessMat( surfaceCopy, i-1, j ) +
						accessMat( surfaceCopy, i+1, j ) +
						accessMat( surfaceCopy, i, j-1 ) +
						accessMat( surfaceCopy, i, j+1 ) ) / 4;
			/* 4.2.4. Compute the maximum residual difference (absolute value) */

			if(step==0 && gpuNum_deactivated[0]==num_focal){
				//time1=cp_Wtime();
				//init<<<blockSize,gridSize>>>(gpuResiduals,rows,columns);
				compute_residual<<<gridSize,blockSize>>>(gpuSurface,gpuSurfaceCopy,rows,columns,gpuResiduals);

				//int numValues = nearestUpperPow2;
				int redSize = nearestUpperPow2;
				int blockSizeR = (1024);
				int sharedMemorySize = blockSizeR * sizeof(float);
				while ( redSize > 1 )
				{
					int baseNumBlocks = redSize/blockSizeR;

					int additionalBlock;
					if(redSize%blockSizeR==0)
						additionalBlock = 0;
					else
						additionalBlock = 1;

					int numBlocks = baseNumBlocks + additionalBlock;
					//printf("numB %d size %d\n",numBlocks,redSize);
					//if(numBlocks==1) exit(0);
					reduce_kernel<<< numBlocks, blockSizeR, sharedMemorySize >>>(gpuResiduals, gpuResiduals, redSize);
					redSize = numBlocks;
				}
				cudaMemcpyAsync(&global_residual, gpuResiduals, sizeof(float), cudaMemcpyDeviceToHost,0);

				//printf("glob %f\n",global_residual);

				//	printf("reesiduo %f\n",global_residual);
				//time2+=cp_Wtime()-time1;
			}

}

		/* If the global residual is lower than THRESHOLD, we have reached enough stability, stop simulation at the end of this iteration */

		/* 4.3. Move teams */
		if(gpuNum_deactivated[0]<num_focal){

			move_teams<<<tamGridTeams,tamBlockTeams>>>(gpuTeam,gpuFocal,num_teams,num_focal);

		}
		/* 4.4. Team actions */
		//cudaMemcpy(surface,gpuSurface,sizeof(float)*rows*columns,cudaMemcpyDeviceToHost);
		//initInt<<<gridSize,blockSize>>>()
		compute_heat_reduction<<<tamGridTeams,tamBlockTeams>>>(gpuTeam,gpuAux,num_teams,rows,columns);

		#ifdef UNROLL
		int *aux;
		aux = (int *)malloc( sizeof(int) * (size_t)rows * (size_t)columns );

		cudaMemcpy(aux,gpuAux,sizeof(int)*rows*columns,cudaMemcpyDeviceToHost);
		for( i=0;i<rows;i++){

		for( j=0;j<columns;j++)
			printf("%d ",aux[i*columns+j]);
			printf("\n" );
		}
		exit(0);
		#endif

		reduce_heat3<<<gridSize,blockSize>>>(gpuSurface,gpuAux,rows,columns);


		#ifdef DEBUG
				/* 4.5. DEBUG: Print the current state of the simulation at the end of each iteration */
				cudaMemcpy(teams,gpuTeam,sizeof(Team)*num_teams,cudaMemcpyDeviceToHost);
				cudaMemcpy(surface,gpuSurface,sizeof(float)*rows*columns,cudaMemcpyDeviceToHost);



				print_status( iter, rows, columns, surface, num_teams, teams, num_focal, focal, global_residual );
		#endif // DEBUG

	if( gpuNum_deactivated[0] == num_focal && global_residual < THRESHOLD ) flag_stability = 1;
	}
	cudaMemcpy(surface,gpuSurface,sizeof(float)*rows*columns,cudaMemcpyDeviceToHost);

		//cudaMemcpy(focal,gpuFocal,sizeof(FocalPoint)*num_focal,cudaMemcpyDeviceToHost);
//cudaFree(gpuSurface);
//cudaFree(gpuSurfaceCopy);
//cudaFree(gpuTeam);
//cudaFree(gpuFocal);
//printf("time1 %f\n",time2);
/*
 *
 * STOP HERE: DO NOT CHANGE THE CODE BELOW THIS POINT
 *
 */

	/* 5. Stop global time */
	cudaDeviceSynchronize();
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
