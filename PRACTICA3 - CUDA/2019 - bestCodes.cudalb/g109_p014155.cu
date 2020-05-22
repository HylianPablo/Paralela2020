// User: g109@88.4.56.57 
// ExecutionRequest[P:'AniviaLaCriofenix.cu',P:1,T:1,args:'',q:'cudalb'] 
// May 16 2019 19:42:55
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

template <unsigned int dimblock>
__device__ void reducirWarp(volatile float *datos,int tid) {

        if(dimblock>=64) datos[tid]=fmax(datos[tid],datos[tid+32]);
        if(dimblock>=32) datos[tid]=fmax(datos[tid],datos[tid+16]);
        if(dimblock>=16) datos[tid]=fmax(datos[tid],datos[tid+8]);
        if(dimblock>=8) datos[tid]=fmax(datos[tid],datos[tid+4]);
        if(dimblock>=4) datos[tid]=fmax(datos[tid],datos[tid+2]);
        if(dimblock>=2) datos[tid]=fmax(datos[tid],datos[tid+1]);
}

//Funcion para inicializar surfaceG y surfaceCopyG.
__global__ void inicializarSurface( float *surfaceG, float *surfaceCopyG, int rows, int columns ) {

        int globalid=(threadIdx.x+blockIdx.x*blockDim.x)+(threadIdx.y+blockIdx.y*blockDim.y)*blockDim.x*gridDim.x;
        int total=rows*columns;
        if(globalid<total){
                surfaceG[globalid]=0.0f;
                surfaceCopyG[globalid]=0.0f;
        }
}

//Funcion
__global__ void activarPuntosFocales( FocalPoint *focalG, int *nd, int num_focal, int iter ) {

        int globalid=(threadIdx.x+blockIdx.x*blockDim.x)+(threadIdx.y+blockIdx.y*blockDim.y)*blockDim.x*gridDim.x;
        if(globalid<num_focal){
                nd[0] = 0;
                if ( focalG[globalid].start == iter ) {
                        focalG[globalid].active = 1;
                }
                if ( focalG[globalid].active == 2 ){
                        atomicAdd(nd,1);
                }
        }
}

//Funcion para actualizar el calor en los puntos focales activos. (x,y):coordenadas, h:calor.
__global__ void actualizarCalor( float *surfaceG, int x, int y, int h, int columns ) {

        int globalid=(threadIdx.x+blockIdx.x*blockDim.x)+(threadIdx.y+blockIdx.y*blockDim.y)*blockDim.x*gridDim.x;
        int pos=x*columns+y;
        if (pos==globalid){
                accessMat(surfaceG,x,y)=h;
        }
}

//no se usa.
__global__ void actualizarCalor2( float *surfaceG, FocalPoint *focalG, int rows, int columns, int num_focal ) {

        int globalid=(threadIdx.x+blockIdx.x*blockDim.x)+(threadIdx.y+blockIdx.y*blockDim.y)*blockDim.x*gridDim.x;
        if(globalid<num_focal){
               // if ( focalG[globalid].active != 1 ){
               // }
               // else{
               if ( focalG[globalid].active == 1 ){
                       // int x = focalG[globalid].x;
                       // int y = focalG[globalid].y;
                       // accessMat( surfaceG, x, y ) = focalG[globalid].heat;
                       accessMat( surfaceG, focalG[globalid].x, focalG[globalid].y ) = focalG[globalid].heat;
                }
        }
}

//Funcion para copiar los valores de surfaceG a surfaceCopyG.
__global__ void copiarSurface( float *surfaceG, float *surfaceCopyG, int rows, int columns ) {

        int globalid=(threadIdx.x+blockIdx.x*blockDim.x)+(threadIdx.y+blockIdx.y*blockDim.y)*blockDim.x*gridDim.x;
        int total=rows*columns;
        if(globalid<total){
                int x=globalid/columns;
                int y=globalid%columns;
                if( (x==0) || (x==rows-1) || (y==0) || (y==columns-1)){
                }
                else{
                        surfaceCopyG[globalid]=surfaceG[globalid];
                }
        }
}

//Funcion para actualizar los valores del surfaceG
__global__ void actualizarSurface( float *surfaceG, float *surfaceCopyG,int rows, int columns ) {

        int globalid=(threadIdx.x+blockIdx.x*blockDim.x)+(threadIdx.y+blockIdx.y*blockDim.y)*blockDim.x*gridDim.x;
        int total=rows*columns;
        if(globalid<total){  
                //if(globalid>=columns){
                    int x=globalid/columns;
                    int y=globalid%columns;
                    if( (x==0) || (x==rows-1) || (y==0) || (y==columns-1)){
                    }
                    else{
                            accessMat( surfaceG, x, y ) = (
                                accessMat( surfaceCopyG, x-1, y ) + 
                                accessMat( surfaceCopyG, x+1, y ) +
                                accessMat( surfaceCopyG, x, y-1 ) + 
                                accessMat( surfaceCopyG, x, y+1 ) ) / 4;
                    }
                //}
        }
}
/*
__device__ __forceinline__ float atomicMaxF (float * addr, float value) {
    float old;
    old = (value >= 0) ? __float_as_int(atomicMax((int *)addr, __float_as_int(value))) :
         __uint_as_float(atomicMin((unsigned int *)addr, __float_as_uint(value)));

    return old;
}

// Add “val” to “*data”. Return old value.
float atomicMF(float *data, float val){
float old, newval, curr = *data;
do {
	// Generate new value from current data
	old = curr;
	//newval = curr + val;
	newval = fmaxf(curr + val);
	// Attempt to swap old <-> new.
	curr = atomicCAS(data, old, newval);
	// Repeat if value has changed in the meantime.
} while(curr != old);
return old;
}

__device__ float atomicCAS_f32(float *p, float cmp, float val) {
	return __int_as_float(atomicCAS((int *) p, __float_as_int(cmp), __float_as_int(val)));
}
*/

//Funcion para calcular el maximo. Devuelve un array con el maximo de cada bloque.
template <unsigned int dimblock>
__global__ void maximo2( float *surfaceG, float *surfaceCopyG, float *gr, int rows, int columns ) {

        int globalid=(threadIdx.x+blockIdx.x*blockDim.x)+(threadIdx.y+blockIdx.y*blockDim.y)*blockDim.x*gridDim.x;
        int total=rows*columns;
       // int borde=globalid%columns;
        int bloque=blockIdx.x+blockIdx.y*gridDim.x;
        extern __shared__ float datos[];

        //Primer nivel de reduccion, lee de la memoria global global y escribe en la memoria compartida.
        int tid = threadIdx.y*blockDim.x+threadIdx.x;
         if(globalid<total){
              //  if( (globalid<columns) || (globalid>total-columns) || (borde==0) || (borde==columns-1) ){
              //          datos[tid]=0;
              //  }
               // else{
                        if (surfaceG[globalid]-surfaceCopyG[globalid] >= 0){
                                if (surfaceG[globalid+columns]-surfaceCopyG[globalid+columns] >= 0){
                                        datos[tid] = fmax((surfaceG[globalid]-surfaceCopyG[globalid]),(surfaceG[globalid+columns]-surfaceCopyG[globalid+columns]));
                                }
                                else{
                                        datos[tid] = fmax((surfaceG[globalid]-surfaceCopyG[globalid]),((surfaceG[globalid+columns]-surfaceCopyG[globalid+columns])*-1.0f));
                                }
                        }
                        else{
                                if (surfaceG[globalid+columns]-surfaceCopyG[globalid+columns] >= 0){
                                        datos[tid] = fmax(((surfaceG[globalid]-surfaceCopyG[globalid])*-1.0f),(surfaceG[globalid+columns]-surfaceCopyG[globalid+columns]));
                                }
                                else{
                                        datos[tid] = fmax(((surfaceG[globalid]-surfaceCopyG[globalid])*-1.0f),((surfaceG[globalid+columns]-surfaceCopyG[globalid+columns])*-1.0f));
                                }
                        }
               // }
        }
       // else{
        //        datos[tid]=0;
       // }
        __syncthreads();

        //Hacer la reducción en memoria compartida.
        for(unsigned int s=(blockDim.x*blockDim.y)/2; s>32; s>>=1) {
                if (tid < s) {
                        datos[tid]=fmax(datos[tid],datos[tid+s]);
                }
               // __syncthreads();  //+++++++++++++++++++++++++++++++++++++++++
        }
	if(dimblock>=1024){
                if(tid<512){
                        datos[tid]=fmax(datos[tid],datos[tid+512]);
                        //__syncthreads();
                }
        }
        if(dimblock>=512){
                if(tid<256){
                        datos[tid]=fmax(datos[tid],datos[tid+256]);
                        //__syncthreads();
                }
        }
        if(dimblock>=256){
                if(tid<128){
                        datos[tid]=fmax(datos[tid],datos[tid+128]);
                        //__syncthreads();
                }
        }
        if(dimblock>=128){
                if(tid<64){
                        datos[tid]=fmax(datos[tid],datos[tid+64]);
                       // __syncthreads();
                }
        }

        if (tid < 32){
                reducirWarp<dimblock>(datos, tid);
        }
        __syncthreads();
        //Escribir el resultado de este hilo a la memoria global.
        if(tid == 0){
               	gr[bloque] = datos[0];
                //__threadfence();
                //__threadfence_block();
                //atomicExch(&gr[0],fmaxf(gr[0],datos[0]));
                //atomicMF(gr,datos[0]);
        }
}

//Funcion para reducir el calor. (i,j):Coordenadas del equipo antincendios. radius:Radio de accion.
__global__ void reducirCalor2( float *surfaceG, int i, int j, int rows, int columns, int radius ) {

        int globalid=(threadIdx.x+blockIdx.x*blockDim.x)+(threadIdx.y+blockIdx.y*blockDim.y)*blockDim.x*gridDim.x;
        //int y=(threadIdx.x+blockIdx.x*blockDim.x);
        //int x=(globalid-y)/columns;
        int x,y;
        int total=rows*columns;
        if(globalid<total){
                x=globalid/columns;
                y=globalid%columns;
        }
	if(globalid<total){
                if ( x<1 || x>=rows-1 || y<1 || y>=columns-1 ){
                }
                else{
                float dx = i - x;
                float dy = j - y;
                float distance = sqrtf( dx*dx + dy*dy );
                if ( distance <= radius ) {
                        int pos=x*columns+y;
                        if(pos==globalid){
                                accessMat( surfaceG, x, y ) = accessMat( surfaceG, x, y ) * 0.75f;/*( 1 - 0.25 );*/ // Team efficiency factor
                        }
                }
                }
        }

}

//Funcion para reducir el calor. (i,j):Coordenadas del equipo antincendios. radius:Radio de accion.
__global__ void reducirCalor3( float *surfaceG, Team *teamsG, FocalPoint *focalG, int rows, int columns, int num_teams ) {

        int globalid=(threadIdx.x+blockIdx.x*blockDim.x)+(threadIdx.y+blockIdx.y*blockDim.y)*blockDim.x*gridDim.x;
        int bloque=blockIdx.x+blockIdx.y*gridDim.x;
        int bloques=gridDim.x*gridDim.y;
        int tid = threadIdx.y*blockDim.x+threadIdx.x;
        int x,y,t;
        int total=rows*columns;
        if(globalid<total){
                x=globalid/columns;
                y=globalid%columns;
                t=globalid%num_teams;

                if(globalid<num_teams){
                        int target = teamsG[t].target;
                        if ( target != -1 && focalG[target].x == teamsG[t].x && focalG[target].y == teamsG[t].y && focalG[target].active == 1 )
                                focalG[target].active = 2;
                }
                //if(bloques>=num_teams){
                        if(bloque<num_teams && tid==0){
                                t=bloque;
                                //for(t=0;t<num_teams;t++){
                                //if(globalid==0){
                                int radius;
                                // Influence area of fixed radius depending on type
                                if ( teamsG[t].type == 1 ) radius = 3;
                                else radius = 9;
                                for( int i=teamsG[t].x-radius; i<=teamsG[t].x+radius; i++ ) {
                                        for( int j=teamsG[t].y-radius; j<=teamsG[t].y+radius; j++ ) {
                                                if ( i<1 || i>=rows-1 || j<1 || j>=columns-1 ) continue; // Out of the heated surface
                                                float dx = teamsG[t].x - i;
                                                float dy = teamsG[t].y - j;
                                                float distance = sqrtf( dx*dx + dy*dy );
                                                if ( distance <= radius ) {
                                                        int pos=i*columns+j;
                                                        //float a = ((surfaceG[pos]*-1.0f)+(surfaceG[pos]*0.75f));
                                                        accessMat( surfaceG, i, j ) = accessMat( surfaceG, i, j ) * ( 1 - 0.25 ); // Team efficiency factor
                                                        //atomicExch(&surfaceG[pos],surfaceG[pos]*0.75f);
                                                        //atomicAdd(&surfaceG[pos],((surfaceG[pos]*-1)+(surfaceG[pos]-(surfaceG[pos]*0.25))));
                                                        //atomicAdd(&surfaceG[pos],((surfaceG[pos]*-1.0f)+(surfaceG[pos]*0.75f)));
                                                        //atomicAdd(&surfaceG[pos],a);
                                                }
                                        }
                                }
                        }
	}
}


__device__ float atomicMul(float* address, float val) { 
  int* address_as_int = (int*)address; 
  int old = *address_as_int, assumed; 
  do { 
    assumed = old; 
    old = atomicCAS(address_as_int, assumed, __float_as_int(val * __int_as_float(assumed))); 
 } while (assumed != old); 
 return __int_as_float(old);
}

__global__ void reducirCalor4( float *surfaceG, Team *teamsG, FocalPoint *focalG, int rows, int columns, int num_teams ) {

        int globalid=(threadIdx.x+blockIdx.x*blockDim.x)+(threadIdx.y+blockIdx.y*blockDim.y)*blockDim.x*gridDim.x;
	if(globalid<num_teams){
		int t=globalid;
 			// 4.4. Team actions
                        // 4.4.1. Deactivate the target focal point when it is reached //
                        int target = teamsG[t].target;
                        if ( target != -1 && focalG[target].x == teamsG[t].x && focalG[target].y == teamsG[t].y
                                && focalG[target].active == 1 )
                                focalG[target].active = 2;

                        // 4.4.2. Reduce heat in a circle around the team //
                        int radius;
                        // Influence area of fixed radius depending on type
                        if ( teamsG[t].type == 1 ) radius = RADIUS_TYPE_1;
                        else radius = RADIUS_TYPE_2_3;
                        for( int i=teamsG[t].x-radius; i<=teamsG[t].x+radius; i++ ) {
                                for( int j=teamsG[t].y-radius; j<=teamsG[t].y+radius; j++ ) {
                                        if ( i<1 || i>=rows-1 || j<1 || j>=columns-1 ) continue; // Out of the heated surface
                                        float dx = teamsG[t].x - i;
                                        float dy = teamsG[t].y - j;
                                        float distance = sqrtf( dx*dx + dy*dy );
                                        if ( distance <= radius ) {
						int pos=i*columns+j;
                                                //accessMat( surfaceG, i, j ) = accessMat( surfaceG, i, j ) * ( 1 - 0.25 ); // Team efficiency factor
						atomicMul(&surfaceG[pos],0.75f);
                                        }
                                }
                        }
                }
}

__global__ void reducirCalor5( float *surfaceG, Team *teamsG, FocalPoint *focalG, int rows, int columns, int num_teams ) {

    int globalid=(threadIdx.x+blockIdx.x*blockDim.x)+(threadIdx.y+blockIdx.y*blockDim.y)*blockDim.x*gridDim.x;
    int bloque=blockIdx.x+blockIdx.y*gridDim.x;
    int tid = threadIdx.y*blockDim.x+threadIdx.x;
	int t=bloque;
	int i,j,pos,x,y;
	if(bloque<num_teams){
			// 4.4. Team actions
            // 4.4.1. Deactivate the target focal point when it is reached //
            //int target = teamsG[t].target;
            //if ( target != -1 && focalG[target].x == teamsG[t].x && focalG[target].y == teamsG[t].y )
            //        focalG[target].active = 2;

            // 4.4.2. Reduce heat in a circle around the team //
            int radius;
            // Influence area of fixed radius depending on type
            //if ( teamsG[t].type == 1 ) radius = RADIUS_TYPE_1;
            //else radius = RADIUS_TYPE_2_3;

    	//if(radius==RADIUS_TYPE_1){
        if ( teamsG[t].type == 1 ){
                radius = RADIUS_TYPE_1;
    			int newradius=radius*2+1;

    			if(tid<newradius*newradius){
    				x=tid/newradius;
    				y=tid%newradius;
    				i=teamsG[t].x+(x-radius);
    				j=teamsG[t].y+(y-radius);

                    if ( i<1 || i>=rows-1 || j<1 || j>=columns-1 ){
    				}// Out of the heated surface
    				else{
                        float dx = teamsG[t].x - i;
                        float dy = teamsG[t].y - j;
                        float distance = dx*dx + dy*dy ;
                        if ( distance <= radius*radius ) {
        					pos=i*columns+j;
        					atomicMul(&surfaceG[pos],0.75f);
                        }
    				}
    			}
        }
    	else{
                radius = RADIUS_TYPE_2_3;
    			int newradius=radius*2+1;
                int aux = (newradius*newradius)/2;

    			//if(tid<(newradius*newradius)/2){
                if(tid<aux){
    				x=tid/newradius;
    				y=tid%newradius;
    				i=teamsG[t].x+(x-radius);
    				j=teamsG[t].y+(y-radius);

                    if ( i<1 || i>=rows-1 || j<1 || j>=columns-1 ){
    				}// Out of the heated surface
    				else{
                        float dx = teamsG[t].x - i;
                        float dy = teamsG[t].y - j;
                        float distance = dx*dx + dy*dy ;
                        if ( distance <= radius*radius ) {
        					pos=i*columns+j;
        					atomicMul(&surfaceG[pos],0.75f);
                        }
    				}

    				//tid=tid+((newradius*newradius)/2)+1;
                    tid=tid+aux+1;
    				x=tid/newradius;
    				y=tid%newradius;
    				i=teamsG[t].x+(x-radius);
    				j=teamsG[t].y+(y-radius);
                    if ( i<1 || i>=rows-1 || j<1 || j>=columns-1 ){
    				}// Out of the heated surface
    				else{
                        float dx = teamsG[t].x - i;
                        float dy = teamsG[t].y - j;
                        float distance = dx*dx + dy*dy ;
                        if ( distance <= radius*radius ) {
        					pos=i*columns+j;
        					atomicMul(&surfaceG[pos],0.75f);
                        }
    				}

    			}
    			//else if(tid==(int)(newradius*newradius)/2){
                else if(tid==aux){
    				x=tid/newradius;
    				y=tid%newradius;
    				i=teamsG[t].x+(x-radius);
    				j=teamsG[t].y+(y-radius);
                    if ( i<1 || i>=rows-1 || j<1 || j>=columns-1 ){
    				}// Out of the heated surface
    				else{
                        float dx = teamsG[t].x - i;
                        float dy = teamsG[t].y - j;
                        float distance = dx*dx + dy*dy ;
                        if ( distance <= radius*radius ) {
        					pos=i*columns+j;
        					atomicMul(&surfaceG[pos],0.75f);
                        }
    				}
    			}
    	}
	}
}
/*
__global__ void reducirCalor6( float *surfaceG, Team *teamsG, FocalPoint *focalG, int rows, int columns, int num_teams, int *r ) {

        int globalid=(threadIdx.x+blockIdx.x*blockDim.x)+(threadIdx.y+blockIdx.y*blockDim.y)*blockDim.x*gridDim.x;
        int bloque=blockIdx.x+blockIdx.y*gridDim.x;
        int tid = threadIdx.y*blockDim.x+threadIdx.x;
	int t=bloque;
	int i,j,pos,x,y;
	if(globalid<num_teams){
			// 4.4. Team actions
                        // 4.4.1. Deactivate the target focal point when it is reached //
                        int target = teamsG[t].target;
                        if ( target != -1 && focalG[target].x == teamsG[t].x && focalG[target].y == teamsG[t].y
                                && focalG[target].active == 1 )
                                focalG[target].active = 2;
	}
                        // 4.4.2. Reduce heat in a circle around the team //
                        int radius;
                        // Influence area of fixed radius depending on type
                        if ( teamsG[t].type == 1 ) radius = RADIUS_TYPE_1;
                        else radius = RADIUS_TYPE_2_3;
	if(globalid>=3){
			int newradius=radius*2+1;
			if(tid<newradius*newradius){
				x=tid/newradius;
				y=tid%newradius;
				i=teamsG[t].x+(x-radius);
				j=teamsG[t].y+(y-radius);
                                if ( i<1 || i>=rows-1 || j<1 || j>=columns-1 ){
				}// Out of the heated surface
				else{
                                float dx = teamsG[t].x - i;
                                float dy = teamsG[t].y - j;
                                float distance = dx*dx + dy*dy ;
                                if ( distance <= radius*radius ) {
					pos=i*columns+j;
					atomicMul(&surfaceG[pos],0.75f);
                                }
				}
		}	
	}
                
}
*/

//Funcion para mover equipos.
__global__ void moverEquipos( Team *teamsG, FocalPoint *focalG, int rows, int columns, int num_teams, int num_focal ) {
	int globalid=(threadIdx.x+blockIdx.x*blockDim.x)+(threadIdx.y+blockIdx.y*blockDim.y)*blockDim.x*gridDim.x;
        if(globalid<num_teams){
                /* 4.3. Move teams */
                /* 4.3.1. Choose nearest focal point */
                float distance = FLT_MAX;
                int target = -1;
                for( int j=0; j<num_focal; j++ ) {
                        if ( focalG[j].active != 1 ) continue; // Skip non-active focal points
                        float dx = focalG[j].x - teamsG[globalid].x;
                        float dy = focalG[j].y - teamsG[globalid].y;
                        float local_distance = sqrtf( dx*dx + dy*dy );
                        if ( local_distance < distance ) {
                                distance = local_distance;
                                target = j;
                        }
                }
                /* 4.3.2. Annotate target for the next stage */
                teamsG[globalid].target = target;

                /* 4.3.3. No active focal point to choose, no movement */
                if ( target != -1 ){
                        /* 4.3.4. Move in the focal point direction */
                        if ( teamsG[globalid].type == 1 ) {
                                // Type 1: Can move in diagonal
                                if ( focalG[target].x < teamsG[globalid].x ) teamsG[globalid].x--;
                                if ( focalG[target].x > teamsG[globalid].x ) teamsG[globalid].x++;
                                if ( focalG[target].y < teamsG[globalid].y ) teamsG[globalid].y--;
                                if ( focalG[target].y > teamsG[globalid].y ) teamsG[globalid].y++;
                        }
                        else if ( teamsG[globalid].type == 2 ) {
                                // Type 2: First in horizontal direction, then in vertical direction
                                if ( focalG[target].y < teamsG[globalid].y ) teamsG[globalid].y--;
                                else if ( focalG[target].y > teamsG[globalid].y ) teamsG[globalid].y++;
                                else if ( focalG[target].x < teamsG[globalid].x ) teamsG[globalid].x--;
                                else if ( focalG[target].x > teamsG[globalid].x ) teamsG[globalid].x++;
                        }
                        else {
                                // Type 3: First in vertical direction, then in horizontal direction
                                if ( focalG[target].x < teamsG[globalid].x ) teamsG[globalid].x--;
                                else if ( focalG[target].x > teamsG[globalid].x ) teamsG[globalid].x++;
                                else if ( focalG[target].y < teamsG[globalid].y ) teamsG[globalid].y--;
                                else if ( focalG[target].y > teamsG[globalid].y ) teamsG[globalid].y++;
                        }
                }
        }
}

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
	float *surfaceG;  //surface en la GPU.
        float *surfaceCopyG;  //surfaceCopy en la GPU.
        float *surface3;  //surfaceCopy en la GPU.
        float *gr;  //global_residual en la GPU.
        int *nd; 
//	int *r = (int *)malloc( sizeof(int) * (num_teams+1) );
//	int *rG;
        Team *teamsG;
        FocalPoint *focalG;
        cudaError_t error;

        //Calculo de los tamaños de bloque y de grid.
        int tbx=32;
        int tby=8;
        int dimblock=tbx*tby;
        int tgx,tgy;

        if (rows%tbx==0){
                tgx=rows/tbx;
        }else{
                tgx=rows/tbx+1;
        }
        if (columns%tby==0){
                tgy=columns/tby;
        }else{
                tgy=columns/tby+1;
        }

        //printf("%d %d %d %d\n",tbx,tby,tgx,tgy);
        dim3 tamBlock(tbx,tby);
        dim3 tamGrid(tgx,tgy);

        error=cudaMalloc((void**) &surfaceG, sizeof(float) * (size_t)rows * (size_t)columns );
        if ( error != cudaSuccess )
                printf("Error malloc 1: %s\n", cudaGetErrorString( error ) );

        error=cudaMalloc((void**) &surfaceCopyG, sizeof(float) * (size_t)rows * (size_t)columns );
        if ( error != cudaSuccess )
                printf("Error malloc 2: %s\n", cudaGetErrorString( error ) );

        error=cudaMalloc((void**) &gr, sizeof(float) * tgx * tgy ); //Cada bloque calculará su propio global_residual
        //error=cudaMalloc((void**) &gr, sizeof(float) * 1 ); //Cada bloque calculará su propio global_residual
        if ( error != cudaSuccess )
                printf("Error malloc 3: %s\n", cudaGetErrorString( error ) );

        error=cudaMalloc((void**) &teamsG, sizeof(Team) * (size_t) num_teams );
        if ( error != cudaSuccess )
                printf("Error malloc 4: %s\n", cudaGetErrorString( error ) );

        error=cudaMalloc((void**) &focalG, sizeof(FocalPoint) * (size_t) num_focal );
        if ( error != cudaSuccess )
                printf("Error malloc 5: %s\n", cudaGetErrorString( error ) );

        error=cudaMalloc((void**) &nd, sizeof(int) * 1 );
        if ( error != cudaSuccess )
                printf("Error malloc 6: %s\n", cudaGetErrorString( error ) );
/*
        error=cudaMalloc((void**) &rG, sizeof(int) * (num_teams+1) );
        if ( error != cudaSuccess )
                printf("Error malloc 7: %s\n", cudaGetErrorString( error ) );
*/
/*
	// 3. Initialize surface //
        inicializarSurface<<<tamGrid,tamBlock>>>(surfaceG,surfaceCopyG,rows,columns);
        error = cudaGetLastError();
        if ( error != cudaSuccess )
                printf("Error inicializar surface: %s\n", cudaGetErrorString( error ) );
*/
/*
	int ra;
	r[0]=0;
	for(int p=1;p<num_teams+1;p++){
		if ( teams[p-1].type == 1 ) ra = RADIUS_TYPE_1;
                else ra = RADIUS_TYPE_2_3;
		r[p]=r[p-1]+((ra*2+1)*(ra*2+1));
	}
*/
/*	for(int p=0;p<num_teams+1;p++){
		printf("%d ",r[p]); 
	}
*/

	/* 4. Simulation */
	int iter;
	int flag_stability = 0;
	int first_activation = 0;
    float *glores=(float *)malloc( sizeof(float) * tgx * tgy );
	for( iter=0; iter<max_iter && ! flag_stability; iter++ ) {

		/* 4.1. Activate focal points */
        	int num_deactivated = 0;
         	for( i=0; i<num_focal; i++ ) {
             		if ( focal[i].start == iter ) {
                 		focal[i].active = 1;
                		first_activation = 1;
             	}
             		// Count focal points already deactivated by a team
             		else if ( focal[i].active == 2 ) num_deactivated++;
         	}

		if(first_activation==0)continue;

/*
		int *numdea=(int *)malloc( sizeof(int) * 1 );
                int num_deactivated = 0;

                cudaMemcpy(focalG,focal, sizeof(FocalPoint )* num_focal,cudaMemcpyHostToDevice);
                error = cudaGetLastError();
                if ( error != cudaSuccess )
                        printf("Error enviar focal: %s\n", cudaGetErrorString( error ) );

                activarPuntosFocales<<<tamGrid,tamBlock>>>(focalG,nd,num_focal,iter);
                error = cudaGetLastError();
                if ( error != cudaSuccess )
                        printf("Error actualizar calor: %s\n", cudaGetErrorString( error ) );

                error=cudaMemcpy(numdea, nd, sizeof(int)*1,cudaMemcpyDeviceToHost);
                if ( error != cudaSuccess )
                        printf("Error recuperar global_residual: %s\n", cudaGetErrorString( error ) );

                num_deactivated=numdea[0];
*/
/*
	 	error=cudaMemcpy(focal,focalG, sizeof(FocalPoint) * num_focal,cudaMemcpyDeviceToHost);
        	if ( error != cudaSuccess )
                	printf("Error recuperar surface: %s\n", cudaGetErrorString( error ) );
*/

                /* 4.2. Propagate heat (10 steps per each team movement) */
                //float *glores=(float *)malloc( sizeof(float) * tgx * tgy );
                //float *glores=(float *)malloc( sizeof(float) * 1 );
                float global_residual=0.0f;
/*
	 	error=cudaMemcpy(surfaceG,surface, sizeof(float) * rows * columns,cudaMemcpyHostToDevice);
        	if ( error != cudaSuccess )
                	printf("Error recuperar surface: %s\n", cudaGetErrorString( error ) );
*/
/*
	 	error=cudaMemcpy(surfaceCopyG,surfaceCopy, sizeof(float) * rows * columns,cudaMemcpyHostToDevice);
        	if ( error != cudaSuccess )
                	printf("Error recuperar surface: %s\n", cudaGetErrorString( error ) );
*/
	 	error=cudaMemcpy(focalG,focal, sizeof(FocalPoint) * num_focal,cudaMemcpyHostToDevice);
        	if ( error != cudaSuccess )
                	printf("Error recuperar surface: %s\n", cudaGetErrorString( error ) );


            /**************************
            STEP 0
            ***************************/

            actualizarCalor2<<<tamGrid,tamBlock>>>(surfaceG,focalG,rows,columns,num_focal);
            error = cudaGetLastError();
            if ( error != cudaSuccess )
                    printf("Error actualizar calor: %s\n", cudaGetErrorString( error ) );

            surface3=surfaceG;
            surfaceG=surfaceCopyG;
            surfaceCopyG=surface3;

            actualizarSurface<<<tamGrid,tamBlock>>>(surfaceG,surfaceCopyG,rows,columns);
            error = cudaGetLastError();
            if ( error != cudaSuccess )
                    printf("Error actualizar surface: %s\n", cudaGetErrorString( error ) );

            if(num_focal==num_deactivated){
                switch(dimblock){
                    case 1024:
                            maximo2<1024><<<tamGrid,tamBlock,sizeof(float)*tbx*tby>>>(surfaceG,surfaceCopyG,gr,rows,columns);
                            break;
                    case 512:
                            maximo2<512><<<tamGrid,tamBlock,sizeof(float)*tbx*tby>>>(surfaceG,surfaceCopyG,gr,rows,columns);
                            break;
                    case 256:
                            maximo2<256><<<tamGrid,tamBlock,sizeof(float)*tbx*tby>>>(surfaceG,surfaceCopyG,gr,rows,columns);
                            break;
                    case 128:
                            maximo2<128><<<tamGrid,tamBlock,sizeof(float)*tbx*tby>>>(surfaceG,surfaceCopyG,gr,rows,columns);
                            break;
                    case 64:
                            maximo2<64><<<tamGrid,tamBlock,sizeof(float)*tbx*tby>>>(surfaceG,surfaceCopyG,gr,rows,columns);
                            break;
                    case 32:
                            maximo2<32><<<tamGrid,tamBlock,sizeof(float)*tbx*tby>>>(surfaceG,surfaceCopyG,gr,rows,columns);
                            break;
                    case 16:
                            maximo2<16><<<tamGrid,tamBlock,sizeof(float)*tbx*tby>>>(surfaceG,surfaceCopyG,gr,rows,columns);
                            break;
                    case 8:
                            maximo2<8><<<tamGrid,tamBlock,sizeof(float)*tbx*tby>>>(surfaceG,surfaceCopyG,gr,rows,columns);
                            break;
                    case 4:
                            maximo2<4><<<tamGrid,tamBlock,sizeof(float)*tbx*tby>>>(surfaceG,surfaceCopyG,gr,rows,columns);
                            break;
                    case 2:
                            maximo2<2><<<tamGrid,tamBlock,sizeof(float)*tbx*tby>>>(surfaceG,surfaceCopyG,gr,rows,columns);
                            break;
                    case 1:
                            maximo2<1><<<tamGrid,tamBlock,sizeof(float)*tbx*tby>>>(surfaceG,surfaceCopyG,gr,rows,columns);
                            break;
                    default:
                            maximo2<128><<<tamGrid,tamBlock,sizeof(float)*tbx*tby>>>(surfaceG,surfaceCopyG,gr,rows,columns);
                            break;
                }

                error = cudaGetLastError();
                if ( error != cudaSuccess )
                        printf("Error calcular maximo: %s\n", cudaGetErrorString( error ) );
                error=cudaMemcpy(glores, gr, sizeof(float)*tgx*tgy,cudaMemcpyDeviceToHost);
                if ( error != cudaSuccess )
                        printf("Error recuperar global_residual: %s\n", cudaGetErrorString( error ) );

                for (i=0;i<(tgx*tgy);i++){

                    if(global_residual>=THRESHOLD){ 
                        break;
                    }
                    if(glores[i]>global_residual){
                            global_residual=glores[i];
                    }
                }
            }


            //STEP 1-9
            int step;
            for( step=1; step<10; step++ )  {

                    actualizarCalor2<<<tamGrid,tamBlock>>>(surfaceG,focalG,rows,columns,num_focal);
                    error = cudaGetLastError();
                    if ( error != cudaSuccess )
                            printf("Error actualizar calor: %s\n", cudaGetErrorString( error ) );

                    surface3=surfaceG;
                    surfaceG=surfaceCopyG;
                    surfaceCopyG=surface3;

                    actualizarSurface<<<tamGrid,tamBlock>>>(surfaceG,surfaceCopyG,rows,columns);
                    error = cudaGetLastError();
                    if ( error != cudaSuccess )
                            printf("Error actualizar surface: %s\n", cudaGetErrorString( error ) );

                    /*
                        if (step==0 && num_focal==num_deactivated){


                            switch(dimblock){
                                case 1024:
                                        maximo2<1024><<<tamGrid,tamBlock,sizeof(float)*tbx*tby>>>(surfaceG,surfaceCopyG,gr,rows,columns);
                                        break;
                                case 512:
                                        maximo2<512><<<tamGrid,tamBlock,sizeof(float)*tbx*tby>>>(surfaceG,surfaceCopyG,gr,rows,columns);
                                        break;
                                case 256:
                                        maximo2<256><<<tamGrid,tamBlock,sizeof(float)*tbx*tby>>>(surfaceG,surfaceCopyG,gr,rows,columns);
                                        break;
                                case 128:
                                        maximo2<128><<<tamGrid,tamBlock,sizeof(float)*tbx*tby>>>(surfaceG,surfaceCopyG,gr,rows,columns);
                                        break;
                                case 64:
                                        maximo2<64><<<tamGrid,tamBlock,sizeof(float)*tbx*tby>>>(surfaceG,surfaceCopyG,gr,rows,columns);
                                        break;
                                case 32:
                                        maximo2<32><<<tamGrid,tamBlock,sizeof(float)*tbx*tby>>>(surfaceG,surfaceCopyG,gr,rows,columns);
                                        break;
                                case 16:
                                        maximo2<16><<<tamGrid,tamBlock,sizeof(float)*tbx*tby>>>(surfaceG,surfaceCopyG,gr,rows,columns);
                                        break;
                                case 8:
                                        maximo2<8><<<tamGrid,tamBlock,sizeof(float)*tbx*tby>>>(surfaceG,surfaceCopyG,gr,rows,columns);
                                        break;
                                case 4:
                                        maximo2<4><<<tamGrid,tamBlock,sizeof(float)*tbx*tby>>>(surfaceG,surfaceCopyG,gr,rows,columns);
                                        break;
                                case 2:
                                        maximo2<2><<<tamGrid,tamBlock,sizeof(float)*tbx*tby>>>(surfaceG,surfaceCopyG,gr,rows,columns);
                                        break;
                                case 1:
                                        maximo2<1><<<tamGrid,tamBlock,sizeof(float)*tbx*tby>>>(surfaceG,surfaceCopyG,gr,rows,columns);
                                        break;
                                default:
                                        maximo2<128><<<tamGrid,tamBlock,sizeof(float)*tbx*tby>>>(surfaceG,surfaceCopyG,gr,rows,columns);
                                        break;
                            }

                            error = cudaGetLastError();
                            if ( error != cudaSuccess )
                                    printf("Error calcular maximo: %s\n", cudaGetErrorString( error ) );
                            error=cudaMemcpy(glores, gr, sizeof(float)*tgx*tgy,cudaMemcpyDeviceToHost);
                            //error=cudaMemcpy(glores, gr, sizeof(float)*1,cudaMemcpyDeviceToHost);
                            if ( error != cudaSuccess )
                                    printf("Error recuperar global_residual: %s\n", cudaGetErrorString( error ) );

                            for (i=0;i<(tgx*tgy);i++){

                				if(global_residual>=THRESHOLD){ 
                					break;
                				}
                                if(glores[i]>global_residual){
                                        global_residual=glores[i];
                                }
                            }
                        }
                    */
            }
            
/*
	 	error=cudaMemcpy(surface,surfaceG, sizeof(float) * rows * columns,cudaMemcpyDeviceToHost);
        	if ( error != cudaSuccess )
                	printf("Error recuperar surface: %s\n", cudaGetErrorString( error ) );
*/
/*
	 	error=cudaMemcpy(surfaceCopy,surfaceCopyG, sizeof(float) * rows * columns,cudaMemcpyDeviceToHost);
        	if ( error != cudaSuccess )
                	printf("Error recuperar surface: %s\n", cudaGetErrorString( error ) );
*/
/*
	 	error=cudaMemcpy(focal,focalG, sizeof(FocalPoint) * num_focal,cudaMemcpyDeviceToHost);
        	if ( error != cudaSuccess )
                	printf("Error recuperar surface: %s\n", cudaGetErrorString( error ) );
*/
		/* If the global residual is lower than THRESHOLD, we have reached enough stability, stop simulation at the end of this iteration */
		if( num_deactivated == num_focal && global_residual < THRESHOLD ) flag_stability = 1;


if(num_deactivated!=num_focal){
		// 4.3. Move teams //
		for( t=0; t<num_teams; t++ ) {
			// 4.3.1. Choose nearest focal point //
			float distance = FLT_MAX;
			int target = -1;

			for( j=0; j<num_focal; j++ ) {
				if ( focal[j].active != 1 ) continue; // Skip non-active focal points
				float dx = focal[j].x - teams[t].x;
				float dy = focal[j].y - teams[t].y;
				float local_distance = dx*dx + dy*dy ;
				if ( local_distance < distance ) {
					distance = local_distance;
					target = j;
				}
			}

			// 4.3.2. Annotate target for the next stage //
			teams[t].target = target;

			// 4.3.3. No active focal point to choose, no movement //
			if ( target == -1 ) continue; 

			// 4.3.4. Move in the focal point direction //
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

/*		cudaMemcpy(teamsG,teams, sizeof(Team )* num_teams,cudaMemcpyHostToDevice);
                error = cudaGetLastError();
                if ( error != cudaSuccess )
                        printf("Error enviar teams: %s\n", cudaGetErrorString( error ) );

                moverEquipos<<<tamGrid,tamBlock>>>(teamsG,focalG,rows,columns,num_teams,num_focal);
                error = cudaGetLastError();
                if ( error != cudaSuccess )
                        printf("Error mover equipos: %s\n", cudaGetErrorString( error ) );

                cudaMemcpy(teams,teamsG, sizeof(Team )* num_teams,cudaMemcpyDeviceToHost);
                error = cudaGetLastError();
                if ( error != cudaSuccess )
                        printf("Error recuperar teams: %s\n", cudaGetErrorString( error ) );
*/
}

	 	error=cudaMemcpy(teamsG,teams, sizeof(Team) * num_teams,cudaMemcpyHostToDevice);
        	if ( error != cudaSuccess )
                	printf("Error recuperar surface: %s\n", cudaGetErrorString( error ) );

        for( t=0; t<num_teams; t++ ) {
            /* 4.4.1. Deactivate the target focal point when it is reached */
            int target = teams[t].target;
            if ( target != -1 && focal[target].x == teams[t].x && focal[target].y == teams[t].y )
                focal[target].active = 2;
        }
		reducirCalor5<<<tamGrid,tamBlock>>>(surfaceG,teamsG,focalG,rows,columns,num_teams);
                error = cudaGetLastError();
                if ( error != cudaSuccess )
                        printf("Error reducir calor: %s\n", cudaGetErrorString( error ) );
                /*
                cudaMemcpy(focal,focalG, sizeof(FocalPoint )* num_focal,cudaMemcpyDeviceToHost);
                error = cudaGetLastError();
                if ( error != cudaSuccess )
                        printf("Error enviar focal: %s\n", cudaGetErrorString( error ) );
                */
/*
	 	error=cudaMemcpy(surface,surfaceG, sizeof(float) * rows * columns,cudaMemcpyDeviceToHost);
        	if ( error != cudaSuccess )
                	printf("Error recuperar surface: %s\n", cudaGetErrorString( error ) );

	 	error=cudaMemcpy(surfaceCopy,surfaceCopyG, sizeof(float) * rows * columns,cudaMemcpyDeviceToHost);
        	if ( error != cudaSuccess )
                	printf("Error recuperar surface: %s\n", cudaGetErrorString( error ) );
*/
		// 4.4. Team actions 
/*		for( t=0; t<num_teams; t++ ) {
			// 4.4.1. Deactivate the target focal point when it is reached //
			int target = teams[t].target;
			if ( target != -1 && focal[target].x == teams[t].x && focal[target].y == teams[t].y 
				&& focal[target].active == 1 )
				focal[target].active = 2;

			// 4.4.2. Reduce heat in a circle around the team //
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
						accessMat( surface, i, j ) = accessMat( surface, i, j ) * ( 1 - 0.25 ); // Team efficiency factor
					}
				}
			}
		}
*/
		/* 4.4. Team actions */
/*                for( t=0; t<num_teams; t++ ) {
                        int target = teams[t].target;
                        if ( target != -1 && focal[target].x == teams[t].x && focal[target].y == teams[t].y
                                && focal[target].active == 1 )
                                focal[target].active = 2;

                        int radius;
                        // Influence area of fixed radius depending on type
                        if ( teams[t].type == 1 ) radius = RADIUS_TYPE_1;
                        else radius = RADIUS_TYPE_2_3;
                        i=teams[t].x;
                        j=teams[t].y;
                        reducirCalor2<<<tamGrid,tamBlock>>>(surfaceG,i,j,rows,columns,radius);
                        error = cudaGetLastError();
                        if ( error != cudaSuccess )
                                printf("Error reducir calor: %s\n", cudaGetErrorString( error ) );

		}

	 	error=cudaMemcpy(surface,surfaceG, sizeof(float) * rows * columns,cudaMemcpyDeviceToHost);
        	if ( error != cudaSuccess )
                	printf("Error recuperar surface: %s\n", cudaGetErrorString( error ) );

	 	error=cudaMemcpy(surfaceCopy,surfaceCopyG, sizeof(float) * rows * columns,cudaMemcpyDeviceToHost);
        	if ( error != cudaSuccess )
                	printf("Error recuperar surface: %s\n", cudaGetErrorString( error ) );
*/
#ifdef DEBUG
		/* 4.5. DEBUG: Print the current state of the simulation at the end of each iteration */
		print_status( iter, rows, columns, surface, num_teams, teams, num_focal, focal, global_residual );
#endif // DEBUG
	}
	 	error=cudaMemcpy(surface,surfaceG, sizeof(float) * rows * columns,cudaMemcpyDeviceToHost);
        	if ( error != cudaSuccess )
                	printf("Error recuperar surface: %s\n", cudaGetErrorString( error ) );
	
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
