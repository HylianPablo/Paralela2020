
CFLAGS= -I. -O3
CFLAGS_DEBUG= -I. -g -O0
LIBS=-lm
CC=mpicc

evolution: funeralDeGhana.c
	$(CC) $(CFLAGS) $< -o $@ $(LIBS)

debug: funeralDeGhana.c
	$(CC) $(CFLAGS_DEBUG) -DDEBUG $< -o $@ $(LIBS)

original: original-mpi.c
	$(CC) $(CFLAGS) $< -o $@ $(LIBS)

original_debug: original-mpi.c
	$(CC) $(CFLAGS_DEBUG) -DDEBUG $< -o $@ $(LIBS)

print: verbose_print.c
	$(CC) $(CFLAGS_DEBUG) -DDEBUG $< -o $@ $(LIBS)

clean:
	rm -f evolution debug original original_debug print

