
CFLAGS= -I. -O3
CFLAGS_DEBUG= -I. -g -O0
LIBS=-lm
CC=mpicc

evolution: funeralDeGhana.c
	$(CC) $(CFLAGS) $< -o $@ $(LIBS)

debug: evolution_mpi.c
	$(CC) $(CFLAGS_DEBUG) -DDEBUG $< -o $@ $(LIBS)

original: original-mpi.c
	$(CC) $(CFLAGS) $< -o $@ $(LIBS)

original_debug: original-mpi.c
	$(CC) $(CFLAGS_DEBUG) -DDEBUG $< -o $@ $(LIBS)

print: verbose_print.c
	$(CC) $(CFLAGS_DEBUG) -DDEBUG $< -o $@ $(LIBS)

simulation: funeralDeGhana.c
	$(CC) $(CFLAGS) -DCP_TABLON $< -o $@ $(LIBS)

kuma: kuma.c
	$(CC) $(CFLAGS) $< -o evolution $(LIBS)

gabbanos: manalda.c
	$(CC) $(CFLAGS) $< -o evolution $(LIBS)

clean:
	rm -f evolution debug original original_debug print

