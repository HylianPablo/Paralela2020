
CFLAGS= -I. -O3 -DCUDA_CHECK --compiler-options -Wall
CFLAGS_DEBUG= -I. -O0 -G -DCUDA_CHECK --compiler-options -Wall
CFLAGS_TEST= -I. -O3 -DCUDA_CHECK -DDEVELOPMENT --compiler-options -Wall
LIBS=
CC=nvcc

evolution: miPanaMiguel.cu *.h
	$(CC) $(CFLAGS) $< -o $@ $(LIBS)

debug: evolution.cu *.h
	$(CC) $(CFLAGS_DEBUG) -DDEBUG $< -o $@ $(LIBS)

original: original.cu *.h
	$(CC) $(CFLAGS) $< -o $@ $(LIBS)

odebug: original.cu *.h
	$(CC) $(CFLAGS_DEBUG) -DDEBUG $< -o $@ $(LIBS)

tgz:
	tar -czf evolution_cuda.tgz Makefile miPanaMiguel.cu *.h

clean:
	rm -f evolution debug original

