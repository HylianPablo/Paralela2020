
CFLAGS= -I. -O3 -fopenmp
CFLAGS_DEBUG= -I. -g -O0
LIBS=-lm

evolution: evolution.c
	gcc $(CFLAGS) $< -o $@ $(LIBS)

debug: evolution.c
	gcc $(CFLAGS_DEBUG) -DDEBUG $< -o $@ $(LIBS)

clean:
	rm -f evolution debug

safe: original.c
	gcc $(CFLAGS) $< -o evolution $(LIBS)

safe-debug: original.c
	gcc $(CFLAGS_DEBUG) -DDEBUG $< -o debug $(LIBS)

