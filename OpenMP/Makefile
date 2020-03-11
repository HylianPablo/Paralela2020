CFLAGS= -I. -O3 -Wall -Werror -W
CFLAGS_DEBUG= -I. -g -O0
LIBS=-lm -fopenmp

evolution: evolution.c
	gcc $(CFLAGS) $< -o $@ $(LIBS)

debug: evolution.c
	gcc $(CFLAGS_DEBUG) -DDEBUG $< -o $@ $(LIBS)

clean:
	rm -f evolution debug original x *.out

safe: original.c
	gcc $(CFLAGS) $< $(LIBS)

safe-debug: original.c
	gcc $(CFLAGS_DEBUG) -DDEBUG $< -o debug $(LIBS)

pablo:	x.c
	gcc $(CFLAGS) $< $(LIBS)
