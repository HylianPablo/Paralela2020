
CFLAGS= -I. -O3
CFLAGS_DEBUG= -I. -g -O0
LIBS=-lm

evolution: evolution.c
	gcc $(CFLAGS) $< -o $@ $(LIBS)

debug: evolution.c
	gcc $(CFLAGS_DEBUG) -DDEBUG $< -o $@ $(LIBS)

clean:
	rm -f evolution debug

