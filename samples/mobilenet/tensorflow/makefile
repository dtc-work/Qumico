target = main.c out_c/qumico.c
execfile = mobilenet_c
asmfile = mobilenet_c.asm

CC = gcc
STD = c99
OPTIMIZE = -Ofast -fopenmp
DEBUG_OPT = -pg -g


$(execfile):	$(target)
	$(CC) $(target) $(DEBUG_OPT) -lm -o $(execfile) $(OPTIMIZE) -std=$(STD)

asm:
	$(CC) $(target) $(DEBUG_OPT) -lm -S $(OPTIMIZE) -std=$(STD)

clean:
	rm $(execfile)

