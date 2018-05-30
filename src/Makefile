
CC := mpicc

PSRS : PSRS.o
	$(CC) -o PSRS PSRS.o

PSRS.o : PSRS.c
	$(CC) -c PSRS.c -o PSRS.o

clean :
	-rm -f PSRS *.o
