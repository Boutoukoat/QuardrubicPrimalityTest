
GGG = clang++ -O3 -Wall -march=native -fomit-frame-pointer
GGG = g++ -O3 -Wall -march=native -fomit-frame-pointer -fexpensive-optimizations

OBJ = quardrubic_primality_main.o \
      quardrubic_primality.o \
      quardrubic_primality_alloc.o \
      quardrubic_primality_precompute.o \
      expression_parser.a

quardrubic: $(OBJ)
	$(GGG) -static -o quardrubic $(OBJ) -lgmp -lpthread -lm

quardrubic_primality_main.o: quardrubic_primality_main.cpp quardrubic_primality.h quardrubic_primality_alloc.h bison.gmp_expr.tab.h
	$(GGG) -c -o quardrubic_primality_main.o quardrubic_primality_main.cpp

quardrubic_primality_alloc.o: quardrubic_primality_alloc.cpp quardrubic_primality_alloc.h
	$(GGG) -c -o quardrubic_primality_alloc.o quardrubic_primality_alloc.cpp

quardrubic_primality_precompute.o: quardrubic_primality_precompute.cpp quardrubic_primality_precompute.h
	$(GGG) -c -o quardrubic_primality_precompute.o quardrubic_primality_precompute.cpp

quardrubic_primality.o: quardrubic_primality.cpp quardrubic_primality.h quardrubic_primality_precompute.h
	$(GGG) -Wno-unused-function -c -o quardrubic_primality.o quardrubic_primality.cpp

expression_parser.a : bison.gmp_expr.o lex.gmp_expr.o
	ar vr expression_parser.a bison.gmp_expr.o lex.gmp_expr.o

bison.gmp_expr.o : bison.gmp_expr.tab.c bison.gmp_expr.h
	$(GGG) -c -o bison.gmp_expr.o bison.gmp_expr.tab.c

bison.gmp_expr.tab.c bison.gmp_expr.tab.h : parser.y
	bison -d parser.y

lex.gmp_expr.o : lex.gmp_expr.c
	$(GGG) -Wno-unused-function -DYY_BUF_SIZE=65540 -DYYLMAX=65540 -c -o lex.gmp_expr.o lex.gmp_expr.c

lex.gmp_expr.c : parser.l bison.gmp_expr.tab.h
	flex parser.l

check: quardrubic
	./quardrubic -st

clean:
	rm -f ./quardrubic $(OBJ) bison.gmp_expr.o bison.gmp_expr.tab.c bison.gmp_expr.tab.h lex.gmp_expr.o lex.gmp_expr.c


