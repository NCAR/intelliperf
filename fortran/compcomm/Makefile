FC := gfortran
SRC := compcomm

run: build
	./${SRC}.exe

build: clean
	${FC} ${SRC}.F90 -o ${SRC}.exe

clean:
	rm -f ${SRC}.o ${SRC}.exe
