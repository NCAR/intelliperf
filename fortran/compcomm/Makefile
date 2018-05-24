#FC := gfortran
FC := ifort
SRC := compcomm

EXTRAE_HOME := /glade/p/tdd/asap/contrib/cheyenne_packages/extrae/3.5.1
EXTRAE_CONFIG_FILE=extrae.xml

PAPI_HOME := /glade/p/tdd/asap/contrib/cheyenne_packages/papi_2017/5.5.1
XML_HOME := /glade/apps/opt/libxml2/2.9.0/gnu/4.7.2
UNWIND_HOME := /glade/p/tdd/asap/contrib/cheyenne_packages/libunwind/1.2

PAPILIB := -L${PAPI_HOME}/lib -lpapi
XMLLIB := -L${XML_HOME}/lib -lxml2
UNWINDLIB := -L${XML_HOME}/lib -lunwind
LIBERTYLIB := /usr/lib64/libiberty.a
EXTRAELIB := -L${EXTRAE_HOME}/lib -lseqtrace

ALLLIBS := ${LAPACKLIBS} ${PAPILIB} ${XMLLILB} ${UNWINDLIB} ${LIBERTYLIB} ${EXTRAELIB}

LIBPATHS := ${EXTRAE_HOME}/lib:${XML_HOME}/lib:${XML_HOME}/lib:${PAPI_HOME}/lib

PRERUN := export EXTRAE_HOME=${EXTRAE_HOME}; export EXTRAE_CONFIG_FILE=${EXTRAE_CONFIG_FILE}; export LD_LIBRARY_PATH=${LIBPATHS}:${LD_LIBRARY_PATH}

FOLDING_HOME := /glade/p/tdd/asap/contrib/cheyenne_packages/folding/1.0.2
FOLD_DIR := folding

CLUSTERING_HOME := /glade/p/tdd/asap/contrib/cheyenne_packages/clustering/2.6.6
CLUSTER_DIR := clustering

OPTS := -g ${ALLLIBS}

run: build
	${PRERUN}; ./${SRC}.exe

build: clean
	${FC} ${SRC}.F90 ${OPTS} -o ${SRC}.exe

clean: cleanprv
	rm -f ${SRC}.o ${SRC}.exe extrae_module.mod

cleanprv:
	rm -f ${SRC}.exe.prv ${SRC}.exe.pcf ${SRC}.exe.row
	rm -rf set-0

prv: cleanprv
	${EXTRAE_HOME}/bin/mpi2prv -f TRACE.mpits -o .prv

fold: run
	rm -rf ${FOLD_DIR}; mkdir ${FOLD_DIR}; cd ${FOLD_DIR}; ${FOLDING_HOME}/bin/folding ../${SRC}.exe.prv "User function"


