
PRERUN := module load python numpy scipy matplotlib scikit-learn

WORKDIR := /glade/u/home/youngsun/work2/perfessay/echam-PSrad/work/folding/PSrad.exe
LABELLED := PSrad.exe.codeblocks.fused.any.any.any.slope.labelled.csv

lrtmrng:
	${PRERUN}; python lrtm_rng.py ${WORKDIR}/${LABELLED}
