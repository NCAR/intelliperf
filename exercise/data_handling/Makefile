extrae:
	python read_extrae_raw.py ../../data/mg2/mg2.folded

folding:
	python read_folding_csv.py ../../data/mg2/mg2.slope.csv

label:
	python generate_labels.py /glade/u/home/youngsun/work2/perfessay/echam-PSrad/work/folding/PSrad.exe/PSrad.exe.codeblocks.fused.any.any.any.slope.csv -r 1-35 -o ${PWD}/PSrad.exe.codeblocks.fused.any.any.any.slope.labelled.csv

random:
	python data_split.py
	
	
