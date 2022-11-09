
PCACHE:= $(wildcard */__pycache__)

clean: 
	ls ${PCACHE} 

env:
	conda activate vision
