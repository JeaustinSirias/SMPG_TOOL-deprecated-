#python3
##--------------------------------Makefile------------------------------------
#shortcuts
TEST = ./test/

#Run SMPG script
#*****************************************************************************
run: SMPG

SMPG:	
	pip install -r requirements.txt \
	&& python3 $(TEST)main.py


