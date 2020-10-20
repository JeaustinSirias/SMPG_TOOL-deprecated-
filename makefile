#python3
##--------------------------------Makefile------------------------------------
#shortcuts
TEST = ./test/

#Run SMPG script
#*****************************************************************************
run: SMPG

SMPG:
	python3 $(TEST)main.py

