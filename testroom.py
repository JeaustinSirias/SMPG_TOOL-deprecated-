import numpy as np
import matplotlib.pyplot as plt
import pickle
from io import *



output_snack = pickle.load(open('output_snack', 'rb'))
#print(output_snack[1][0])

lip = []
p = 5
k = 0
while p > 1:
	p = p - k
	k = 1
	lip.append(p)
lip = np.sort(lip)
print(lip)

