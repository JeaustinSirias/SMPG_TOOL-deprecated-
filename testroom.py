import numpy as np
import matplotlib.pyplot as plt
import pickle
from io import *



output_snack = np.array(pickle.load(open('data_dp', 'rb')))
#print(output_snack[1][0])

print(output_snack.shape)

'''
lip = []
p = 5
k = 0
while p > 1:
	p = p - k
	k = 1
	lip.append(p)
lip = np.sort(lip)


lista = np.array([1, 2, 3, 4, 5, 6, None, None, None])

x = np.arange(0, len(lista), 1)



y = [2, 5, 6, 8, 2, 4, 9, 2]

k = [None]*(36 - len(y))


print(len(y + k))
'''



