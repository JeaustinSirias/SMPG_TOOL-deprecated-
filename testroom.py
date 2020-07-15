import numpy as np
import matplotlib.pyplot as plt
import pickle
from io import *



output_snack = pickle.load(open('output_snack', 'rb'))

print(output_snack[1][0])
