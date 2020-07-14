import numpy as np
import matplotlib.pyplot as plt
import pickle
from io import *
import pandas as pd


#A class dedicated to compute long term avg. rainfall stats
class LT_procedures(): 
	def __init__(self, init_year, end_year, init_dek, end_dek):

		self.fst_dek = init_dek
		self.lst_dek = end_dek
		self.fst_year = init_year
		self.lst_year = end_year
		self.dek_dictionary = pickle.load(open('dekads_dictionary', 'rb')) #a dictionary of dekads
		self.raw_data = pickle.load(open('data', 'rb')) #WHOLE RAW DATA

##############################################################################################################################################

	def compute_median(self, data_in): #data_in is the raw input file

		store_dek = [] #an array to store data_in as individual years throughtout a single location
		dek_number = 36*(len(np.arange(self.fst_year, self.lst_year, 1))) #read the amount of dekads in data_in 

		for i in np.arange(36, dek_number+36, 36):
			com = data_in[i-36:i]
			store_dek.append(com)

		store_dek = np.array(store_dek)
		#once we get store_dek array then we need to transpose it in order to get their medians
		store_dek_T = store_dek.transpose()

		#now we get the median for each 
		LT_mean = []
		for i in np.arange(0, 36, 1):
			mean = np.median(store_dek_T[i])
			LT_mean.append(mean)

		#As an extra we get an array that contains the current year dekads
		current_year = data_in[dek_number:]

		#OUTPUTS
		return np.array([LT_mean, current_year, store_dek])
		#[a, b, c]

##############################################################################################################################################

	def get_median_for_whole_data(self): #to get the median for all location in a row

		raw_years = [] #It'll be an array to store input data, but now in years
		actual_year = []
		full_data_median = []#an array which contains the historical median for all completed years available
		for i in np.arange(0, len(self.raw_data), 1):

			a = self.compute_median(self.raw_data[i])
			actual_year.append(a[1])
			raw_years.append(a[2])
			full_data_median.append(a[0])

		output = np.array([full_data_median, actual_year, raw_years])

		#we need the OUTPUT
		dek_data = open('output_snack', 'wb') #to save whole data separated in dekads [n_locations, n_years, 36]. Only takes completed years
		pickle.dump(output, dek_data)
		dek_data.close()

##############################################################################################################################################

	def rainfall_accumulations(self):

		output_snack = pickle.load(open('output_snack', 'rb'))
		yrs_window = np.arange(self.fst_year, self.lst_year, 1)

		#to get the rainfall accumutaions for all years except for the current one
		n = 0
		skim = []
		for k in np.arange(0, len(output_snack[2]), 1):
			remain = []
			skim.append(remain)
			for j in np.arange(0, len(yrs_window), 1):
				sumV = []
				remain.append(sumV)
				for i in np.arange(self.dek_dictionary[self.fst_dek]-1, self.dek_dictionary[self.lst_dek], 1):
					n = n + output_snack[2][k][j][i]
					sumV.append(n)
					if len(sumV) == len(np.arange(self.dek_dictionary[self.fst_dek]-1, self.dek_dictionary[self.lst_dek], 1)):
						n = 0

		#to get the rainfall accumulation for current year in selected dekad window
		n = 0 
		acumulado_por_estacion = []
		for k in np.arange(0, len(output_snack[1]), 1):
			acumulado_ano_actual = []
			acumulado_por_estacion.append(acumulado_ano_actual)

			for i in np.arange(self.dek_dictionary[self.fst_dek]-1, self.dek_dictionary[self.lst_dek], 1):
				n = n + output_snack[1][k][i]
				acumulado_ano_actual.append(n)
				if len(acumulado_ano_actual) == len(np.arange(self.dek_dictionary[self.fst_dek]-1, self.dek_dictionary[self.lst_dek], 1)):
					n = 0

		accumutaions = np.array(np.array(skim), np.array(acumulado_ano_actual))

		return print(acumulado_ano_actual)

##############################################################################################################################################
##############################################################################################################################################
##############################################################################################################################################







sample = LT_procedures(1981, 2020, '1-Jan', '1-May')
sample.get_median_for_whole_data()
sample.rainfall_accumulations()









