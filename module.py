import numpy as np
import matplotlib.pyplot as plt
import pickle
from io import *
import pandas as pd
from scipy.stats import rankdata
from collections import defaultdict



class LT_procedures(): 
	def __init__(self, init_year, end_year, init_dek, end_dek, init_clim, end_clim):

		self.init_clim = init_clim #start climatology
		self.end_clim = end_clim #end_climatology
		self.fst_dek = init_dek
		self.lst_dek = end_dek
		self.fst_year = init_year
		self.lst_year = end_year
		self.dek_dictionary = pickle.load(open('./datapath/dekads_dictionary', 'rb')) #a dictionary of dekads
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
			mean = np.mean(store_dek_T[i])
			LT_mean.append(mean)

		#As an extra we get an array that contains the current year dekads
		current_year = list(data_in[dek_number:])
		k = [None]*(36 - len(current_year))
		current_year_None = np.array(current_year + k) #to fill missing dekads with null characters

		#OUTPUTS
		return np.array([LT_mean, current_year_None, store_dek, np.array(current_year)])
		#[a, b, c]

##############################################################################################################################################

	def get_median_for_whole_data(self): #to get the median for all location in-a-row

		raw_years = [] #It'll be an array to store input data, but now by location
		actual_year = []
		actual_year_no_None = []
		full_data_median = []#an array which contains the historical median for all completed years available
		for i in np.arange(0, len(self.raw_data[2]), 1):

			a = self.compute_median(self.raw_data[2][i])
			actual_year.append(a[1])
			raw_years.append(a[2])
			full_data_median.append(a[0])
			actual_year_no_None.append(a[3])

		#****************************************CLIMATOLOGY********************************
		#we're gonna setup a way to fix climatology. For this we'll make
		#a dictionary with all years gotten in the header like: {1985:0, 1986:1, ... 2018:33}
		#so when user inputs its climatology window, program will user bouds as passwords in this dictionary

		year_1 = int(self.raw_data[1][0][0:4]) #absolute first year
		year_2 = int(self.raw_data[1][-1][0:4]) #absolute last year
		years = np.arange(year_1, year_2, 1)
		linspace = np.arange(0, len(years), 1)
		clim_dict = dict(zip(years, linspace)) #a dictionary which will help me to choose climatology

		clim = []
		for i in np.arange(0, len(raw_years), 1):
			add = raw_years[i][clim_dict[self.init_clim]:clim_dict[self.end_clim]+1]
			clim.append(add) 

		clim = np.array(clim)

		#to get the mean for climatology window
		mean_clim = [] #to store mean data from climatology
		for i in np.arange(0, len(clim), 1):
			get = []
			mean_clim.append(get)
			for k in np.arange(0, len(clim[0].transpose()), 1):
				store = np.mean(clim[i].transpose()[k])
				get.append(store)

		#OUTPUT OPERATIONS
		#this output gives an array with these specs: [average based ]					  [location labels]	[climatology_graph1]
		output = np.array([full_data_median, actual_year, raw_years, actual_year_no_None, self.raw_data[0], mean_clim])

		dek_data = open('./datapath/output_snack', 'wb') #to save whole data separated in dekads [n_locations, n_years, 36]. Only takes completed years
		pickle.dump(output, dek_data)
		dek_data.close()

	
		return output # np.array([full_data_median, actual_year, raw_years])

##############################################################################################################################################

	def rainfall_accumulations(self):

		output_snack = self.get_median_for_whole_data()   #pickle.load(open('output_snack', 'rb'))
		yrs_window = np.arange(self.fst_year, self.lst_year, 1)

		#return print(len(output_snack[1]))
	
		#to get the rainfall accumulations for all years except for the current one
		n = 0
		skim = [] #OUTPUT 1
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

		#auxilliar step to set accumulations as a dictionary, where the password is their corresponding years
		skim = np.array(skim)

		skim_dictionary = []
		for i in np.arange(0, len(skim), 1):
			cat =  dict(zip(np.arange(self.fst_year, self.lst_year, 1), skim[i]))
			skim_dictionary.append(cat)
	
		#to get the rainfall accumulation for current year in selected dekad window
		n = 0 
		acumulado_por_estacion = [] #OUTPUT 2
		for k in np.arange(0, len(output_snack[1]), 1):
			acumulado_ano_actual = []
			acumulado_por_estacion.append(acumulado_ano_actual)

			for i in np.arange(self.dek_dictionary[self.fst_dek]-1, self.dek_dictionary[self.lst_dek], 1):
				if output_snack[1][k][i] == None:
					n = 0
					break
				else:
					n = n + output_snack[1][k][i]
					acumulado_ano_actual.append(n)
					if len(acumulado_ano_actual) == len(np.arange(self.dek_dictionary[self.fst_dek]-1, self.dek_dictionary[self.lst_dek], 1)):
						n = 0

		acumulado_por_estacion = np.array(acumulado_por_estacion)
		#return print(np.array(acumulado_por_estacion).shape)
		
		output = np.array([skim, acumulado_por_estacion.transpose(), np.array(skim_dictionary)])
		accumulation = open('./datapath/accumulations', 'wb') #to save rainfall accumulations array as [past_years_accum, current_year_accum, past_years_dict]
		pickle.dump(output, accumulation)
		accumulation.close()

		
		return np.array([skim, acumulado_por_estacion.transpose(), np.array(skim_dictionary)])

##############################################################################################################################################
#ANALOG YEARS
	
	def sum_error_sqr(self): #computes the square of substraction between the biggest accumulations 
		
		accumulations = self.rainfall_accumulations()
		#print(self.accumulations[1].transpose()[0])
		error_sqr = []
		for i in np.arange(0, len(accumulations[0]), 1):
			local_sums = []
			error_sqr.append(local_sums)
			for j in np.arange(0, len(accumulations[0][0]), 1):

				sqr_error = (accumulations[1].transpose()[i][-1] - accumulations[0][i][j][-1])**2
				local_sums.append(sqr_error)

		error_sqr = np.array(error_sqr) #this array must have a shape like [num_locations, num_years]
		
		sum_error_sqr_rank = []
		for i in np.arange(0, len(error_sqr), 1):
			rank =  rankdata(error_sqr[i], method = 'ordinal')
			sum_error_sqr_rank.append(rank)

		return np.array(sum_error_sqr_rank) #ULTIMATE OUTPUT

##############################################################################################################################################

	def sum_dekad_error(self):

		accumulations = self.rainfall_accumulations()
		output_snack = self.get_median_for_whole_data()

		
		global_substraction = []
		for i in np.arange(0, len(accumulations[0]), 1): #for each location in the array
			mid_substraction = []
			global_substraction.append(mid_substraction)
			for j in np.arange(0, len(accumulations[0][0]), 1): #drive by each year 
				spec_substraction = []
				mid_substraction.append(spec_substraction)
				for k in np.arange(0, len(accumulations[0][0][0])): #while visiting every dek from the chosen window by the user

					a = output_snack[1][i][self.dek_dictionary[self.fst_dek]-1:self.dek_dictionary[self.lst_dek]] #current year for chosen dekads
					b = output_snack[2][i][j][self.dek_dictionary[self.fst_dek]-1:self.dek_dictionary[self.lst_dek]] #raw past year for chosen dekads

					if a[k] == None:
						break
					else:
						subs_sqr = (a[k] - b[k])**2
						spec_substraction.append(subs_sqr)

		global_substraction = np.array(global_substraction) #[num_locations, num_years, num_dek]

		#Let's compute the sum of each array 
		total_sum = [] #this is my ultimate output
		for i in np.arange(0, len(global_substraction), 1):
			med_sum = []
			total_sum.append(med_sum)
			for j in np.arange(0, len(global_substraction[0]), 1):
				sum_ex = np.sum(global_substraction[i][j]) #sum execution
				med_sum.append(sum_ex)

		total_sum = np.array(total_sum)

		sum_dekad_error_rank = []
		for i in np.arange(0, len(total_sum), 1):
			rank = rankdata(total_sum[i], method = 'ordinal')
			sum_dekad_error_rank.append(rank)

		return np.array(sum_dekad_error_rank) #ULTIMATE OUTPUT!

##############################################################################################################################################	

	def get_analog_years(self):

		#both must have the same shape
		call_sum_error_sqr = self.sum_error_sqr() #we call the resulting RANK FOR SUM ERROR SQUARE
		call_sum_dekad_error = self.sum_dekad_error() #we call the resulting RANK FOR SUM DEKAD error_sqr

		#first we sum the ranks of each error for their corresponding year i.e 1981, 1990, 2000...
		analog_yrs_immed = []  #MUST HAVE THE SAME SIZE AS ITS PARENTS [num_locations, num_years]
		for i in np.arange(0, len(call_sum_error_sqr), 1):
			med_yrs = []
			analog_yrs_immed.append(med_yrs)
			for j in np.arange(0, len(call_sum_error_sqr[0]), 1):

				summ = call_sum_error_sqr[i][j] + call_sum_dekad_error[i][j]
				med_yrs.append(summ)

		#now we must rank analog_yrs_immed

		analog_yrs = []
		analog_yrs_ranked = []
		for i in np.arange(0, len(analog_yrs_immed), 1):
			com = rankdata(analog_yrs_immed[i], method = 'ordinal')
			com_ranked = rankdata(analog_yrs_immed[i], method = 'dense')
			analog_yrs.append(com)
			analog_yrs_ranked.append(com_ranked)



		#we better create a dictionary so we can link the ranks to their corresponding year
		time_window = np.arange(self.fst_year, self.lst_year, 1)
		analog_yrs_dict = []
		for i in np.arange(0, len(analog_yrs), 1):
			dictionary = dict(zip(np.array(analog_yrs[i]), time_window))
			analog_yrs_dict.append(dictionary)


		#also we need to create a dictionary which sets up by rank. There may be several analog years that share the same rank. 
		ranked = []
		for j in np.arange(0, len(analog_yrs_ranked), 1):

			data_dict = defaultdict(list)
			tupla = []
			k = 0
			for i in time_window:
				m = (i, analog_yrs_ranked[j][k])
				tupla.append(m)
				k = k + 1

			for k, v in tupla:
				data_dict[v].append(k)

			ranked.append(dict(data_dict))


		export = open('./datapath/analogs', 'wb')
		pickle.dump(analog_yrs_dict, export)
		export.close()

		return np.array(analog_yrs_dict) #these are the ranks for analog years
		#return print(np.array(ranked).shape)


##############################################################################################################################################




#t = LT_procedures(1981, 2020, '1-Feb', '3-May', 1981, 2010)
#z = print(t.get_analog_years())






