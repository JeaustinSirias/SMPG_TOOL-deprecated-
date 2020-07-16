import numpy as np
import matplotlib.pyplot as plt
import pickle
from io import *
import pandas as pd
from scipy.stats import rankdata


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

	def get_median_for_whole_data(self): #to get the median for all location in-a-row

		raw_years = [] #It'll be an array to store input data, but now in years
		actual_year = []
		full_data_median = []#an array which contains the historical median for all completed years available
		for i in np.arange(0, len(self.raw_data), 1):

			a = self.compute_median(self.raw_data[i])
			actual_year.append(a[1])
			raw_years.append(a[2])
			full_data_median.append(a[0])

		#output = np.array([full_data_median, actual_year, raw_years])
		return np.array([full_data_median, actual_year, raw_years])

		#we need the OUTPUT
		#dek_data = open('output_snack', 'wb') #to save whole data separated in dekads [n_locations, n_years, 36]. Only takes completed years
		#pickle.dump(output, dek_data)
		#dek_data.close()
		
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

		#auxilliar step to set accumulations as a dictionary where the password is their corresponding year
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
		
##############################################################################################################################################
##############################################################################################################################################
##############################################################################################################################################


#A class dedicated to compute and get analog years
class analog_years(LT_procedures): #we inherit LT_procedures class properties
	#def __init__(self):

	accumulations = np.array(pickle.load(open('./datapath/accumulations', 'rb'))) #include accumulations vector
	output_snack = LT_procedures(1985, 2019, '1-Feb', '1-May').get_median_for_whole_data() #pickle.load(open('output_snack', 'rb'))
	
##############################################################################################################################################

	def sum_error_sqr(self): #computes the square of substraction between the biggest accumulations 
		
		#print(self.accumulations[1].transpose()[0])
		error_sqr = []
		for i in np.arange(0, len(self.accumulations[0]), 1):
			local_sums = []
			error_sqr.append(local_sums)
			for j in np.arange(0, len(self.accumulations[0][0]), 1):

				sqr_error = (self.accumulations[1].transpose()[i][-1] - self.accumulations[0][i][j][-1])**2
				local_sums.append(sqr_error)

		error_sqr = np.array(error_sqr) #this array must have a shape like [num_locations, num_years]
		
		sum_error_sqr_rank = []
		for i in np.arange(0, len(error_sqr), 1):
			rank =  rankdata(error_sqr[i], method = 'ordinal')
			sum_error_sqr_rank.append(rank)

		return np.array(sum_error_sqr_rank) #ULTIMATE OUTPUT
		
		#return print(sum_error_sqr_rank) #this array must have a shape like [num_locations, num_years]

##############################################################################################################################################
	
	def sum_dekad_error(self):
		
		global_substraction = []
		for i in np.arange(0, len(self.accumulations[0]), 1): #for each location in the array
			mid_substraction = []
			global_substraction.append(mid_substraction)
			for j in np.arange(0, len(self.accumulations[0][0]), 1): #drive by each year 
				spec_substraction = []
				mid_substraction.append(spec_substraction)
				for k in np.arange(0, len(self.accumulations[0][0][0])): #while visiting every dek from the chosen window by the user

					a = self.output_snack[1][i][self.dek_dictionary[self.fst_dek]-1:self.dek_dictionary[self.lst_dek]] #current year for chosen dekads
					b = self.output_snack[2][i][j][self.dek_dictionary[self.fst_dek]-1:self.dek_dictionary[self.lst_dek]] #raw past year for chosen dekads

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

		#return print(np.argsort(total_sum[0]))
		#return print(rankdata(total_sum[0], method = 'ordinal'))

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
		for i in np.arange(0, len(analog_yrs_immed), 1):
			com = rankdata(analog_yrs_immed[i], method = 'ordinal')
			analog_yrs.append(com)

		#we better create a dictionary so we can link the ranks to their corresponding year
		time_window = np.arange(self.fst_year, self.lst_year, 1)

		analog_yrs_dict = []
		for i in np.arange(0, len(analog_yrs), 1):
			dictionary = dict(zip(np.array(analog_yrs[i]), time_window))
			analog_yrs_dict.append(dictionary)

		export = open('analogs', 'wb')
		pickle.dump(analog_yrs_dict, export)
		export.close()
		#return print(self.sum_error_sqr()) #these are the ultimate analog years result.

##############################################################################################################################################
##############################################################################################################################################
##############################################################################################################################################

'''
class proccess_data_to_plot():
	def __init__(self, analog_input): #analog input will be the amount of analog years selected by user

		self.analogs_dictionary = np.array(pickle.load(open('analogs', 'rb')))
		self.analog_num = analog_input
		self.accumulations = np.array(pickle.load(open('accumulations', 'rb'))) #we're gonna need the third output

##############################################################################################################################################

	def get_analog_accumulation(self): #It'll filter only the analog years

		#identify the analog years passwords. If user inputs 5, it returns [1, 2, 3, 4, 5]. Each number is a password
		analogs = []; k = 0
		while self.analog_num > 1:
			self.analog_num = self.analog_num - k
			k = 1
			analogs.append(self.analog_num)
		analogs = np.sort(analogs)

		#we get the ANALOG YEARS ARRAY i.e [2000, 1982, 1995,...] they'll be my passwords to get their respective data
		accumulation_analog_yrs = []
		for i in np.arange(0, len(self.analogs_dictionary), 1): #for each location available
			camp = []
			accumulation_analog_yrs.append(camp)
			for j in np.arange(0, len(analogs), 1):
				get = self.analogs_dictionary[i][analogs[j]]
				camp.append(get)

		accumulation_analog_yrs = np.array(accumulation_analog_yrs)

		return accumulation_analog_yrs

#############################################################################################
	
	def prep_graph2_data(self):
		accumulation_analog_yrs = self.get_analog_accumulation()
		#now we get the data for every analog year (accumulations)
		analog_curves = []
		for i in np.arange(0, len(self.accumulations[2]), 1):
			curves = []
			analog_curves.append(curves)
			for j in np.arange(0, len(accumulation_analog_yrs[0]), 1):
				com = self.accumulations[2][i][accumulation_analog_yrs[i][j]]
				curves.append(com)

		return np.array(analog_curves)#np.array(analog_curves)
		

		#return print(self.accumulations[2][0])
		#pass



		#return print(self.accumulations[2][0])#print(self.accumulations[2][0][1985])		
'''





class1 = LT_procedures(1985, 2019, '1-Jan', '1-May')
class1.rainfall_accumulations()


'''
sample = LT_procedures(1985, 2019, '1-Feb', '1-May')
#sample.get_median_for_whole_data()
sample.rainfall_accumulations()

select = analog_years(1985, 2019, '1-Feb', '1-May')
print(select.get_analog_years()

kit = proccess_data_to_plot(5)
#print(kit.get_analog_accumulation())
#print(kit.prep_graph2_data())
'''








