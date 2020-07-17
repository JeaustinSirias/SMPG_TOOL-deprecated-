import numpy as np
#import matplotlib.pyplot as plt
import pickle
from io import *
#import pandas as pd
from scipy.stats import rankdata

#A module to prepare data to be plotted

class proccess_data_to_plot():
	def __init__(self, analog_input): #analog input will be the amount of analog years selected by user

		self.analogs_dictionary = np.array(pickle.load(open('./datapath/analogs', 'rb')))
		self.analog_num = analog_input
		self.accumulations = np.array(pickle.load(open('./datapath/accumulations', 'rb'))) #we're gonna need the third output
		self.output_snack = pickle.load(open('./datapath/output_snack', 'rb')) #[full_data_median, current_year, raw_years]

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

		return np.array(accumulation_analog_yrs)

##############################################################################################################################################

	def get_graph2_curves(self): #now we get the data for every analog year (accumulations)

		years = self.get_analog_accumulation()

		analog_curves = []
		for i in np.arange(0, len(self.accumulations[2]), 1):
			curves = []
			analog_curves.append(curves)
			for j in np.arange(0, len(years[0]), 1):
				com = self.accumulations[2][i][years[i][j]]
				curves.append(com)

		analog_curves = np.array(analog_curves) #it contains ONLY the curves for chosen analog years

		#To get the accumulated rainfall median FOR PAST YEARS
		external = []
		for i in np.arange(0, len(analog_curves), 1):
			n = np.array(analog_curves[i].transpose())
			external.append(n)
		external = np.array(external)

		accumulated_median = []
		for i in np.arange(0, len(external), 1):
			com = []
			accumulated_median.append(com)
			for j in np.arange(0, len(external[0]), 1):
				m = np.median(external[i][j])
				com.append(m)
		accumulated_median = np.array(accumulated_median)

		#THIS ARRAY CONTAINS THE NEEDED Y-AXIS DATA TO PLOT THE SECOND FIGURE
		return np.array([analog_curves, accumulated_median, self.accumulations[1]]) #np.array(analog_curves)

##############################################################################################################################################
	
	def get_graph3_curves(self): #It'll be the assembly

		graph2_curves = self.get_graph2_curves()

		
		assembly = []
		for i in np.arange(0, len(graph2_curves[0]), 1):
			asem = []
			assembly.append(asem)

			for j in np.arange(0, len(graph2_curves[0][0]), 1):
				link = list(graph2_curves[2].transpose()[i]) +  list(graph2_curves[0][i][j][len(graph2_curves[2].transpose()[0]):])  
	

		return assembly







class1 = proccess_data_to_plot(5)
z = class1.get_graph3_curves()

print(z)
