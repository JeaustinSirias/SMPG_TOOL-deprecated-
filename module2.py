import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle
from io import *
#import pandas as pd
from scipy.stats import rankdata

#A module to prepare data to be plotted

class proccess_data_to_plot():
	def __init__(self, analog_input, init_yr, end_yr, init_dek, end_dek): #analog input will be the amount of analog years selected by user

		self.init_yr = init_yr
		self.end_yr = end_yr
		self.init_dek = init_dek
		self.end_dek = end_dek

		self.analogs_dictionary = np.array(pickle.load(open('./datapath/analogs', 'rb')))
		self.analog_num = analog_input
		self.accumulations = np.array(pickle.load(open('./datapath/accumulations', 'rb'))) #we're gonna need the third output
		self.output_snack = pickle.load(open('./datapath/output_snack', 'rb')) #[full_data_median, current_year, raw_years]
		self.dek_dictionary = pickle.load(open('./datapath/dekads_dictionary', 'rb')) #a dictionary of dekads

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
		return np.array([analog_curves, accumulated_median, self.accumulations[1], years]) #np.array(analog_curves)

##############################################################################################################################################
	
	def get_graph3_curves(self): #It'll be the assembly

		graph2_curves = self.get_graph2_curves()

		assembly = []
		for i in np.arange(0, len(graph2_curves[0]), 1):
			asem = []
			assembly.append(asem)

			for j in np.arange(0, len(graph2_curves[0][0]), 1):
				link = list(graph2_curves[2].transpose()[i]) +  list(graph2_curves[0][i][j][len(graph2_curves[2].transpose()[0]):]) 
				asem.append(link) 
	

		return np.array([np.array(assembly).transpose(), graph2_curves[0], graph2_curves[1], graph2_curves[2], graph2_curves[3]])

##############################################################################################################################################

	def plot_report(self):
		g3 = self.get_graph3_curves()

		#we need to plot a 3 subplots report 
		for i in np.arange(0, len(g3[1]), 1):

			fig = plt.figure(num = i, tight_layout = True, figsize = (11, 8)) #figure number. There will be a report for each processed location
			fig_grid = gridspec.GridSpec(2,2) #we set a 2x2 grid space to place subplots
			avg_plot = fig.add_subplot(fig_grid[0, :])
			seasonal_accum_plot = fig.add_subplot(fig_grid[1, 0])
			ensemble_plot = fig.add_subplot(fig_grid[1, 1])

			#AVG AND CURRENT RAINFALL SEASON:
			avg_plot.plot(np.arange(0, 36, 1), self.output_snack[0][i], color = 'r', lw = 4, label = 'LT Avg. {init} - {end}'.format(init = self.init_yr, end = self.end_yr - 1))
			avg_plot.bar(np.arange(0, len(self.output_snack[3][0]), 1), self.output_snack[3][i], color = 'b', label = 'Current year = {yr}'.format(yr = self.end_yr))
			avg_plot.legend()
			avg_plot.set_title('Average & current rainfall season: location {num}'.format(num = i))
			avg_plot.set_ylabel('Rainfall [mm]')
			avg_plot.set_xlabel('Dekadals')
			avg_plot.set_xticks(np.arange(0, 36, 1))
			avg_plot.set_xticklabels(('1-Jan', '2-Jan', '3-Jan', '1-Feb', '2-Feb', '3-Feb', '1-Mar', '2-Mar', '3-Mar', '1-Apr', '2-Apr', '3-Apr', '1-May', '2-May', '3-May', '1-Jun',
		 			'2-Jun', '3-Jun', '1-Jul', '2-Jul', '3-Jul', '1-Aug', '2-Aug', '3-Aug', '1-Sep', '2-Sep', '3-Sep', '1-Oct', '2-Oct', '3-Oct', '1-Nov', '2-Nov', '3-Nov', '1-Dec', '2-Dec', '3-Dec'), rotation = 'vertical')
			avg_plot.grid()

			#ENSEMBLE AND SEASONAL ACCUMULATIONS:
			for j in np.arange(0, len(g3[1][0]), 1):

				#SEASONAL ACUMULATIONS
				seasonal_accum_plot.plot(np.arange(0, len(g3[1][0][0]), 1), g3[1][i][j], lw = 2, label = 'Analog {num}: {yr}'.format(num = j+1, yr = g3[4][i][j])) #accumulation curves

				#ESEMBLE
				ensemble_plot.plot(np.arange(0, len(g3[1][0][0]), 1), g3[0].transpose()[i][j], lw = 2, label = 'Analog {num}: {yr}'.format(num = j+1, yr = g3[4][i][j]))

			#SEASONAL ACCUMULATIONS
			seasonal_accum_plot.plot(np.arange(0, len(g3[1][0][0]), 1), g3[2][i], color = 'r', lw = 5, label = 'LTM') #average
			seasonal_accum_plot.plot(np.arange(0, len(g3[3].transpose()[0]), 1), g3[3].transpose()[i], color = 'b', lw = 5, label = '{}'.format(self.end_yr)) #current year
			seasonal_accum_plot.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), fancybox=True, shadow=True, ncol=3)
			seasonal_accum_plot.set_title('Seasonal accumulations')
			seasonal_accum_plot.set_ylabel('Accum. rainfall [mm]')
			seasonal_accum_plot.set_xlabel('Dekadals')
			seasonal_accum_plot.set_xticks(np.arange(0, len(g3[1][0][0]), 1))
			seasonal_accum_plot.set_xticklabels(list(self.dek_dictionary.keys())[self.dek_dictionary[self.init_dek]-1:self.dek_dictionary[self.end_dek]], rotation = 'vertical')
			seasonal_accum_plot.grid()

			#ENSEMBLE
			ensemble_plot.plot(np.arange(0, len(g3[1][0][0]), 1), g3[2][i], color = 'r', lw = 5, label = 'LTM') #average
			ensemble_plot.plot(np.arange(0, len(g3[3].transpose()[0]), 1), g3[3].transpose()[i], color = 'b', lw = 5, label = '{}'.format(self.end_yr)) #current year
			ensemble_plot.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), fancybox=True, shadow=True, ncol=3)
			ensemble_plot.set_xticks(np.arange(0, len(g3[1][0][0]), 1))
			ensemble_plot.set_xticklabels(list(self.dek_dictionary.keys())[self.dek_dictionary[self.init_dek]-1:self.dek_dictionary[self.end_dek]], rotation = 'vertical')
			ensemble_plot.set_title('Ensemble')
			ensemble_plot.set_ylabel('Accumulated rainfall [mm]')
			ensemble_plot.set_xlabel('Dekadals')
			ensemble_plot.grid()

			fig.align_labels()

		return plt.show()
	
##############################################################################################################################################


#class1 = proccess_data_to_plot(6, 1985, 2019, '1-May', '3-Nov')
#class1.plot_report()



'''
class1 = proccess_data_to_plot(6)
#g2 = class1.get_graph2_curves()
g3 = class1.get_graph3_curves()

print(g3[3].transpose()[0])

#print(g3[0].transpose().shape)
#print(g3[1][17][4])

output_snack = pickle.load(open('./datapath/output_snack', 'rb')) #[full_data_median, current_year, raw_years]

#print(output_snack[3][0])


for i in np.arange(0, len(g3[1]), 1):

	fig = plt.figure(num = i, tight_layout = True)
	gs = gridspec.GridSpec(2,2)
	ax = fig.add_subplot(gs[0, :])
	ax1 = fig.add_subplot(gs[1, 0])
	ax2 = fig.add_subplot(gs[1, 1])
	#fig, (ax1, ax2) = plt.subplots(1, 2)

	ax.plot(np.arange(0, 36, 1), output_snack[0][i], color = 'r', lw = 2)
	ax.bar(np.arange(0, len(output_snack[3][0]), 1), output_snack[3][i], color = 'b')
	ax.grid()
	

	for j in np.arange(0, len(g3[1][0]), 1):
		#ax.plot(np.arange(0, len(g3[1][0][0]), 1), g3[0].transpose()[i][j])
		ax2.plot(np.arange(0, len(g3[1][0][0]), 1), g3[0].transpose()[i][j])
		ax1.plot(np.arange(0, len(g3[1][0][0]), 1), g3[1][i][j])

		ax1.set_ylabel('Accumulated rainfall [mm]')
		ax1.set_xlabel('Dekadals')

		ax2.set_ylabel('Accumulated rainfall [mm]')
		ax2.set_xlabel('Dekadals')

	ax1.grid()
	ax2.grid()

	ax2.set_title('Ensemble')
	ax1.set_title('seasonal accumulations')


	ax1.plot(np.arange(0, len(g3[1][0][0]), 1), g3[2][i], color = 'r', lw = 3)
	ax2.plot(np.arange(0, len(g3[1][0][0]), 1), g3[2][i], color = 'r', lw = 3)
	ax1.plot(np.arange(0, len(g3[3].transpose()[0]), 1), g3[3].transpose()[i], color = 'b', lw = 3)


	ax.set_title('Average & current rainfall season: location {num}'.format(num = i))
	ax.set_ylabel('Rainfall [mm]')
	ax.set_xlabel('Dekadals')
	ax.set_xticks(np.arange(0, 36, 1))
	ax.set_xticklabels(('1-Jan', '2-Jan', '3-Jan', '1-Feb', '2-Feb', '3-Feb', '1-Mar', '2-Mar', '3-Mar', '1-Apr', '2-Apr', '3-Apr', '1-May', '2-May', '3-May', '1-Jun',
		 			'2-Jun', '3-Jun', '1-Jul', '2-Jul', '3-Jul', '1-Aug', '2-Aug', '3-Aug', '1-Sep', '2-Sep', '3-Sep', '1-Oct', '2-Oct', '3-Oct', '1-Nov', '2-Nov', '3-Nov', '1-Dec', '2-Dec', '3-Dec'), rotation = 'vertical')

	fig.align_labels()

plt.show()
'''
		


'''
for i in np.arange(0, len(g3[0].transpose()), 1):

	plt.figure(num = i, figsize = (10, 6))
	for k in g3[0].transpose()[i]:
		plt.plot(np.arange(0, len(g3[0].transpose()[0][0]), 1), k)

plt.show()
'''
'''
for i in np.arange(0, len(g3[1].transpose()), 1):
	fig = plt.figure(num = i)
	for k in np.arange(0, len(g3[0][0][0]), 1):
		fig, (ax1, ax2) = plt.subplots(1, 2)
		ax1.plot(np.arange(0, 22, 1), g3[1][i][k])
		ax2.plot(np.arange(0, 22, 1), g3[0].transpose()[i][k])

plt.show()
'''