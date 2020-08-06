import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle
from io import *
#import pandas as pd
from scipy.stats import rankdata

#A module to prepare data to be plotted

class proccess_data_to_plot():
	def __init__(self, analog_input, init_yr, end_yr, init_dek, end_dek, init_clim, end_clim): #analog input will be the amount of analog years selected by user

		self.init_yr = init_yr
		self.end_yr = end_yr

		self.init_dek = init_dek
		self.end_dek = end_dek

		self.init_clim = init_clim
		self.end_clim = end_clim

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
		for i in np.arange(0, len(self.analogs_dictionary[0]), 1): #for each location available
			camp = []
			accumulation_analog_yrs.append(camp)
			for j in np.arange(0, len(analogs), 1):
				get = self.analogs_dictionary[0][i][analogs[j]]
				camp.append(get)

		return np.array(accumulation_analog_yrs)


##############################################################################################################################################
	
	def get_graph2_curves(self): #now we get the data for every analog year (accumulations)

		years = self.get_analog_accumulation()
		
		analog_curves = [] #these are analog years to plot
		#choosing analogs to plot
		for i in np.arange(0, len(self.accumulations[2]), 1):
			curves = []
			analog_curves.append(curves)
			for j in np.arange(0, len(years[0]), 1):
				com = self.accumulations[2][i][years[i][j]]
				curves.append(com)
		
		analog_curves = np.array(analog_curves) #it contains ONLY the curves for chosen analog years

		'''
		#choosing analog years found in climatology window. This is usefull to calc climatology based in rainfall accumulations
		analog_curves_clim = [] #it'll only contain the curves for analog years between chosen climatology window
		for i in np.arange(0, len(self.accumulations[2]), 1):
			curves_clim = []
			analog_curves_clim.append(curves_clim)
			for j in np.arange(0, len(years[0]), 1):
				#if self.end_clim >= years[i][j] and years[i][j] >= self.init_clim: #si el ano analogo esta entre 2010 y 1981
				if years[i][j] <= self.end_clim and years[i][j] >= self.init_clim:
					com_clim = self.accumulations[2][i][years[i][j]]
					curves_clim.append(com_clim)
		'''

		#CHOOSING YEARS TO GET CLIMATOLOGY BASED MEAN CURVE

		clim_window = np.arange(self.init_clim, self.end_clim + 1, 1) #a linspace with preffered years (keys for dict) in climatoloy

		clim_accumulations = []
		for i in np.arange(0, len(analog_curves), 1):
			get = []
			clim_accumulations.append(get)
			for k in clim_window:
				yr = self.accumulations[2][i][k]
				get.append(yr)
	
		clim_accumulations = np.array(clim_accumulations)
		
		#To get the accumulated rainfall mean FOR PAST YEARS for chosen analogs in CLIMATOLOGY ONLY!
		external = []
		for i in np.arange(0, len(analog_curves), 1):
			n = clim_accumulations[i].transpose()
			external.append(n)
		#external = np.array(external).transpose()

		accumulated_median = []
		for i in np.arange(0, len(external), 1):
			com = []
			accumulated_median.append(com)
			for j in np.arange(0, len(external[0]), 1):
				m = np.mean(external[i][j])
				com.append(m)

		accumulated_median = np.array(accumulated_median) #THIS IS THE CLIMATOLOGY


		#WE'RE GONNA GET STATICS BASED IN PLOTTABLE INFO FOR GRAPH 2
		biggest_accum_row = [] #an array to hold the lastest accumulations for each year in chosen analogs
		for i in np.arange(0, len(analog_curves), 1):
			z = []
			biggest_accum_row.append(z)
			for j in np.arange(0, len(analog_curves[0]), 1):
				k = analog_curves[i][j][-1]
				z.append(k)
		biggest_accum_row = np.array(biggest_accum_row)


		#also we need to compute the LTM based in analog_curves so:
		analog_curves_T = [] #it transposes the years to get rows for the same dek
		for i in np.arange(0, len(analog_curves), 1):
			n = analog_curves[i].transpose()
			analog_curves_T.append(n)

		LTM = [] #long term mean based in analog years
		for i in np.arange(0, len(analog_curves_T), 1):
			p = []
			LTM.append(p)
			for j in np.arange(0, len(analog_curves_T[0]), 1):
				k = np.mean(analog_curves_T[i][j])
				p.append(k)

		LTM = np.array(LTM)



		#GET STANDARD DEVIATION, 33rd, 67th pertenciles, std+avg, std-avg
		statics = [] #runup will be a statics array like [std_dev, 33rd, 67th, std+avg, std-avg]
		for i in np.arange(0, len(biggest_accum_row), 1):

			thrd = np.percentile(biggest_accum_row[i], 33)
			sixth = np.percentile(biggest_accum_row[i], 67)
			dev = np.std(biggest_accum_row[i])
			std_add = LTM[i][-1] + dev 
			std_sub = LTM[i][-1] - dev

			statics.append([dev, thrd, sixth, std_add, std_sub])
		

		#THIS ARRAY CONTAINS THE NEEDED Y-AXIS DATA TO PLOT THE SECOND FIGURE
		return np.array([analog_curves, accumulated_median, self.accumulations[1], years, statics, clim_accumulations ]) #np.array(analog_curves)
		#return statics
		#return np.array(analog_curves).shape
	
		
##############################################################################################################################################

	def get_graph3_curves(self): #It'll be the assembly

		graph2_curves = self.get_graph2_curves()

		#this loop will take the [-1] element form the accumulated current year array and will start a new accumulation from this 
		#point for each location, in every past year until the dekad window ends i.e if my current year ends on 3-May dek, but muy chosen
		#dekad window ends on 1-Aug, then it'll create a (num_loc, num_years, 1-May - 1-Aug) array 
		
		#SETTING UP ENSEMBLE: it calculates the ensemble for all past years in dataset. 
		assembly = [] #it'll store the ensemble array
		for i in np.arange(0, len(graph2_curves[0]), 1): #for each location. We're only taking the size!
			n = graph2_curves[2].transpose()[i][-1]
			asem = []
			assembly.append(asem)

			for j in np.arange(0, len(self.output_snack[2][0]), 1): #for each location 
				stamp = []
				asem.append(stamp)

				for k in np.arange(len(self.output_snack[3][0]), self.dek_dictionary[self.end_dek], 1):
					n = n + self.output_snack[2][i][j][k]
					stamp.append(n)

					if len(stamp) == len(np.arange(len(self.output_snack[3][0]), self.dek_dictionary[self.end_dek], 1)):
						n = graph2_curves[2].transpose()[i][-1]


		#PREPARING ENSEMBLE ARRAY
		#the next loop is to cat the ensemble to current year 
		ensemble = []
		for i in np.arange(0, len(assembly), 1): #for each location read
			scat = []
			ensemble.append(scat)
			for j in np.arange(0, len(assembly[0]), 1): #for each year read

				link = list(graph2_curves[2].transpose()[i]) + list(assembly[i][j]) #cat curren year deks and ensembled deks
				scat.append(link)

		ensemble = np.array(ensemble)

		#now we choose which year I have to keep in
		#create a dictionary to simplify ensembled curves selection according to chosen analog years by users
		yrs = np.arange(self.init_yr, self.end_yr, 1)
		num_yrs = np.arange(0, len(yrs), 1)
		dictionary = dict(zip(yrs, num_yrs))

		#save ensemble for only analog years chosen
		ensemble_analogs = [] #ordinary analog years ensemble curves
		ensemble_analogs_clim = [] #specific analog year ensemble curves in climatology window
		for i in np.arange(0, len(ensemble), 1):
			get = []
			get_clim = []
			ensemble_analogs.append(get)
			ensemble_analogs_clim.append(get_clim)
			for j in graph2_curves[3][i]:

				if j <= self.end_clim and j >= self.init_clim: #select analog year data in climatology window
					choose_yr_clim = dictionary[j]
					choose_array_clim = ensemble[i][choose_yr_clim]
					get_clim.append(choose_array_clim)

				choose_yr = dictionary[j]
				choose_array = ensemble[i][choose_yr]
				get.append(choose_array)

		ensemble_analogs = np.array(ensemble_analogs)

		#get mean data for ensemble, but based in analog years inside climatology window bounds
		ensemble_avg = []
		for i in np.arange(0, len(ensemble_analogs_clim), 1): #for each location 
			z = np.array(ensemble_analogs_clim[i]).transpose()
			avg = []
			ensemble_avg.append(avg)
			for j in np.arange(0, len(z), 1):
				k = np.mean(z[j])
				avg.append(k)

		#FINALLY WE GET STATICS FOR ENSEMBLE: std_dev, 33rd, 67th, avg+std, avg-std
		#first of all we're gonna get the last row in ensemble accumulations 
		biggest_accum_row = [] #an array to hold the lastest accumulations for each year in chosen analogs
		for i in np.arange(0, len(ensemble_analogs), 1):
			z = []
			biggest_accum_row.append(z)
			for j in np.arange(0, len(ensemble_analogs[0]), 1):
				k = ensemble_analogs[i][j][-1]
				z.append(k)

		biggest_accum_row = np.array(biggest_accum_row)

		statics_E = [] #statics will be a statics array like [std_dev, 33rd, 67th, std+avg, std-avg]
		for i in np.arange(0, len(biggest_accum_row), 1):

			thrd = np.percentile(biggest_accum_row[i], 33)
			sixth = np.percentile(biggest_accum_row[i], 67)
			dev = np.std(biggest_accum_row[i])
			std_add = ensemble_avg[i][-1] + dev 
			std_sub = ensemble_avg[i][-1] - dev

			statics_E.append([dev, thrd, sixth, std_add, std_sub])

		statics_E = np.array(statics_E)




		#return np.array([ensemble.transpose(), graph2_curves[0], graph2_curves[1], graph2_curves[2], graph2_curves[3], np.array(ensemble_avg), graph2_curves[4], statics_E ])
		return statics_E
		#return np.arange(len(graph2_curves[2].transpose()[0]), self.dek_dictionary[self.end_dek], 1) #len(list(self.output_snack[2][0][0][0])), 1)
		#return graph2_curves[2].transpose()[0][14]

##############################################################################################################################################

	def plot_report(self):

		#========================================THIS SPACE IS TO PREPARE DATA TO BE LAUNCHED IN PLOTS================================= 
		g3 = self.get_graph3_curves()

		#kk = self.seasonal_accumulations_plotting()




		x = np.arange(0, len(g3[1][0][0]), 1) #this is the seasonal linspace i.e, 1-Jan - 3-Jun

		#we need to plot a 3 subplots report 
		for i in np.arange(0, len(g3[1]), 1):

			#*******************************************SUBPLOT SETUP******************************************************
			
			#plots:
			fig = plt.figure(num = i, tight_layout = True, figsize = (16, 9)) #figure number. There will be a report for each processed location
			fig_grid = gridspec.GridSpec(2,3) #we set a 2x2 grid space to place subplots
			avg_plot = fig.add_subplot(fig_grid[0, 0:2])
			seasonal_accum_plot = fig.add_subplot(fig_grid[1, 0])
			ensemble_plot = fig.add_subplot(fig_grid[1, 1])
			#tables:
			Asummary = fig.add_subplot(fig_grid[0, 2])
			Bsummary = fig.add_subplot(fig_grid[1, 2])



			#plt.subplots_adjust(left=0.5, bottom=0.5)

			#AVG AND CURRENT RAINFALL SEASON:
			avg_plot.plot(np.arange(0, 36, 1), self.output_snack[-1][i], color = 'r', lw = 4, label = 'LT Avg [climatology based]: {init} - {end}'.format(init = self.init_clim, end = self.end_clim))
			avg_plot.bar(np.arange(0, len(self.output_snack[3][0]), 1), self.output_snack[3][i], color = 'b', label = 'Current year: {yr}'.format(yr = self.end_yr))
			#avg_plot.bar([self.output_snack[3][0][-1]], self.output_snack[3][i][-1], color = 'purple') #label = 'Current year: {yr}'.format(yr = self.end_yr))
			avg_plot.legend()
			try:
				avg_plot.set_title('Average & current rainfall season: {num}'.format(num = self.output_snack[4][i]))

			except:
				avg_plot.set_title('Average & current rainfall season: location {num}'.format(num = i))

			avg_plot.set_ylabel('Rainfall [mm]')
			avg_plot.set_xlabel('Dekadals')
			avg_plot.set_xticks(np.arange(0, 36, 1))
			avg_plot.set_xticklabels(('1-Jan', '2-Jan', '3-Jan', '1-Feb', '2-Feb', '3-Feb', '1-Mar', '2-Mar', '3-Mar', '1-Apr', '2-Apr', '3-Apr', '1-May', '2-May', '3-May', '1-Jun',
		 			'2-Jun', '3-Jun', '1-Jul', '2-Jul', '3-Jul', '1-Aug', '2-Aug', '3-Aug', '1-Sep', '2-Sep', '3-Sep', '1-Oct', '2-Oct', '3-Oct', '1-Nov', '2-Nov', '3-Nov', '1-Dec', '2-Dec', '3-Dec'), rotation = 'vertical')
			avg_plot.grid()


			#120% to 80% avg range plot
			seasonal_accum_plot.fill_between(x, (g3[2][i])*1.2, (g3[2][i])*0.8, color = 'lightblue' )
			ensemble_plot.fill_between(x, (g3[2][i])*1.2, (g3[2][i])*0.8, color = 'lightblue' )



			#ENSEMBLE AND SEASONAL ACCUMULATIONS:
			for j in np.arange(0, len(g3[1][0]), 1):

				#SEASONAL ACUMULATIONS
				seasonal_accum_plot.plot(x, g3[1][i][j], lw = 2, label = '{yr}'.format(yr = g3[4][i][j])) #accumulation curves

				#ESEMBLE
				ensemble_plot.plot(x, g3[0].transpose()[i][j], lw = 2, label = '{yr}'.format(yr = g3[4][i][j]))

			#SEASONAL ACCUMULATIONS
			seasonal_accum_plot.plot(x, g3[2][i], color = 'r', lw = 5, label = 'LTM') #average
			seasonal_accum_plot.plot(np.arange(0, len(g3[3].transpose()[0]), 1), g3[3].transpose()[i], color = 'b', lw = 5, label = '{}'.format(self.end_yr)) #current year

			#statics
			seasonal_accum_plot.plot([x[-1]], [g3[6][i][3]], marker='^', markersize=7, color="green", label = 'Avg+Std')
			seasonal_accum_plot.plot([x[-1]], [g3[6][i][4]], marker='^', markersize=7, color="green", label = 'Avg-Std')
			seasonal_accum_plot.plot([x[-1]], [g3[6][i][1]], marker='s', markersize=7, color="k", label = '33rd pct')
			seasonal_accum_plot.plot([x[-1]], [g3[6][i][2]], marker='s', markersize=7, color="k", label = '67th pct')


			seasonal_accum_plot.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), fancybox=False, shadow=True, ncol=4)
			seasonal_accum_plot.set_title('Seasonal accumulations')
			seasonal_accum_plot.set_ylabel('Accum. rainfall [mm]')
			#seasonal_accum_plot.set_xlabel('Dekadals')
			seasonal_accum_plot.set_xticks(np.arange(0, len(g3[1][0][0]), 1))
			seasonal_accum_plot.set_xticklabels(list(self.dek_dictionary.keys())[self.dek_dictionary[self.init_dek]-1:self.dek_dictionary[self.end_dek]], rotation = 'vertical')
			seasonal_accum_plot.grid()


			#ENSEMBLE
			#ensemble_plot.plot(np.arange(0, len(g3[1][0][0]), 1), g3[5][i], '--', color = 'k', lw = 2, label = 'ELTM')
			ensemble_plot.plot(x, g3[2][i], color = 'r', lw = 5, label = 'LTM') #average
			ensemble_plot.plot(x, g3[5][i], '--', color = 'k', lw = 2, label = 'ELTM')

			#statics
			ensemble_plot.plot([np.arange(0, len(g3[1][0][0]), 1)[-1]], [g3[6][i][3]], marker='^', markersize=7, color="green", label = 'Avg+Std')
			ensemble_plot.plot([np.arange(0, len(g3[1][0][0]), 1)[-1]], [g3[6][i][4]], marker='^', markersize=7, color="green", label = 'Avg-Std')
			ensemble_plot.plot([np.arange(0, len(g3[1][0][0]), 1)[-1]], [g3[6][i][1]], marker='s', markersize=7, color="k", label = '33rd pct')
			ensemble_plot.plot([np.arange(0, len(g3[1][0][0]), 1)[-1]], [g3[6][i][2]], marker='s', markersize=7, color="k", label = '67th pct')

			#statics ensemble
			ensemble_plot.plot([np.arange(0, len(g3[1][0][0]), 1)[-1]], [g3[-1][i][3]], marker='^', markersize=7, color="orange", label = 'E_Avg+Std')
			ensemble_plot.plot([np.arange(0, len(g3[1][0][0]), 1)[-1]], [g3[-1][i][4]], marker='^', markersize=7, color="orange", label = 'E_Avg-Std')
			ensemble_plot.plot([np.arange(0, len(g3[1][0][0]), 1)[-1]], [g3[-1][i][1]], marker='s', markersize=7, color="blue", label = 'E_33rd pct')
			ensemble_plot.plot([np.arange(0, len(g3[1][0][0]), 1)[-1]], [g3[-1][i][2]], marker='s', markersize=7, color="blue", label = 'E_67th pct')



			ensemble_plot.plot(np.arange(0, len(g3[3].transpose()[0]), 1), g3[3].transpose()[i], color = 'b', lw = 5, label = '{}'.format(self.end_yr)) #current year
			ensemble_plot.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), fancybox=True, shadow=True, ncol=4)
			ensemble_plot.set_xticks(np.arange(0, len(g3[1][0][0]), 1))
			ensemble_plot.set_xticklabels(list(self.dek_dictionary.keys())[self.dek_dictionary[self.init_dek]-1:self.dek_dictionary[self.end_dek]], rotation = 'vertical')
			ensemble_plot.set_title('Ensemble')
			ensemble_plot.set_ylabel('Accumulated rainfall [mm]')
			#ensemble_plot.set_xlabel('Dekadals')
			ensemble_plot.grid()

			####################################TABLES################################################

			#CLIAMOTOLOGICAL ANALYISIS
			row = ['Seasonal Avg', 'Seasonal Std', 'Seasonal median', 'Total at current dek.', 'LTA Value', 'LTA percentage']
			txt = [[127, 129], [234, 333], [190, 123], [127, 129], [234, 333], [190, 123]]
			col = ('Analogs', 'All years')
			Asummary.set_title('climatological analysis')
			Asummary.axis('tight')
			Asummary.axis('off')
			Asummary.table(rowLabels = row, colLabels = col, cellText = txt, loc = 'center', cellLoc = 'center', bbox = [0.25, 0.35, 0.8, 0.6], colColours = ['coral']*len(col), rowColours = ['greenyellow']*len(row))
			
			#OUTLOOK
			outlook_row = ['Above normal', 'Normal', 'Below normal']
			data = [[12, 23], [45, 32], [45, 12]]
			Asummary.table(rowLabels = outlook_row, colLabels = col, cellText = data, cellLoc = 'center', bbox = [0.35, -0.1, 0.5, 0.3])

			#CURRENT YEAR ANALYSIS
			Bsummary.set_title('Current year analysis')
			Bsummary.axis('tight')
			Bsummary.axis('off')
			Bsummary.table(rowLabels = row, colLabels = col, cellText = txt, loc = 'center', cellLoc = 'center', bbox = [0.25, 0.35, 0.8, 0.6])

			#ANALOG YEARS 
			analog_row = []; analog_data = []; z = 0; y = 6
			while y > 1:
				y = y - z
				ar = 'analog {top}'.format(top = y)
				ad = [self.analogs_dictionary[1][i][y]]
				analog_row.append(ar)
				analog_data.append(ad)
				z = 1

			Bsummary.table(rowLabels = analog_row, colLabels = ['Years'], cellText = analog_data, cellLoc = 'center', bbox = [0.15, -0.4, 0.8, 0.6] )

			
			fig.align_labels()

		return plt.show()

##############################################################################################################################################



#class1 = proccess_data_to_plot(39, 1981, 2020, '1-Feb', '3-May', 1981, 2010)
#print(class1.get_graph3_curves())

