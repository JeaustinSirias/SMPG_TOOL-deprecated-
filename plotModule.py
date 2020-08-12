import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle
from io import *
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

	def seasonal_accumulations_plotting(self): #a function that calcs the output to plot seasonal accumulations 

		analog_yrs = self.analogs_dictionary[2] #it contains the analog years according to user's choice
		currentYearLastDek = len(self.accumulations[1].transpose()[0])-1

		analog_curves = [] #these are analog years to plot
		#choosing analogs to plot
		for i in np.arange(0, len(self.accumulations[2]), 1):
			curves = []
			analog_curves.append(curves)
			for j in np.arange(0, len(analog_yrs[0]), 1):
				com = self.accumulations[2][i][analog_yrs[i][j]]
				curves.append(com)

		analog_curves = np.array(analog_curves) #it contains ONLY the curves for chosen analog years

		
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

		#WE'RE GONNA GET STATICS BASED IN PLOTTABLE INFO FOR GRAPH 2
		biggest_accum_row = [] #an array to hold the lastest accumulations for each year in chosen analogs
		for i in np.arange(0, len(analog_curves), 1):
			z = []
			biggest_accum_row.append(z)
			for j in np.arange(0, len(analog_curves[0]), 1):
				k = analog_curves[i][j][-1]
				z.append(k)
		biggest_accum_row = np.array(biggest_accum_row)


		#We need to get the next stats:
		#SEASONAL AVERAGE
		#SEASONAL STD DEV
		#SEASONAL MEDIAN

		stats = [] #runup will be a statics array like [std_dev, 33rd, 67th, std+avg, std-avg]
		for i in np.arange(0, len(biggest_accum_row), 1):

			#thrd = np.percentile(biggest_accum_row[i], 33)
			#sixth = np.percentile(biggest_accum_row[i], 67)
			stDev = np.std(biggest_accum_row[i])
			#Avg = np.mean(biggest_accum_row[i])
			Med = np.median(biggest_accum_row[i])
			#std_add = LTM[i][-1] + dev 
			#std_sub = LTM[i][-1] - dev

			stats.append([LTM[i][-1], stDev, Med, LTM[i][currentYearLastDek]])
		#statics = np.array(statics)
		

		return [analog_curves, stats, analog_yrs]
		#return LTM[0][currentYearLastDek]

##############################################################################################################################################
	
	def seasonal_accumulations(self): #a funcion that calcs

		clim_window = np.arange(self.init_clim, self.end_clim + 1, 1) #a linspace with preffered years (keys for dict) in climatology
		accumulations = self.accumulations[2] #complete accumulations array(dictionary) for all location in every past year
		full_accum =  self.accumulations[0]
		currentYearLastDek = len(self.accumulations[1].transpose()[0])-1 #last dek as positional number in current year

	
		#we need the accumulation row at the end of season for all locations in one array, so:
		endSeasonRow_full = []
		for i in np.arange(0, len(full_accum), 1):
			t = []
			endSeasonRow_full.append(t)
			for j in np.arange(0, len(full_accum[0]), 1):

				lstNum = full_accum[i][j][-1]
				t.append(lstNum)

		endSeasonRow_full = np.array(endSeasonRow_full)

		#first we start by calculating the LTM red line which will be permanent in seasonal accumulations plot:
		#CHOOSING DATA TO GET CLIMATOLOGY BASED IN MEAN CURVE
		clim_accumulations = []
		endSeasonRow = []
		for i in np.arange(0, len(accumulations), 1):
			get = []
			get_c = []
			clim_accumulations.append(get)
			endSeasonRow.append(get_c)
			for k in clim_window:
				yr = accumulations[i][k]
				z = accumulations[i][k][-1]
				get.append(yr)
				get_c.append(z)

		endSeasonRow = np.array(endSeasonRow)
		clim_accumulations = np.array(clim_accumulations)

		#To get the accumulated rainfall mean FOR PAST YEARS for chosen analogs in CLIMATOLOGY ONLY!
		external = []
		for i in np.arange(0, len(clim_accumulations), 1):
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

		seasonal_climatology = np.array(accumulated_median) #THIS IS THE CLIMATOLOGY


		#****************************************COMPUTING STATS***************************************
		#Seasonal average
			#seasonal StDev
				#seasonal median
					#33rd percentil
						#67rd percentil 
							#StDev + Avg
								#StdDev - Avg

		statics = [] 
		for i in np.arange(0, len(endSeasonRow_full), 1): #for every location select an endSeasonRow array and get its stats

			thrd = np.percentile(endSeasonRow_full[i], 33)
			sixth = np.percentile(endSeasonRow_full[i], 67)
			stDev = np.std(endSeasonRow_full[i])
			std_add = seasonal_climatology[i][-1] + stDev
			std_sub = seasonal_climatology[i][-1] - stDev
			Med = np.median(endSeasonRow_full[i])

			statics.append([seasonal_climatology[i][-1], stDev, Med, thrd, sixth, std_add, std_sub, seasonal_climatology[i][currentYearLastDek]])
		statics = np.array(statics)
			

		return [seasonal_climatology, statics]

##############################################################################################################################################

	def ensemble_plotting(self):

		#we need again the analog curves
		years = self.analogs_dictionary[2]
		accumulations = self.accumulations[1].transpose()
		
		analog_curves = [] #these are analog years to plot
		#choosing analogs to plot
		for i in np.arange(0, len(self.accumulations[2]), 1):
			curves = []
			analog_curves.append(curves)
			for j in np.arange(0, len(years[0]), 1):
				com = self.accumulations[2][i][years[i][j]]
				curves.append(com)
		
		analog_curves = np.array(analog_curves) #it contains ONLY the curves for chosen analog years


		#this loop will take the [-1] element form the accumulated current year array and will start a new accumulation from this 
		#point for each location, in every past year until the dekad window ends i.e if my current year ends on 3-May dek, but muy chosen
		#dekad window ends on 1-Aug, then it'll create a (num_loc, num_years, 1-May - 1-Aug) array 

		#SETTING UP ENSEMBLE: it calculates the ensemble for all past years in dataset. 
		assembly = [] #it'll store the ensemble array
		for i in np.arange(0, len(analog_curves), 1): #for each location. We're only taking the size!
			n = accumulations[i][-1]
			asem = []
			assembly.append(asem)

			for j in np.arange(0, len(self.output_snack[2][0]), 1): #for each location 
				stamp = []
				asem.append(stamp)

				for k in np.arange(len(self.output_snack[3][0]), self.dek_dictionary[self.end_dek], 1):
					n = n + self.output_snack[2][i][j][k]
					stamp.append(n)

					if len(stamp) == len(np.arange(len(self.output_snack[3][0]), self.dek_dictionary[self.end_dek], 1)):
						n = accumulations[i][-1]

		#PREPARING ENSEMBLE ARRAY
		#the next loop is to cat the ensemble to current year 
		ensemble = []
		for i in np.arange(0, len(assembly), 1): #for each location read
			scat = []
			ensemble.append(scat)
			for j in np.arange(0, len(assembly[0]), 1): #for each year read

				link = list(accumulations[i]) + list(assembly[i][j]) #cat curren year deks and ensembled deks
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
			for j in years[i]:

				if j <= self.end_clim and j >= self.init_clim: #select analog year data in climatology window
					choose_yr_clim = dictionary[j]
					choose_array_clim = ensemble[i][choose_yr_clim]
					get_clim.append(choose_array_clim)

				choose_yr = dictionary[j]
				choose_array = ensemble[i][choose_yr]
				get.append(choose_array)

		ensemble_curves = np.array(ensemble_analogs) #this is my plotable output


		#GET MEAN DATA FOR ENSEMBLE ANALOGS INSIDE CLIMATOLOGY
		ensemble_avg = []
		for i in np.arange(0, len(ensemble_analogs_clim), 1): #for each location 
			z = np.array(ensemble_analogs_clim[i]).transpose()
			avg = []
			ensemble_avg.append(avg)
			for j in np.arange(0, len(z), 1):
				k = np.mean(z[j])
				avg.append(k)

		#GET STATS FOR PLOTTABLE ENSENSEMBLE

		endSeasonRow = [] #an array to hold the lastest accumulations for each year in chosen analogs
		for i in np.arange(0, len(analog_curves), 1):
			z = []
			endSeasonRow.append(z)
			for j in np.arange(0, len(analog_curves[0]), 1):
				k = ensemble_curves[i][j][-1]
				z.append(k)

		#****************************************COMPUTING STATS FOR PLOTTABLE ENSEMBLE***************************************
		#Seasonal average
			#seasonal StDev
				#seasonal median
					#33rd percentil
						#67rd percentil 
							#StDev + Avg
								#StdDev - Avg

		stats_EP = [] #PLOTABLE ENSEMBLE STATICS
		for i in np.arange(0, len(endSeasonRow), 1): #for each location in the array do:

			Med = np.median(endSeasonRow[i])
			thrd = np.percentile(endSeasonRow[i], 33)
			sixth = np.percentile(endSeasonRow[i], 67)
			stDev = np.std(endSeasonRow[i])
			std_add = ensemble_avg[i][-1] + stDev
			std_sub = ensemble_avg[i][-1] - stDev

			stats_EP.append([ensemble_avg[i][-1], stDev, Med, thrd, sixth, std_add, std_sub])


		#****************************************COMPUTING STATS FOR ENSEMBLE***************************************
		#Seasonal average
			#seasonal StDev
				#seasonal median
					#33rd percentil
						#67rd percentil 



		endSeasonRow_full = []
		for i in np.arange(0, len(ensemble), 1):
			k = []
			endSeasonRow_full.append(k)
			for j in np.arange(0, len(ensemble[0]), 1):
				t = ensemble[i][j][-1]
				k.append(t)

		stats_E = []
		for i in np.arange(0, len(endSeasonRow), 1): #for each location in the array do:

			Avg = np.mean(endSeasonRow_full[i])
			Med = np.median(endSeasonRow_full[i])
			thrd = np.percentile(endSeasonRow_full[i], 33)
			sixth = np.percentile(endSeasonRow_full[i], 67)
			stDev = np.std(endSeasonRow_full[i])
			std_add = ensemble_avg[i][-1] + stDev
			std_sub = ensemble_avg[i][-1] - stDev

			stats_E.append([Avg, stDev, Med, thrd, sixth])


						

		return [ensemble_curves, ensemble_avg, stats_EP, stats_E]
		
##############################################################################################################################################

	def generate_reports(self):

		#================================THIS SPACE IS TO PREPARE DATA TO BE LAUNCHED=======================================

		#Function callouts:
		current_yr_accum = self.accumulations[1].transpose() #contains the rainfall accumulated curve for current year. Plot in BLUE!
		stamp = self.seasonal_accumulations_plotting() #accumulation curves for seasonal accumulations plot
		stamp2 = self.seasonal_accumulations() 
		stamp3 = self.ensemble_plotting() #ensemble data to plot 

		#data
		climCurve = stamp2[0] #constant red lined curve based in climatology for seasonal accumulations and ensemble plots
		seasonalStats = stamp2[1] #it contains the points ad the end of season: 33rd, 67th, stDev +/- Avg
		analog_curves = stamp[0]
		analog_yrs = stamp[2]
		analog_stats1 = stamp[1]#it contains the seasonal accumulation stats but based on analog years
		curves_E = stamp3[0]
		E_avgCurve = stamp3[1]
		ensembleStats = stamp3[2]
		ensembleStatsFull = stamp3[3]

		#=================================================SETTING UP LINSPACES (x-axes)=======================================

		locNum = np.arange(0, len(analog_curves), 1) #it is a linspace that contains the array with locations number size
		yrSize = np.arange(0, len(analog_curves[0]), 1) #this is a linspace that contains the analog years size
		seasonSize = np.arange(0, len(analog_curves[0][0]), 1) #a linspace with size of the season 
		currentYr = np.arange(0, len(current_yr_accum[0]), 1) #current year linspace size
		seasonLabels = list(self.dek_dictionary.keys())[self.dek_dictionary[self.init_dek]-1:self.dek_dictionary[self.end_dek]]

		#=================================================SETTING UP SUBPLOTS/loop starts here=======================================

		for i in locNum:

			fig = plt.figure(num = i, tight_layout = True, figsize = (16, 9)) #figure number. There will be a report for each processed location
			fig_grid = gridspec.GridSpec(2,3) #we set a 2x2 grid space to place subplots
			avg_plot = fig.add_subplot(fig_grid[0, 0:2])
			seasonal_accum_plot = fig.add_subplot(fig_grid[1, 0])
			ensemble_plot = fig.add_subplot(fig_grid[1, 1])
			#tables:
			Asummary = fig.add_subplot(fig_grid[0:2, 2])
			#Bsummary = fig.add_subplot(fig_grid[1, 2])

			#AVG AND CURRENT RAINFALL SEASON:
			avg_plot.plot(np.arange(0, 36, 1), self.output_snack[-1][i], color = 'r', lw = 4, label = 'LT Avg [climatology based]: {init} - {end}'.format(init = self.init_clim, end = self.end_clim))
			avg_plot.bar(np.arange(0, len(self.output_snack[3][0]), 1), self.output_snack[3][i], color = 'b', label = 'Current year: {yr}'.format(yr = self.end_yr))
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


			#SEASONAL ACCUMULATIONS AND ENSEMBLE
			for j in yrSize:

				#SEASONAL ACUMULATIONS
				seasonal_accum_plot.plot(seasonSize, analog_curves[i][j], lw = 2, label = '{yr}'.format(yr = analog_yrs[i][j])) #accumulation curves

				#ENSEMBLE
				ensemble_plot.plot(seasonSize, curves_E[i][j], lw = 2, label = '{yr}'.format(yr = analog_yrs[i][j]))




			###############SEASONAL ACCUMULATIONS
			seasonal_accum_plot.plot(seasonSize, climCurve[i], color = 'r', lw = 5, label = 'LTM') #average
			seasonal_accum_plot.plot(currentYr, current_yr_accum[i], color = 'b', lw = 5, label = '{}'.format(self.end_yr)) #current year
			seasonal_accum_plot.fill_between(seasonSize, (climCurve[i])*1.2, (climCurve[i])*0.8, color = 'lightblue' ) #120 - 80% curve


			#stats
			seasonal_accum_plot.plot([seasonSize[-1]], [seasonalStats[i][5]], marker='^', markersize=7, color="green", label = 'Avg+Std')
			seasonal_accum_plot.plot([seasonSize[-1]], [seasonalStats[i][6]], marker='^', markersize=7, color="green", label = 'Avg-Std')
			seasonal_accum_plot.plot([seasonSize[-1]], [seasonalStats[i][3]], marker='s', markersize=7, color="k", label = '33rd pct')
			seasonal_accum_plot.plot([seasonSize[-1]], [seasonalStats[i][4]], marker='s', markersize=7, color="k", label = '67th pct')


			seasonal_accum_plot.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), fancybox=False, shadow=True, ncol=5, fontsize = 7.5)
			seasonal_accum_plot.set_title('Seasonal accumulations')
			seasonal_accum_plot.set_ylabel('Accum. rainfall [mm]')
			#seasonal_accum_plot.set_xlabel('Dekadals')
			seasonal_accum_plot.set_xticks(seasonSize)
			seasonal_accum_plot.set_xticklabels(seasonLabels, rotation = 'vertical')
			seasonal_accum_plot.grid()



			###############ENSEMBLE
			ensemble_plot.plot(seasonSize, E_avgCurve[i], '--', color = 'k', lw = 2, label = 'ELTM')
			ensemble_plot.plot(seasonSize, climCurve[i], color = 'r', lw = 5, label = 'LTM') #average
			ensemble_plot.fill_between(seasonSize, (climCurve[i])*1.2, (climCurve[i])*0.8, color = 'lightblue' ) #120 - 80% curve
			ensemble_plot.plot(currentYr, current_yr_accum[i], color = 'b', lw = 5, label = '{}'.format(self.end_yr)) #current year

			#statics
			ensemble_plot.plot([seasonSize[-1]], [seasonalStats[i][5]], marker='^', markersize=7, color="green", label = 'Avg+Std')
			ensemble_plot.plot([seasonSize[-1]], [seasonalStats[i][6]], marker='^', markersize=7, color="green", label = 'Avg-Std')
			ensemble_plot.plot([seasonSize[-1]], [seasonalStats[i][3]], marker='s', markersize=7, color="k", label = '33rd pct')
			ensemble_plot.plot([seasonSize[-1]], [seasonalStats[i][4]], marker='s', markersize=7, color="k", label = '67th pct')

			#ensemble statics
			ensemble_plot.plot([seasonSize[-1]], [ensembleStats[i][5]], marker='^', markersize=7, color="orange", label = 'E_Avg+Std')
			ensemble_plot.plot([seasonSize[-1]], [ensembleStats[i][6]], marker='^', markersize=7, color="orange", label = 'E_Avg-Std')
			ensemble_plot.plot([seasonSize[-1]], [ensembleStats[i][3]], marker='s', markersize=7, color="blue", label = 'E_33rd pct')
			ensemble_plot.plot([seasonSize[-1]], [ensembleStats[i][4]], marker='s', markersize=7, color="blue", label = 'E_67th pct')




			ensemble_plot.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), fancybox=True, shadow=True, ncol=5, fontsize = 7.5)
			ensemble_plot.set_xticks(seasonSize)
			ensemble_plot.set_xticklabels(seasonLabels, rotation = 'vertical')
			ensemble_plot.set_title('Ensemble')
			ensemble_plot.set_ylabel('Accumulated rainfall [mm]')
			#ensemble_plot.set_xlabel('Dekadals')
			ensemble_plot.grid()


			#================================================TABLES==================================================
			#ALL TABLES WILL NEED THE SAME COLUMNS: [ANALOGS, ALL YEARS]
			#SETUP
			Asummary.set_title('Summary Statistics. Season') #{a} - {b}'.format(a = self.init_dek, b = self.end_dek))
			Asummary.axis('tight')
			Asummary.axis('off')
			col = ('Analog years', 'All years')

			colC = ['coral']*len(col)
			rowC = ['lightsteelblue']
			headerColor = ['palegreen']


			#=======================ANALOG YEARS RANKING TABLE===================

			analog_row = []; analog_data = []; z = 0; y = self.analog_num
			while y > 1:
				y = y - z
				ar = 'Top {top}'.format(top = y)
				ad = [self.analogs_dictionary[1][i][y]]
				analog_row.append(ar)
				analog_data.append(ad)
				z = 1

			Asummary.table(rowLabels = analog_row, colLabels = ['Analog years ranking'], cellText = analog_data, cellLoc = 'center', bbox = [0.1, 0.7, 0.8, 0.3], colColours = colC, rowColours = rowC*len(analog_row))

			#====================CLIMATOLOGICAL ANALYSIS TABLE===================

			LTAvalP = round(analog_stats1[i][0], 2)
			LTAval = round(seasonalStats[i][0], 2)


			LTA_percP = (current_yr_accum[i][-1]/analog_stats1[i][3])*100
			LTA_perc =  (current_yr_accum[i][-1]/seasonalStats[i][-1])*100

			#HEADER
			Asummary.table(cellText = [[None]], colLabels = ['Climatological Analysis'], bbox = [0.2, 0.55, 0.7, 0.12 ], colColours = headerColor)


			row = ['Seasonal Average', 'Seasonal Std. Dev', 'Seasonal median', 'Total at current dek.', 'LTA Value', 'LTA percentage']
			txt = [[LTAvalP, LTAval], 
					[round(analog_stats1[i][1], 1), round(seasonalStats[i][1], 2)], 
					[round(analog_stats1[i][2], 2), round(seasonalStats[i][2], 2)], 
					[round(current_yr_accum[i][-1], 2), round(current_yr_accum[i][-1], 2)], 
					[round(analog_stats1[i][3],2), round(seasonalStats[i][-1], 2)], 
					[round(LTA_percP, 2), round(LTA_perc, 2)]]

			Asummary.table(rowLabels = row, colLabels = col, cellText = txt, loc = 'center', cellLoc = 'center', bbox = [0.2, 0.32, 0.7, 0.3], colColours = colC, rowColours = rowC*len(row))

			#====================CURRENT YEAR ANALYSIS [ENSEMBLE] TABLE===================

			#HEADER
			Asummary.table(cellText = [[None]], colLabels = ['Current year analysis: {yr}'.format(yr = self.end_yr)], bbox = [0.2, 0.16, 0.7, 0.12 ], colColours = headerColor)


			seasonalAvgP = round(ensembleStats[i][0], 2)
			seasonalAvg = round(ensembleStatsFull[i][0], 2)
			
			LTApercP = (seasonalAvgP/LTAvalP)*100
			LTAperc = (seasonalAvg/LTAval)*100

			row_B = ['Ensemble Average', 'Ensemble Std. Dev', 'Ensemble median', '33rd. Percentile', '67th Percentile', 'LTA Val. End of season', 'LTA Percentage']
			data_B =[[seasonalAvgP, seasonalAvg], 
					[round(ensembleStats[i][1], 2), round(ensembleStatsFull[i][1], 2)], 
					[round(ensembleStats[i][2], 2), round(ensembleStatsFull[i][2], 2)], 
					[round(ensembleStats[i][3], 2), round(ensembleStatsFull[i][3], 2)], 
					[round(ensembleStats[i][4], 2), round(ensembleStatsFull[i][4], 2)], 
					[LTAvalP, LTAval], 
					[round(LTApercP, 2), round(LTAperc, 2)]]

			Asummary.table(rowLabels = row_B, colLabels = col, cellText = data_B, loc = 'center', cellLoc = 'center', bbox = [0.2, -0.08, 0.7, 0.3], colColours = colC, rowColours = rowC*len(row_B))

			#===================OUTLOOK PROBABILITY TABLE=======================

			#HEADER
			Asummary.table(cellText = [[None]], colLabels = ['Outlook: Probability at the end of season'], bbox = [0.2, -0.23, 0.7, 0.12 ], colColours = headerColor)

			outlook_row = ['Above normal', 'Normal', 'Below normal']
			data = [['None', 'None'], ['None', 'None'], ['None', 'None']]
			Asummary.table(rowLabels = outlook_row, colLabels = col, cellText = data, cellLoc = 'center', bbox = [0.2, -0.36, 0.7, 0.2], colColours = colC, rowColours = rowC*len(outlook_row))















			fig.align_labels()
		
		return plt.show()

##############################################################################################################################################

#sample = proccess_data_to_plot(None, 1981, 2020, '1-Feb', '3-May', 1981, 2010)
#sample.generate_reports()
#print(sample.seasonal_accumulations_plotting())