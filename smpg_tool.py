from tkinter import *
import tkinter.messagebox
import ttk
import numpy as np
#import numpy.random.common
#import numpy.random.bounded_integers
#import numpy.random.entropy
import matplotlib.pyplot as plt
from scipy.stats import rankdata
from collections import defaultdict
import matplotlib.gridspec as gridspec
import pandas as pd
from tkinter.ttk import *
#from tkinter.filedialog import askopenfile 
import os
import sys
from tkinter import filedialog
from cycler import cycler
import random


#functions
##################################################################################################################################

def compute_median(init_yr, end_yr, data_in, fctStatuts): #data_in is the raw input file

	store_dek = [] #an array to store data_in as individual years throughtout a single location
	dek_number = 36*(len(np.arange(init_yr, end_yr, 1))) #read the amount of dekads in data_in 

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

	if fctStatuts == True: #if true, then programs starts to compute data considering last current year dek as a forecast
		#As an extra we get an array that contains the current year dekads
		current_year = list(data_in[dek_number:-1]) #current year deks reaches to the dek before forecast
		current_year_fct = list(data_in[dek_number:])
		k = [None]*(36 - len(current_year))
		q = [None]*(36 - len(current_year_fct))
		current_year_None = np.array(current_year + k) #to fill missing dekads with null characters
		current_year_None_fct = np.array(current_year_fct + k) #to fill missing dekads with null characters


	else:
		current_year = list(data_in[dek_number:])
		current_year_None_fct = [None]

		k = [None]*(36 - len(current_year))
		current_year_None = np.array(current_year + k) #to fill missing dekads with null characters

	fctDek = [data_in[-1]]

	#k = [None]*(36 - len(current_year))
	#current_year_None = np.array(current_year + k) #to fill missing dekads with null characters

	#OUTPUTS
	return np.array([LT_mean, current_year_None, store_dek, np.array(current_year), fctDek, current_year_None_fct])
	#[a, b, c]

##############################################################################################################################################

def get_median_for_whole_data(raw_data, init_yr, end_yr, init_clim, end_clim, fct): #to get the median for all location in-a-row

	raw_years = [] #It'll be an array to store input data, but now by location
	actual_year = []
	actual_year_no_None = []
	full_data_median = []#an array which contains the historical median for all completed years available
	forecast = []
	current_year_fct = []
	for i in np.arange(0, len(raw_data[2]), 1):

		a = compute_median(init_yr, end_yr, raw_data[2][i], fct)
		actual_year.append(a[1])
		raw_years.append(a[2])
		full_data_median.append(a[0])
		actual_year_no_None.append(a[3])
		forecast.append(a[4])
		current_year_fct.append(a[5])

	#****************************************CLIMATOLOGY********************************
	#we're gonna setup a way to fix climatology. For this we'll make
	#a dictionary with all years gotten in the header like: {1985:0, 1986:1, ... 2018:33}
	#so when user inputs its climatology window, program will user bouds as passwords in this dictionary

	year_1 = int(raw_data[1][0][0:4]) #absolute first year
	year_2 = int(raw_data[1][-1][0:4]) #absolute last year
	years = np.arange(year_1, year_2, 1)
	linspace = np.arange(0, len(years), 1)
	clim_dict = dict(zip(years, linspace)) #a dictionary which will help me to choose climatology

	clim = []
	for i in np.arange(0, len(raw_years), 1):
		add = raw_years[i][clim_dict[init_clim]:clim_dict[end_clim]+1]
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
	output = np.array([full_data_median, actual_year, raw_years, actual_year_no_None, raw_data[0], mean_clim, forecast, current_year_fct])

	#dek_data = open('./datapath/output_snack', 'wb') #to save whole data separated in dekads [n_locations, n_years, 36]. Only takes completed years
	#pickle.dump(output, dek_data)
	#dek_data.close()

	return output # np.array([full_data_median, actual_year, raw_years])

##############################################################################################################################################

def rainfall_accumulations(init_yr, end_yr, fst_dek, lst_dek, dek_dictionary, output_snack, fctStatuts ):

	yrs_window = np.arange(init_yr, end_yr, 1)
	linspace = np.arange(dek_dictionary[fst_dek]-1, dek_dictionary[lst_dek], 1)
	fctAcummulation = [] #stores purple accumulation rainfall curve if forecast is computed

	#to get the rainfall accumulations for all years except for the current one
	n = 0
	skim = [] #OUTPUT 1
	for k in np.arange(0, len(output_snack[2]), 1):
		remain = []
		skim.append(remain)
		for j in np.arange(0, len(yrs_window), 1):
			sumV = []
			remain.append(sumV)
			for i in linspace:
				n = n + output_snack[2][k][j][i]
				sumV.append(n)
				if len(sumV) == len(linspace):
					n = 0

	#auxilliar step to set accumulations as a dictionary, where the password is their corresponding years
	skim = np.array(skim)

	skim_dictionary = []
	for i in np.arange(0, len(skim), 1):
		cat =  dict(zip(yrs_window, skim[i]))
		skim_dictionary.append(cat)

	#to get the rainfall accumulation for current year in selected dekad window
	n = 0 
	acumulado_por_estacion = [] #OUTPUT 2
	for k in np.arange(0, len(output_snack[1]), 1):
		acumulado_ano_actual = []
		acumulado_por_estacion.append(acumulado_ano_actual)

		for i in linspace:
			if output_snack[1][k][i] == None:
				n = 0
				break
			else:
				n = n + output_snack[1][k][i]
				acumulado_ano_actual.append(n)
				if len(acumulado_ano_actual) == len(linspace):
					n = 0

	if fctStatuts == True: #When forecast mode is computed

		n = 0 
		#fctAcummulation = [] #OUTPUT 2
		for k in np.arange(0, len(output_snack[7]), 1):
			acumulado_ano_actual = []
			fctAcummulation.append(acumulado_ano_actual)

			for i in linspace:
				if output_snack[7][k][i] == None:
					n = 0
					break
				else:
					n = n + output_snack[7][k][i]
					acumulado_ano_actual.append(n)
					if len(acumulado_ano_actual) == len(linspace):
						n = 0

		#fctAcummulation = np.array(fctAcummulation)


	acumulado_por_estacion = np.array(acumulado_por_estacion)

	return np.array([skim, acumulado_por_estacion.transpose(), np.array(skim_dictionary), np.array(fctAcummulation).transpose()])

##############################################################################################################################################

def sum_error_sqr(accumulations): #computes the square of substraction between the biggest accumulations 
		
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

def sum_dekad_error(fst_dek, lst_dek, accumulations, dek_dictionary, output_snack):

	global_substraction = []
	for i in np.arange(0, len(accumulations[0]), 1): #for each location in the array
		mid_substraction = []
		global_substraction.append(mid_substraction)
		for j in np.arange(0, len(accumulations[0][0]), 1): #drive by each year 
			spec_substraction = []
			mid_substraction.append(spec_substraction)
			for k in np.arange(0, len(accumulations[0][0][0])): #while visiting every dek from the chosen window by the user

				a = output_snack[1][i][dek_dictionary[fst_dek]-1:dek_dictionary[lst_dek]] #current year for chosen dekads
				b = output_snack[2][i][j][dek_dictionary[fst_dek]-1:dek_dictionary[lst_dek]] #raw past year for chosen dekads

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

def get_analog_years(init_yr, end_yr, analog_num, call_sum_error_sqr, call_sum_dekad_error):

	#both must have the same shape
	#call_sum_error_sqr = sum_error_sqr() #we call the resulting RANK FOR SUM ERROR SQUARE
	#call_sum_dekad_error = sum_dekad_error(fst_dek, lst_dek) #we call the resulting RANK FOR SUM DEKAD error_sqr

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
	time_window = np.arange(init_yr, end_yr, 1)
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

	#to get the analog years according to the user's choice
	analogs = []; k = 0
	while analog_num > 1:
		analog_num = analog_num - k
		k = 1
		analogs.append(analog_num)
	analogs = np.sort(analogs)

	#we get the ANALOG YEARS ARRAY i.e [2000, 1982, 1995,...] they'll be my passwords to get their respective data
	accumulation_analog_yrs = []
	for i in np.arange(0, len(analog_yrs_dict), 1): #for each location available
		camp = []
		accumulation_analog_yrs.append(camp)
		for j in np.arange(0, len(analogs), 1):
			get = analog_yrs_dict[i][analogs[j]]
			camp.append(get)

	analog_array = np.array([analog_yrs_dict, ranked, accumulation_analog_yrs])

	return analog_array

##############################################################################################################################################

def seasonal_accumulations_plotting(accumulations, analogs_dictionary): #a function that calcs the output to plot seasonal accumulations 

	analog_yrs = analogs_dictionary[2] #it contains the analog years according to user's choice
	currentYearLastDek = len(accumulations[1].transpose()[0])-1

	analog_curves = [] #these are analog years to plot
	#choosing analogs to plot
	for i in np.arange(0, len(accumulations[2]), 1):
		curves = []
		analog_curves.append(curves)
		for j in np.arange(0, len(analog_yrs[0]), 1):
			com = accumulations[2][i][analog_yrs[i][j]]
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
	
def seasonal_accumulations(init_clim, end_clim, accumulations): #a funcion that calcs

	clim_window = np.arange(init_clim, end_clim + 1, 1) #a linspace with preffered years (keys for dict) in climatology
	accumulation = accumulations[2] #complete accumulations array(dictionary) for all location in every past year
	full_accum =  accumulations[0]
	currentYearLastDek = len(accumulations[1].transpose()[0])-1 #last dek as positional number in current year

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
	for i in np.arange(0, len(accumulation), 1):
		get = []
		get_c = []
		clim_accumulations.append(get)
		endSeasonRow.append(get_c)
		for k in clim_window:
			yr = accumulation[i][k]
			z = accumulation[i][k][-1]
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

def ensemble_plotting(init_yr, end_yr, end_dek, init_clim, end_clim, output_snack, accumulations, analogs_dictionary, dek_dictionary):

	#we need again the analog curves
	years = analogs_dictionary[2]
	accumulation = accumulations[1].transpose()
	linspace = np.arange(len(output_snack[3][0]), dek_dictionary[end_dek], 1)
	
	analog_curves = [] #these are analog years to plot
	#choosing analogs to plot
	for i in np.arange(0, len(accumulations[2]), 1):
		curves = []
		analog_curves.append(curves)
		for j in np.arange(0, len(years[0]), 1):
			com = accumulations[2][i][years[i][j]]
			curves.append(com)
	
	analog_curves = np.array(analog_curves) #it contains ONLY the curves for chosen analog years


	#this loop will take the [-1] element form the accumulated current year array and will start a new accumulation from this 
	#point for each location, in every past year until the dekad window ends i.e if my current year ends on 3-May dek, but muy chosen
	#dekad window ends on 1-Aug, then it'll create a (num_loc, num_years, 1-May - 1-Aug) array 

	#SETTING UP ENSEMBLE: it calculates the ensemble for all past years in dataset. 
	assembly = [] #it'll store the ensemble array
	for i in np.arange(0, len(analog_curves), 1): #for each location. We're only taking the size!
		n = accumulation[i][-1]
		asem = []
		assembly.append(asem)

		for j in np.arange(0, len(output_snack[2][0]), 1): #for each location 
			stamp = []
			asem.append(stamp)

			for k in linspace:
				n = n + output_snack[2][i][j][k]
				stamp.append(n)

				if len(stamp) == len(linspace):
					n = accumulation[i][-1]

	#PREPARING ENSEMBLE ARRAY
	#the next loop is to cat the ensemble to current year 
	ensemble = []
	for i in np.arange(0, len(assembly), 1): #for each location read
		scat = []
		ensemble.append(scat)
		for j in np.arange(0, len(assembly[0]), 1): #for each year read

			link = list(accumulation[i]) + list(assembly[i][j]) #cat curren year deks and ensembled deks
			scat.append(link)

	ensemble = np.array(ensemble)

	#now we choose which year I have to keep in
	#create a dictionary to simplify ensembled curves selection according to chosen analog years by users
	yrs = np.arange(init_yr, end_yr, 1)
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

			if j <= end_clim and j >= init_clim: #select analog year data in climatology window
				choose_yr_clim = dictionary[j]
				choose_array_clim = ensemble[i][choose_yr_clim]
				get_clim.append(choose_array_clim)

			choose_yr = dictionary[j]
			choose_array = ensemble[i][choose_yr]
			get.append(choose_array)

	ensemble_curves = np.array(ensemble_analogs) #this is my plotable output


	#GET MEAN DATA FOR ENSEMBLE ANALOGS INSIDE CLIMATOLOGY ***changed: ensemble mean is now computed including all analogs
	ensemble_avg = []
	for i in np.arange(0, len(ensemble_analogs), 1): #for each location 
		z = np.array(ensemble_analogs[i]).transpose()
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

	#OUTLOOK for ensemble
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

	return [ensemble_curves, ensemble_avg, stats_EP, stats_E, endSeasonRow, endSeasonRow_full]

##############################################################################################################################################

def outlook_calc(endSeasonRow, stats):

	ok = []
	yearsNum = len(endSeasonRow[0])
	for i in np.arange(0, len(endSeasonRow), 1):
		a = 0; n = 0; b = 0
		trd = stats[i][3]
		sxth = stats[i][4]
		for j in np.arange(0, yearsNum, 1):

			subject = endSeasonRow[i][j]

			if subject >= sxth:
				a += 1

			elif sxth > subject >= trd:
				n += 1

			elif subject < trd:
				b += 1

		above = (a/yearsNum)*100
		normal = (n/yearsNum)*100
		below = (b/yearsNum)*100

		ok.append([above, normal, below])

	return ok
##############################################################################################################################################

def round2Darray(inputA):

	output = []
	for i in np.arange(0, len(inputA), 1):
		op = []
		output.append(op)
		for j in np.arange(0, len(inputA[0]), 1):
			out = int(round(inputA[i][j]))
			op.append(out)

	return output

##############################################################################################################################################
def generate_reports(init_yr, end_yr, init_dek, end_dek, init_clim, end_clim, analogRank, output_snack, accumulations, stamp, stamp2, stamp3, dek_dictionary, analogs_dictionary, dirName, saveStatus, dispStatus, fctStatus):

	#================================THIS SPACE IS TO PREPARE DATA TO BE LAUNCHED=====================================

	#Function callouts:
	current_yr_accum = round2Darray(accumulations[1].transpose()) #contains the rainfall accumulated curve for current year. Plot in BLUE!
	current_yr_fct = accumulations[3].transpose()
	#stamp = seasonal_accumulations_plotting() #accumulation curves for seasonal accumulations plot
	#stamp2 = seasonal_accumulations(init_clim, end_clim)
	#stamp3 = ensemble_plotting(init_yr, end_yr, lst_dek, init_clim, end_clim) #ensemble data to plot 

	#data
	climCurve = stamp2[0] #constant red lined curve based in climatology for seasonal accumulations and ensemble plots
	seasonalStats = np.array(round2Darray(stamp2[1])) #it contains the points ad the end of season: 33rd, 67th, stDev +/- Avg
	analog_curves = stamp[0]
	analog_yrs = stamp[2]
	analog_stats1 = round2Darray(stamp[1])#it contains the seasonal accumulation stats but based on analog years
	curves_E = stamp3[0]
	E_avgCurve = stamp3[1]
	ensembleStats = round2Darray(stamp3[2])
	ensembleStatsFull = np.array(round2Darray(stamp3[3]))

	endSeasonRow = stamp3[4]
	endSeasonRow_full = stamp3[5]
	

	#ADITIONAL CALC IS MADE HERE TO GET OUTLOOK:
	outlook_E = round2Darray(outlook_calc(endSeasonRow, seasonalStats))
	outlook = np.array(round2Darray(outlook_calc(endSeasonRow_full, seasonalStats)))


	#============================================SETTING UP LINSPACES (x-axes)=======================================

	locNum = np.arange(0, len(analog_curves), 1) #it is a linspace that contains the array with locations number size
	yrSize = np.arange(0, len(analog_curves[0]), 1) #this is a linspace that contains the analog years size
	seasonSize = np.arange(0, len(analog_curves[0][0]), 1) #a linspace with size of the season 
	currentYr = np.arange(0, len(current_yr_accum[0]), 1) #current year linspace size
	seasonLabels = list(dek_dictionary.keys())[dek_dictionary[init_dek]-1:dek_dictionary[end_dek]]
	colorPalette = ["#9b59b6", "#3498db", "#95a5a6", 
						"#e74c3c", "#34495e", "#2ecc71", 
							'#FFC312', '#C4E538', '#12CBC4', 
								'#FDA7DF', '#ED4C67', '#EE5A24', 
									'#009432', '#0652DD', '#9980FA', 
										'#B53471', '#EA2027', '#006266', 
											'#4CAF50', '#F97F51', '#EAB543', 
												'#BDC581', '#1B9CFC', '#58B19F', 
													'#3ae374', '#e17055', '#795548', 
														'#546E7A', '#00897B', '#5f27cd', 
															'#84817a', '#9e1c22', '#27ae60', 
																'#d35400', '#ffda79', '#a4b0be', 
																	'#ce6f71', '#286e95', '#d3af0e', 
																		'#116c25', '#71647d', '#22b455']	

	#============================================OUTPUT DATAFRAME SUMMARY===================================================================

	#OUTPUT SUMMARY DATAFRAME
	LTAcalcs = [] #this is an auxiliar array to simplify getting % of LTA at current dek and % of LTA at the end of the season
	for i in locNum:

		#LTAvalP = analog_stats1[i][0]
		#LTAval = seasonalStats[i][0]
		LTA_percP = int(round((current_yr_accum[i][-1]/analog_stats1[i][3])*100))
		LTA_perc =  int(round((current_yr_accum[i][-1]/seasonalStats[i][-1])*100))
		#seasonalAvgP = ensembleStats[i][0]
		#seasonalAvg = ensembleStatsFull[i][0]
		LTApercP = int(round((ensembleStats[i][0]/analog_stats1[i][0])*100))
		LTAperc = int(round((ensembleStatsFull[i][0]/seasonalStats[i][0])*100))

		LTAcalcs.append([LTApercP, LTA_perc, LTApercP, LTAperc])

	LTAcalcs = np.array(LTAcalcs)
	LTAcalcsT = LTAcalcs.transpose()



	ok_E = outlook.transpose()
	#stats1 = seasonalStats.transpose()
	#stats2 = ensembleStatsFull.transpose()
	datas = {'Code': output_snack[4], #codes/names for locations
		 		'pctofavgatdek': LTAcalcsT[1],
		 			'pctofavgatEOS': LTAcalcsT[3],
		 				'Above': ok_E[0],
		 					'Normal': ok_E[1],
		 						'Below': ok_E[2]
			}

	colNames = ['Code', 'pctofavgatdek', 'pctofavgatEOS', 'Above', 'Normal', 'Below']
	frame = pd.DataFrame(datas, columns = colNames)


	'''
	#ADVANCED SUMMARY
	advancedData = {'Code': output_snack[4],
						'Seasonal Avg': stats1[0],
							'Seasonal StDev': stats1[1],
								'Seasonal Median': stats1[2],
									'Total at current Dek': ['None']*len(yrSize),
										'LTA Value': stats1[-1],
											'LTA Percentage': LTAcalcsT[1],
												'Ensemble Avg': ['None']*len(yrSize),
													'Ensemble StDev': ['None']*len(yrSize),
														'Ensemble Median': ['None']*len(yrSize),
															'E_33rd. Perc.': ['None']*len(yrSize),
																'E_67th. Perc': ['None']*len(yrSize),
																	'E_LTA Value': ['None']*len(yrSize),
																		'E_LTA Perc': LTAcalcsT[3],
																			'Above Prob': ok_E[0],
																				'Normal Prob': ok_E[1],
																					'Below Prob': ok_E[2]

					}

	colNamesAd = ['Code', 'Seasonal Avg', 'Seasonal StDev', 'Seasonal Median', 
					'Total at current Dek', 'LTA Value', 'LTA Percentage', 
						'Ensemble Avg', 'Ensemble StDev', 'Ensemble Median', 
							'E_33rd. Perc.', 'E_67th. Perc', 'E_LTA Value', 
								'E_LTA Perc', 'Above Prob', 'Normal Prob', 'Below Prob']

	frameAd = pd.DataFrame(advancedData, columns = colNamesAd)
	'''

	if saveStatus == True:
		frame.to_csv('{dir}/summary.csv'.format(dir = dirName), index = False)
		#frameAd.to_csv('{dir}/advancedSummary.csv'.format(dir = dirName), index = False)

	#=====================================SETTING UP SUBPLOTS/loop starts here=======================================

	for i in locNum:

		fig = plt.figure(num = i, tight_layout = True, figsize = (16, 9)) #figure number. There will be a report for each processed location
		fig.canvas.set_window_title('Code: {}'.format(output_snack[4][i]))
		fig_grid = gridspec.GridSpec(2,3) #we set a 2x2 grid space to place subplots
		avg_plot = fig.add_subplot(fig_grid[0, 0:2])
		seasonal_accum_plot = fig.add_subplot(fig_grid[1, 0])
		ensemble_plot = fig.add_subplot(fig_grid[1, 1])
		#tables:
		Asummary = fig.add_subplot(fig_grid[0:2, 2])
		#color palettes setup 
		seasonal_accum_plot.set_prop_cycle(cycler('color', colorPalette))
		ensemble_plot.set_prop_cycle(cycler('color', colorPalette))

		#AVG AND CURRENT RAINFALL SEASON:
		avg_plot.plot(np.arange(0, 36, 1), output_snack[5][i], color = 'r', lw = 4, label = 'LT Avg [climatology based]: {init} - {end}'.format(init = init_clim, end = end_clim))
		avg_plot.bar(np.arange(0, len(output_snack[3][0]), 1), output_snack[3][i], color = 'b', label = 'Current year: {yr}'.format(yr = end_yr))
		
		if fctStatus == True:
			avg_plot.bar([len(np.arange(0, len(output_snack[3][0]), 1))], output_snack[6][i], color = 'm', label = 'Forecasted dekadal')



		avg_plot.legend()

		try:
			avg_plot.set_title('Average & current rainfall season: {num}'.format(num = output_snack[4][i]))

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
			seasonal_accum_plot.plot(seasonSize, analog_curves[i][j], lw = 1.5, label = '{yr}'.format(yr = analog_yrs[i][j])) #accumulation curves

			#ENSEMBLE
			ensemble_plot.plot(seasonSize, curves_E[i][j], lw = 1.5, label = '{yr}'.format(yr = analog_yrs[i][j]))


		###############SEASONAL ACCUMULATIONS
		seasonal_accum_plot.plot(seasonSize, climCurve[i], color = 'r', lw = 5, label = 'LTM') #average

		if fctStatus == True:
			x_ax = np.arange(0, len(current_yr_fct[0]), 1)
			seasonal_accum_plot.plot(x_ax, current_yr_fct[i], color = 'm', lw = 5, label = 'Forecast') #FORECAST

		seasonal_accum_plot.plot(currentYr, current_yr_accum[i], color = 'b', lw = 5, label = '{}'.format(end_yr)) #current year
		seasonal_accum_plot.fill_between(seasonSize, (climCurve[i])*1.2, (climCurve[i])*0.8, color = 'lightblue', label = '120-80%' ) #120 - 80% curve


		#stats
		seasonal_accum_plot.plot([seasonSize[-1]], [seasonalStats[i][5]], marker='^', markersize=7, color="green", label = 'Avg+Std')
		seasonal_accum_plot.plot([seasonSize[-1]], [seasonalStats[i][6]], marker='^', markersize=7, color="green", label = 'Avg-Std')
		seasonal_accum_plot.plot([seasonSize[-1]], [seasonalStats[i][3]], marker='s', markersize=7, color="k", label = '33rd pct')
		seasonal_accum_plot.plot([seasonSize[-1]], [seasonalStats[i][4]], marker='s', markersize=7, color="k", label = '67th pct')


		seasonal_accum_plot.legend(loc='upper center', bbox_to_anchor=(0.5, -0.28), fancybox=False, shadow=True, ncol=5, fontsize = 7.5)
		seasonal_accum_plot.set_title('Seasonal accumulations')
		seasonal_accum_plot.set_ylabel('Accumulated rainfall [mm]')
		seasonal_accum_plot.set_xlabel('Dekadals')
		seasonal_accum_plot.set_xticks(seasonSize)
		seasonal_accum_plot.set_xticklabels(seasonLabels, rotation = 'vertical')
		seasonal_accum_plot.grid()


		###############ENSEMBLE
		ensemble_plot.plot(seasonSize, E_avgCurve[i], '--', color = 'k', lw = 2, label = 'ELTM')
		ensemble_plot.plot(seasonSize, climCurve[i], color = 'r', lw = 5, label = 'LTM') #average
		ensemble_plot.fill_between(seasonSize, (climCurve[i])*1.2, (climCurve[i])*0.8, color = 'lightblue', label = '120-80%' ) #120 - 80% curve
		ensemble_plot.plot(currentYr, current_yr_accum[i], color = 'b', lw = 5, label = '{}'.format(end_yr)) #current year

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


		ensemble_plot.legend(loc='upper center', bbox_to_anchor=(0.5, -0.28), fancybox=True, shadow=True, ncol=5, fontsize = 7.5)
		ensemble_plot.set_xticks(seasonSize)
		ensemble_plot.set_xticklabels(seasonLabels, rotation = 'vertical')
		ensemble_plot.set_title('Ensemble')
		ensemble_plot.set_ylabel('Accumulated rainfall [mm]')
		ensemble_plot.set_xlabel('Dekadals')
		ensemble_plot.grid()


		#================================================TABLES==================================================
		#ALL TABLES WILL NEED THE SAME COLUMNS: [ANALOGS, ALL YEARS]
		#SETUP
		Asummary.set_title('Summary Statistics') #{a} - {b}'.format(a = self.init_dek, b = self.end_dek))
		Asummary.axis('tight')
		Asummary.axis('off')
		col = ('Analog years', 'All years')

		colC = ['coral']*len(col)
		rowC = ['lightsteelblue']
		headerColor = ['palegreen']


		#=======================ANALOG YEARS RANKING TABLE===================

		analog_row = []; analog_data = []; z = 0; y = analogRank
		while y > 1:
			y = y - z
			ar = 'Top {top}'.format(top = y)
			ad = [analogs_dictionary[1][i][y]]
			analog_row.append(ar)
			analog_data.append(ad)
			z = 1

		Asummary.table(rowLabels = analog_row, colLabels = ['Analog years ranking'], cellText = analog_data, cellLoc = 'center', bbox = [0.1, 0.7, 0.8, 0.3], colColours = colC, rowColours = rowC*len(analog_row))

		#====================CLIMATOLOGICAL ANALYSIS TABLE===================


		LTAvalP = analog_stats1[i][0]
		LTAval = seasonalStats[i][0]


		#LTA_percP = int(round((current_yr_accum[i][-1]/analog_stats1[i][3])*100))
		#LTA_perc =  int(round((current_yr_accum[i][-1]/seasonalStats[i][-1])*100))

		#HEADER
		Asummary.table(cellText = [[None]], colLabels = ['Climatological Analysis'], bbox = [0.2, 0.55, 0.7, 0.12 ], colColours = headerColor)


		row = ['Seasonal Average', 'Seasonal Std. Dev', 'Seasonal median', 'Total at current dek.', 'LTA Value', 'Current Dek. LTA %']
		txt = [[LTAvalP, LTAval], 
				[analog_stats1[i][1], seasonalStats[i][1]], 
				[analog_stats1[i][2], seasonalStats[i][2]], 
				[current_yr_accum[i][-1], current_yr_accum[i][-1]], 
				[analog_stats1[i][3], seasonalStats[i][-1]], 
				[LTAcalcs[i][0], LTAcalcs[i][1]]]

		Asummary.table(rowLabels = row, colLabels = col, cellText = txt, loc = 'center', cellLoc = 'center', bbox = [0.2, 0.32, 0.7, 0.3], colColours = colC, rowColours = rowC*len(row))

		#====================CURRENT YEAR ANALYSIS [ENSEMBLE] TABLE===================

		#HEADER
		Asummary.table(cellText = [[None]], colLabels = ['Current year analysis: {yr}'.format(yr = end_yr)], bbox = [0.2, 0.16, 0.7, 0.12 ], colColours = headerColor)


		#seasonalAvgP = ensembleStats[i][0]
		#seasonalAvg = ensembleStatsFull[i][0]
		
		#LTApercP = int(round((seasonalAvgP/LTAvalP)*100))
		#LTAperc = int(round((seasonalAvg/LTAval)*100))

		row_B = ['Ensemble Average', 'Ensemble Std. Dev', 'Ensemble median', '33rd. Percentile', '67th. Percentile', 'LTA Value', 'End of Season LTA %']
		data_B =[[ensembleStats[i][0], ensembleStatsFull[i][0]], 
				[ensembleStats[i][1], ensembleStatsFull[i][1]], 
				[ensembleStats[i][2], ensembleStatsFull[i][2]], 
				[ensembleStats[i][3], ensembleStatsFull[i][3]], 
				[ensembleStats[i][4], ensembleStatsFull[i][4]], 
				[LTAvalP, LTAval], 
				[LTAcalcs[i][2], LTAcalcs[i][3]]]

		Asummary.table(rowLabels = row_B, colLabels = col, cellText = data_B, loc = 'center', cellLoc = 'center', bbox = [0.2, -0.08, 0.7, 0.3], colColours = colC, rowColours = rowC*len(row_B))

		#===================OUTLOOK PROBABILITY TABLE=======================

		#HEADER
		Asummary.table(cellText = [[None]], colLabels = ['Outlook: Probability at the end of season'], bbox = [0.2, -0.23, 0.7, 0.12 ], colColours = headerColor)

		outlook_row = ['Above normal', 'Normal', 'Below normal']
		data = [[round(outlook_E[i][0]), round(outlook[i][0])], [round(outlook_E[i][1]), round(outlook[i][1])], [round(outlook_E[i][2]), round(outlook[i][2])]]
		Asummary.table(rowLabels = outlook_row, colLabels = col, cellText = data, cellLoc = 'center', bbox = [0.2, -0.36, 0.7, 0.2], colColours = colC, rowColours = rowC*len(outlook_row))

		fig.align_labels()
		if saveStatus == True:

			#fig.savefig('{key}{dir}'.format(dir = dirName, key = output_snack[4][i]))
			fig.savefig('{dir}/{key}_report'.format(dir = dirName, key = output_snack[4][i]))


	if dispStatus == True:
		return plt.show()

	else:
		return 0


####################################################################################################################################


class mainFrame():

	def __init__(self, master):

		self.titulo = master.title('SMPG-TOOL alpha_1.1c')

		self.dek_dictionary = {'1-Jan': 1, '2-Jan': 2, 
								'3-Jan': 3, '1-Feb': 4, 
								'2-Feb': 5, '3-Feb': 6, 
								'1-Mar': 7, '2-Mar': 8, 
								'3-Mar': 9, '1-Apr': 10, 
								'2-Apr': 11, '3-Apr': 12, 
								'1-May': 13, '2-May': 14, 
								'3-May': 15, '1-Jun': 16, 
								'2-Jun': 17, '3-Jun': 18, 
								'1-Jul': 19, '2-Jul': 20, 
								'3-Jul': 21, '1-Aug': 22, 
								'2-Aug': 23, '3-Aug': 24, 
								'1-Sep': 25, '2-Sep': 26, 
								'3-Sep': 27, '1-Oct': 28, 
								'2-Oct': 29, '3-Oct': 30, 
								'1-Nov': 31, '2-Nov': 32, 
								'3-Nov': 33, '1-Dec': 34, 
								'2-Dec': 35, '3-Dec': 36}

		self.background = PhotoImage(file = './back.gif')
		self.bg = Canvas(master, width = 800, height = 100 )
		self.bg.pack()
		self.cv_img = self.bg.create_image(0, 0, image = self.background, anchor = 'nw')

		self.frame = Frame(master, width = 500, height = 400)
		self.frame.pack()
		
		self.year_lst = np.arange(1980, 2021, 1)
		self.dekad_lst = ['1-Jan', '2-Jan', '3-Jan', '1-Feb', '2-Feb', '3-Feb', '1-Mar', '2-Mar', '3-Mar', '1-Apr', '2-Apr', '3-Apr', '1-May', '2-May', '3-May', '1-Jun',
		 					'2-Jun', '3-Jun', '1-Jul', '2-Jul', '3-Jul', '1-Aug', '2-Aug', '3-Aug', '1-Sep', '2-Sep', '3-Sep', '1-Oct', '2-Oct', '3-Oct', '1-Nov', '2-Nov', 
		 					'3-Nov', '1-Dec', '2-Dec', '3-Dec']
		self.analogs_lst = np.arange(1, 40, 1)
		self.variable_analogs_lst = IntVar(self.frame)
		self.variable_init_dekad = StringVar(self.frame)
		self.variable_end_dekad = StringVar(self.frame)

		self.radio_button = IntVar(self.frame)
		self.variable_rank = IntVar(self.frame)
		self.variable_rank.set('')

		self.out = []
		self.fileOpen = ''

		#CHECKBUTTON
		self.check = IntVar(self.frame)
		self.disp = IntVar(self.frame)



		#self.variable_end = IntVar(self.frame)
		#self.variable_init.set(self.year_lst[0])
		#self.variable_end.set(self.year_lst[0])

		#climatology
		self.variable_init_clim = IntVar(self.frame)
		self.variable_end_clim = IntVar(self.frame)
		#self.variable_init_clim.set()

##############################################################################################################################################

		#LABELS
		self.label0 = Label(self.frame, text = 'Set up climatology window')
		self.label0.grid(row = 1, column = 2, columnspan = 4)

		#self.label_clim1 = Label(self.frame, text = 'From:')
		#self.label_clim1.grid(row = 1, column = 1, padx = 10)

		#self.label_clim2 = Label(self.frame, text = 'To:')
		#self.label_clim2.grid(row = 1, column = 3, padx = 10)

		#self.labelz = Label(self.frame, text = 'Choose analysis preferences')
		#self.labelz.grid(row = 0, column = 1, columnspan = 4)

		self.labelz = Label(self.frame, text = 'Define a season to monitor')
		self.labelz.grid(row = 3, column = 2, columnspan = 3)


		self.label1 = Label(self.frame, text = 'Initial year:')
		self.label1.grid(row = 2, column = 1, padx = 10)

		self.label2 = Label(self.frame, text = 'Final year:')
		self.label2.grid(row = 2, column = 3, padx = 10)

	
		self.label3 = Label(self.frame, text = 'From:')
		self.label3.grid(row = 4, column = 1, sticky = E, padx =  5)

		self.label4 = Label(self.frame, text = 'to:')
		self.label4.grid(row = 4, column = 3, sticky = E, padx = 5)
	

		self.label5 = Label(self.frame, text = 'Select the number of analog years to compute:')
		self.label5.grid(row = 5, column = 1, pady = 15, columnspan = 3)

		self.label6 = Label(self.frame, text = 'Specify the max rank of analog years to show:')
		self.label6.grid(row = 6, column = 1, columnspan = 3)

		self.label7 = Label(self.frame, text = 'Computing preferences')
		self.label7.grid(row = 5, column = 0)

##############################################################################################################################################		
		#MENUS

		self.init_clim = ttk.Combobox(self.frame, textvariable = self.variable_init_clim, values = tuple(self.year_lst))
		self.init_clim.grid(row = 2, column = 2, pady = 15)

		self.end_clim = ttk.Combobox(self.frame, textvariable = self.variable_end_clim, values = tuple(self.year_lst))
		self.end_clim.grid(row = 2, column = 4)
		
		#start year option menu
		#self.ano_init = ttk.Combobox(self.frame, textvariable = self.variable_init, values = tuple(self.year_lst))
		#self.ano_init.grid(row = 1, column = 2)
		#self.ano_init.pack()

		#end year option menu
		#self.ano_fin = ttk.Combobox(self.frame, textvariable = self.variable_end, values = tuple(self.year_lst))
		#self.ano_fin.grid(row = 1, column = 4)
		#self.ano_fin.pack()

		#first dekad menu
		self.start_dekad = ttk.Combobox(self.frame, textvariable = self.variable_init_dekad, values = tuple(self.dekad_lst))
		self.start_dekad.grid(row = 4, column = 2, pady = 10)

		#end dekad menu
		self.end_dekad = ttk.Combobox(self.frame, textvariable = self.variable_end_dekad, values = tuple(self.dekad_lst))
		self.end_dekad.grid(row = 4, column = 4)
		
		#ANALOG YEARS MENU
		self.analog_menu  = ttk.Combobox(self.frame, textvariable = self.variable_analogs_lst, values = tuple(self.analogs_lst))
		self.analog_menu.grid(row = 5, column = 4)

		#RANK SELECTION MENU
		self.rank_menu  = ttk.Combobox(self.frame, textvariable = self.variable_rank, values = tuple(self.analogs_lst))
		self.rank_menu.grid(row = 6, column = 4)

	
##############################################################################################################################################	
		#BUTTONS

		
		self.load_data_btn = Button(self.frame, text = 'Advanced settings')
		self.load_data_btn.grid(row = 8, column = 0, sticky = W)
	
		
		self.LT_avg_btn = Button(self.frame, text = 'GENERATE REPORTS', command = lambda: mainFrame.gen_rep(self,
																												str(self.start_dekad.get()), 
																												str(self.end_dekad.get()),
																												int(self.init_clim.get()),
																												int(self.end_clim.get()),
																												int(self.analog_menu.get()), 
																												int(self.rank_menu.get())))
																																																					
		self.LT_avg_btn.grid(row = 2, column = 0, columnspan = 1)
		
	
		#browse button
		self.browse_btn = Button(self.frame, text = 'Browse Files', command = lambda: mainFrame.open_file(self), width = 25)
		self.browse_btn.grid(row = 0, column = 0, pady = 20, columnspan = 2, sticky = W+E)

		self.entry = Entry(self.frame, width = 54, text=self.fileOpen)
		self.entry.configure({"background": "red"})
		self.entry.grid(row = 0, column = 2, columnspan = 3, padx = 0, sticky = E)

		self.help_btn = Button(self.frame, text = 'Clear', command = lambda: mainFrame.clearFiles(self))
		#self.help_btn.configure(bg = 'red')
		self.help_btn.grid(row = 8, column = 4, pady = 15)
		
		self.fct = Radiobutton(self.frame, text = 'Forecast', variable = self.radio_button, value = 0)
		self.fct.grid(row = 6, column = 0, sticky = NW)

		self.analysis = Radiobutton(self.frame, text = 'Observed data', variable = self.radio_button, value = 1)
		self.analysis.grid(row = 7, column = 0, sticky = NW)

		self.save_check = Checkbutton(self.frame, text = 'Save reports', variable = self.check)
		self.save_check.grid(row = 3, column = 0, sticky = W)

		self.disp_check = Checkbutton(self.frame, text = 'Display reports', variable = self.disp)
		self.disp_check.grid(row = 4, column = 0, sticky = W)

##############################################################################################################################################

	def open_file(self):
		try:
			file = filedialog.askopenfile(mode ='r', filetypes =[('csv files', '*.csv')]) 
			indir = str(file.name)
			self.entry.delete(0, END)
			self.entry.insert(0, indir)

		except AttributeError:
			return 0

		if file is not None: 
			data = pd.read_csv(file, header = None)
			df = pd.DataFrame(data)

			#SETUP HEADER AS STRING LIKE 'YEAR|DEK' FIRST 4 CHARACTERS DEFINE YEAR AND LAST 2 CHARACTERS DEFINE ITS DEK
			header = list(df.loc[0][1:])
			header_str = []
			for i in np.arange(0, len(header), 1):
				head =  str(header[i])[0:6]
				header_str.append(head)

			locNames = np.array(df.loc[1:][0]); locs = [] #locations' names
			for i in np.arange(0, len(locNames), 1):
				try:
					key = str(int(locNames[i]))
					locs.append(key)

				except ValueError:

					key = locNames[i]
					locs.append(key)


			#OUTPUT: returns a 3rd dim array with this features: [locations'_tags, header, raw data]
			output = np.array([locs, np.array(header_str), np.array(df.loc[1:]).transpose()[1:].transpose()])
			self.out = output
			tkinter.messagebox.showinfo('Data loaded!', 'Input dataset goes from {init} to {end}'.format(init = output[1][0][0:4], end = output[1][-1][0:4]))


			'''
			array_out = open('data', 'wb') #write binary
			pickle.dump(output, array_out)
			tkinter.messagebox.showinfo('Data loaded!', 'Input dataset goes from {init} to {end}'.format(init = output[1][0][0:4], end = output[1][-1][0:4]))
			array_out.close()
			del(array_out)
			'''
##############################################################################################################################################

	def gen_rep(self, fst_dek, lst_dek, init_clim, end_clim, analog_num, analogRank):

		#params
		raw_data = self.out
		init_yr = int(self.out[1][0][0:4])
		end_yr = int(self.out[1][-1][0:4])
		dek_dictionary = self.dek_dictionary
		yrsWindow = len(np.arange(init_yr, end_yr, 1))

		if init_clim >= end_clim:
			tkinter.messagebox.showerror('Error while computing climatology', 'End year must be greater than init. year' )
			return 0

		elif init_clim < init_yr:
			tkinter.messagebox.showerror('Error while computing climatology', 'Initial year cannot be less than {} for this dataset'.format(init_yr))
			return 0
		elif end_clim > end_yr:
			tkinter.messagebox.showerror('Error while computing climatology', 'End year cannot be greater than {} for this dataset'.format(end_yr))
			return 0

		elif analog_num == 0 or analog_num == 1:
			tkinter.messagebox.showerror('warning', 'More than 1 analog year must be chosen')
			return 0

		elif analog_num > yrsWindow:
			tkinter.messagebox.showerror('Choice exceeds available analogs', 'There are {} years as Max.'.format(yrsWindow))
			return 0

		if self.check.get() == 1:
			try:
				dir_name = filedialog.askdirectory() # asks user to choose a directory
				os.chdir(dir_name) #changes your current directory
				curr_directory = os.getcwd()

			
				#dir_name = filedialog.asksaveasfile(initialfile = 'report',title = "Select file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
				#os.chdir(dir_name) #changes your current directory
				#curr_directory = dir_name
				status = True
			#return curr_directory

			except FileNotFoundError:
				pass

			except TypeError:
				pass
		else:
			status = False
			curr_directory = None


		if self.disp.get() == 1:
			disp_rep = True

		else:
			disp_rep = False


		output_snack = get_median_for_whole_data(raw_data, init_yr, end_yr, init_clim, end_clim, True)
		accumulations = rainfall_accumulations(init_yr, end_yr, fst_dek, lst_dek, dek_dictionary, output_snack, True)

		call_sum_error_sqr = sum_error_sqr(accumulations) #we call the resulting RANK FOR SUM ERROR SQUARE
		call_sum_dekad_error = sum_dekad_error(fst_dek, lst_dek, accumulations, dek_dictionary, output_snack) #we call the resulting RANK FOR SUM DEKAD error_sqr
		analogs_dictionary = get_analog_years(init_yr, end_yr, analog_num, call_sum_error_sqr, call_sum_dekad_error)

		stamp = seasonal_accumulations_plotting(accumulations, analogs_dictionary)
		stamp2 = seasonal_accumulations(init_clim, end_clim, accumulations)
		stamp3 = ensemble_plotting(init_yr, end_yr, lst_dek, init_clim, end_clim, output_snack, accumulations, analogs_dictionary, dek_dictionary)
		
		plot = generate_reports(init_yr, end_yr, fst_dek, lst_dek, init_clim, end_clim, analogRank, output_snack, accumulations, stamp, stamp2, stamp3, dek_dictionary, analogs_dictionary, curr_directory, status, disp_rep, True)

		tkinter.messagebox.showinfo('Notification!', 'Done! Reports computed')
##############################################################################################################################################

	def clearFiles(self):

		#clear menus:
		self.analog_menu.set('')
		self.start_dekad.set('')
		self.end_dekad.set('')
		self.init_clim.set('')
		self.end_clim.set('')
		
		tkinter.messagebox.showinfo('status', 'Clearing')

		python = sys.executable
		os.execl(python, python, *sys.argv)

##############################################################################################################################
root = Tk()



main = mainFrame(root)
root.mainloop()

