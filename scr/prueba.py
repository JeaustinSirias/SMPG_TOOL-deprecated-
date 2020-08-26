
#==========MODULES==============
import numpy as np
import numpy.random.common
import numpy.random.bounded_integers
import numpy.random.entropy
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import rankdata
from collections import defaultdict
import matplotlib.gridspec as gridspec

#=============================================================FUNCTIONS=======================================================================
##############################################################################################################################################

def input_data(input_d):
	data = pd.read_csv(input_d, header = None,)
	df = pd.DataFrame(data)

	#SETUP HEADER AS STRING LIKE 'YEAR|DEK' FIRST 4 CHARACTERS DEFINE YEAR AND LAST 2 CHARACTERS DEFINE ITS DEK
	header = list(df.loc[0][1:-1])
	header_str = []
	for i in np.arange(0, len(header), 1):
		head =  str(header[i])[0:6]
		header_str.append(head)

	raw = np.array(df.loc[1:]).transpose()[1:-1].transpose()
	scenarios = np.array(df.loc[1:]).transpose()[-1].transpose()
	scenarios = [float(scenarios[i]) for i in np.arange(0, len(scenarios), 1) if scenarios[i] != None]

	#returns a 3rd dim array with this features: [locations'_tags, header, raw data]
	return np.array([np.array(df.loc[1:][0]), np.array(header_str), raw, scenarios])
	
	
##############################################################################################################################################

def compute_median(init_yr, end_yr, data_in): #data_in is the raw input file

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

		#As an extra we get an array that contains the current year dekads
		current_year = list(data_in[dek_number:])
		k = [None]*(36 - len(current_year))
		current_year_None = np.array(current_year + k) #to fill missing dekads with null characters

		#OUTPUTS
		return np.array([LT_mean, current_year_None, store_dek, np.array(current_year)])
		#[a, b, c]

##############################################################################################################################################

def get_median_for_whole_data(raw_data, init_yr, end_yr, init_clim, end_clim): #to get the median for all location in-a-row

		raw_years = [] #It'll be an array to store input data, but now by location
		actual_year = []
		actual_year_no_None = []
		full_data_median = []#an array which contains the historical median for all completed years available
		for i in np.arange(0, len(raw_data[2]), 1):

			a = compute_median(init_yr, end_yr, raw_data[2][i])
			actual_year.append(a[1])
			raw_years.append(a[2])
			full_data_median.append(a[0])
			actual_year_no_None.append(a[3])

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
		output = np.array([full_data_median, actual_year, raw_years, actual_year_no_None, raw_data[0], mean_clim])

		#dek_data = open('./datapath/output_snack', 'wb') #to save whole data separated in dekads [n_locations, n_years, 36]. Only takes completed years
		#pickle.dump(output, dek_data)
		#dek_data.close()

		return output # np.array([full_data_median, actual_year, raw_years])

##############################################################################################################################################

def rainfall_accumulations(init_yr, end_yr, fst_dek, lst_dek):

		yrs_window = np.arange(init_yr, end_yr, 1)
		linspace = np.arange(dek_dictionary[fst_dek]-1, dek_dictionary[lst_dek], 1)

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

		acumulado_por_estacion = np.array(acumulado_por_estacion)
	
		return np.array([skim, acumulado_por_estacion.transpose(), np.array(skim_dictionary)])

##############################################################################################################################################

def sum_error_sqr(): #computes the square of substraction between the biggest accumulations 
		
		dekNum = len(accumulations[1].transpose()[0]) - 1 #number of dekads in current year
		
		error_sqr = []
		for i in np.arange(0, len(accumulations[0]), 1):
			local_sums = []
			error_sqr.append(local_sums)
			for j in np.arange(0, len(accumulations[0][0]), 1):

				sqr_error = (accumulations[1].transpose()[i][-1] - accumulations[0][i][j][dekNum])**2
				local_sums.append(sqr_error)

		error_sqr = np.array(error_sqr) #this array must have a shape like [num_locations, num_years]
		
		sum_error_sqr_rank = []
		for i in np.arange(0, len(error_sqr), 1):
			rank =  rankdata(error_sqr[i], method = 'ordinal')
			sum_error_sqr_rank.append(rank)

		
		#return (accumulations[0][0][0][-3] - accumulations[1].transpose()[0][-1])**2
		return np.array(sum_error_sqr_rank) #ULTIMATE OUTPUT
		#return len(accumulations[1].transpose()[0])

##############################################################################################################################################

def sum_dekad_error(fst_dek, lst_dek):

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

def get_analog_years(init_yr, end_yr, analog_num):

		#both must have the same shape
		call_sum_error_sqr = sum_error_sqr() #we call the resulting RANK FOR SUM ERROR SQUARE
		call_sum_dekad_error = sum_dekad_error(fst_dek, lst_dek) #we call the resulting RANK FOR SUM DEKAD error_sqr

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

def seasonal_accumulations_plotting(): #a function that calcs the output to plot seasonal accumulations 

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
	
def seasonal_accumulations(init_clim, end_clim): #a funcion that calcs

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

def ensemble_plotting(init_yr, end_yr, end_dek, init_clim, end_clim):

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

	#return endSeasonRow
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

			elif sxth > subject > trd:
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

	return np.array(output)

##############################################################################################################################################
def generate_reports(init_yr, end_yr, init_dek, end_dek, init_clim, end_clim, analogRank):

	#================================THIS SPACE IS TO PREPARE DATA TO BE LAUNCHED=====================================

	#Function callouts:
	current_yr_accum = round2Darray(accumulations[1].transpose()) #contains the rainfall accumulated curve for current year. Plot in BLUE!
	stamp = seasonal_accumulations_plotting() #accumulation curves for seasonal accumulations plot
	stamp2 = seasonal_accumulations(init_clim, end_clim)
	stamp3 = ensemble_plotting(init_yr, end_yr, lst_dek, init_clim, end_clim) #ensemble data to plot 

	#data
	climCurve = stamp2[0] #constant red lined curve based in climatology for seasonal accumulations and ensemble plots
	seasonalStats = round2Darray(stamp2[1]) #it contains the points ad the end of season: 33rd, 67th, stDev +/- Avg
	analog_curves = stamp[0]
	analog_yrs = stamp[2]
	analog_stats1 = round2Darray(stamp[1])#it contains the seasonal accumulation stats but based on analog years
	curves_E = stamp3[0]
	E_avgCurve = stamp3[1]
	ensembleStats = round2Darray(stamp3[2])
	ensembleStatsFull = round2Darray(stamp3[3])

	endSeasonRow = stamp3[4]
	endSeasonRow_full = stamp3[5]

	

	#ADITIONAL CALC IS MADE HERE TO GET OUTLOOK:

	#outlook_E = outlook_calc(endSeasonRow, seasonalStats)
	outlook = outlook_calc(endSeasonRow_full, seasonalStats)


	#============================================SETTING UP LINSPACES (x-axes)=======================================

	locNum = np.arange(0, len(analog_curves), 1) #it is a linspace that contains the array with locations number size
	yrSize = np.arange(0, len(analog_curves[0]), 1) #this is a linspace that contains the analog years size
	seasonSize = np.arange(0, len(analog_curves[0][0]), 1) #a linspace with size of the season 
	currentYr = np.arange(0, len(current_yr_accum[0]), 1) #current year linspace size
	seasonLabels = list(dek_dictionary.keys())[dek_dictionary[init_dek]-1:dek_dictionary[end_dek]]



	LTAcalcs = []
	for i in locNum:

		LTAvalP = analog_stats1[i][0]
		LTAval = seasonalStats[i][0]
		LTA_percP = int(round((current_yr_accum[i][-1]/analog_stats1[i][3])*100))
		LTA_perc =  int(round((current_yr_accum[i][-1]/seasonalStats[i][-1])*100))
		seasonalAvgP = ensembleStats[i][0]
		seasonalAvg = ensembleStatsFull[i][0]
		LTApercP = int(round((seasonalAvgP/LTAvalP)*100))
		LTAperc = int(round((seasonalAvg/LTAval)*100))

		LTAcalcs.append([LTApercP, LTA_perc, LTApercP, LTAperc])

	#=====================================SETTING UP SUBPLOTS/loop starts here=======================================

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
		avg_plot.plot(np.arange(0, 36, 1), output_snack[-1][i], color = 'r', lw = 4, label = 'LT Avg [climatology based]: {init} - {end}'.format(init = init_clim, end = end_clim))
		avg_plot.bar(np.arange(0, len(output_snack[3][0]), 1), output_snack[3][i], color = 'b', label = 'Current year: {yr}'.format(yr = end_yr))
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
			seasonal_accum_plot.plot(seasonSize, analog_curves[i][j], lw = 2, label = '{yr}'.format(yr = analog_yrs[i][j])) #accumulation curves

			#ENSEMBLE
			ensemble_plot.plot(seasonSize, curves_E[i][j], lw = 2, label = '{yr}'.format(yr = analog_yrs[i][j]))


		###############SEASONAL ACCUMULATIONS
		seasonal_accum_plot.plot(seasonSize, climCurve[i], color = 'r', lw = 5, label = 'LTM') #average
		seasonal_accum_plot.plot(currentYr, current_yr_accum[i], color = 'b', lw = 5, label = '{}'.format(end_yr)) #current year
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


		LTA_percP = int(round((current_yr_accum[i][-1]/analog_stats1[i][3])*100))
		LTA_perc =  int(round((current_yr_accum[i][-1]/seasonalStats[i][-1])*100))

		#HEADER
		Asummary.table(cellText = [[None]], colLabels = ['Climatological Analysis'], bbox = [0.2, 0.55, 0.7, 0.12 ], colColours = headerColor)


		row = ['Seasonal Average', 'Seasonal Std. Dev', 'Seasonal median', 'Total at current dek.', 'LTA Value', 'LTA percentage']
		txt = [[LTAvalP, LTAval], 
				[analog_stats1[i][1], seasonalStats[i][1]], 
				[analog_stats1[i][2], seasonalStats[i][2]], 
				[current_yr_accum[i][-1], current_yr_accum[i][-1]], 
				[analog_stats1[i][3], seasonalStats[i][-1]], 
				[LTA_percP, LTA_perc]]

		Asummary.table(rowLabels = row, colLabels = col, cellText = txt, loc = 'center', cellLoc = 'center', bbox = [0.2, 0.32, 0.7, 0.3], colColours = colC, rowColours = rowC*len(row))

		#====================CURRENT YEAR ANALYSIS [ENSEMBLE] TABLE===================

		#HEADER
		Asummary.table(cellText = [[None]], colLabels = ['Current year analysis: {yr}'.format(yr = end_yr)], bbox = [0.2, 0.16, 0.7, 0.12 ], colColours = headerColor)


		seasonalAvgP = ensembleStats[i][0]
		seasonalAvg = ensembleStatsFull[i][0]
		
		LTApercP = int(round((seasonalAvgP/LTAvalP)*100))
		LTAperc = int(round((seasonalAvg/LTAval)*100))

		row_B = ['Seasonal Average', 'Seasonal Std. Dev', 'Seasonal median', '33rd. Percentile', '67th Percentile', 'LTA Value', 'LTA Percentage']
		data_B =[[seasonalAvgP, seasonalAvg], 
				[ensembleStats[i][1], ensembleStatsFull[i][1]], 
				[ensembleStats[i][2], ensembleStatsFull[i][2]], 
				[ensembleStats[i][3], ensembleStatsFull[i][3]], 
				[ensembleStats[i][4], ensembleStatsFull[i][4]], 
				[LTAvalP, LTAval], 
				[LTApercP, LTAperc]]

		Asummary.table(rowLabels = row_B, colLabels = col, cellText = data_B, loc = 'center', cellLoc = 'center', bbox = [0.2, -0.08, 0.7, 0.3], colColours = colC, rowColours = rowC*len(row_B))

		#===================OUTLOOK PROBABILITY TABLE=======================

		#HEADER
		Asummary.table(cellText = [[None]], colLabels = ['Outlook: Probability at the end of season'], bbox = [0.2, -0.23, 0.7, 0.12 ], colColours = headerColor)

		outlook_row = ['Above normal', 'Normal', 'Below normal']
		data = [['None', 'None'], ['None', 'None'], ['None', 'None']]
		Asummary.table(rowLabels = outlook_row, colLabels = col, cellText = data, cellLoc = 'center', bbox = [0.2, -0.36, 0.7, 0.2], colColours = colC, rowColours = rowC*len(outlook_row))

		fig.align_labels()
	
	return ensembleStatsFull.transpose()  #plt.show()
	

##############################################################################################################################################



#=============
#    MAIN
#=============

raw_data = input_data('/home/jussc_/Desktop/SMPG_TOOL_DEV/datapath/ejemplo1.csv')
#raw_data2 = input_data('/home/jussc_/Desktop/SMPG_TOOL_DEV/datapath/exp.csv')


#print(raw_data)
#print(raw_data2)


init_yr = int(raw_data[1][0][0:4])
end_yr = int(raw_data[1][-1][0:4])
analogRank = 5
analog_num = 34
fst_dek = '1-Feb'
lst_dek = '3-May'
init_clim = 1985
end_clim = 2010
dek_dictionary = {'1-Jan': 1, '2-Jan': 2, 
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


output_snack = get_median_for_whole_data(raw_data, init_yr, end_yr, init_clim, end_clim)
accumulations = rainfall_accumulations(init_yr, end_yr, fst_dek, lst_dek)
suma = sum_error_sqr()
err = sum_dekad_error(fst_dek, lst_dek)
analogs_dictionary = get_analog_years(init_yr, end_yr, analog_num)

ranked = analogs_dictionary[1]
locations = raw_data[0]



def AnalogsTab(dictionary, locNames):

	data = []
	for i in np.arange(0, len(dictionary), 1):
		scan = []
		data.append(scan)
		for j in np.arange(1, 11, 1):
			rank = dictionary[i][j]
			scan.append(rank)

	#organize table components
	summary = np.array(data).transpose()
	colNames = ['analog_{}'.format(i) for i in np.arange(1, 11, 1)]
	data = dict(zip(colNames, summary))

	#generate table
	df = pd.DataFrame(data = data, index = locNames)
	table = df.to_csv('./analogs_years.csv')


AnalogsTab(ranked, locations)
















#analogs = analogs_dictionary[1]
#locations = raw_data[0]



#plot = generate_reports(init_yr, end_yr, fst_dek, lst_dek, init_clim, end_clim, analogRank)
#print(plot)








