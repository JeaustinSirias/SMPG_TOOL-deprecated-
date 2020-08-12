from tkinter import *
import tkinter.messagebox
import ttk
import numpy as np
from io import *
import pandas as pd
from tkinter.ttk import *
from tkinter.filedialog import askopenfile 
import os #operating system
from tkinter import filedialog


#local packages
from prueba import *


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

		self.background = PhotoImage(file = '/home/jussc_/Downloads/back.gif')
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

		#CHECKBUTTON
		self.check = IntVar(self.frame)



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
		self.label0.grid(row = 0, column = 1, columnspan = 4)

		#self.label_clim1 = Label(self.frame, text = 'From:')
		#self.label_clim1.grid(row = 1, column = 1, padx = 10)

		#self.label_clim2 = Label(self.frame, text = 'To:')
		#self.label_clim2.grid(row = 1, column = 3, padx = 10)

		#self.labelz = Label(self.frame, text = 'Choose analysis preferences')
		#self.labelz.grid(row = 0, column = 1, columnspan = 4)

		self.labelz = Label(self.frame, text = 'Define a season to monitor')
		self.labelz.grid(row = 2, column = 1, columnspan = 4, pady = 25)


		self.label1 = Label(self.frame, text = 'Initial year:')
		self.label1.grid(row = 1, column = 1, padx = 10)

		self.label2 = Label(self.frame, text = 'Final year:')
		self.label2.grid(row = 1, column = 3, padx = 10)

	
		self.label3 = Label(self.frame, text = 'From:')
		self.label3.grid(row = 3, column = 1,)

		self.label4 = Label(self.frame, text = 'to:')
		self.label4.grid(row = 3, column = 3)
	

		self.label5 = Label(self.frame, text = 'Select the number of analog years to compute:')
		self.label5.grid(row = 4, column = 1, pady = 25, columnspan = 3)

		self.label6 = Label(self.frame, text = 'Specify max. rank to show:')
		self.label6.grid(row = 5, column = 1, columnspan = 3)

		self.label7 = Label(self.frame, text = 'Computing preferences')
		self.label7.grid(row = 4, column = 0)

##############################################################################################################################################		
		#MENUS

		self.init_clim = ttk.Combobox(self.frame, textvariable = self.variable_init_clim, values = tuple(self.year_lst))
		self.init_clim.grid(row = 1, column = 2)

		self.end_clim = ttk.Combobox(self.frame, textvariable = self.variable_end_clim, values = tuple(self.year_lst))
		self.end_clim.grid(row = 1, column = 4)
		
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
		self.start_dekad.grid(row = 3, column = 2)

		#end dekad menu
		self.end_dekad = ttk.Combobox(self.frame, textvariable = self.variable_end_dekad, values = tuple(self.dekad_lst))
		self.end_dekad.grid(row = 3, column = 4)
		
		#ANALOG YEARS MENU
		self.analog_menu  = ttk.Combobox(self.frame, textvariable = self.variable_analogs_lst, values = tuple(self.analogs_lst))
		self.analog_menu.grid(row = 4, column = 4)

		#RANK SELECTION MENU
		self.rank_menu  = ttk.Combobox(self.frame, textvariable = self.variable_rank, values = tuple(self.analogs_lst))
		self.rank_menu.grid(row = 5, column = 4)

	
##############################################################################################################################################	
		#BUTTONS

		
		self.load_data_btn = Button(self.frame, text = 'Advanced settings')
		self.load_data_btn.grid(row = 6, column = 2, pady = 25)
	
		
		self.LT_avg_btn = Button(self.frame, text = 'GENERATE REPORTS', command = lambda: mainFrame.gen_rep(self,
																												str(self.start_dekad.get()), 
																												str(self.end_dekad.get()),
																												int(self.init_clim.get()),
																												int(self.end_clim.get()),
																												int(self.analog_menu.get()), 
																												int(self.rank_menu.get())))
																												
																											
																											
		self.LT_avg_btn.grid(row = 2, column = 0)
		
	
		#browse button
		self.browse_btn = Button(self.frame, text = 'Browse Files', command = lambda: mainFrame.open_file(self))
		self.browse_btn.grid(row = 0, column = 0, pady = 20)

		self.help_btn = Button(self.frame, text = 'Clear', command = lambda: mainFrame.clearFiles(self))
		#self.help_btn.configure(bg = 'red')
		self.help_btn.grid(row = 6, column = 4, pady = 25)
		
		self.fct = Radiobutton(self.frame, text = 'Forecast', variable = self.radio_button, value = 0)
		self.fct.grid(row = 5, column = 0)

		self.analysis = Radiobutton(self.frame, text = 'Observed data', variable = self.radio_button, value = 1)
		self.analysis.grid(row = 6, column = 0)

		self.save_check = Checkbutton(self.frame, text = 'Save reports', variable = self.check)
		self.save_check.grid(row = 3, column = 0)

##############################################################################################################################################

	def open_file(self):

		file = askopenfile(mode ='r', filetypes =[('csv files', '*.csv')]) 
		if file is not None: 
			data = pd.read_csv(file, header = None)
			df = pd.DataFrame(data)

			#SETUP HEADER AS STRING LIKE 'YEAR|DEK' FIRST 4 CHARACTERS DEFINE YEAR AND LAST 2 CHARACTERS DEFINE ITS DEK
			header = list(df.loc[0][1:])
			header_str = []
			for i in np.arange(0, len(header), 1):
				head =  str(header[i])[0:6]
				header_str.append(head)

			#OUTPUT: returns a 3rd dim array with this features: [locations'_tags, header, raw data]
			output = np.array([np.array(df.loc[1:][0]), np.array(header_str), np.array(df.loc[1:]).transpose()[1:].transpose()])
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

		if self.check.get() == 1:
			try:
				dir_name = filedialog.askdirectory() # asks user to choose a directory
				os. chdir(dir_name) #changes your current directory
				curr_directory = os.getcwd()
			#return curr_directory

			except FileNotFoundError:
				pass

			except TypeError:
				pass

		if analog_num == 0 or analog_num == 1:
			tkinter.messagebox.showerror('warning', 'More than 1 analog year must be chosen')

		else:
			raw_data = self.out
			init_yr = int(self.out[1][0][0:4])
			end_yr = int(self.out[1][-1][0:4])
			dek_dictionary = self.dek_dictionary

			output_snack = get_median_for_whole_data(raw_data, init_yr, end_yr, init_clim, end_clim)
			accumulations = rainfall_accumulations(init_yr, end_yr, fst_dek, lst_dek, dek_dictionary, output_snack)

			call_sum_error_sqr = sum_error_sqr(accumulations) #we call the resulting RANK FOR SUM ERROR SQUARE
			call_sum_dekad_error = sum_dekad_error(fst_dek, lst_dek, accumulations, dek_dictionary, output_snack) #we call the resulting RANK FOR SUM DEKAD error_sqr
			analogs_dictionary = get_analog_years(init_yr, end_yr, analog_num, call_sum_error_sqr, call_sum_dekad_error)

			stamp = seasonal_accumulations_plotting(accumulations, analogs_dictionary)
			stamp2 = seasonal_accumulations(init_clim, end_clim, accumulations)
			stamp3 = ensemble_plotting(init_yr, end_yr, lst_dek, init_clim, end_clim, output_snack, accumulations, analogs_dictionary, dek_dictionary)
			
			plot = generate_reports(init_yr, end_yr, fst_dek, lst_dek, init_clim, end_clim, analogRank, output_snack, accumulations, stamp, stamp2, stamp3, dek_dictionary, analogs_dictionary)

##############################################################################################################################################

	def clearFiles(self):

		#clear menus:
		self.analog_menu.set('')
		self.start_dekad.set('')
		self.end_dekad.set('')
		self.init_clim.set('')
		self.end_clim.set('')

		'''
		#clear files
		try:
			for i in ['data', './datapath/output_snack', './datapath/accumulations', './datapath/analogs']:
				os.remove(i)

		except FileNotFoundError:
			a = 0
		'''
		
		tkinter.messagebox.showinfo('status', 'All cleared!')

##############################################################################################################################
root = Tk()



main = mainFrame(root)
root.mainloop()

