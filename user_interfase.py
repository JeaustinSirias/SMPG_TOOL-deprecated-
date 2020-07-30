from tkinter import *
import tkinter.messagebox
import ttk
import numpy as np
from io import *
import pandas as pd
from tkinter.ttk import *
from tkinter.filedialog import askopenfile 
import os

#local packages
from module import *
from module2 import *




class mainFrame():

	def __init__(self, master):

		self.titulo = master.title('SMPG-TOOL alpha_1.0b')

		self.background = PhotoImage(file = '/home/jussc_/Downloads/new.gif')
		self.bg = Canvas(master, width = 800, height = 100 )
		self.bg.pack()
		self.cv_img = self.bg.create_image(0, 0, image = self.background, anchor = 'nw')



		self.frame = Frame(master, width = 500, height = 400)
		self.frame.pack()
		#self.title = Label(self.bg, text = 'WELCOME TO THIS BETA VERSION OF SMPG TOOL', font = 50)
		#self.title.pack(side = TOP)
		#self.title.grid(row = 0, column = 1, pady = 25, columnspan = 6)
		#self.title.config(justify = 'center')

		self.year_lst = np.arange(1980, 2021, 1)
		self.dekad_lst = ['1-Jan', '2-Jan', '3-Jan', '1-Feb', '2-Feb', '3-Feb', '1-Mar', '2-Mar', '3-Mar', '1-Apr', '2-Apr', '3-Apr', '1-May', '2-May', '3-May', '1-Jun',
		 					'2-Jun', '3-Jun', '1-Jul', '2-Jul', '3-Jul', '1-Aug', '2-Aug', '3-Aug', '1-Sep', '2-Sep', '3-Sep', '1-Oct', '2-Oct', '3-Oct', '1-Nov', '2-Nov', 
		 					'3-Nov', '1-Dec', '2-Dec', '3-Dec']
		self.analogs_lst = np.arange(1, 40, 1)
		self.variable_analogs_lst = IntVar(self.frame)
		self.variable_init_dekad = StringVar(self.frame)
		self.variable_end_dekad = StringVar(self.frame)

		self.radio_button = IntVar(self.frame)
		self.variable_init = IntVar(self.frame)
		self.variable_end = IntVar(self.frame)
		self.variable_init.set(self.year_lst[0])
		self.variable_end.set(self.year_lst[0])

		#climatology
		self.variable_init_clim = IntVar(self.frame)
		self.variable_end_clim = IntVar(self.frame)
		#self.variable_init_clim.set()

##############################################################################################################################################

		#LABELS
		self.label0 = Label(self.frame, text = 'Set up climatology window')
		self.label0.grid(row = 2, column = 1, columnspan = 4)

		self.label_clim1 = Label(self.frame, text = 'From:')
		self.label_clim1.grid(row = 1, column = 1, padx = 10)

		self.label_clim2 = Label(self.frame, text = 'To:')
		self.label_clim2.grid(row = 1, column = 3, padx = 10)

		self.labelz = Label(self.frame, text = 'Choose analysis preferences')
		self.labelz.grid(row = 0, column = 1, columnspan = 4)

		self.labelz = Label(self.frame, text = 'Define a season to monitor')
		self.labelz.grid(row = 4, column = 1, columnspan = 4, pady = 25)


		self.label1 = Label(self.frame, text = 'Initial year:')
		self.label1.grid(row = 3, column = 1, padx = 10)

		self.label2 = Label(self.frame, text = 'Final year:')
		self.label2.grid(row = 3, column = 3, padx = 10)

	
		self.label3 = Label(self.frame, text = 'From:')
		self.label3.grid(row = 5, column = 1,)

		self.label4 = Label(self.frame, text = 'to:')
		self.label4.grid(row = 5, column = 3)
	

		self.label5 = Label(self.frame, text = 'Select the number of analog years to compute:')
		self.label5.grid(row = 6, column = 1, pady = 25, columnspan = 3)

		self.label6 = Label(self.frame, text = 'Specify the top by rank:')
		self.label6.grid(row = 7, column = 1, pady = 25, columnspan = 3)

		self.label7 = Label(self.frame, text = 'Computing preferences')
		self.label7.grid(row = 5, column = 0)

##############################################################################################################################################		
		#MENUS

		self.init_clim = ttk.Combobox(self.frame, textvariable = self.variable_init_clim, values = tuple(self.year_lst))
		self.init_clim.grid(row = 3, column = 2)

		self.end_clim = ttk.Combobox(self.frame, textvariable = self.variable_end_clim, values = tuple(self.year_lst))
		self.end_clim.grid(row = 3, column = 4)
		
		#start year option menu
		self.ano_init = ttk.Combobox(self.frame, textvariable = self.variable_init, values = tuple(self.year_lst))
		self.ano_init.grid(row = 1, column = 2)
		#self.ano_init.pack()

		#end year option menu
		self.ano_fin = ttk.Combobox(self.frame, textvariable = self.variable_end, values = tuple(self.year_lst))
		self.ano_fin.grid(row = 1, column = 4)
		#self.ano_fin.pack()

		#first dekad menu
		self.start_dekad = ttk.Combobox(self.frame, textvariable = self.variable_init_dekad, values = tuple(self.dekad_lst))
		self.start_dekad.grid(row = 5, column = 2)

		#end dekad menu
		self.end_dekad = ttk.Combobox(self.frame, textvariable = self.variable_end_dekad, values = tuple(self.dekad_lst))
		self.end_dekad.grid(row = 5, column = 4)
		
		#ANALOG YEARS MENU
		self.analog_menu  = ttk.Combobox(self.frame, textvariable = self.variable_analogs_lst, values = tuple(self.analogs_lst))
		self.analog_menu.grid(row = 6, column = 4)
	
##############################################################################################################################################	
		#BUTTONS

		self.load_data_btn = Button(self.frame, text = 'COMPUTE INPUT DATA', command = lambda: mainFrame.compute_input_data(self, 
																															int(self.ano_init.get()), 
																															int(self.ano_fin.get()), 
																															str(self.start_dekad.get()), 
																															str(self.end_dekad.get()),
																															int(self.init_clim.get()),
																															int(self.end_clim.get())))
		self.load_data_btn.grid(row = 2, column = 0, pady = 25)

		
		self.LT_avg_btn = Button(self.frame, text = 'GENERATE REPORTS', command = lambda: mainFrame.gen_reports(self, 
																												int(self.analog_menu.get()), 
																												int(self.ano_init.get()), 
																												int(self.ano_fin.get()), 
																												str(self.start_dekad.get()), 
																												str(self.end_dekad.get()),
																												int(self.init_clim.get()),
																												int(self.end_clim.get())))
																											
		self.LT_avg_btn.grid(row = 4, column = 0)
		
	
		#browse button
		self.browse_btn = Button(self.frame, text = 'Browse Files', command = lambda: mainFrame.open_file(self))
		self.browse_btn.grid(row = 0, column = 0, pady = 20)

		self.help_btn = Button(self.frame, text = 'Clear', command = lambda: mainFrame.clearFiles(self))
		#self.help_btn.configure(bg = 'red')
		self.help_btn.grid(row = 8, column = 4, pady = 25)
		

		self.fct = Radiobutton(self.frame, text = 'Forecast', variable = self.radio_button, value = 0)
		self.fct.grid(row = 6, column = 0)

		self.analysis = Radiobutton(self.frame, text = 'Analysis', variable = self.radio_button, value = 1)
		self.analysis.grid(row = 7, column = 0)

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
			array_out = open('data', 'wb') #write binary
			pickle.dump(output, array_out)
			tkinter.messagebox.showinfo('Data loaded!', 'Input dataset goes from {init} to {end}'.format(init = output[1][0][0:4], end = output[1][-1][0:4]))
			array_out.close()
			del(array_out)

##############################################################################################################################################

	def compute_input_data(self, init_year, end_year, fst_dek, lst_dek, init_clim, end_clim):

		if init_year == end_year:
			tkinter.messagebox.showinfo('warning', 'Intial year and end year cannot be the same')


		else:
			run = LT_procedures(init_year, end_year, fst_dek, lst_dek, init_clim, end_clim)
			run.get_analog_years()
			tkinter.messagebox.showinfo('status', 'Dataset succesfully computed')

##############################################################################################################################################

	def gen_reports(self, an_years, init_year, end_year, init_dek, end_dek, init_clim, end_clim):

		if an_years == 0 or an_years == 1:
			tkinter.messagebox.showerror('warning', 'More than 1 analog year must be chosen')

		else:
			report = proccess_data_to_plot(an_years, init_year, end_year, init_dek, end_dek, init_clim, end_clim)
			report.plot_report()

##############################################################################################################################################
	
	def clearFiles(self):
		#clear menus:
		self.ano_init.set('')
		self.ano_fin.set('')
		self.analog_menu.set('')
		self.start_dekad.set('')
		self.end_dekad.set('')
		self.init_clim.set('')
		self.end_clim.set('')

		#clear files
		try:
			for i in ['data', './datapath/output_snack', './datapath/accumulations', './datapath/analogs']:
				os.remove(i)

		except FileNotFoundError:
			a = 0
		
		tkinter.messagebox.showinfo('status', 'All cleared!')

##############################################################################################################################################
root = Tk()
#root.config(bg = 'blue')


main = mainFrame(root)
root.mainloop()






'''
class myFirstFrame():
	
	#PROPIEDADES
	def __init__(self):

		self.master = Tk()
		self.titulo = self.master.title('SMPG-TOOL alpha_1.0a')
		#self.wm_iconbitmap('earth.ico')
		self.frame =  Frame(self.master)
		self.frame.pack()
		#self.get()
		
	#METODOS


	#add a label to frame
	def label(self, texto, fila, columna):
		Label(self.frame, text = texto).grid(row = fila, column = columna)

	#add a label to my frame
	def button(self, texto, command):
		#self.master = raiz
		self.boton = Button(self.master, text = texto, command = lambda: command).pack()
	

	def textbox(self, fila, columna, Xspace, Yspace):
		self.box = Entry(self.frame)
		self.box.grid(row = fila, column = columna, padx = Xspace, pady = Yspace)
		#return self.box.get()

gui = myFirstFrame()


#etiquetas
gui.label('First year:', 1, 1)
gui.label('End year:', 1, 3)

#ventanas de entrada
ano_init = gui.textbox(1, 2, 5, 5)
ano_fin = gui.textbox(1, 4, 5, 5)




#botones
#gui.button('RUN LT AVERAGES', mean_plot(ano_init.get(), ano_fin.get()))
#gui.button('Get accumulations', graph_1().func(str()))

gui.master.mainloop()
'''








'''

root = Tk()

root.title('ventana')

#root.iconbitmap('/home/jussc_/Desktop/smpg-tool/earth_1.gif')
#root.geometry


root.call('wm', 'iconphoto', root._w, PhotoImage(file = 'earth_1.gif'))

myFrame = Frame()
myFrame.pack(fill = 'both', expand = 'True')
myFrame.config(bg = 'red', width = '650', height = '350')
#myFrame.config(cursor = 'hand2')
root.mainloop()


#variableLabel = Label(contenedor, opciones)

'''