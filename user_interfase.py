from tkinter import *
import tkinter.messagebox
import ttk
import numpy as np
from io import *
from import_data import input_data
import pandas as pd
from modules import *
from main import mean_plot
from tkinter.ttk import *
from tkinter.filedialog import askopenfile 




class mainFrame():

	def __init__(self, master):

		self.titulo = master.title('SMPG-TOOL alpha_1.0a')

		self.background = PhotoImage(file = '/home/jussc_/Downloads/view1.gif')
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
		self.variable_init_dekad = StringVar(self.frame)
		self.variable_end_dekad = StringVar(self.frame)

		self.variable_init = IntVar(self.frame)
		self.variable_end = IntVar(self.frame)
		self.variable_init.set(self.year_lst[0])
		self.variable_end.set(self.year_lst[0])

		#background
		#self.background = PhotoImage(file = '/home/jussc_/Downloads/view.gif')
		#self.bg_label = Label(self.frame, compound = CENTER, image = self.background).grid()
		
		
		#LABELS
		self.label1 = Label(self.frame, text = 'Start year:')
		self.label1.grid(row = 1, column = 1)

		self.label2 = Label(self.frame, text = 'End year:')
		self.label2.grid(row = 1, column = 3)

		self.label3 = Label(self.frame, text = 'Choose starting dekad')
		self.label3.grid(row =2, column = 2)

		self.label4 = Label(self.frame, text = 'Choose last dekad')
		self.label4.grid(row = 3, column = 2)



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
		self.start_dekad.grid(row = 2, column = 3)

		#end dekad menu
		self.end_dekad = ttk.Combobox(self.frame, textvariable = self.variable_end_dekad, values = tuple(self.dekad_lst))
		self.end_dekad.grid(row = 3, column = 3)


		#BUTTONS
		self.load_data_btn = Button(self.frame, text = 'LOAD DATA')
		self.load_data_btn.grid(row = 2, column = 0, pady = 25)

		self.LT_avg_btn = Button(self.frame, text = 'RUN LT AVERAGES', command = lambda: mean_plot(int(self.ano_init.get()), int(self.ano_fin.get())))
		self.LT_avg_btn.grid(row = 3, column = 0, pady = 25)

		self.accumulations_btn =  Button(self.frame, text = 'Get accumulations history', command = lambda: graph_1().func(str(self.start_dekad.get()), str(self.end_dekad.get()), int(self.ano_init.get()), int(self.ano_fin.get()) ))
		self.accumulations_btn.grid(row = 4, column = 0, pady = 25)

		#browse button
		self.browse_btn = Button(self.frame, text = 'Browse Files', command = lambda: mainFrame.open_file(self))
		self.browse_btn.grid(row = 1, column = 0, pady = 25)

		self.help_btn = Button(self.frame, text = 'Help/Quick tutorial')
		#self.help_btn.configure(bg = 'red')
		self.help_btn.grid(row = 4, column = 4)


	def open_file(self):

		file = askopenfile(mode ='r', filetypes =[('csv files', '*.csv')]) 
		if file is not None: 
			data = pd.read_csv(file, header = None)
			df = pd.DataFrame(data)
			array_out = open('data', 'wb') #write binary
			pickle.dump(np.array(df.loc[:]), array_out)
			tkinter.messagebox.showinfo('status', 'Data loaded')
			array_out.close()
			del(array_out)




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