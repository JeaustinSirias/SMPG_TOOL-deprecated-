import numpy as np
import matplotlib.pyplot as plt
import pickle
from io import *
import plotly.graph_objects as go
from plotly.colors import n_colors
import pandas as pd
from collections import defaultdict
import math
from PIL import Image





#OUTPUT SUMMARY DATAFRAME
locNum = np.arange(0, 18, 1)

datas = {'Code': ['None']*len(locNum),
		 'pctofavgatdek': ['None']*len(locNum),
		 'pctofavgatEOS': ['None']*len(locNum),
		 'Above': ['None']*len(locNum),
		 'Normal': ['None']*len(locNum),
		 'Below': ['None']*len(locNum)
		
			}

colNames = ['Code', 'pctofavgatdek', 'pctofavgatEOS', 'Above', 'Normal', 'Below']
frame = pd.DataFrame(datas, columns = colNames )

frame.to_csv('./summary.csv', index = False)




'''
k = {1:[2010, 2011], 2:[1980, 1988, 2009], 3:[2003], 4:[2005, 2000], 5:[1997]}
analog_col = []; analog_data = []; z = 0

y = 5
while y > 1:
	y = y - z
	ar = 'analog {top}'.format(top = y)
	ad = [k[y]]
	analog_col.append(ar)
	analog_data.append(ad)
	z = 1

plt.figure()
plt.axis('tight')
plt.axis('off')
plt.table(colLabels = ['Years'], rowLabels = analog_col, cellText = analog_data, loc = 'center', cellLoc = 'center', bbox = [0.25, 0.35, 0.8, 0.6])
plt.show()


print(int('2018'))
'''		






'''
import plotly.graph_objects as go
from plotly.colors import n_colors
import numpy as np
np.random.seed(1)

colors = n_colors('rgb(255, 200, 200)', 'rgb(200, 0, 0)', 9, colortype='rgb')
a = np.random.randint(low=0, high=9, size=10)
b = np.random.randint(low=0, high=9, size=10)
c = np.random.randint(low=0, high=9, size=10)

fig = go.Figure(data=[go.Table(
  header=dict(
    values=['<b>Column A</b>', '<b>Column B</b>', '<b>Column C</b>'],
    line_color='white', fill_color='white',
    align='center',font=dict(color='black', size=12)
  ),
  cells=dict(
    values=[a, b, c],
    line_color=[np.array(colors)[a],np.array(colors)[b], np.array(colors)[c]],
    fill_color=[np.array(colors)[a],np.array(colors)[b], np.array(colors)[c]],
    align='center', font=dict(color='white', size=11)
    ))
])

fig.show()
'''















'''
space = {}


key = [ 7, 7, 7, 1, 3, 4, 5, 5, 1]
val = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
data_dict = defaultdict(list)

tupla = []
k = 0 
for i in key:
	m = (i, val[k])
	tupla.append(m)
	k = k + 1

for k, v in tupla:
	data_dict[k].append(v)

print(dict(data_dict))



plt.figure()
plt.plot([4], [5], marker='o', markersize=10, color="red")
plt.show()

'''




'''
def input_data(input_d):
	data = pd.read_csv(input_d, header = None,)
	df = pd.DataFrame(data)

	#SETUP HEADER AS STRING LIKE 'YEAR|DEK' FIRST 4 CHARACTERS DEFINE YEAR AND LAST 2 CHARACTERS DEFINE ITS DEK
	header = list(df.loc[0][1:])
	header_str = []
	for i in np.arange(0, len(header), 1):
		head =  str(header[i])[0:6]
		header_str.append(head)

	
	locNames = np.array(df.loc[1:][0])

	locs = []
	for i in np.arange(0, len(locNames), 1):
		try:
			key = str(int(locNames[i]))
			locs.append(key)

		except ValueError:

			key = locNames[i]
			locs.append(key)


	#returns a 3rd dim array with this features: [locations'_tags, header, raw data]
	return np.array([locs, np.array(header_str), np.array(df.loc[1:]).transpose()[1:].transpose()])

print(input_data('ejemplo3.csv')[0])
'''




'''
row = ['Seasonal Avgs', 'Seasonal std', 'Seasonal 33rd', 'Seasonal 67th']
txt = [[127, 129], [234, 333], [190, 123], [222, 345]]
col = ('Analogs', 'All years')

plt.figure()
plt.table(rowLabels = row, colLabels = col, cellText = txt)
plt.show()




fig, ax = plt.subplots(1)

# Hide axes
ax.xaxis.set_visible(False) 
ax.yaxis.set_visible(False)

# Table from Ed Smith answer
clust_data = np.random.random((10,3))
collabel=("col 1", "col 2", "col 3")
ax.table(cellText=clust_data,colLabels=collabel,loc='center')
plt.show()
'''


'''
rows = ['%d year' % x for x in (100, 50, 20, 10, 5)]
print(rows)



data = [[ 66386, 174296,  75131, 577908,  32015],
        [ 58230, 381139,  78045,  99308, 160454],
        [ 89135,  80552, 152558, 497981, 603535],
        [ 78415,  81858, 150656, 193263,  69638],
        [139361, 331509, 343164, 781380,  52269]]

columns = ('Freeze', 'Wind', 'Flood', 'Quake', 'Hail')
rows = ['%d year' % x for x in (100, 50, 20, 10, 5)]

values = np.arange(0, 2500, 500)
value_increment = 1000

# Get some pastel shades for the colors
colors = plt.cm.BuPu(np.linspace(0, 0.5, len(rows)))
n_rows = len(data)

index = np.arange(len(columns)) + 0.3
bar_width = 0.4

# Initialize the vertical-offset for the stacked bar chart.
y_offset = np.zeros(len(columns))

# Plot bars and create text labels for the table
cell_text = []
for row in range(n_rows):
    plt.bar(index, data[row], bar_width, bottom=y_offset, color=colors[row])
    y_offset = y_offset + data[row]
    cell_text.append(['%1.1f' % (x / 1000.0) for x in y_offset])
# Reverse colors and text labels to display the last value at the top.
colors = colors[::-1]
cell_text.reverse()
print(cell_text)

# Add a table at the bottom of the axes
the_table = plt.table(cellText=cell_text,
                      rowLabels=rows,
                      rowColours=colors,
                      colLabels=columns,
                      loc='bottom')

# Adjust layout to make room for the table:
plt.subplots_adjust(left=0.2, bottom=0.2)

plt.ylabel("Loss in ${0}'s".format(value_increment))
plt.yticks(values * value_increment, ['%d' % val for val in values])
plt.xticks([])
plt.title('Loss by Disaster')

plt.show()
'''



'''
lip = []
p = 5
k = 0
while p > 1:
	p = p - k
	k = 1
	lip.append(p)
lip = np.sort(lip)


lista = np.array([1, 2, 3, 4, 5, 6, None, None, None])

x = np.arange(0, len(lista), 1)



y = [2, 5, 6, 8, 2, 4, 9, 2]

k = [None]*(36 - len(y))


print(len(y + k))
'''



