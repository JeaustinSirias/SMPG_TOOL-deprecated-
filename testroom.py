import numpy as np
import matplotlib.pyplot as plt
import pickle
from io import *
import plotly.graph_objects as go
from plotly.colors import n_colors
import pandas as pd


def input_data(input_d):
	data = pd.read_csv(input_d, header = None,)
	df = pd.DataFrame(data)

	#SETUP HEADER AS STRING LIKE 'YEAR|DEK' FIRST 4 CHARACTERS DEFINE YEAR AND LAST 2 CHARACTERS DEFINE ITS DEK
	header = list(df.loc[0][1:])
	header_str = []
	for i in np.arange(0, len(header), 1):
		head =  str(header[i])[0:6]
		header_str.append(head)

	#returns a 3rd dim array with this features: [locations'_tags, header, raw data]
	return np.array([np.array(df.loc[1:][0]), np.array(header_str), np.array(df.loc[1:]).transpose()[1:].transpose()])

print(input_data('data_hg.csv')[2].shape)












'''
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



