from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
from copy import deepcopy
import numpy as np
import pandas as pd
# import geopandas as gpd

# plotly.offline.init_notebook_mode()
# plotly.offline.init_notebook_mode(connected=True)

import matplotlib.pyplot as plt

plt.style.use('ggplot')


# figDir = "C:\\Users\\swmishr\\Documents\\All-Program\\App\\GeoSpatial-Analysis\\Figures\\"

class GlobalPlot(object):
    def __init__(self):
        pass
    
    def get_Plot_info(self):
        plotInfo = dict(
            plot_type='scatter, histogram2Dcontours',
            plot_mode="lines, markers, lines+markers, lines+markers+text",
            marker_conf=dict(
                size='3 to 10',
                opacity='0.3 to 0.7',
                color='black, white , etc. When using group on a third column then pass in the column values;',
                colorscale='Viridis etc. Use when performing a group by on a third column values. Optimal when many values in 3rd column',
                showscale='True or False : Should you choose to activate the color scale or not'
            )
        )
        
        return plotInfo
    
    def get_GeoPlot_info(self):
        pass
    
    def set_figure(self):
        pass
    
    def base_plot(self):
        pass
    
    def add_plot(self):
        pass


class GeoPlot():
    def __init__(self):
        # GlobalPlot.__init__(self)
        self.newPlot = None
        self.basePlot = None
        self.axPointer = None
        
        # We basically make a default subplot, However this can be overridden by calling
        # the set_figure function
        # self.fig, self.ax = plt.subplots(1, 1,
        #                                  figsize=(10, 10),
        #                                  facecolor='w', edgecolor='k')
    
    def set_figure(self, numRows=1, numColumns=1, lenXaxis=40, lenYaxis=15):
        self.fig, self.ax = plt.subplots(numRows,
                                         numColumns,
                                         figsize=(lenXaxis, lenYaxis),
                                         facecolor='w', edgecolor='k')
        if numRows > 1 or numColumns > 1:
            self.ax = self.ax.ravel()
            self.axPointer = 0
    
    def set_data(self, dataIN):
        self.dataIN = dataIN
    
    def base_plot(self, color='white', dataIN=[]):
        if any(dataIN):
            self.dataIN = dataIN
        
        if self.axPointer != None:
            self.basePlot = self.dataIN.plot(ax=self.ax[self.axPointer], color=color)
            self.axPointer += 1
        else:
            self.basePlot = self.dataIN.plot(ax=self.ax, color=color)
    
    def add_plot(self, dataIN, color='red', shapePlot=False):
        if not shapePlot:
            if type(dataIN) == list:
                dataIN = pd.DataFrame({
                    'x': [dataIN[0]],
                    'y': [dataIN[1]]
                })
            elif type(dataIN) == np.ndarray:
                dataIN = pd.DataFrame({
                    'x': dataIN[:, 0],
                    'y': dataIN[:, 1]
                })
            elif isinstance(dataIN, pd.DataFrame):
                dataIN = pd.DataFrame({
                    'x': dataIN.iloc[:, 0],
                    'y': dataIN.iloc[:, 1]
                })
            else:
                raise ValueError('GeoPlot: The input format of dataIN doesnt match any format specified.')
            
            self.newPlot = dataIN.plot(x='x', y='y', kind='scatter', ax=self.basePlot, color=color)
        else:
            self.newPlot = dataIN.plot(ax=self.basePlot, color=color)
        
        self.basePlot = self.newPlot
        # else:
        #     # When plotting the third layer
        #     self.newPlot = dataIN.plot(x='x', y='y', kind='scatter', ax=self.newPlot, color='red')
    
    def show(self, plotFileName='Temp-Plot'):
        plt.savefig(plotFileName + ".png")
        plt.show()

