import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")


class Plot():
    def __init__(self, rows=1, columns=1, fig_size=[8, 8]):
        fig_size = tuple(fig_size)
        f, axs = plt.subplots(rows, columns, figsize=fig_size)
        self.axs_ind = 0
        
        if rows == 1 and columns == 1:
            self.axs = [axs]
        else:
            self.axs = axs.ravel()
    
    def vizualize(self, data, colX, colY=None, label_col=None, viz_type='bar', params={}):
        '''
            params : title,
            data should be a data frame
        '''
        if viz_type == 'hist':
            data = data.reset_index().drop('index', 1)
            sns.distplot(data, ax=self.axs[self.axs_ind])
            if 'title' in params:
                self.axs[self.axs_ind].set_title(params['title'])
        
        if viz_type == 'scatter':
            data.plot.scatter(x=colX, y=colY, c=label_col, colormap='Dark2', ax=self.axs[self.axs_ind])
        
        if viz_type == 'bar':
            if not colY:
                data_grpd = data.groupby(colX).size().rename('count').reset_index()
                percentage = np.array(data_grpd['count']) / sum(np.array(data_grpd['count']))
                ax = self.axs[self.axs_ind]
                tot_cnt = float(len(data_grpd))
                sns.barplot(x=colX, y="count", data=data_grpd, ax=ax)
            else:
                ax = self.axs[self.axs_ind]
                data[colY] = data[colY].astype('float')
                percentage = np.array(data[colY], dtype=float) / sum(np.array(data[colY], dtype=float))
                sns.barplot(x=colX, y=colY, data=data, ax=ax)
            
            if 'title' in params:
                ax.set_title(params['title'])
            
            for e, p in enumerate(ax.patches):
                # print ('dsadasdasdasdasdadasdadas')
                height = p.get_height()
                ax.text(p.get_x() + p.get_width() / 2.,
                        height + 3,
                        '%s %s' % (str(round(percentage[e] * 100, 3)), str('%')),
                        ha="center")
        
        if viz_type == 'countplot':
            ''' Prefer when labels are not highly imbalance, The plot would render nicely'''
            ax = self.axs[self.axs_ind]
            tot_cnt = float(len(data))
            sns.countplot(x=colX, hue=label_col, data=data, ax=ax)
            for p in ax.patches:
                height = p.get_height()
                ax.text(p.get_x() + p.get_width() / 2.,
                        height + 5,
                        '%s %s' % (str(round((height / tot_cnt) * 100, 3)), str('%')),
                        ha="center")
        
        if viz_type == 'line':
            ax = self.axs[self.axs_ind]
            #             if 'title' in params:
            #                 assert len(params['title']) == len(data.columns())
            for col_names in data.columns:
                ax.plot(np.array(data[col_names]))
            
            ax.legend(list(data.columns), loc=4)
            
            if 'title' in params:
                ax.set_title(params['title'])
        
        self.axs_ind += 1
    
    def show(self):
        plt.show()