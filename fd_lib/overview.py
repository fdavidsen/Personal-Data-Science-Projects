from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



class Overview:
    ''' Find out what is different from normal situations. '''
    
    def __init__(self, data, series, only_outliers=False):
        self.data = data.drop(series.name, axis=1)
        self.target = data[series.name]
        self.series = data[series]
        self.series_name = series.name.title()
        self.only_outliers = only_outliers
        self._fit()
        
    def _fit(self):
        IQR = 0
        self.threshold = defaultdict(dict)
        for feature in self.data._get_numeric_data():
            if self.only_outliers is True:
                IQR = self.data[feature].quantile(0.75) - self.data[feature].quantile(0.25)
            
            self.threshold[feature]['lower'] = self.data[feature].quantile(0.25) - (1.5 * IQR)
            self.threshold[feature]['upper'] = self.data[feature].quantile(0.75) + (1.5 * IQR)
            self.threshold[feature]['mean'] = self.data[feature].mean()
            
        frame = defaultdict(list)
        self.plot = {}
        error = []
        for index, row in self.series.iterrows():
            for feature in self.data._get_numeric_data():
                if row[feature] < self.threshold[feature]['lower']:
                    frame['Index'].append(index)
                    frame['Feature'].append(feature)
                    frame['Value'].append(row[feature])
                    frame['Mean'].append(self.threshold[feature]['mean'])
                    
                elif row[feature] > self.threshold[feature]['upper']:
                    frame['Index'].append(index)
                    frame['Feature'].append(feature)
                    frame['Value'].append(row[feature])
                    frame['Mean'].append(self.threshold[feature]['mean'])
                    
            self.frame = pd.DataFrame(frame)
            
            try:
                for feature in self.data._get_numeric_data():
                    features = self.frame[self.frame['Index'] == index]['Feature'].values

                    item = pd.DataFrame(row[features]).T
                    item['Legend'] = self.series_name

                    mean = pd.DataFrame(self.data[features].mean()).T
                    mean['Legend'] = 'Mean'

                    bind = pd.concat([item, mean]).melt('Legend')
                    self.plot[index] = bind
            except KeyError as e:
                error.append(e)
                
        if error:
            print('No records found. Everything is normal.')
                
    def get_frame(self, index):
        title = 'The {}th Observation with A {} Score of {}'.format(index, self.series_name, self.target[index])
        df = self.frame[self.frame['Index'] == index]
        df = df.reset_index().drop(['index', 'Index'], axis=1)
        return df.style.set_caption(title)
        
    def get_plot(self, index, color=['blue', 'green'], fontsize=15, pad=15, figsize=None):
        title = 'The {}th Observation with A {} Score of {}'.format(index, self.series_name, self.target[index])
        fig, ax = plt.subplots()
        
        sns.barplot(
            ax      = ax,
            x       = 'variable',
            y       = 'value',
            hue     = 'Legend',
            data    = self.plot[index],
            palette = sns.color_palette(color)
        )
        ax.set_title(title, fontdict={'fontsize': fontsize}, pad=pad)
        ax.set_xlabel('Feature')
        ax.set_ylabel('Value')
        
        if figsize is not None:
            fig.set_size_inches(figsize)
        
        plt.plot()
        
    def get_index(self, index=None):
        array = np.array(self.frame['Index'].unique())
        
        if index is None:
            return array
        else:
            return array[index]