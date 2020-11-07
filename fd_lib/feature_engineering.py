from itertools import combinations
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin



class BestPair:
    ''' Find out the feature pairs with the highest correlation (positive and negative). '''
    
    def __init__(self, X, y, how='product'):
        self.X = X
        self.y = y
        self.how = how
        self._fit()
        
    def _fit(self):
        self.corr_list = []
        for row, col in combinations(self.X._get_numeric_data(), r=2):
            if self.how == 'product':
                self._corr_append('product', self.product(row, col), row, col)
                
            elif self.how == 'distance':
                self._corr_append('distance', self.distance(row, col), row, col)
                
            elif self.how == 'all':
                self._corr_append('product', self.product(row, col), row, col)
                self._corr_append('distance', self.distance(row, col), row, col)
                
        self.corr_list = sorted(self.corr_list, key=lambda item: -item[3])
    
    def product(self, row, col):
        return self.X[row] * self.X[col]
        
    def distance(self, row, col):
        return (self.X[row]**2 + self.X[col]**2)**0.5
    
    def _corr_append(self, how, new_col, row, col):
        self.corr_list.append((how, row, col, self.y.corr(new_col)))
        
    def get_list(self, top=1):
        upper_top = self.corr_list[:top]
        lower_top = self.corr_list[-top:]
        
        return upper_top + lower_top
        
    def mold(self, top=1):
        top_corr = self.get_list(top)
        
        how_space = max([len(item[0]) for item in top_corr])
        row_space = max([len(item[1]) for item in top_corr])
        col_space = max([len(item[2]) for item in top_corr])
        
        for how, row, col, score in top_corr:
            line = '[{:>%(how_space)s}]   {:>%(row_space)s} & {:<%(col_space)s}   | {:.3f}' % locals()
            print(line.format(how.title(), row, col, score))

            

class CustomFeature(BaseEstimator, TransformerMixin):
    ''' Select custom features to add to the feature matrix. '''
    
    def __init__(self, columns):
        self.columns = columns
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        X = X.copy()
        for how, row, col in self.columns:
            if how == 'product':
                result = X[row] * X[col]
                
            elif how == 'distance':
                result = (self.X[row]**2 + self.X[col]**2)**0.5
            
            X['{}_{}'.format(row, col)] = result
            
        return X