import pandas as pd
class GetDummies():
    def __init__(self,  part=0.01):
        self.part = part
        self.unique_values = {}
        self.cat_columns = []
        self.num_columns = []

    def fit(self, df):
        self.cat_columns = list(df.columns[df.dtypes == 'object'])
        self.num_columns = list(df.columns[df.dtypes == 'int']) + list(df.columns[df.dtypes == 'float'])
        for i in self.cat_columns:
            tmp = df[i].value_counts(normalize=True).reset_index()
            tmp = tmp[tmp[i] >= self.part]
            self.unique_values[i] = list(tmp['index'].unique())

    def transform(self, df, inplace=False):
        if inplace:
            columns = []
            for i in self.cat_columns:
                for j in range(len(self.unique_values[i])):
                    columns.append(str(i + '=' + str(self.unique_values[i][j])))
                    if i in df.columns:
                        df[i + '=' + str(self.unique_values[i][j])] = \
                        df[i].apply(lambda x: 1 if x == str(self.unique_values[i][j]) else 0)
            return df
        else:
            df_ = df.copy()
            columns = []
            for i in self.cat_columns:
                for j in range(len(self.unique_values[i])):
                    columns.append(str(i + '=' + str(self.unique_values[i][j])))
                    if i in df_.columns:
                        df_[i + '=' + str(self.unique_values[i][j])] = \
                        df_[i].apply(lambda x: 1 if x == str(self.unique_values[i][j]) else 0)
            return df_
        
class TargetPercentageEncoding():
    def __init__(self, target_column, part=0.05, filltype='mean'):
        self.part = part
        self.target_column = target_column
        self.fit_dict = {}
        self.cat_columns = []
        self.filltype=filltype
    def fit(self, df):
        self.cat_columns = list(df.columns[df.dtypes == 'object'])
        self.cat_columns_fill = {}
        mean_v = df[self.target_column].mean()
        for i in self.cat_columns:
            tmp = df[[i,self.target_column]].groupby(i)[self.target_column].mean().reset_index()
            tmp_ = df.groupby(i).size().reset_index().rename(columns={i:i, 0 : 'CNT'})
            tmp_['CNT'] = tmp_['CNT'] / tmp_['CNT'].sum()
            tmp = tmp.merge(tmp_)
            tmp = tmp[tmp['CNT'] >= self.part]
            if self.filltype=='mean':
                self.cat_columns_fill[i] = mean_v
            else:
                self.cat_columns_fill[i] = 0
            self.fit_dict[i] = pd.Series(tmp[self.target_column].values,index = tmp[i]).to_dict()
    
    def transform(self, df, inplace = False):
        if inplace:
            for i in self.cat_columns:
                val = self.cat_columns_fill[i]
                df[i+'_TARGET_PERCENTAGE'] = df[i].map(self.fit_dict[i]).fillna(val)
            return df
        else:
            df_ = df.copy()
            for i in self.cat_columns:
                val = self.cat_columns_fill[i]
                df_[i+'_TARGET_PERCENTAGE'] = df_[i].map(self.fit_dict[i]).fillna(val)
            return df_
        
class ValueCountEncoding():
    def __init__(self):
        self.fit_dict = {}
        self.cat_columns = []
        
    def fit(self, df):
        self.cat_columns = list(df.columns[df.dtypes == 'object'])
        for i in self.cat_columns:
            tmp = df.groupby(i).size().reset_index().rename(columns={i:i, 0 : 'CNT'})
            tmp['CNT'] = tmp['CNT'] / tmp['CNT'].sum()
            self.fit_dict[i] = pd.Series(tmp['CNT'].values,index = tmp[i]).to_dict()
    
    def transform(self, df, inplace = False):
        if inplace:
            for i in self.cat_columns:
                df[i+'_FREQUENCY'] = df[i].map(self.fit_dict[i]).fillna(0)
            return df
        else:
            df_ = df.copy()
            for i in self.cat_columns:
                df_[i+'_FREQUENCY'] = df_[i].map(self.fit_dict[i]).fillna(0)
            return df_
class FillNa():
    def __init__(self, target_column, method='Median'):
        self.fit_dict = {}
        self.cat_columns = []
        self.num_columns = []
        self.target_column = target_column
        self.method = method
        
    def fit(self, df):
        self.cat_columns = list(df.columns[df.dtypes == 'object'])
        self.num_columns = list(set(df.columns) - set(self.cat_columns))
        if set(self.target_column).issubset(self.num_columns):
            self.num_columns = [x for x in self.num_columns if x not in self.target_column]
        else:
            self.cat_columns = [x for x in self.cat_columns if x not in self.target_column]
            
        self.cat_columns_with_na = [x for x in self.cat_columns if df[x].isnull().sum()>0]
        self.num_columns_with_na = [x for x in self.num_columns if df[x].isnull().sum()>0]
        
        for i in self.cat_columns:
            self.fit_dict[i] = df[i].mode()
        if self.method == 'median':
            for i in self.num_columns:
                self.fit_dict[i] = df[i].median()
        elif self.method == 'mean':
            for i in self.num_columns:
                self.fit_dict[i] = df[i].mean()
        elif self.method == 'default':
            for i in self.num_columns:
                self.fit_dict[i] = -99999
            for i in self.cat_columns:
                self.fit_dict[i] = '-99999'
        else:
            return 'Unexpected value of parameter method'
    
    def transform(self, df, inplace=False):
        if inplace:
            for i in self.cat_columns_with_na:
                df[i + '_IS_NAN'] = df[i].isna().astype(int)
                df[i] = df[i].fillna(self.fit_dict[i][0])
            for i in self.num_columns_with_na:
                df[i + '_IS_NAN'] = df[i].isna().astype(int)
                df[i] = df[i].fillna(float(self.fit_dict[i]))
        else:
            df_ = df.copy()
            for i in self.cat_columns:
                if i in self.cat_columns_with_na:
                    df_[i + '_IS_NAN'] = df_[i].isna().astype(int)
                if df[i].isnull().sum()>0:
                    df_[i] = df_[i].fillna(self.fit_dict[i][0])
            for i in self.num_columns:
                if i in self.num_columns_with_na:
                    df_[i + '_IS_NAN'] = df_[i].isna().astype(int)
                if df[i].isnull().sum()>0:
                    df_[i] = df_[i].fillna(float(self.fit_dict[i]))
            return df_