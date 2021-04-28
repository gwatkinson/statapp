# Import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load raw data
def load_df(path, form):
    """
    Load a DataFrame.
    
    Parameters
    ----------
    path : str
        Path to the DataFrame.
    form : str
        Type of the file.
        
        Can be in :
        
            * stata
            * pickle
    
    Returns
    -------
    DataFrame
        Return the loaded DataFrame.
    """
    try:
        if form == 'stata':
            tmp = pd.read_stata(path)
        elif form == 'pickle':
            tmp = pd.read_pickle(path)
        else:
            print("Wrong format")
            tmp = None
    except FileNotFoundError:
        print("File not found")
        tmp = None
    
    return tmp
    
def load_data(dfs=['data_firm_level', 'patents_firm_merge', 'cites']):
    """
    Return a list of the wanted DataFrame.
    
    Parameters
    ----------
    dfs : list[str], default ['data_firm_level', 'patents_firm_merge', 'cites']
        List of the wanted df.
    
    Notes
    -----
    The valid values are :
    
        * data_firm_level
        * data_patent_level
        * cites
        * firm_innovation_v2
        * patents_xi
        * patent_values
        * patents_firm_merge
        * patent_cites
        * firm_cites 
    
    Examples
    --------
    Load the three important DataFrames :
    >>> data_firm_level, patents_firm_merge, cites = load_data(dfs=['data_firm_level', 'patents_firm_merge', 'cites'])
    
    Load a generated DataFrame :
    >>> firm_cites = load_data(dfs=['firm_cites'])
    
    Returns
    -------
    list[DataFrame]
        List containing the wanted DataFrames in the right order.
    """
    d = {
        "data_firm_level" : ("../data/Firm_patent/data_firm_level.dta", 'stata'),
        "data_patent_level" : ("../data/Patent_level_data/data_patent_level.dta", 'stata'),
        "cites" : ("../data/Patent_level_data/USPatent_1926-2010/cites/cites.dta", 'stata'),
        "firm_innovation_v2" : ("../data/Patent_level_data/USPatent_1926-2010/firm_innovation/firm_innovation_v2.dta", 'stata'),
        "patents_xi" : ("../data/Patent_level_data/USPatent_1926-2010/patents_xi/patents_xi.dta", 'stata'),
        "patent_values" : ("../data/Patent_level_data/Patent_CRSP_match_1929-2017/patent_values/patent_values.dta", 'stata'),
        "patents_firm_merge" : ("../data/Firm_patent/patents_firm_merge.dta", 'stata'),
        "patent_cites" : ("../data/derived_data/cites/patent_cites.pkl", 'pickle'),
        "firm_cites" : ("../data/derived_data/cites/firm_cites.pkl", 'pickle'),
    }
    l = []
    
    for df in dfs:
        l.append(load_df(*d[df]))

    return l

#  Patent level functions
def format_patent_data(patent_data, keep=["year", "permno", "patent_class"], na_cols=["patent_class"], date_cols=[]):
    """
    Format the raw DataFrame.
    
    Parameters
    ----------
    patent_data : DataFrame
        The raw patent level DataFrame.
    keep : list[str], default ["idate", "year", "permno", "patent_class"]
        The columns to keep.
    na_cols : list[str], default ["patent_class", "xi"]
        The columns to where to drop the missing values.
    date_cols : list[str], default ["idate"]
        The date columns to format.
    
    Notes
    -----
    The columns of the default patent level data are :
    
    ```
    ['index', 'patnum', 'fdate', 'idate', 'pdate', 'permno', 'patent_class',
    'subclass', 'ncites', 'xi', 'year', 'Npats', 'Tcw', 'Tsm', 'tcw', 'tsm',
    '_merge']
    ```
    
    Returns
    -------
    DataFrame
        The formatted DataFrame.
    """
    tmp = patent_data.copy()
    tmp["patent_class"] = pd.to_numeric(tmp["patent_class"], errors="coerce")
    tmp = tmp.dropna(subset=na_cols)
    tmp["patent_class"] = tmp["patent_class"].astype(int)

    for col in date_cols:
        a = pd.to_datetime(tmp[col], format="%m/%d/%Y", errors="coerce")
        tmp[col] = a
    
    tmp = tmp[keep]

    return tmp

def add_dumies(patent_data, prefix='pc', extra={'xi' : 'mean'}):
    """
    Add patent class dummies.
    
    Parameters
    ----------
    patent_data : DataFrame
        The formatted patent DataFrame.
    prefix : str, default 'class'
        The prefix to add to the dummies.
    extra : dict{str : str}, default {'xi' : 'mean'}
        Extra argument to :func:`pandas.DataFrame.agg()`.
    
    Returns
    -------
    DataFrame
        The firm and year level dummy DataFrame.
    """
    tmp = patent_data.copy()
    tmp = pd.get_dummies(data=tmp, columns=["patent_class"], prefix=[prefix])
    agg_dict = {}
    agg_dict.update(extra)
    agg_dict.update({s : 'sum' for s in tmp.columns if s.startswith(prefix)})
    firm_df = tmp.groupby(by=["permno", "year"]
    ).agg(agg_dict).reset_index()

    return firm_df


# Firm patent level functions
class AddLag(base.BaseEstimator, base.TransformerMixin):
    
    def __init__(self, numLags, groupCol='permno', timevar='year', col_prefix='pc', sufix='lag', set_index=True, dropna=True, filter_rows=True):
        self.numLags = numLags
        self.groupCol = groupCol
        self.timevar = timevar
        self.col_prefix = col_prefix + '_'
        self.sufix = sufix
        self.set_index = set_index
        self.dropna = dropna
        self.filter_rows = filter_rows
        
    def fit(self, X, y=None):
        self.X = X
        return self
    
    def transform(self, X):
        tmp = self.X.copy()
        cols = tmp.columns[tmp.columns.str.startswith(self.col_prefix)]

        if self.filter_rows:
            comp = (tmp.groupby([self.groupCol])[self.groupCol].count() > self.numLags).rename('n').reset_index()
            comp = comp[comp.n][self.groupCol]
            tmp = tmp[tmp[self.groupCol].isin(comp)]

        for i in range(1, self.numLags + 1):
            for col in cols:
                tmp[col + '_' + self.sufix  + '_' + str(i)] = tmp.groupby([self.groupCol])[col].shift(i) 
            
        if self.dropna:
            tmp = tmp.dropna()
            tmp = tmp.reset_index(drop=True)

        if self.set_index:
            tmp = tmp.set_index([self.groupCol, self.timevar])
            
        return tmp.astype(int)

# class AddConcurrence(base.BaseEstimator, base.TransformerMixin):
#     # FIXME : doesn't work
#     def __init__(self, permno, compvar='permno', timevar='year', col_prefix='pc', prefix='comp'):
#         self.permno = permno
#         self.compvar = compvar
#         self.timevar = timevar
#         self.col_prefix = col_prefix + '_'
#         self.prefix = prefix
        
#     def fit(self, X, y=None):
#         self.X = X
#         return self
    
#     def transform(self, X):
#         tmp = self.X.copy()
#         sub = tmp[tmp[compvar]==permno].drop(columns=[compvar])
#         years = sub[self.timevar]
#         sub = sub.set_index(self.timevar)
#         comps = tmp[compvar].unique()
#         cols = tmp.columns[tmp.columns.str.startswith(self.col_prefix)]

#         for comp in comps:
#             if comp != permno:
#                 compsub = tmp[tmp[compvar]==comp].drop(columns=[compvar]).set_index(self.timevar)
#                 tmp[self.prefix  + '_' + str(i) + '_' + col] = tmp.groupby([self.groupCol])[col].shift(i) 

#         if self.set_index:
#             tmp = tmp.set_index([self.groupCol, self.timevar])
            
#         return tmp.astype(int)

# Cite data functions
def add_permno(cites, patents, how='inner', save=True):
    """
    Add citing and cited company id to the cite table.
    
    Drops missing value (a lot are missing, this is a problem TODO).
    
    Parameters
    ----------
    cites : DataFrame
        The DataFrame containing the citations.
    patents : DataFrame
        The DataFrame linking the `permno` to the `patnum`.
        
        Must contain those two columns.
    how : str, default 'inner'
        The merge method.
        
        See :func:`pandas.merge` for details.
    save : bool, default True
        Weither to save the DataFrame as a pickle.
    
    Returns
    -------
    DataFrame
        New cite DataFrame with the companies' ids.    
    """
    assert all(col in patents.columns for col in ['patnum', 'permno']), "Missing the required columns 'patnum' and 'permno'."
    
    tmp1 = pd.merge(cites, patents[['patnum', 'permno']], how=how, left_on='citing', right_on='patnum')
    tmp1 = tmp1.drop(columns=['patnum']).rename(columns={'permno' : 'citing_permno'})
    tmp2 = pd.merge(tmp1, patents[['patnum', 'permno']], how=how, left_on='cited', right_on='patnum')
    tmp2 = tmp2.drop(columns=['patnum']).rename(columns={'permno' : 'cited_permno'})
    
    if save:
        tmp2.to_pickle('../data/derived_data/cites/patent_cites.pkl')
        
    return tmp2

def patent_to_firm_cites(cites, methods=['count', 'freq'], save=True):
    """
    Group the patents by the citing and cited companies.
    
    Parameters
    ----------
    cites : DataFrame
        The DataFrame containing the citations and the companies cited/citing.
    methods : list[str], default ['count', 'freq']
        The methods to use for the new columns.
        
        Values must be in `['count', 'freq']`.
        
        'count' adds the number of patents between the cited and citing companies.
         
        'freq' divides count by the total number of patents of the citing company.
    save : bool, default True
        Weither to save the DataFrame as a pickle.
    
    Returns
    -------
    DataFrame
        New firm DataFrame with the columns `citing_permno`, `cited_permno`, `count` and/or `freq`.        
    """
    tmp = cites.groupby(['citing_permno', 'cited_permno']).count().drop(columns=['cited']).rename(columns={'citing': 'count'}).reset_index()
    
    if 'freq' in methods:
        grp = tmp.groupby('citing_permno')
        tmp['freq'] = grp['count'].transform(lambda x : x/sum(x))
    
    if 'count' not in methods:
        tmp = tmp.drop(columns=['count'])
    
    if save:
        tmp.to_pickle('../data/derived_data/cites/firm_cites.pkl')
    
    return tmp

def cite_hist(permno, permno_cites, method='freq', show=True, save=True, path=None, dpi=500, figsize=(10,7)):
    """
    Plot and/or save a histogramm for the given firm id.
    
    Parameters
    ----------
    permno : int
        The index of the firm to plot.
    permno_cites : DataFrame
        The firm DataFrame with `count` and/or `freq`.
    method : str, default 'freq'
        The type of the plot ('freq' or 'count').
    show : bool, default True
        Weither to show the figure.
    save : bool, default True
        Weither to save the figure.
    path : str, default None
        Where to save, if `None`, saves in f"../images/cites/hist_{method}_{permno}".
    dpi : float, default 500
        The quality of the image.
    figsize : tuple[float], default (10,7)
        The size of the figure.
    
    Returns
    -------
    None
    """
    assert method in permno_cites.columns, "The wanted method is not possible with the given DataFrame."
    
    if save and path is None:
        path = f"../images/cites/hist_{method}_{permno}"
        
    fig, ax = plt.subplots(figsize=figsize)
    label = 'Nombre de brevets' if method=='count' else 'Frequence'
    permno_cites[permno_cites['citing_permno'] == permno].sort_values('cited_permno').plot(x='cited_permno', y=method, label=label, ax=ax)
    plt.axvline(x=int(permno), color='red', linestyle='dotted', linewidth=0.7, label=f'Entreprise {permno}')
    ax.set_title(f'Histogramme des entreprises citées par {permno}')
    ax.set_xlabel(f'Index des entreprises citées par l\'entreprise {permno}')
    ax.set_ylabel(label)
    plt.legend()
    
    if save:
        fig.savefig(path, dpi=dpi)
    if show:
        plt.show()

# Firm level functions

