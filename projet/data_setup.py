# Import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Patent level functions


# Cite data functions
def add_permno(cites, patents, how='inner'):
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
    
    Returns
    -------
    DataFrame
        New cite DataFrame with the company's id.    
    """
    assert all(col in patents.columns for col in ['patnum', 'permno']), "Missing the required columns 'patnum' and 'permno'."
    
    tmp1 = pd.merge(cites, patents[['patnum', 'permno']], how=how, left_on='citing', right_on='patnum')
    tmp1 = tmp1.drop(columns=['patnum']).rename(columns={'permno' : 'citing_permno'})
    tmp2 = pd.merge(tmp1, patents[['patnum', 'permno']], how=how, left_on='cited', right_on='patnum')
    tmp2 = tmp2.drop(columns=['patnum']).rename(columns={'permno' : 'cited_permno'})
    # TODO : Add save option
    return tmp2

def pat_to_permno_cites(cites, methods=['count', 'freq']):
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
    
    # TODO : Add save option
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

