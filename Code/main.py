# %% [markdown]
# # Clustering Mapper
# %% [markdown]
# ## Étapes
#
# * Lisser par rapport au temps (B)
# * Passer au log
# * Enlever les index
# * Normaliser
# * ACP (JB)
# * km.cover(n = 20, cov = 0.5) (G)
# * km.map(ACP, data, cover)
# * Clustering (JB/M)
# * Créer le graph (M)
# %% [markdown]
# ## Importation des modules
# %% [markdown]
# ### Import des modules de bases

# %%
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# %% [markdown]
# ### Pour normaliser les données
#
# Separating out the features
#
#     x = df.loc[:, features].values
#
# Standardizing the features
#
#     x = StandardScaler().fit_transform(x)

# %%
from sklearn.preprocessing import StandardScaler

# %% [markdown]
# ### Pour faire l'ACP
#
# Initialise la classe
#
#     pca = PCA(n_components=2)
#
# Fit le modèle
#
#     principalComponents = pca.fit_transform(x)
#
# Transforme en df pandas
#
#     principalDf = pd.DataFrame(data = principalComponents
#                 , columns = ['principal component 1', 'principal component 2'])
#     finalDf = pd.concat([df[index]], principalDf, axis = 1)

# %%
from sklearn.decomposition import PCA

# %% [markdown]
# ### Pour faire le clustering
#
# En utilisant sklearn :
#
#     model = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='single')
#     model.fit(X)
#     labels = model.labels_

# %%
from sklearn.cluster import AgglomerativeClustering

# %% [markdown]
# En utilisant scipy :
#
#     link = sch.linkage(y, method='single", metric='...')
#     dendrogram = sch.dendrogram(link)
#
# Voir https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html

# %%
import scipy.cluster.hierarchy as sch

# %% [markdown]
# ### Keppler Mapper

# %%
import kmapper as km
from kmapper import jupyter  # Creates custom CSS full-size Jupyter screen

# %% [markdown]
# ## Chargement des données

# %%
data_firm_level = pd.read_stata("../Data/Firm_patent/data_firm_level.dta")
data_patent_level = pd.read_stata("../Data/Patent_level_data/data_patent_level.dta")
cites = pd.read_stata("../Data/Patent_level_data/USPatent_1926-2010/cites/cites.dta")
firm_innovation_v2 = pd.read_stata(
    "../Data/Patent_level_data/USPatent_1926-2010/firm_innovation/firm_innovation_v2.dta"
)
patents_xi = pd.read_stata(
    "../Data/Patent_level_data/USPatent_1926-2010/patents_xi/patents_xi.dta"
)
patent_values = pd.read_stata(
    "../Data/Patent_level_data/Patent_CRSP_match_1929-2017/patent_values/patent_values.dta"
)

# %% [markdown]
# ## Utilisation de la base merged

# %%
patents_firm_merge = pd.read_stata("../Data/Firm_patent/patents_firm_merge.dta")


# %%
patents_firm_merge


# %%
datetime_df = patents_firm_merge
for col in ["fdate", "idate", "pdate"]:
    datetime_df[col] = pd.to_datetime(
        patents_firm_merge[col], infer_datetime_format=True, errors="coerce"
    )


# %%
datetime_df.dtypes


# %%
datetime_df.count() / len(datetime_df)


# %%
full_df = datetime_df.dropna(subset=["xi", "ncites", "tcw", "tsm"])


# %%
full_df.count() / len(full_df)


# %%
full_df.groupby("permno")


# %%
full_df.drop(["fdate", "pdate", "year", "_merge"], axis=1)


# %%
index_names = [
    c
    for c in patents_firm_merge.columns
    if c
    not in [
        "index",
        "patnum",
        "fdate",
        "idate",
        "pdate",
        "permno",
        "year",
        "Npats",
        "_merge",
        "patent_class",
        "subclass",
    ]
]


# %%
df = patents_firm_merge[feature_names]


# %%
df.dtypes


# %%
# Some sample data
from sklearn import datasets

data, labels = datasets.make_circles(n_samples=5000, noise=0.03, factor=0.3)

# Initialize
mapper = km.KeplerMapper(verbose=1)

# Fit to and transform the data
projected_data = mapper.fit_transform(data, projection=[0, 1])  # X-Y axis

# Create dictionary called 'graph' with nodes, edges and meta-information
graph = mapper.map(projected_data, data)

# Visualize it
html = mapper.visualize(
    graph,
    path_html="make_circles_keplermapper_output.html",
    title="make_circles(n_samples=5000, noise=0.03, factor=0.3)",
)

# Inline display
# jupyter.display(path_html="http://mlwave.github.io/tda/word2vec-gender-bias.html")
jupyter.display(path_html="make_circles_keplermapper_output.html")


# %%
# projected_data
from sklearn.decomposition import PCA

x = df.values
# preprocessing avant PCA

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scale = scaler.fit(X)
X_scaled = scaler.transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
PC = pca.fit_transform(X_scaled_df)
p_Df = pd.DataFrame(data=PC, columns=["principal component 1", "principal component 2"])
p_Df.head()

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
projected_data = pd.DataFrame(
    data=principalComponents, columns=["principal component 1", "principal component 2"]
)

projected_dataframe = pd.concat([projected_data, df[["target"]]], axis=1)

fig = plt.figure(figsize=(1000, 1000))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel("Principal Component 1", fontsize=15)
ax.set_ylabel("Principal Component 2", fontsize=15)
ax.set_title("2 component PCA", fontsize=20)
targets = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
colors = ["r", "g", "b"]
for target, color in zip(targets, colors):
    indicesToKeep = finalDf["target"] == target
    ax.scatter(
        finalDf.loc[indicesToKeep, "principal component 1"],
        finalDf.loc[indicesToKeep, "principal component 2"],
        c=color,
        s=50,
    )
ax.legend(targets)
ax.grid()

