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
from kmapper.cover import Cover
from kmapper import jupyter  # Creates custom CSS full-size Jupyter screen

# %% [markdown]
# ## Chargement des données

# %%
# data_firm_level = pd.read_stata("../Data/Firm_patent/data_firm_level.dta")
# data_patent_level = pd.read_stata("../Data/Patent_level_data/data_patent_level.dta")
# cites = pd.read_stata("../Data/Patent_level_data/USPatent_1926-2010/cites/cites.dta")
# firm_innovation_v2 = pd.read_stata("../Data/Patent_level_data/USPatent_1926-2010/firm_innovation/firm_innovation_v2.dta")
# patents_xi = pd.read_stata("../Data/Patent_level_data/USPatent_1926-2010/patents_xi/patents_xi.dta")
# patent_values = pd.read_stata("../Data/Patent_level_data/Patent_CRSP_match_1929-2017/patent_values/patent_values.dta")

# %% [markdown]
# ## Utilisation de la base merged
# %% [markdown]
# ### Récupération des données en dataframe pandas

# %%
patents_firm_merge = pd.read_stata("../Data/Firm_patent/patents_firm_merge.dta")


# %%
patents_firm_merge


# %%
patents_firm_merge.count() / len(patents_firm_merge)

# %% [markdown]
# ### On garde les grandes entreprises

# %%
patents_firm_merge[patents_firm_merge["permno"] == 12490.0]


# %%
patents_firm_merge["permno"].nunique()


# %%
big_firms = (
    patents_firm_merge.groupby("permno")["Npats"]
    .mean()
    .sort_values(ascending=False)
    .iloc[10:13]
)


# %%
big_firms_index = big_firms.reset_index()["permno"].values


# %%
reduced_data = patents_firm_merge[patents_firm_merge["permno"].isin(big_firms_index)]


# %%
reduced_data

# %% [markdown]
# ### Utilise les index données dans la df et convertit les dates

# %%
datetime_df = reduced_data
for col in ["fdate", "idate", "pdate"]:
    a = pd.to_datetime(reduced_data[col], format="%m/%d/%Y", errors="coerce")
    datetime_df[col] = a
datetime_df.set_index("index", inplace=True)


# %%
datetime_df.dtypes

# %% [markdown]
# ### On enlève les lignes incomplètes
#
# On voit le pourcentage de lignes non vides pour chaques colonnes :

# %%
datetime_df.count() / len(datetime_df)


# %%
full_df = datetime_df.dropna(subset=["xi", "ncites", "tcw", "tsm"])


# %%
full_df.count() / len(full_df)


# %%
full_df

# %% [markdown]
# ### On lisse les données numériques par rapport au temps

# %%
features = ["xi", "Tcw", "Tsm", "tcw", "tsm", "ncites"]
SMA_features = ["SMA_" + l for l in features]


# %%
full_df[SMA_features] = (
    full_df.sort_values(by="idate")
    .groupby(["permno", "patent_class"])[features]
    .rolling(window=5, min_periods=1)  # =5?
    .mean()
    .reset_index(level=[0, 1], drop=True)
    .rename(columns={l: "SMA_" + l for l in features})
)


# %%
for l in features:
    full_df["log_" + l] = np.log(1 + full_df["SMA_" + l])

# %% [markdown]
# ### On normalise les données numériques lissées et passées au log

# %%
matrix = full_df[["log_xi", "log_Tcw", "log_Tsm", "log_tcw", "log_tsm", "log_ncites"]]


# %%
matrix


# %%
normalised_matrix = StandardScaler().fit_transform(matrix)


# %%
normalised_matrix

# %% [markdown]
# ### On fait une ACP sur cette matrice
#
# Puis on rajoute les indices

# %%
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(normalised_matrix)
principalDf = pd.DataFrame(data=principalComponents, columns=["PC1", "PC2"])


# %%
projected_data = pd.concat(
    [matrix.reset_index()["index"], principalDf], axis=1
).set_index("index")


# %%
projected_data

# %% [markdown]
# ### On applique le Mapper Algorithm

# %%
# Initialize
mapper = km.KeplerMapper(verbose=1)


# %%
proj_matrix = mapper.fit_transform(
    X=normalised_matrix, projection=PCA(n_components=2)
)  # , scaler=StandardScaler())


# %%
proj_matrix


# %%
# Create dictionary called 'graph' with nodes, edges and meta-information
graph = mapper.map(
    lens=proj_matrix, X=normalised_matrix, cover=Cover(n_cubes=20, perc_overlap=0.5)
)  # , clusterer=AgglomerativeClustering(n_clusters=[2], linkage="single"))


# %%
# Visualize it
html = mapper.visualize(
    graph, path_html="../docs/MapperCluster2.html", title="Mapper Clustering Algorithm"
)

# Inline display
# jupyter.display(path_html="../docs/MapperCluster.html")
