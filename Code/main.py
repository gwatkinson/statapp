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
datetime_df = patents_firm_merge.set_index("index")
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
full_df


# %%
features = ["xi", "Tcw", "Tsm", "tcw", "tsm", "ncites"]
SMA_features = ["SMA_" + l for l in features]


# %%
full_df[SMA_features] = (
    full_df.sort_values(by="idate")
    .groupby(["permno", "patent_class"])[features]
    .rolling(window=5, min_periods=1)
    .mean()
    .reset_index(level=[0, 1], drop=True)
    .rename(columns={l: "SMA_" + l for l in features})
)


# %%
for l in features:
    full_df["log_" + l] = np.log(1 + full_df["SMA_" + l])


# %%
matrix = full_df[["log_xi", "log_Tcw", "log_Tsm", "log_tcw", "log_tsm", "log_ncites"]]


# %%
matrix


# %%
normalised_matrix = StandardScaler().fit_transform(matrix)


# %%
normalised_matrix


# %%
PCA = PCA(n_components=2)
principalComponents = PCA.fit_transform(normalised_matrix)
principalDf = pd.DataFrame(data=principalComponents, columns=["PC1", "PC2"])


# %%
matrix.reset_index()["index"]


# %%
projected_data = pd.concat(
    [matrix.reset_index()["index"], principalDf], axis=1
).set_index("index")


# %%
projected_data


# %%
# Initialize
mapper = km.KeplerMapper(verbose=0)


# %%
# Cover
cov = Cover(n_cubes=20, perc_overlap=0.5)


# %%
# Create dictionary called 'graph' with nodes, edges and meta-information
graph = mapper.map(projected_data, normalised_matrix, cover=cov)


# %%
# Visualize it
html = mapper.visualize(
    graph, path_html="../docs/MapperCluster.html", title="Mapper Clustering Algorithm"
)

# Inline display
jupyter.display(path_html="../docs/MapperCluster.html")


# %%
graph

