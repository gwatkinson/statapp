"""Implement the Mapper algorithm"""
# # Import the class
# import kmapper as km
# from kmapper.cover import Cover

# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# from sklearn.cluster import AgglomerativeClustering

# # Some sample data
# from sklearn import datasets

# data, labels = datasets.make_circles(n_samples=20000, noise=0.03, factor=0.3)

# # Initialize
# mapper = km.KeplerMapper(1)

# # Fit to and transform the data
# projected_data = mapper.fit_transform(data, projection=PCA(n_components=2), scaler=StandardScaler())

# # Create dictionary called 'graph' with nodes, edges and meta-information
# graph = mapper.map(projected_data, data, cover=Cover(n_cubes=10, perc_overlap=0.5), clusterer=AgglomerativeClustering(10, linkage="single"))

# # Visualize it
# mapper.visualize(
#     graph,
#     path_html="../docs/test_km_20k.html",
#     title="Size 20000, PCA, AgglomerativeClustering",
# )
