# Import the class
import kmapper as km

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
mapper.visualize(
    graph,
    path_html="docs/make_circles_keplermapper_output.html",
    title="make_circles(n_samples=5000, noise=0.03, factor=0.3)",
)

