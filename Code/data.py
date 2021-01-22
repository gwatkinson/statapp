import pandas as pd
import kmapper as km

# Some sample data
from sklearn import datasets


def import_data():
    data_firm_level = pd.read_stata(r"..\Data\Firm_patent\data_firm_level.dta")

    patents_firm_merge = pd.read_stata(r"..\Data\Firm_patent\patents_firm_merge.dta")

    data_patent_level = pd.read_stata(
        r"..\Data\Patent_level_data\data_patent_level.dta"
    )

    cites = pd.read_stata(
        r"..\Data\Patent_level_data\USPatent_1926-2010\cites\cites.dta"
    )

    firm_innovation_v2 = pd.read_stata(
        r"..\Data\Patent_level_data\USPatent_1926-2010\firm_innovation\firm_innovation_v2.dta"
    )

    patents_xi = pd.read_stata(
        r"..\Data\Patent_level_data\USPatent_1926-2010\patents_xi\patents_xi.dta"
    )

    patent_values = pd.read_stata(
        r"..\Data\Patent_level_data\Patent_CRSP_match_1929-2017\patent_values\patent_values.dta"
    )

    return {
        "data_firm_level": data_firm_level,
        "patents_firm_merge": patents_firm_merge,
        "data_patent_level": data_patent_level,
        "cites": cites,
        "firm_innovation_v2": firm_innovation_v2,
        "patents_xi": patents_xi,
        "patent_values": patent_values,
    }


def main():
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
        path_html="make_circles_keplermapper_output.html",
        title="make_circles(n_samples=5000, noise=0.03, factor=0.3)",
    )


if __name__ == "__main__":
    dict = import_data()
