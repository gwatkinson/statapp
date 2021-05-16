"""Generate the useful dataframes from the raw datasets."""

# Import
import data_setup as ds


def generate_dataframes():
    """
    Generate and save the derived DataFrames to the derived data folder.
    """
    # Load raw dfs
    data_firm_level, patents_firm_merge, cites = ds.load_data(dfs=["data_firm_level", "patents_firm_merge", "cites"])

    # Format patent data
    patent_data = ds.format_patent_data(
        patents_firm_merge,
        keep=["year", "permno", "xi", "Npats", "Tcw", "Tsm", "patent_class"],
        na_cols=["patent_class", "xi"],
        save=True,
    )
    print("Saving patent data")

    # Patent distribution
    distributions = ds.add_dumies(
        patent_data,
        extra={"xi": "mean", "Npats": "first", "Tcw": "first", "Tsm": "first"},
        save=True,
    )
    print("Saving distributions")

    # Patent level cites
    patent_cites = ds.add_permno(cites, patents_firm_merge, save=True)
    print("Saving patent cites")

    # Firm level cites
    firm_cites = ds.patent_to_firm_cites(patent_cites, save=True)
    print("Saving firm cites")


if __name__ == "__main__":
    generate_dataframes()