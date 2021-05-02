"""
Module to store graphs and ploting functions.
"""


def cite_hist(
    permno,
    permno_cites,
    method="freq",
    show=True,
    save=True,
    path=None,
    dpi=500,
    figsize=(10, 7),
):
    """
    Plot and/or save a histogram of the cited firms by the given firm id.

    Parameters
    ----------
    permno : int
        The index of the firm to plot.
    permno_cites : DataFrame
        The firm DataFrame with `count` and/or `freq`.
    method : str, default 'freq'
        The type of the plot ('freq' or 'count').
    show : bool, default True
        Whether to show the figure.
    save : bool, default True
        Whether to save the figure.
    path : str, default None
        Where to save.

        If `None`, saves in f"../images/cites/hist_{method}_{permno}".
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
    label = "Nombre de brevets" if method == "count" else "Frequence"
    permno_cites[permno_cites["citing_permno"] == permno].sort_values("cited_permno").plot(
        x="cited_permno", y=method, label=label, ax=ax
    )
    plt.axvline(
        x=int(permno),
        color="red",
        linestyle="dotted",
        linewidth=0.7,
        label=f"Entreprise {permno}",
    )
    ax.set_title(f"Histogramme des entreprises citées par {permno}")
    ax.set_xlabel(f"Index des entreprises citées par l'entreprise {permno}")
    ax.set_ylabel(label)
    plt.legend()

    if save:
        fig.savefig(path, dpi=dpi)
    if show:
        plt.show()
