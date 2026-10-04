"""Small plotting conveniences. Polars frames go to seaborn as pandas (`.to_pandas()`), which
is the only place pandas is needed."""

from __future__ import annotations

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns


def setup() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams["figure.figsize"] = (10, 4)
    pl.Config.set_tbl_rows(20)
    pl.Config.set_fmt_str_lengths(60)


def rate_plot(rates: pl.DataFrame, title: str, x: str = "low", ax=None):
    """Plots `binned_rate` / `categorical_rate` output: rate per bin, bar width ~ rows."""
    ax = ax or plt.subplots()[1]
    data = rates.drop_nulls(x).to_pandas()
    sns.lineplot(data=data, x=x, y="rate", marker="o", ax=ax)
    ax.set_title(title)
    ax.set_ylabel("rate")
    return ax


def loglog_hist(values: pl.Series, title: str, ax=None):
    """Histogram of a heavy-tailed count (reviews per user / game) on log-log axes."""
    ax = ax or plt.subplots()[1]
    counts = values.value_counts().sort(values.name)
    ax.loglog(counts[values.name].to_numpy(), counts["count"].to_numpy(), ".", alpha=0.6)
    ax.set_title(title)
    ax.set_xlabel(values.name)
    ax.set_ylabel("frequency")
    return ax
