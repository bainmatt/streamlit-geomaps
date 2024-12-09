"""
Plotting customization.
"""

import matplotlib as mpl
from cycler import cycler


def customize_plots() -> None:
    r"""
    Set custom matplotlib rcParams.

    Handles default styling for all matplotlib plotting functions and
    functions built upon matplotlib (e.g., Seaborn plotters).

    Notes
    -----
    Contrast with the default matplotlib rc file:
    https://matplotlib.org/stable/users/explain/customizing.html#the-matplotlibrc-file.

    Examples
    --------
    .. plot::

        >>> import matplotlib
        >>> import matplotlib.pyplot as plt
        >>> from src.stylesheet import customize_plots

        Define some data

        >>> import numpy as np
        >>> x = np.linspace(0, 2 * np.pi, 1000)

        Define a couple plotting functions

        >>> def single_panel_plot(x):
        ...     ax = plt.subplot(111)
        ...     ax.plot(x, np.sin(x), label="$sin(x)$")
        ...     ax.plot(x, np.cos(x), label="$cos(x)$")
        ...     ax.set(xlabel="$x$", frame_on=False)
        ...     ax.set_ylabel("$f(x)$", rotation=0)
        ...     ax.legend()
        ...     ax.grid()
        ...     plt.show()

        >>> def multi_panel_plot(x):
        ...     fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
        ...     axs = axs.flatten()
        ...     for i, ax in enumerate(axs, start=1):
        ...         ax.plot(x, np.sin(i/2 * x))
        ...         ax.plot(x, np.cos(i/2 * x))
        ...         ax.text(
        ...             max(x) - 0.1, -1 + 0.1,
        ...             f"$a = {i/2}$",
        ...             size=12,
        ...             ha="right",
        ...             bbox={
        ...                 "boxstyle": "round",
        ...                 "alpha": 0.8,
        ...                 "facecolor": "white",
        ...             }
        ...         )
        ...         ax.grid()
        ...     fig.supylabel("$f(x)$", rotation=0)
        ...     fig.supxlabel("$x$", x=.5)
        ...     axs[0].legend(
        ...         labels=["$sin(ax)$", "$cos(ax)$"],
        ...         ncol=1,
        ...         loc="lower left",
        ...     )
        ...     plt.show()

        Create the plots

        ..
            The first two examples are before styling;
            the following two are after styling.
            # >>> single_panel_plot()  # /doctest: +SKIP
            # >>> multi_panel_plot()  # /doctest: +SKIP

        >>> customize_plots()
        >>> single_panel_plot(x)  # /doctest: +SKIP
        >>> multi_panel_plot(x)  # /doctest: +SKIP
    """

    # -- General properties --------------------------------------------------

    # Font face and sizes
    mpl.rcParams['font.family'] = 'sans-serif'
    # NOTE: will need to be reverted to the default for use in a notebook.
    # mpl.rcParams['font.sans-serif'] = "Verdana"
    mpl.rcParams['font.size'] = 9               # default font sizes
    mpl.rcParams['axes.titlesize'] = 14         # large
    mpl.rcParams['axes.labelsize'] = 12         # medium
    mpl.rcParams['xtick.labelsize'] = 10        # medium
    mpl.rcParams['ytick.labelsize'] = 10        # medium
    mpl.rcParams['legend.fontsize'] = 10        # medium
    mpl.rcParams['legend.title_fontsize'] = 11  # None (same as default axes)
    mpl.rcParams['figure.titlesize'] = 15       # large (suptitle size)
    mpl.rcParams['figure.labelsize'] = 13       # large (sup[x|y]label size)

    # Spines and ticks
    mpl.rcParams['axes.spines.top'] = True
    mpl.rcParams['axes.spines.right'] = True
    mpl.rcParams['axes.linewidth'] = .7
    mpl.rcParams['axes.edgecolor'] = 'black'
    mpl.rcParams['xtick.direction'] = 'out'
    mpl.rcParams['ytick.direction'] = 'out'
    mpl.rcParams['xtick.major.size'] = 0        # default: 3.5
    mpl.rcParams['ytick.major.size'] = 0        # default: 3.5
    # mpl.rcParams['xtick.major.width'] =  0.8
    # mpl.rcParams['ytick.major.width'] =  0.8

    # Grid
    # lines at {major, minor, both} ticks
    mpl.rcParams['axes.grid.which'] = 'major'
    mpl.rcParams['grid.linestyle'] = '-'
    mpl.rcParams['grid.color'] = '#CCCCCC'
    mpl.rcParams['grid.linewidth'] = 0.8
    mpl.rcParams['grid.alpha'] = .2

    # Label placement
    mpl.rcParams['axes.titlelocation'] = 'center'  # {left, right, center}
    mpl.rcParams['axes.titlepad'] = 7.5  # 6
    mpl.rcParams['axes.labelpad'] = 7.5  # 4
    # mpl.rcParams['xtick.major.pad'] = 3.5  # dist to major tick label in pts
    # mpl.rcParams['ytick.major.pad'] = 3.5

    # Discrete color cycle (and continuous map)
    mpl.rcParams['axes.prop_cycle'] = cycler(
        color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    )
    # mpl.rcParams['axes.prop_cycle'] = cycler(
    #     color=sns.color_palette("PiYG", n_colors=6)
    # )
    # mpl.rcParams['axes.prop_cycle'] = cycler(
    #     color=mpl.cm.PiYG(np.linspace(0.15, .85, 6))  # type: ignore
    # )

    # Legend properties
    mpl.rcParams['legend.loc'] = 'best'
    mpl.rcParams['legend.frameon'] = False

    # Legend padding
    # mpl.rcParams['legend.borderpad'] =  0.4     # border whitespace
    # mpl.rcParams['legend.labelspacing'] = 0.5   # vert space between entries
    mpl.rcParams['legend.handlelength'] = 1.35    # length of the legend lines
    # mpl.rcParams['legend.handleheight'] = 0.7   # height of the legend handle
    mpl.rcParams['legend.handletextpad'] = 0.8    # space btwn leg lines/text
    mpl.rcParams['legend.borderaxespad'] = 0.5    # border btwn axes/leg edge
    mpl.rcParams['legend.columnspacing'] = 1.0    # column separation

    # Space-filling object properties (e.g., polygons/circles, bars/scatter)
    mpl.rcParams['patch.edgecolor'] = 'black'  # if forced, else not filled
    mpl.rcParams['patch.force_edgecolor'] = 1
    mpl.rcParams['patch.linewidth'] = .4       # edgewidth (default: .5)

    # -- Object-specific properties ------------------------------------------

    # Scatter properties
    # mpl.rcParams['scatter.edgecolors'] = 'black'  # alt: 'face' (match edges)

    # Line properties
    mpl.rcParams['lines.markersize'] = 6
    mpl.rcParams['lines.linewidth'] = 2

    # Bar properties
    # NOTE: No global styling parameter exists for the following:
    # mpl.rcParams['bar.width'] = 0.8

    # Error properties
    mpl.rcParams['errorbar.capsize'] = 3

    # NOTE: No global styling parameter exists for the following:
    # mpl.rcParams['errorbar.color'] = 'black'
    # mpl.rcParams['errorbar.linewidth'] = 1.5

    # Contour properties
    # if `none`, falls back to line.linewidth
    mpl.rcParams['contour.linewidth'] = 1

    # Histogram properties
    # hist.bins: 10  # the default number of histogram bins or 'auto'

    # Box properties
    # box
    mpl.rcParams['boxplot.boxprops.linewidth'] = 0      # box outline (0.5)
    # mpl.rcParams['boxplot.boxprops.color'] = 'none'   # alt: 'black' (check)

    # box line to cap
    mpl.rcParams['boxplot.whiskerprops.linewidth'] = .65
    mpl.rcParams['boxplot.whiskerprops.linestyle'] = '--'
    # mpl.rcParams['boxplot.whiskerprops.color'] = 'black'          # (check)

    # box cap line
    mpl.rcParams['boxplot.capprops.linewidth'] = .75
    # mpl.rcParams['boxplot.capprops.color'] = 'black'              # (check)

    # box median line
    mpl.rcParams['boxplot.medianprops.linewidth'] = 1
    mpl.rcParams['boxplot.medianprops.linestyle'] = '-'
    # mpl.rcParams['boxplot.medianprops.color'] = 'black'           # (check)

    mpl.rcParams['boxplot.meanprops.linewidth'] = 1
    mpl.rcParams['boxplot.meanprops.linestyle'] = '-'
    # mpl.rcParams['boxplot.meanprops.color'] = 'black'             # (check)

    # box scatter
    mpl.rcParams['boxplot.flierprops.markerfacecolor'] = 'none'
    mpl.rcParams['boxplot.flierprops.markeredgewidth'] = .65
    mpl.rcParams['boxplot.flierprops.marker'] = 'o'
    # mpl.rcParams['boxplot.flierprops.markersize'] = 6             # (check)
    # mpl.rcParams['boxplot.flierprops.linewidth'] = 0              # (check)
    # mpl.rcParams['boxplot.flierprops.markeredgecolor'] = 'black'  # (check)
    # mpl.rcParams['boxplot.flierprops.color'] = 'black'            # (check)

    # -- Figure padding ------------------------------------------------------

    # Figure layout
    # auto-make plot elements fit on figure
    mpl.rcParams['figure.autolayout'] = True
    mpl.rcParams['figure.constrained_layout.use'] = True  # apply tight layout

    # Subplot padding (all dims are a fraction of the fig width and height)/
    # NOTE: not compatible with constrained_layout.
    #
    # mpl.rcParams['figure.subplot.left'] = .125    # left side
    # mpl.rcParams['figure.subplot.right'] = 0.9    # right side of subplots
    # mpl.rcParams['figure.subplot.bottom'] = 0.11  # bottom of subplots
    # mpl.rcParams['figure.subplot.top'] = 0.88     # top of subplots

    # Reserved space between subplots
    # mpl.rcParams['figure.subplot.wspace'] = 0.2   # width
    # mpl.rcParams['figure.subplot.hspace'] = 0.2   # height

    # Constrained layout padding (not compatible with autolayout)
    # mpl.rcParams['figure.constrained_layout.h_pad'] = 0.04167
    # mpl.rcParams['figure.constrained_layout.w_pad'] = 0.04167

    # Constrained layout spacing between subplots, relative to subplot sizes.
    # Is much smaller than tight_layout (figure.subplot.{hspace, wspace)
    # as constrained_layout already takes surrounding text
    # (titles, labels, # ticklabels) into account.
    # NOTE: not compatible with autolayout.
    #
    # mpl.rcParams['figure.constrained_layout.hspace'] = 0.02
    # mpl.rcParams['figure.constrained_layout.wspace'] = 0.02

    # -- Other ---------------------------------------------------------------

    # Figure size and quality
    mpl.rcParams['figure.dpi'] = 100         # NOTE: Alters figure size
    mpl.rcParams['figure.figsize'] = (5, 5)  # (6, 4), (6.4, 4.8)

    # Figure saving settings
    mpl.rcParams['savefig.transparent'] = False
    mpl.rcParams['savefig.format'] = 'svg'  # {png, ps, pdf, svg}
    mpl.rcParams['savefig.dpi'] = 330

    # Set format/quality of inline figures in Jupyter notebooks
    # %config InlineBackend.figure_format = 'svg'


def main():
    import doctest
    doctest.testmod(verbose=True)

    # from src.workflow import doctest_function
    # doctest_function(customize_plots, globs=globals())


if __name__ == "__main__":
    main()
