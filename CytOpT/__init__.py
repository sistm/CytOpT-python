
try:
    from CytOpT.CytOpT import CytOpT
    from CytOpT.descentAscent import cytopt_desasc
    from CytOpT.minMaxSwapping import cytopt_minmax
    from CytOpT.labelPropSto import robbinsWass, c_transform, h_function, cost
    from CytOpT.plots import *

except ImportError:
    import sys

    _, e, _ = sys.exc_info()
    from sys import stderr

    stderr.write('''\
    Try installing and importing CytOpT. Error is :''' % e)

__all__ = ("cytopt_desasc",
           "cytopt_minmax",
           "cost",
           "h_function",
           "c_transform",
           "robbinsWass",
           "labelPropSto",
           "CytOpT",
           "Bland_Altman",
           "bar_plot",
           "KL_plot",
           "result_plot")
