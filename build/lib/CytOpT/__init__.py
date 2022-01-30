
try:
    from CytOpT.CytOpt import CytOpT
    from CytOpT.CytOpt_Descent_Ascent import cytopt_desasc
    from CytOpT.CytOpt_MinMax_Swapping import cytopt_minmax
    from CytOpT.Label_Prop_sto import Robbins_Wass, c_transform, h_function, cost
    from CytOpT.CytOpt_plot import *

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
           "Robbins_Wass",
           "Label_Prop_sto",
           "CytOpT",
           "Bland_Altman",
           "bar_plot",
           "KL_plot")
