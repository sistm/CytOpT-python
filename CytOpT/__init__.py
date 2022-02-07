try:
    from CytOpT.CytOpt import CytOpT
    from CytOpT.descentAscent import cytoptDesasc
    from CytOpT.labelPropSto import labelPropSto, cost, hFunction, robbinsWass, cTransform
    from CytOpT.minmaxSwapping import cytoptMinmax
    from CytOpT.plots import *

except ImportError:
    import sys

    _, e, _ = sys.exc_info()
    from sys import stderr

    stderr.write('''\
    Try installing and importing CytOpT. Error is :''' % e)

__all__ = ("CytOpT",
           "cytoptDesasc",
           "cytoptMinmax",
           "cost",
           "hFunction",
           "cTransform",
           "robbinsWass",
           "labelPropSto",
           "BlandAltman",
           "barPlot",
           "KLPlot",
           "resultPlot")
