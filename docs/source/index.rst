.. CytOpT documentation master file, created by
   sphinx-quickstart on Sat Jan 15 17:34:33 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

CytOpT documentation
====================
The CytOpT is new package includes a new algorithm using regularized optimal transport to directly estimate the different cell population proportions from a biological sample characterized with flow cytometry measurements. Algorithm is based on the regularized Wasserstein metric to compare cytometry measurements from different samples, thus accounting for possible mis-alignment of a given cell population across sample (due to technical variability from the technology of measurements).

Approach
          Supervised learning technique based on the Wasserstein metric that
          is used to estimate an optimal re-weighting of class proportions in
          a mixture model from a source distribution (with known segmentation
          into cell sub-populations) to fit a target distribution with unknown segmentation.

Description
               A new algorithm, referred to as CytOpT, using regularized
               optimal transport to directly estimate the different cell
               population proportions from a biological sample characterized
               with flow cytometry measurements. It is based on the regularized
               Wasserstein metric to compare cytometry measurements from
               different samples, thus accounting for possible mis-alignment
               of a given cell population across sample (due to technical variability from the technology of measurements).

Overview
               The methods implemented in this package are detailed in the following article:
               Paul Freulon, Jérémie Bigot, Boris P. Hejblum. CytOpT: Optimal Transport with Domain Adaptation for Interpreting Flow Cytometry data https://arxiv.org/abs/2006.09003


The project homepage is https://github.com/sistm/CytOpt-python.

See ``README`` for the installation instructions.


.. include:: readme_link.rst
.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
