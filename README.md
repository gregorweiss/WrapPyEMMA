# WrapPyEMMA
This is a simple wrapper for the python library PyEMMA to enable flexible analytics scripts that are capable of running 
on HPC and workstations.

One can easily load, (re)calculate, or pipeline the discretization steps for Markov State modeling 
of Molecular Dynamics data. The wrapper is suitable for large data sets with high dimensionality, 
thus, memory intensive projects.

The user can pass bool arguments to `wrapper.feat.get_feat`, `wrapper.tica.get_tica`, and `wrapper.kmeans.get_kmeans` to 
decide whether the intermediate results should be (re)calculated or retrieved from existing file by loading the 
data into memory or pipelining it from storage. Additionally, one can use a stored model/object, for instance from the
tICA or clustering steps, to transform new data without (re)calculation using the full data set.

The file `custom/features.py` holds a simple featurization function, that can be customized to system specific needs.
PyEMMA specific plotting and parameter selection utilities are provided in `wrapper/util.py`. For quick usage, examples 
for some wrappers and utilities are provided in `example_scripts`.

For more details on installation and usage of PyEMMA see: http://emma-project.org