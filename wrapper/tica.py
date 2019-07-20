import pyemma
import numpy as np

from .util import Forcer
from .feat import pipe_feat, get_feat


def pipe_tica(traj_list, feat, files, lag=2, var_cutoff=0.95, forceFeat=False):
    """Pipelining time-lagged independent component analyses

    The user can request the slow, but memory saving, pipelining approach
    of PyEmma

    Parameters
    ----------
    traj_list : list of strings
        MD trajectory file names
    feat : class pyemma.coordinate.MDFeaturize
       User specified features.
    files : dict
        File names for data in or to storage.
    lag : int (default 2)
        Lagtime for tICA analysis.
    var_cutoff : float (default 0.95)
        Defines the cutoff based on cumulative variance of the tICA.
    forceFeat : bool
        Whether features should be recalculated.
    """
    pipe = pipe_feat(traj_list, feat, files, forceFeat)

    tica_obj = pyemma.coordinates.tica(lag=lag, var_cutoff=var_cutoff, kinetic_map=True)
    pipe.add_element(tica_obj)
    print("Add tICA object to pipeline.")

    return pipe


def get_tica(traj_list, feat, files, lag=2, var_cutoff=0.95, ndims=1e6,
             pipeline=False, forceCalc=False, forceModel=False, forceFeat=False):
    """Wrapper for time-lagged independent component analyses

    We choose different sources based on user decision and available
    data. The user can request a slow, but memory saving, pipelining or
    force recalculation. If neither is the case, we try to retrieve
    data from storage. If the data can not be found on storage, it is
    calculated from a stored model or newly created model.

    Parameters
    ----------
    traj_list : list of strings
        MD trajectory file names
    feat : class pyemma.coordinate.MDFeaturize
       User specified features.
    files : dict
        File names for data in or to storage.
        Important:
        - files['cumvarFile']
        - files['ticaFile']
        - files['ticaModel']
    lag : int (default 2)
        Lagtime for tICA analysis.
    var_cutoff : float (default 0.95)
        Defines the cutoff based on cumulative variance of the tICA.
    ndims: int (1e6 just something large so it doesn't interfere with var_cutoff)
        Number of tICA dimensions to use.
    pipeline : bool (False)
        Whether data analysis should be piped an thus save memory but slow.
    forceCalc : bool (False)
        Whether tICA should be (re-)calculated.
    forceModel : bool (False)
        forceCalc : bool (False)
        Whether tICA should be (re-)modeled.
    forceFeat : bool
        Whether features should be recalculated.
    """
    ########################################################
    # Create TICA object if memory is too small for data
    ########################################################
    if pipeline:
        pipe = pipe_feat(traj_list, feat, files, forceFeat)

        tica_obj = pyemma.coordinates.tica(lag=lag, var_cutoff=var_cutoff, kinetic_map=True)
        pipe.add_element(tica_obj)
        print("Add tICA object to pipeline.")

        pipe.parametrize()
        tica_output = tica_obj.get_output()

        return tica_obj, tica_output

    #####################
    # Get TICA ...
    try:
        try:
            if forceFeat or forceCalc or forceModel:
                raise Forcer()

            tica_obj = pyemma.load(files['ticaModel'])
            data = np.load((files['ticaFile']), mmap_mode='r')
            cumvar = np.load(files['cumvarFile'])
            varcutoff_dims = [n for n, i in enumerate(cumvar) if i >= var_cutoff][0]
            cut_dims = min(ndims, varcutoff_dims)
            tica_output = [np.array(traj[:, :cut_dims], dtype='float32') for traj in data]
            print("tICs from storage")

            return tica_obj, tica_output
    # ... from data ...
        except:
            if forceFeat or forceCalc:
                raise Forcer()

            tica_obj = pyemma.load(files['ticaModel'])
            inp = get_feat(traj_list, feat, files, forceFeat)
            tica_output = tica_obj.transform(inp)
            print("tICs from model")

            return tica_obj, tica_output
    # ... from model ...

    except:
        inp = get_feat(traj_list, feat, files, forceFeat)

        tica_obj = pyemma.coordinates.tica(inp, lag=lag, var_cutoff=var_cutoff, kinetic_map=True)
        tica_output = tica_obj.get_output()
        print("tICs from calculation")

        tica_obj.save(files['ticaModel'], overwrite=True)
        np.save(files['ticaFile'], tica_output)
        np.save(files['cumvarFile'], tica_obj.cumvar)

        return tica_obj, tica_output
    # ... or calculate it.
    #######################