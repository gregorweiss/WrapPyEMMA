import pyemma
import numpy as np

from glob import glob

from .util import Forcer

def pipe_feat(traj_list, feat, files, force=False):
    """Initiate pipeline with featurization

    The user can request the slow, but memory saving, pipelining approach
    of PyEmma.

    Parameters
    ----------
    traj_list : list of strings
        MD trajectory file names
    feat : class pyemma.coordinate.Featurize
       User specified features.
    files : dict
        File names for data in or to storage.
        Important:
        - files['featTraj']
    force : bool
        Whether features should be recalculated.
    """
    pipe = pyemma.coordinates.pipeline([])

    try:  # Try pipelining from feature files.
        if force:
            raise Forcer()

        featFile_list = sorted(glob(files['featTraj'] + '*.npy'))
        inp = pyemma.coordinates.source(featFile_list)
        pipe.add_element(inp)
        print("Add features to pipeline.")

        return pipe

    except:  # Else pipeline the MD trajectory files.
        inp = pyemma.coordinates.source(traj_list, features=feat)
        pipe.add_element(inp)

        print("Add featurizer to pipeline.")

        return pipe


def get_feat(traj_list, feat, files, force=False):
    """Wrapper for featurization

    We choose different sources based on user decision and available
    data. The user can request a slow, but memory saving, pipelining or
    force recalcutlation. If neither is the case, we try to retrieve
    from storage. If the data can not be found on storage it is
    calculated from a stored model or newly created model and stored.

    Parameters
    ----------
    traj_list : list of strings
        MD trajectory file names
    feat : class pyemma.coordinate.Featurize
        User specified features.
    files : dict
        File names for data in or to storage.
        Important:
        - files['featTraj']
    force : bool
        Whether features should be recalculated.
    """
    ######################################################
    # Push featurized trajectories fully into memory ...
    try:
        if force:
            raise Forcer()

        featFile_list = sorted(glob(files['featTraj'] + '*.npy'))
        feat_output = [np.array(np.load((featFile)), dtype='float32') for featFile in featFile_list]
        print("Features from storage")

        return feat_output
    # ... from data ...
    except:
        feat_output = pyemma.coordinates.load(traj_list, features=feat)
        for i in range(len(feat_output)):
            np.save(files['featTraj'] + str(i) + '.npy', feat_output[i])
        print("Features from calculation")

        return feat_output
    # ... or calculate it.
    #######################################################
