import pyemma
import numpy as np

from .util import Forcer
from .tica import pipe_tica, get_tica


def get_kmeans(traj_list, feat, files, kclusters=2, lag=2, var_cutoff=0.95, ndims=1e6,
               pipeline=False, forceCalc=False, forceModel=False,
               forceCalcTICA=False, forceModelTICA=False, forceFeat=False):
    """Wrapper for KMeans clustering

    We choose different sources based on user decision and available
    data. The user can request a slow, but memory saving, pipelining or
    force recalcutlation. If neither is the case, we try to retrieve
    data from storage. If the data can not be found on storage, it is
    calculated from a stored model or newly created model.

    Parameters
    ----------
    traj_list : list of strings
        MD trajectory file names
    feat : class pyemma.coordinate.Featurize
       User specified features.
    files : dict
        File names for data in or to storage.
        Important:
        - files['clusterFile']
        - files['clusterModel']
    kclusters : int (2)
        Number of clusters
    lag : int (2)
        Lagtime for tICA analysis.
    var_cutoff : float (0.95)
        Defines the cutoff based on cumulative variance of the tICA.
    ndims: int (1e6 just something large so it doesn't interfere with var_cutoff)
        Number of tICA dimensions to use.
    pipeline : bool (False)
        Whether data analysis should be piped an thus save memory but slow.
    forceCalc : bool (False)
        Whether clusters should be recalculated.
    forceModel : bool (False)
        forceCalc : bool (False)
        Whether clusters should be remodeled.
    forceCalcTICA : bool (False)
        Whether tICA should be recalculated
    forceModelTICA : bool (False)
        Whether tICA should be remodeled
    forceFeat : bool
        Whether features should be recalculated.
    """
    ###############################################################
    # Create clustering object if memory is too small for data
    ###############################################################
    if pipeline:
        pipe = pipe_tica(traj_list, feat, files, lag=lag, var_cutoff=var_cutoff, forceFeat=False)
        cluster_obj = pyemma.coordinates.cluster_kmeans(k=kclusters,
                                                        max_iter=500, stride=50)
        pipe.add_element(cluster_obj)
        pipe.parametrize()
        print("Clusters from pipeline.")

        return cluster_obj, cluster_obj.dtrajs

    #################################
    # Get discrete trajectories ...
    try:
        try:
            if forceFeat or forceCalcTICA or forceModelTICA or forceCalc or forceModel:
                raise Forcer()

            cluster_obj = pyemma.load(files['clusterModel'])
            data = np.load((files['clusterFile']))
            dtraj_output = [np.array(traj, dtype='int32') for traj in data]
            print("Clusters from storage")

            return cluster_obj, dtraj_output
        # ... from model ...
        except:
            if forceFeat or forceCalcTICA or forceModelTICA or forceCalc:
                raise Forcer()

            cluster_obj = pyemma.load(files['clusterModel'])
            tica_obj, tica_output = get_tica(traj_list, feat, files, lag=lag, var_cutoff=var_cutoff, ndims=ndims,
                                             forceModel=forceModelTICA, forceCalc=forceCalcTICA, forceFeat=forceFeat)
            dtraj_output = cluster_obj.transform(tica_output)
            print("Clusters from model")

            return cluster_obj, dtraj_output
    # ... from calculation ...
    except:
        tica_obj, tica_output = get_tica(traj_list, feat, files, lag=lag, var_cutoff=var_cutoff, ndims=ndims,
                                         forceModel=forceModelTICA, forceCalc=forceCalcTICA, forceFeat=forceFeat)

        cluster_obj = pyemma.coordinates.cluster_kmeans(tica_output, k=kclusters,
                                                        max_iter=500, stride=50)
        dtraj_output = cluster_obj.dtrajs

        cluster_obj.save(files['clusterModel'], overwrite=True)
        np.save(files['clusterFile'], dtraj_output)
        print("Clusters from calculation")

        return cluster_obj, dtraj_output
    # ... or calculate it.
    ##########################
