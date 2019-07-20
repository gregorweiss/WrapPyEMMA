import pyemma

from .util import Forcer
from .kmeans import get_kmeans


def get_bmsm(traj_list, feat, files, msmlag=2, kclusters=2, lag=2, var_cutoff=0.95, ndims=1e6,
            forceBMSM=False, pipeline=False,  forceCalcKmeans=False, forceModelKmeans=False,
            forceCalcTICA=False, forceModelTICA=False, forceFeat=False):
    """Wrapper for Markov state modeling

    We choose different sources based on user decision and available
    data. The user can request a slow, but memory saving, pipelining or
    force recalculation. If neither is the case, we try to retrieve
    model from storage.

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
    msmlag : int (2)
        Lagtime for MSM
    kclusters : int (2)
        Number of clusters
    lag : int (2)
        Lagtime for tICA analysis.
    var_cutoff : float (0.95)
        Defines the cutoff based on cumulative variance of the tICA.
    ndims: int (1e6 just something large so it doesn't interfere with var_cutoff)
        Number of tICA dimensions to use.
    forceBMSM : bool (False)
        Whether Bayesian MSM should be recalculated
    pipeline : bool (False)
        Whether data analysis should be piped an thus save memory but slow.
    forceCalcKmeans : bool (False)
        Whether clusters should be recalculated.
    forceModelKmeans : bool (False)
        forceCalc : bool (False)
        Whether clusters should be remodeled.
    forceCalcTICA : bool (False)
        Whether tICA should be recalculated
    forceModelTICA : bool (False)
        Whether tICA should be remodeled
    forceFeat : bool
        Whether features should be recalculated.
    """
    try:
        if forceFeat or forceCalcTICA or forceModelTICA or forceCalcKmeans or forceModelKmeans or forceBMSM:
            raise Forcer()

        msm_obj = pyemma.load(files["msmModel"])
        print("MSM from storage")

        return msm_obj

    except:
        dtraj, dtraj_output = get_kmeans(traj_list, feat, files,
                                         kclusters=kclusters, lag=lag, var_cutoff=var_cutoff, ndims=ndims,
                                         pipeline=pipeline, forceCalc=forceCalcKmeans, forceModel=forceModelKmeans,
                                         forceCalcTICA=forceCalcTICA, forceModelTICA=forceModelTICA, forceFeat=forceFeat)

        msm_obj = pyemma.msm.bayesian_markov_model(dtraj_output, lag=msmlag, dt_traj='0.02 ns')
        msm_obj.save(files["msmModel"])

        return msm_obj