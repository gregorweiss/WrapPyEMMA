import argparse
from glob import glob


def parse():
    """ Creates an object of class argparse.ArgumentParser

    We initialize often occuring parser arguments.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--prefix', type=str, default='./')
    parser.add_argument('-s', '--suffix', type=str, default='traj.xtc')
    parser.add_argument('-t', '--top', type=str, default='./')
    parser.add_argument('-d', '--directory', type=str, default='./')

    parser.add_argument('-l', '--lag', type=int, default=2)
    parser.add_argument('-k', '--kclusters', type=int, default=2)
    parser.add_argument('-ml', '--msmlag', type=int, default=2)
    parser.add_argument('-m', '--mstates', type=int, default=2)

    # TODO: global comprehension how ndims and var_cutoff control the tICA dimensionality.
    parser.add_argument('-nd', '--ndims', type=int, default=10000)
    parser.add_argument('-vc', '--var_cutoff', type=float, default=1.0)

    parser.add_argument('-smp', '--smplen', type=int, default=2)
    parser.add_argument('-off', '--offset', type=int, default=0)

    parser.add_argument('-fF', '--forceCalcFeat', default=False, action='store_true')
    parser.add_argument('-fT', '--forceCalcTICA', default=False, action='store_true')
    parser.add_argument('-fmT', '--forceModelTICA', default=False, action='store_true')
    parser.add_argument('-fC', '--forceCalcClustering', default=False, action='store_true')
    parser.add_argument('-fmC', '--forceModelClustering', default=False, action='store_true')
    parser.add_argument('-fM', '--forceCalcMSM', default=False, action='store_true')
    parser.add_argument('-pipe', '--pipeline', default=False, action='store_true')

    parser.add_argument('-Ftraj', '--FeatureTraj', type=str, default='Feature/feat_traj')
    parser.add_argument('-CV', '--CumVarFile', type=str, default='tICA/CumVar')
    parser.add_argument('-T', '--tICAFile', type=str, default='tICA/TICA')
    parser.add_argument('-TM', '--tICAModel', type=str, default='tICA/tica-obj')
    parser.add_argument('-C', '--ClusterFile', type=str, default='KMeans/dTrajs')
    parser.add_argument('-CM', '--ClusterModel', type=str, default='KMeans/cluster-obj')
    parser.add_argument('-MM', '--MarkovModel', type=str, default='MSM/msm-obj')

    args = parser.parse_args()

    files = {}
    files["featTraj"] = args.directory + args.FeatureTraj
    files["ticaFile"] = args.directory + args.tICAFile + '-' + str(args.lag) + '.npy'
    files["ticaModel"] = args.directory + args.tICAModel + '-' + str(args.lag) + '.npy'
    files["cumvarFile"] = args.directory + args.CumVarFile + '-' + str(args.lag) + '.npy'
    files["clusterFile"] = args.directory + args.ClusterFile
    files["clusterFile"] += '-' + str(args.lag) + '-' + str(args.ndims) + '-' + str(args.kclusters) + '.npy'
    files["clusterModel"] = args.directory + args.ClusterModel
    files["clusterModel"] += '-' + str(args.lag) + '-' + str(args.ndims) + '-' + str(args.kclusters) + '.npy'
    files["msmModel"] = args.directory + args.MarkovModel
    files["msmModel"] += '-' + str(args.lag) + '-' + str(args.ndims) + '-' + str(args.kclusters)
    files["msmModel"] += '-' + str(args.msmlag) + '.npy'

    return args, files


def init(prefix, suffix):
    """
    Parameters
    ----------
    prefix : string
        name of system folder
    suffix : string
        name of file type
    """
    traj_list = sorted(glob(prefix + '*' + suffix))

    return traj_list
