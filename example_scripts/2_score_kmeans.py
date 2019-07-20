"""
Example script that calculates the VAMP2-score for different number of clusters in KMeans.
"""

import warnings

warnings.filterwarnings("ignore")

from ..util.io import parse
from ..util.io import init
from ..custom.features import feat_CA_dist as feat_init
from ..wrapper.kmeans import get_tica
from ..wrapper.util import score_kmeans

args, files = parse()
trajs = init(args.prefix, args.suffix)
feat = feat_init(args.top)

tica, tica_output = get_tica(trajs, feat, files, lag=args.lag, var_cutoff=args.var_cutoff, ndims=args.ndims,
                             pipeline=args.pipeline, forceCalc=args.forceCalcTICA, forceModel=args.forceModelTICA,
                             forceFeat=args.forceCalcFeat)

n_clustercenters = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
score_kmeans(tica_output, n_clustercenters,
             smplen=args.smplen, lag=args.lag, msmlag=args.msmlag, path=args.directory)
