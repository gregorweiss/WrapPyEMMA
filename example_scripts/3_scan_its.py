"""
Example script that calculates the implied timescales for various MSM lagtimes.
"""
import warnings

warnings.filterwarnings("ignore")

from ..util.io import parse
from ..util.io import init
from ..custom.features import feat_CA_dist as feat_init
from ..wrapper.kmeans import get_kmeans
from ..wrapper.util import its_scan

args, files = parse()
trajs = init(args.prefix, args.suffix)
feat = feat_init(args.top)

dtraj, dtraj_output = get_kmeans(trajs, feat, files,
                                 kclusters=args.kclusters, lag=args.lag, var_cutoff=args.var_cutoff, ndims=args.ndims,
                                 pipeline=args.pipeline, forceCalc=args.forceCalcClustering, forceModel=args.forceModelClustering,
                                 forceCalcTICA=args.forceCalcTICA, forceModelTICA=args.forceModelTICA,
                                 forceFeat=args.forceCalcFeat)

lags = [25, 50, 100, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500]
its_scan(dtraj_output, lags, lag=args.lag, ndims=args.ndims, k=args.kclusters, path=args.directory)
