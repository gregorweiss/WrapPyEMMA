"""
Example script that calculates the features and do tICA on them.
Finally the first twelve tICs are plotted using the util.tica_plot
"""

import warnings

warnings.filterwarnings("ignore")

from ..util.io import parse
from ..util.io import init
from ..custom.features import feat_CA_dist as feat_init
from ..wrapper.kmeans import get_tica
from ..wrapper.util import tica_plot

args, files = parse()
trajs = init(args.prefix, args.suffix)
feat = feat_init(args.top)

tica, tica_output = get_tica(trajs, feat, files, lag=args.lag, var_cutoff=args.var_cutoff, ndims=args.ndims,
                             pipeline=args.pipeline, forceCalc=args.forceCalcTICA, forceModel=args.forceModelTICA,
                             forceFeat=args.forceCalcFeat)

for offset in range(0, 12, 2):
    tica_plot(tica_output, offset, lag=tica.lag, output=args.directory)
