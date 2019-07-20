import pyemma
import mdtraj


def feat_CA_dist(topfile):
    """ Featurize distances between CA of amino acids

    Parameters
    ----------
    topfile : string
        name of topology (.gro) file
    """
    feat = pyemma.coordinates.featurizer(topfile)
    traj = mdtraj.load(topfile)

    selstring = "name CA"
    indices = traj.top.select(selstring)

    feat.add__distances(indices=indices, periodic=True)

    return feat
