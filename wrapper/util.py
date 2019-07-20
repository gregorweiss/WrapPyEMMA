import pyemma
import numpy as np

import matplotlib.pyplot as plt


class Forcer(Exception):
    pass


def tica_plot(tica_output, offset, lag=2, output='./'):
    """Plot 2D projection of free energy in tIC space

    Parameters
    ----------
    tica_output :
        tICA data
    offset : int
        Offset to plot tIC(offset) and tIC(offset+1)
    lag : int
        tICA lagtime
    output :  string
        Output path for figure
    """
    reduced_data = np.concatenate(tica_output)[:, offset:(offset + 2)]

    labels = ['tIC {}'.format(offset + i + 1) for i in range(2)]
    plt.rc('font', size=18)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))

    pyemma.plots.plot_feature_histograms(reduced_data, labels, ax=axes[0])
    pyemma.plots.plot_density(*reduced_data.T, ax=axes[1], cbar=False, logscale=True)
    pyemma.plots.plot_free_energy(*reduced_data.T, ax=axes[2], legacy=False)

    for ax in axes.flat[1:]:
        ax.set_xlabel('tIC {}'.format(offset + 1))
        ax.set_ylabel('tIC {}'.format(offset + 2))

    fig.tight_layout()
    fig.savefig(output + '1_tica-' + str(lag) + '-' + str(offset) + '.png')


def score_cv(data, dim, lag, number_of_splits=10, validation_fraction=0.5):
    """Compute a cross-validated VAMP2 score.

    We randomly split the list of independent trajectories into
    a training and a validation set, compute the VAMP2 score,
    and repeat this process several times.

    Parameters
    ----------
    data : list of numpy.ndarrays
        The input tica data.
    dim : int
        Number of processes to score; equivalent to the dimension
        after projecting the data with VAMP2.
    lag : int
        Lag time for the VAMP2 scoring.
    number_of_splits : int, optional, default=10
        How often do we repeat the splitting and score calculation.
    validation_fraction : int, optional, default=0.5
        Fraction of trajectories which should go into the validation
        set during a split.
    """
    # we temporarily suppress very short-lived progress bars
    # with pyemma.util.contexts.settings(show_progress_bars=False):
    nval = int(len(data) * validation_fraction)
    scores = np.zeros(number_of_splits)
    for n in range(number_of_splits):
        ival = np.random.choice(len(data), size=nval, replace=False)
        inputdata = [d for i, d in enumerate(data) if i not in ival]
        vamp = pyemma.coordinates.vamp(inputdata, lag=lag, dim=dim)
        scores[n] = vamp.score([d for i, d in enumerate(data) if i in ival])
    return scores


def score_kmeans(data, n_cc, smplen=2, lag=2, msmlag=2, path='./'):
    """VAMP cross-validation for number of clusters in kmeans

    Parameters
    ----------
    data : list of numpy.ndarrays
        TICA data
    n_cc : list of int
        different cluster numbers for kmeans to test for
    smplen : int
        number of samples
    lag : int
        tICA lagtime
    msmlag : int
        MSM lagtime (decent choice should be greater or equal to tICA lagtime)
    path : string
        output path
    """
    ndims = np.shape(data)[2]
    scores = np.zeros((len(n_cc), smplen))
    for n, k in enumerate(n_cc):
        for m in range(smplen):
            _cl = pyemma.coordinates.cluster_kmeans(data, k=k, max_iter=500, stride=50)
            _msm = pyemma.msm.estimate_markov_model(_cl.dtrajs, msmlag)
            scores[n, m] = _msm.score_cv(_cl.dtrajs, n=1, score_method='VAMP2', score_k=min(10, k))

    plt.rc('font', size=14)
    fig, ax = plt.subplots()
    lower, upper = pyemma.util.statistics.confidence_interval(scores.T.tolist(), conf=0.9)
    ax.fill_between(n_cc, lower, upper, alpha=0.3)
    ax.plot(n_cc, np.mean(scores, axis=1), '-o')
    ax.semilogx()
    ax.set_xlabel('number of cluster centers')
    ax.set_ylabel('VAMP-2 score')
    fig.tight_layout()
    fig.savefig(path + '2_kmeans_score-' + str(lag)
                + '-' + str(ndims) + '-' + str(msmlag) + '.png')
    np.save(path + '2_kmeans_score-' + str(lag)
            + '-' + str(ndims) + '-' + str(msmlag) + '.npy', scores.T.tolist())

    return scores


def its_scan(dtraj_output, lags, nits=20, hmsm=False, mstates=2, lag=2, ndims=2, k=2, path='./'):
    """Scans implied timescales through various MSM lagtimes

    Parameters
    ----------
    dtraj_output : list of numpy.ndarrays
        discrete trajectories
    lags : list of int
        lagtimes
    nits : int (20)
        number of implied time scale that should be plottted
    hmsm : bool (False)
        Whether to use hidden MSM
    mstates : int (2)
        Number of timescales in HMSM, thus, only required if hmsm==False
    lag : int (2)
        tICA lagtime
    ndims : int (2)
        number of tIC dimensions
    k : int (2)
        number of clusters
    path : string ('./')
        output path
    """
    if hmsm:
        its = pyemma.msm.timescales_hmsm(dtraj_output, mstates, lags=lags, nits=nits, errors='bayes')
    else:
        its = pyemma.msm.timescales_msm(dtraj_output, lags=lags, nits=nits, errors='bayes')

    plt.rc('font', size=14)
    its_figure = pyemma.plots.plot_implied_timescales(its, units='ns', dt=0.02)
    fig = its_figure.get_figure()

    # TODO: change from hard coded strings in file names to default strings
    if hmsm:
        fig.savefig(path + "hmsm_itsplot-" + str(lag) + "-" + str(ndims) + "-" + str(k) + ".png")
        np.save(path + "hmsm_mean-" + str(lag) + "-" + str(ndims) + "-" + str(k) + ".npy", its.get_sample_mean())
        np.save(path + "hmsm_std-" + str(lag) + "-" + str(ndims) + "-" + str(k) + ".npy", its.get_sample_std())
        np.save(path + "hmsm_timescales-" + str(lag) + "-" + str(ndims) + "-" + str(k) + ".npy", its.get_timescales())
    else:
        fig.savefig(path + "msm_itsplot-" + str(lag) + "-" + str(ndims) + "-" + str(k) + ".png")
        np.save(path + "msm_mean-" + str(lag) + "-" + str(ndims) + "-" + str(k) + ".npy", its.get_sample_mean())
        np.save(path + "msm_std-" + str(lag) + "-" + str(ndims) + "-" + str(k) + ".npy", its.get_sample_std())
        np.save(path + "msm_timescales-" + str(lag) + "-" + str(ndims) + "-" + str(k) + ".npy", its.get_timescales())
