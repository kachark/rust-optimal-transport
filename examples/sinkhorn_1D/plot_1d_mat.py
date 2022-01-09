
# Author: Remi Flamary <remi.flamary@unice.fr>

import numpy as np
import matplotlib.pylab as pl
from matplotlib import gridspec

def plot1D_mat(a, b, M, title=''):
    r""" Plot matrix :math:`\mathbf{M}`  with the source and target 1D distribution

    Creates a subplot with the source distribution :math:`\mathbf{a}` on the left and
    target distribution :math:`\mathbf{b}` on the top. The matrix :math:`\mathbf{M}` is shown in between.


    Parameters
    ----------
    a : ndarray, shape (na,)
        Source distribution
    b : ndarray, shape (nb,)
        Target distribution
    M : ndarray, shape (na, nb)
        Matrix to plot
    """
    na, nb = M.shape

    gs = gridspec.GridSpec(3, 3)

    xa = np.arange(na)
    xb = np.arange(nb)

    ax1 = pl.subplot(gs[0, 1:])
    pl.plot(xb, b, 'r', label='Target distribution')
    pl.yticks(())
    pl.title(title)

    ax2 = pl.subplot(gs[1:, 0])
    pl.plot(a, xa, 'b', label='Source distribution')
    pl.gca().invert_xaxis()
    pl.gca().invert_yaxis()
    pl.xticks(())

    pl.subplot(gs[1:, 1:], sharex=ax1, sharey=ax2)
    pl.imshow(M, interpolation='nearest')
    pl.axis('off')

    pl.xlim((0, nb))
    pl.tight_layout()
    pl.subplots_adjust(wspace=0., hspace=0.2)

