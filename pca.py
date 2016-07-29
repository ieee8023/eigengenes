import load_data
import matplotlib.font_manager
import numpy
import pylab
import sys

def main(args):
    """ Performs principal component analysis (PCA) on colombos data. """

    if len(args) != 2:
        sys.exit("Usage: python pca.py <organism> <tolerence>")
    organism = args[0]
    tolerence = float(args[1])
    X = load_data.load(organism, False)[0].transpose()
    m, n = X.shape
    Xmean = numpy.mean(X, axis = 0)
    Xstd = numpy.std(X, axis = 0)
    X = numpy.nan_to_num((X - Xmean) / Xstd)
    K, L, J = [], [], []
    print("Running PCA...")
    U, S, V = numpy.linalg.svd(X)
    explained_variance = (S ** 2) / m
    explained_variance_ratio = explained_variance / sum(explained_variance)
    for k, v in enumerate(S):
        print("  First %d components of %d..." %(k + 1, len(S)))
        l = v ** 2
        projection = numpy.dot(X, V[:k + 1].T)
        reconstruction = numpy.dot(projection, V[:k + 1])
        residual = X - reconstruction
        error = sum(numpy.linalg.norm(residual, axis = 0) ** 2) / (2 * m)
        K.append(k + 1)
        L.append(l)
        J.append(error)
        if 1 - sum(explained_variance_ratio[:k + 1]) < tolerence:
            break
    font_prop = matplotlib.font_manager.FontProperties(size = 12)
    pylab.figure(1, figsize = (12, 9), dpi = 500)

    pylab.subplot(2, 1, 1)
    pylab.grid(True)
    pylab.xlabel(r"# of components $k$", fontproperties = font_prop)
    pylab.ylabel(r"eigenvalue $\lambda$", fontproperties = font_prop)
    pylab.plot(K, L, "b-", linewidth = 2, alpha = 0.6)
    pylab.plot(K, L, "r.", linewidth = 2, alpha = 0.6)
    pylab.xlim(min(K), max(K))

    pylab.subplot(2, 1, 2)
    pylab.grid(True)
    pylab.xlabel(r"# of components $k$", fontproperties = font_prop)
    pylab.ylabel(r"reconstruction error $J$", fontproperties = font_prop)
    pylab.plot(K, J, "b-", linewidth = 2, alpha = 0.6)
    pylab.plot(K, J, "r.", linewidth = 2, alpha = 0.6)
    pylab.xlim(min(K), max(K))

    pylab.savefig("%s.pdf" %(organism), format = "pdf", bbox_inches = "tight")
    pylab.close(1)

if __name__ == "__main__":
    main(sys.argv[1:])
