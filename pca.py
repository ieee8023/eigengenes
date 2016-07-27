import load_data
import matplotlib.font_manager as font_manager
import numpy.linalg as linalg
import pylab
import sys
from sklearn.decomposition import PCA

def main(args):
    """ Performs principal component analysis (PCA) on colombos data. """

    if len(args) != 2:
        sys.exit("Usage: python pca.py <organism> <tolerence>")
    organism = args[0]
    tolerence = float(args[1])
    X = load_data.load(organism, False)[0].transpose()
    m, n = X.shape
    K, Y1, Y2 = [], [], []
    print("Running PCA...")
    for k in range(5, n + 1, 5):
        pca = PCA(n_components = k, whiten = True)
        pca.fit(X)
        var_explained = sum(pca.explained_variance_ratio_)
        projections = pca.transform(X)
        reconstruction = pca.inverse_transform(projections)
        residual = X - reconstruction
        error = linalg.norm(residual)
        K.append(k)
        Y1.append(var_explained)
        Y2.append(error)
        print("  n_components = %d, var_explained = %f, error = %f" \
              %(k, round(var_explained, 3), round(error, 3)))
        if var_explained > 1 - tolerence:
            break

    font_prop = font_manager.FontProperties(size = 12)
    pylab.figure(1, figsize = (12, 9), dpi = 500)

    pylab.subplot(2, 1, 1)
    pylab.grid(True)
    pylab.xlabel(r"# of principal components", fontproperties = font_prop)
    pylab.ylabel(r"% variance explained", fontproperties = font_prop)
    pylab.plot(K, Y1, "k-", linewidth = 2, alpha = 0.6)
    pylab.xlim(min(K), max(K))
    pylab.ylim(0, 1)

    pylab.subplot(2, 1, 2)
    pylab.grid(True)
    pylab.xlabel(r"# of principal components", fontproperties = font_prop)
    pylab.ylabel(r"reconstruction error", fontproperties = font_prop)
    pylab.plot(K, Y2, "k-", linewidth = 2, alpha = 0.6)
    pylab.xlim(min(K), max(K))

    pylab.savefig("%s.pdf" %(organism), format = "pdf", bbox_inches = "tight")
    pylab.close(1)

if __name__ == "__main__":
    main(sys.argv[1:])
