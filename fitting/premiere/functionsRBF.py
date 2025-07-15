from scipy.interpolate import Rbf

def buildRBF ( Xm, Ym, function='linear', epsilon=.01, smooth=.01):
    rbfi = Rbf(Xm, Ym, function=function, epsilon=epsilon, smooth=smooth)  # radial basis function interpolator instance
    return rbfi