"""Simulated four hyptohetical experts with different biases, and test their statistical accuracy
Set the "samples" parameter to determine how many samples to draw. Currently set to 10,000, which takes a while.
"""

from math import gamma, e
import numpy as np

from pathlib import Path
from tqdm import tqdm
import json
from scipy.stats import cramervonmises, kstest
from tqdm.auto import tqdm


from anduryl.io.settings import CalibrationMethod
from anduryl.core import crps
from anduryl.core import anderson_darling


def upper_incomplete_gamma(a, x, iterations):
    """
    Implementation for upper incomplete gamma.
    """
    val = 1.0
    for d in reversed(range(1, iterations)):
        val = d * 2 - 1 - a + x + (d * (a - d)) / val
    return ((x**a) * (e ** (-x))) / val


def chi2cdf(x, df, iterations=100):
    """
    Chi squared cdf function. This function can also be used from scipy,
    but to reduce the compilation size a seperate (slighty slower)
    implementation without scipy is written.
    """
    if x == 0.0:
        return 1.0
    else:
        return 1 - upper_incomplete_gamma(0.5 * df, 0.5 * x, iterations) / gamma(0.5 * df)


def ranks(values):
    return (np.argsort(np.argsort(values)) + 0.5) / len(values)


def sa_cooke(vals, quantiles):
    s = dict(zip(*np.unique(np.digitize(vals, quantiles), return_counts=True)))
    s = np.array([s[i] / len(vals) if i in s else 0 for i in range(len(quantiles) + 1)])
    p = np.diff(np.concatenate([[0], quantiles, [1]]))
    idx = s > 0.0
    MI = np.sum(s[idx] * np.log(s[idx] / p[idx]))
    c = 1 - chi2cdf(x=2 * len(vals) * MI, df=len(s) - 1)
    return c


if __name__ == "__main__":

    np.seterr(under="print")

    sa_function = {
        CalibrationMethod.CRPS: lambda x: crps.crps_sa(x)[0],
        CalibrationMethod.Chi2: lambda x: sa_cooke(x, quantiles),
        CalibrationMethod.CVM: lambda x: cramervonmises(rvs=x, cdf=lambda x: x).pvalue,
        CalibrationMethod.KS: lambda x: kstest(rvs=x, cdf=lambda x: x).pvalue,
        CalibrationMethod.AD: lambda x: anderson_darling.ad_sa(x),
    }

    sa_methods = [
        CalibrationMethod.CRPS,
        CalibrationMethod.Chi2,
        CalibrationMethod.CVM,
        CalibrationMethod.KS,
        CalibrationMethod.AD,
    ]

    experts = {
        "Perfectly calibrated": (1.0, 1.0),
        "Overconfident": (0.35, 0.35),
        "Underconfident": (2, 2),
        "Biased": (1, 2),
    }

    N = 50
    # SET THE SAMPLES HERE TO CONTROL SIMULATION TIME
    samples = 10000
    quantiles = np.array([0.05, 0.5, 0.95])
    # quantiles = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
    npoints = list(range(3, N + 1))

    allvals = []
    calis = {}

    for sa_method in sa_methods:
        calis[sa_method.value] = {}
        for name, (a, b) in experts.items():
            calis[sa_method.value][name] = []

    for name, (a, b) in experts.items():
        
        for j in tqdm(range(samples), desc=name):

            cdfvals = np.random.beta(a=a, b=b, size=N)
            # allvals.append(cdfvals)

            for sa_method in sa_methods:

                sa = [sa_function[sa_method](cdfvals[:i]) for i in npoints]
                calis[sa_method.value][name].append(sa)


    with open(Path(__file__).parent / ".." / "data" / "results" / "sampled_sa_scores.json", "w") as f:
        json.dump(calis, f, indent=4)
