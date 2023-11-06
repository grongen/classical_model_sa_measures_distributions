"""This script uses all 5-percentile cases, removes the 2nd and 4th percentile, and tests the ability of Metalog
and PWU to estimate the position of the missing percentiles"""

from pathlib import Path
from tqdm import tqdm
import json
import numpy as np
from io import StringIO
import anduryl
from anduryl.io.settings import Distribution

workingdir = Path(__file__).parent

# Read settings
with open(workingdir / "settings.json", "r") as f:
    dict = json.load(f)

files = dict["files"]

distributions = [Distribution.PWL, Distribution.METALOG]

table = {method: {key: {} for key in files} for method in distributions}

scores = {}
percentiles = {}

casefolder = workingdir / '..' / "data" / 'case-studies'

# Create a EQ DM to enable plot data
# Read settings
with open(Path(__file__).parent / "settings.json", "r") as f:
    dict = json.load(f)
files = dict["files"]

diffs = {
    "25p_pwl": [],
    "25p_ml": [],
    "75p_pwl": [],
    "75p_ml": [],
}

plot = False

# For each case
for key in tqdm(files, total=len(files)):
    scores[key] = {}
    percentiles[key] = {}

    project = anduryl.Project()
    project.io.load_excalibur(casefolder / f"{key}.dtt", casefolder / f"{key}.rls")

    # Only consider 5 percentile cases
    if len(project.assessments.quantiles) == 3:
        continue

    # Save a 3 percentile version of the project
    sio = StringIO()

    # Remove the second and fourth percentile from the project
    project.assessments.remove_quantile(project.assessments.quantiles[3])
    project.assessments.remove_quantile(project.assessments.quantiles[1])

    savemodel = project.io.to_savemodel()
    sio.write(savemodel.model_dump_json())

    project_3p = anduryl.Project()
    sio.seek(0)
    project_3p.io.load_json(sio)
    sio.close()

    project_5p = anduryl.Project()
    project_5p.io.load_excalibur(casefolder / f"{key}.dtt", casefolder / f"{key}.rls")

    lower, upper = project_5p.assessments.get_bounds()

    for exp, exp_estimates in project_5p.assessments.estimates.items():
        for i, (item, est_obj) in enumerate(exp_estimates.items()):
            if project.items.scales[i] == "log":
                continue

            est_3p = project_3p.assessments.estimates[exp][item]
            if None in list(est_3p.estimates.values()):
                continue

            if np.isnan(list(est_3p.estimates.values())).any():
                continue

            q_2nd_est = project_5p.assessments.quantiles[1]
            est_2nd_p = est_obj.estimates[q_2nd_est]
            q_2nd_int_ml = est_3p._cdf_metalog(est_2nd_p)
            q_2nd_int_pwl = est_3p._cdf_pwl(est_2nd_p, lower=lower[i], upper=upper[i])
            diffs["25p_pwl"].append(q_2nd_int_pwl - q_2nd_est)
            diffs["25p_ml"].append(q_2nd_int_ml - q_2nd_est)

            q_4th_est = project_5p.assessments.quantiles[-2]
            est_4th_p = est_obj.estimates[q_4th_est]
            q_4th_int_ml = est_3p._cdf_metalog(est_4th_p)
            q_4th_int_pwl = est_3p._cdf_pwl(est_4th_p, lower=lower[i], upper=upper[i])
            diffs["75p_pwl"].append(q_4th_int_pwl - q_4th_est)
            diffs["75p_ml"].append(q_4th_int_ml - q_4th_est)


# Add to dataframe and export
with open(workingdir / '..' / "data" / "Results" / "differences_3p_5p.json", "w") as f:
    json.dump(diffs, f, indent=4)
