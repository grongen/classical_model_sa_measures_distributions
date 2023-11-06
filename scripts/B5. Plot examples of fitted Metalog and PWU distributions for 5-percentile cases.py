"""This script generates a lot of figures of fitted Metalog and PWU distribuitions,
solely meant to see how these generally look and compare, and find some illustrative examples."""

from pathlib import Path
from tqdm import tqdm
import json
import numpy as np
from io import StringIO
import anduryl
from anduryl.io.settings import CalculationSettings, Distribution

import matplotlib.pyplot as plt


workingdir = Path(__file__).parent

# Read settings
with open(workingdir / "settings.json", "r") as f:
    dict = json.load(f)

files = dict["files"]

distributions = [Distribution.PWL, Distribution.METALOG]

table = {method: {key: {} for key in files} for method in distributions}

scores = {}
percentiles = {}

basedir = workingdir / '..'
casefolder = basedir / "data" / "case-studies"

# Create a EQ DM to enable plot data
# Read settings

with open(basedir / "scripts" / "settings.json", "r") as f:
    dict = json.load(f)
files = dict["files"]

equal_settings_pwl = CalculationSettings(**dict["settings"]["EQ"])
equal_settings_pwl.id = 'PWL'

equal_settings_ml = equal_settings_pwl.model_copy()
equal_settings_ml.distribution = 'Metalog'
equal_settings_ml.id = 'ML'

diffs = {
    '25p_pwl': [],
    '25p_ml': [],
    '75p_pwl': [],
    '75p_ml': [],
}

plot = False

# For each case
for key in tqdm(files, total=len(files)):

    if key not in ["Arkansas", 'bfiq', 'CoveringKids']:
        continue

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
    project_3p.add_results_from_settings(equal_settings_pwl)
    project_3p.add_results_from_settings(equal_settings_ml)
    sio.close()

    project_5p = anduryl.Project()
    project_5p.io.load_excalibur(casefolder / f"{key}.dtt", casefolder / f"{key}.rls")
    project_5p.add_results_from_settings(equal_settings_pwl)
    project_5p.add_results_from_settings(equal_settings_ml)

    lower, upper = project_5p.assessments.get_bounds()

    # Compare the 2nd and 4th percentile based on the 3p project to the estimates percentiles
    plotdata_3p_pwl = project_3p.results['PWL'].get_plot_data()
    plotdata_3p_ml = project_3p.results['ML'].get_plot_data()
    plotdata_5p_pwl = project_5p.results['PWL'].get_plot_data()
    plotdata_5p_ml = project_5p.results['ML'].get_plot_data()
        

    for exp, exp_estimates in project_5p.assessments.estimates.items():
        for i, (item, est_obj) in enumerate(exp_estimates.items()):

            if project.items.scales[i] == 'log':
                continue

            est_3p = project_3p.assessments.estimates[exp][item]
            
            if np.isnan(list(est_3p.estimates.values())).any():
                continue

            fig, ax = plt.subplots(figsize=(8, 5), ncols=1, constrained_layout=True)
            # ax = axs[0]
            ax.plot(plotdata_3p_pwl[(item, exp)].pdf_x, plotdata_3p_pwl[(item, exp)].pdf_y, color='orange', ls='--')
            ax.plot(plotdata_5p_pwl[(item, exp)].pdf_x, plotdata_5p_pwl[(item, exp)].pdf_y, color='.5', ls='--')
            rng = max(est_obj.estimates.values()) - min(est_obj.estimates.values())
            ax.set_xlim(min(est_obj.estimates.values()) - 0.1 * rng, max(est_obj.estimates.values()) + 0.1 * rng)
            
            ax.plot(plotdata_3p_ml[(item, exp)].pdf_x, plotdata_3p_ml[(item, exp)].pdf_y, color='dodgerblue')
            ax.plot(plotdata_5p_ml[(item, exp)].pdf_x, plotdata_5p_ml[(item, exp)].pdf_y, color='.5')
            # axs[0].set_xlim(axs[1].get_xlim())
            
            q_2nd_est = project_5p.assessments.quantiles[1]
            est_2nd_p = est_obj.estimates[q_2nd_est]
            q_2nd_int_ml = est_3p._cdf_metalog(est_2nd_p)
            q_2nd_int_pwl = est_3p._cdf_pwl(est_2nd_p, lower=lower[i], upper=upper[i])
            diffs['25p_pwl'].append(q_2nd_int_pwl - q_2nd_est)
            diffs['25p_ml'].append(q_2nd_int_ml - q_2nd_est)
            
            # print(diffs['25p_pwl'][-1])

            q_4th_est = project_5p.assessments.quantiles[-2]
            est_4th_p = est_obj.estimates[q_4th_est]
            q_4th_int_ml = est_3p._cdf_metalog(est_4th_p)
            q_4th_int_pwl = est_3p._cdf_pwl(est_4th_p, lower=lower[i], upper=upper[i])
            diffs['75p_pwl'].append(q_4th_int_pwl - q_4th_est)
            diffs['75p_ml'].append(q_4th_int_ml - q_4th_est)
            
            ax.set_title(f"PWL: {diffs['25p_pwl'][-1]:.3f} / {diffs['75p_pwl'][-1]:.3f} | ML: {diffs['25p_ml'][-1]:.3f} / {diffs['75p_ml'][-1]:.3f}")
            
            ax.axvline(est_2nd_p, color='.5', ls=':')
            ax.axvline(est_4th_p, color='.5', ls=':')
            
            ax.axvline(est_3p._ppf_pwl(q_2nd_est, lower=lower[i], upper=upper[i]), ls='-.', color='orange')
            ax.axvline(est_3p._ppf_pwl(q_4th_est, lower=lower[i], upper=upper[i]), ls='-.', color='orange')

            ax.axvline(est_3p._ppf_metalog(q_2nd_est), ls='-.', color='dodgerblue')
            ax.axvline(est_3p._ppf_metalog(q_4th_est), ls='-.', color='dodgerblue')


            fig.savefig(basedir / "data" / "figures" / f"3p_5p_comparison/{key}_{i}_{exp}.pdf")
            plt.close(fig)
