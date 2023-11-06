"""This script loops over all cases and calculates the quantile of all items' realizations,
 for all expert judgments, using Metalog and PWU."""

from pathlib import Path
from tqdm import tqdm
import json
import numpy as np
import anduryl
from anduryl.io.settings import CalculationSettings, CalibrationMethod, Distribution

workingdir = Path(__file__).parent / '..'

np.seterr(under="print")

# Read settings
with open(workingdir / "scripts" / "settings.json", "r") as f:
    settings_dict = json.load(f)

files = settings_dict["files"]

# itemopt_settings = CalculationSettings(**dict["settings"]["ITopt"])
# itemnonopt_settings = CalculationSettings(**dict["settings"]["IT"])
globopt_settings = CalculationSettings(**settings_dict["settings"]["GLopt"])
globnonopt_settings = CalculationSettings(**settings_dict["settings"]["GL"])
# equal_settings = CalculationSettings(**dict["settings"]["EQ"])
# user_settings = CalculationSettings(**dict["settings"]["US"])

distributions = [Distribution.PWL, Distribution.METALOG]
sa_methods = [
    CalibrationMethod.Chi2,
    CalibrationMethod.CRPS,
    CalibrationMethod.KS,
    CalibrationMethod.CVM,
    CalibrationMethod.AD,
]
table = {method: {key: {} for key in files} for method in distributions}

scores = {}
weights = {}
percentiles = {}

casefolder = workingdir / 'data' / 'case-studies'

# For each case
for key in tqdm(files, total=len(files)):

    scores[key] = {dist.value: {} for dist in distributions}
    weights[key] = {dist.value: {} for dist in distributions}
    percentiles[key] = {dist.value: {} for dist in distributions}

    project = anduryl.Project()
    project.io.load_excalibur(casefolder / f"{key}.dtt", casefolder / f"{key}.rls")
    actual_idx = project.experts.get_idx("actual")

    # For both the Metalog as Piece-wise linear assumption
    for sa_method in sa_methods:

        for distribution in distributions:

            for settings in [globnonopt_settings, globopt_settings]:

                settings.distribution = distribution
                settings.calibration_method = sa_method

                # Calculate the different DMs
                project.calculate_decision_maker(settings)
            
            # Add statistical accuracies to dict
            sas = project.experts.calibration[:]
            scores[key][distribution.value][sa_method.value] = dict(zip(project.experts.ids, sas.tolist(), strict=True))
            
            # Add weights to dict
            cbs = project.experts.comb_score[:]
            weights[key][distribution.value][sa_method.value] = dict(zip(project.experts.ids, cbs.tolist(), strict=True))

            # Add the score to the overview
            assert len(project.experts.calibration) == (len(actual_idx) + 2)
            
            
            # Add percentiles to dict
            percentiles[key][distribution.value][sa_method.value] = {
                k: np.array(v).tolist()
                for k, v in project.experts._get_realization_percentiles(
                    settings, project.experts.ids
                ).items()
            }

            for settings in [globnonopt_settings, globopt_settings]:
                project.experts.remove_expert(settings.id)

# Add to dataframe and export
with open(workingdir / "data" / "results" / "percentiles_all.json", "w") as f:
    json.dump(percentiles, f, indent=4)

with open(workingdir / "data"/ "results" / "sa_scores_all.json", "w") as f:
    json.dump(scores, f, indent=4)

with open(workingdir / "data"/ "results" / "comb_scores_all.json", "w") as f:
    json.dump(weights, f, indent=4)
