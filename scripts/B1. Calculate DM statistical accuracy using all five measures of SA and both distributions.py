"""This script generates the decision maker statistical accuracy and combination score for the
comparison between all five measures of statistical accuracy."""


from pathlib import Path
from tqdm import tqdm
import json
import numpy as np
import pandas as pd
import itertools
import anduryl
from anduryl.io.settings import CalculationSettings, CalibrationMethod, Distribution
from anduryl.core import metalog

metalog._JOIN_SIDES = False

np.seterr(under="print")

maindir = Path(__file__).parent / '..'

# Read settings
with open(maindir / 'scripts' / "settings.json", "r") as f:
    dict = json.load(f)

files = dict["files"]

itemopt_settings = CalculationSettings(**dict["settings"]["ITopt"])
itemnonopt_settings = CalculationSettings(**dict["settings"]["IT"])
globopt_settings = CalculationSettings(**dict["settings"]["GLopt"])
globnonopt_settings = CalculationSettings(**dict["settings"]["GL"])
equal_settings = CalculationSettings(**dict["settings"]["EQ"])
user_settings = CalculationSettings(**dict["settings"]["US"])

distributions = [Distribution.METALOG, Distribution.PWL]
sa_methods = [
    CalibrationMethod.Chi2,
    CalibrationMethod.CRPS,
    CalibrationMethod.KS,
    CalibrationMethod.CVM,
    CalibrationMethod.AD,
]
table = {method: {file: {} for file in files} for method in distributions}
sas = {}
infos = {}

weights = {}


setting_options = [globopt_settings, globnonopt_settings, equal_settings]

dm_score_distribution = {}
dm_score_sa_method = {}
dm_score_sa_info = {}


# For each case
for key in tqdm(files, total=len(files)):


    project = anduryl.Project()
    project.io.load_excalibur(
        maindir / "data"/"case-studies" / f"{key}.dtt", maindir / "data" /"case-studies" / f"{key}.rls"
    )
    
    # For Erie Carps, the expert with the highest weight did not answer all questions
    # This causes errors, so remove this expert.
    if 'erie' in key.lower():
        project.experts.remove_expert('8')

    actual_idx = project.experts.get_idx("actual")

    # Remove target items (only seeds are relivant for this exercise)
    for i in np.where(project.items.get_idx("target"))[0][::-1]:
        project.items.remove_item(project.items.ids[i])

    # For both the Metalog as Piece-wise linear assumption
    for settings in setting_options:

        # Calculate initial DM weights and DM scores, without interchanging SA methods or distributions
        for distribution, sa_method in itertools.product(distributions, sa_methods):
            settings.distribution = distribution
            settings.calibration_method = sa_method

            # Calculate the different DMs
            project.calculate_decision_maker(settings)
            comb = (key, distribution.value, settings.name, sa_method.value, sa_method.value)
            comb2 = (key, sa_method.value, settings.name, distribution.value, distribution.value)

            weights[comb] = project.experts.comb_score[actual_idx]
            if settings.optimisation:
                weights[comb] = np.where(
                    project.experts.calibration[actual_idx] >= project.main_results.alpha_opt,
                    project.experts.comb_score[actual_idx],
                    0.0,
                )

            weights[comb] /= weights[comb].sum()

            if np.isnan(weights[comb]).any():
                raise ValueError()

            # Add the score to the overview
            assert len(project.experts.comb_score) == (len(actual_idx) + 1)
            dm_score_sa_method[comb] = project.experts.calibration[-1]
            dm_score_sa_info[comb] = project.experts.comb_score[-1]
            dm_score_distribution[comb2] = project.experts.calibration[-1]

            project.experts.remove_expert(settings.id)

        # Get DM for weights from other methods
        for distribution, sa_method in itertools.product(distributions, sa_methods):
            settings.distribution = distribution
            settings.calibration_method = sa_method

            # Apply the DM weights to the other statistical accuracy methods and get the DM performance
            other_methods = sa_methods[:]
            other_methods.remove(sa_method)

            # Apply the DM weights to the other statistical accuracy methods and get the DM performance
            for other_method in other_methods:
                # Assign other method to settings
                user_settings.calibration_method = other_method
                user_settings.distribution = distribution
                # Assign the other method's weights as user weights
                comb = (key, distribution.value, settings.name, sa_method.value, sa_method.value)
                # print('post', comb)

                project.experts.user_weights = weights[comb]
                # Calculate DM with other SA method's weights as user weights
                project.calculate_decision_maker(user_settings)

                # Get the DM score
                comb = (key, distribution.value, settings.name, sa_method.value, other_method.value)
                assert len(project.experts.comb_score) == (len(actual_idx) + 1)
                if np.isnan(project.experts.calibration[-1]):
                    raise ValueError()
                dm_score_sa_method[comb] = project.experts.calibration[-1]
                dm_score_sa_info[comb] = project.experts.comb_score[-1]

                # Remove the expert
                project.experts.remove_expert(user_settings.id)

        # Get DM for weights from other distribution
        for distribution, sa_method in itertools.product(distributions, sa_methods):
            settings.distribution = distribution
            settings.calibration_method = sa_method

            # Apply the DM weights to the other distribution and get the DM performance
            other_dist = distributions[:]
            other_dist.remove(distribution)
            assert len(other_dist) == 1
            other_dist = other_dist[0]

            # Assign other method to settings
            user_settings.calibration_method = sa_method
            user_settings.distribution = other_dist
            # Assign the other method's weights as user weights
            comb = (key, distribution.value, settings.name, sa_method.value, sa_method.value)
            project.experts.user_weights = weights[comb]
            # Calculate DM with other SA method's weights as user weights
            project.calculate_decision_maker(user_settings)

            # Get the DM score
            comb = (key, sa_method.value, settings.name, distribution.value, other_dist.value)
            assert len(project.experts.comb_score) == (len(actual_idx) + 1)
            dm_score_distribution[comb] = project.experts.calibration[-1]

            # Remove the expert
            project.experts.remove_expert(user_settings.id)

# Add to dataframe and export
df = pd.DataFrame.from_dict(dm_score_distribution, orient="index")[0]
df.index = pd.MultiIndex.from_tuples(df.index)
df.index.names = ["Study", "SA_method", "DM", "Distribution_weight", "Distribution_score"]
df.unstack([2, 4]).to_excel(maindir / "data" / "Results" / "DM_distribution_results.xlsx")

# Add to dataframe and export
df = pd.DataFrame.from_dict(dm_score_sa_method, orient="index")[0]
df.index = pd.MultiIndex.from_tuples(df.index)
df.index.names = ["Study", "Distribution", "DM", "SA_weight", "SA_score"]
df.unstack([2, 4]).to_excel(maindir / "data" / "Results" / "DM_results_SA_only.xlsx")

# Add to dataframe and export
df = pd.DataFrame.from_dict(dm_score_sa_info, orient="index")[0]
df.index = pd.MultiIndex.from_tuples(df.index)
df.index.names = ["Study", "Distribution", "DM", "SA_weight", "SA_score"]
df.unstack([2, 4]).to_excel(maindir / "data" / "Results" / "DM_results_SA_info.xlsx")
