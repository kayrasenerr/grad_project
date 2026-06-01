from copy import deepcopy
import numpy as np
import pandas as pd

import statsmodels.api as sm
from scipy.signal import find_peaks, detrend


class FeatureExtractor:
    """
    Extract behavior based features to map dynamic simulation outputs
    to a low dimensional space which is highly informative about
    behavior characteristics.
    """

    def __init__(self) -> None:
        self.eps = 0.0001
        self.feature_col_arr = [
            "Linear",
            "Oscillation",
            "Growth_Decline",
            "Logarithmic_Growth",
            "Positive_Exponential_Growth",
            "Negative_Exponential_Growth",
        ]

    def extract_features(self, input_df: pd.DataFrame, bhv_imp_dict: dict):
        """
        Interface function for feature mining.
        - Does not extract features for behaviors with importance = 0
        """
        feature_col_arr = self._drop_uninterested_cols(bhv_imp_dict=bhv_imp_dict)
        trans_input_df = pd.DataFrame(columns=feature_col_arr)

        freq_col_arr = input_df.columns
        freq_col_arr = freq_col_arr[~freq_col_arr.isin(["Label", "Index"])].values
        for i, input_i in input_df.iterrows():
            input_i_trans_arr = []
            input_i_arr = input_i[freq_col_arr].values
            freq_feature_arr = freq_col_arr.astype(float)

            if bhv_imp_dict["lnr"] > 0:
                lnr_score_i = self._linearity_feature(
                    input_arr=input_i_arr, freq_col_arr=freq_feature_arr
                )
                input_i_trans_arr.append(lnr_score_i)

            if bhv_imp_dict["osc"] > 0:
                #osc_score_i = self._oscillation_feature(input_arr=input_i_arr)
                osc_score_i = self._new_osc_feature(input_arr=input_i_arr)
                input_i_trans_arr.append(osc_score_i)

            if bhv_imp_dict["gad"] > 0:
                gad_score_i = self._growth_decline_feature(input_arr=input_i_arr)
                input_i_trans_arr.append(gad_score_i)

            if (bhv_imp_dict["log"] > 0):
                log_score_i = self._logarithmic_feature(
                    input_arr=input_i_arr, freq_col_arr=freq_feature_arr
                )
                input_i_trans_arr.append(log_score_i)

            if bhv_imp_dict["exp"] > 0:
                exp_score_i = self._positive_exponential_feature(
                    input_arr=input_i_arr, freq_col_arr=freq_feature_arr
                )
                input_i_trans_arr.append(exp_score_i)

            if bhv_imp_dict["nexp"] > 0:
                nexp_score_i = self._negative_exponential_feature(
                    input_arr=input_i_arr, freq_col_arr=freq_feature_arr
                )
                input_i_trans_arr.append(nexp_score_i)

            trans_input_df.loc[i] = input_i_trans_arr
        return trans_input_df

    def _linearity_feature(self, input_arr: np.array, freq_col_arr: np.array):
        lr_feature_arr = (freq_col_arr)[:, None]
        lr_feature_arr = sm.add_constant(lr_feature_arr).astype(float)

        lr_model = sm.OLS(input_arr.astype(float), lr_feature_arr)
        lr_result = lr_model.fit()
        return lr_result.rsquared_adj

    def _oscillation_feature(self, input_arr: np.array):
        input_len = len(input_arr)
        mean_off_input = input_arr - np.mean(input_arr)
        input_acf = np.correlate(mean_off_input, mean_off_input, mode="full")[
            input_len - 1 :
        ]
        input_acf = input_acf / (input_acf[0] + self.eps)

        acf_peak_idx_arr, _ = find_peaks(input_acf[1:])
        acf_peak_max = 0
        if len(acf_peak_idx_arr) > 0:
            acf_peak_max = input_acf[acf_peak_idx_arr + 1].max()

        acf_valley_idx_arr, _ = find_peaks(-input_acf[1:])
        acf_valley_max = 0
        if len(acf_valley_idx_arr) > 0:
            acf_valley_max = np.abs(input_acf[acf_valley_idx_arr + 1]).max()

        return max(acf_peak_max, acf_valley_max)

    def _new_osc_feature(self, input_arr: np.array):
        temp_input_arr = detrend(input_arr)
        peaks, _ = find_peaks(temp_input_arr, prominence=0.01)
        valleys, _ = find_peaks(-temp_input_arr, prominence=0.01)
        
        return max(len(peaks), len(valleys))

    def _growth_decline_feature(self, input_arr: np.array):
        input_peak_idx_arr, _ = find_peaks(input_arr)
        if len(input_peak_idx_arr) == 1:
            peak_value = input_arr[input_peak_idx_arr[0]]
            final_value = input_arr[-1]
            if peak_value * 0.999 > final_value:
                score = 1
            else:
                score = -1
        # Default output if there is not exactly one peak
        else:
            score = -1

        return score

    def _logarithmic_feature(self, input_arr: np.array, freq_col_arr: np.array):
        log_feature_arr = np.log((freq_col_arr + 1))[:, None]
        log_feature_arr = sm.add_constant(log_feature_arr).astype(float)

        log_lr = sm.OLS(input_arr.astype(float), log_feature_arr)
        log_results = log_lr.fit()
        if (log_results.params[1] <= 0) or (log_results.pvalues[1] >= 0.001):
            return 0
        else:
            return log_results.rsquared_adj

    def _positive_exponential_feature(
        self, input_arr: np.array, freq_col_arr: np.array
    ):
        exp_feature_arr = (freq_col_arr)[:, None]
        exp_feature_arr = sm.add_constant(exp_feature_arr).astype(float)
        exp_target_arr = input_arr - input_arr.min()
        exp_target_arr = np.log((exp_target_arr + self.eps).astype(float))

        exp_lr = sm.OLS(exp_target_arr, exp_feature_arr)
        exp_results = exp_lr.fit()
        if (exp_results.params[1] <= 0) or (exp_results.pvalues[1] >= 0.001):
            return 0
        else:
            return exp_results.rsquared_adj

    def _negative_exponential_feature(
        self, input_arr: np.array, freq_col_arr: np.array
    ):
        nexp_feature_arr = np.exp(-freq_col_arr.astype(float))[:, None]
        nexp_feature_arr = sm.add_constant(nexp_feature_arr).astype(float)
        nexp_target_arr = input_arr.max() - input_arr

        nexp_lr = sm.OLS(nexp_target_arr.astype(float), nexp_feature_arr)
        nexp_results = nexp_lr.fit()
        if (nexp_results.params[1] <= 0) or (nexp_results.pvalues[1] >= 0.001):
            return 0
        else:
            return nexp_results.rsquared_adj

    def _drop_uninterested_cols(self, bhv_imp_dict: dict):
        new_col_arr = deepcopy(self.feature_col_arr)
        if bhv_imp_dict["lnr"] == 0:
            new_col_arr.remove("Linear")

        if bhv_imp_dict["osc"] == 0:
            new_col_arr.remove("Oscillation")

        if bhv_imp_dict["gad"] == 0:
            new_col_arr.remove("Growth_Decline")

        if (bhv_imp_dict["log"] == 0):
            new_col_arr.remove("Logarithmic_Growth")

        if (bhv_imp_dict["exp"] == 0):
            new_col_arr.remove("Positive_Exponential_Growth")

        if (bhv_imp_dict["nexp"] == 0):
            new_col_arr.remove("Negative_Exponential_Growth")

        return new_col_arr
