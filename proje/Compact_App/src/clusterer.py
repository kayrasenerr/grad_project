import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from scipy.signal import find_peaks, detrend

from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import KMeans


class Clusterer:
    def __init__(self, n_sub_clusters: int) -> None:
        self.n_clusters = None
        self.n_sub_clusters = n_sub_clusters
        self.n_step1_cluster = 0
        self.linkage = None

    def find_n_clusters(self, bhv_imp_dict: dict):
        n_clusters = 0
        for value in bhv_imp_dict.values():
            if value > 0:
                n_clusters += 1
        self.n_clusters = n_clusters

    def run_clustering(self, trans_input_df: pd.DataFrame, input_df: pd.DataFrame, consider_val: bool, osc_detail_importance: bool):
        result_df = trans_input_df.copy()
        result_df["Cluster"] = None

        feature_col_arr = trans_input_df.columns.values
        result_df = self._clustering_step_1(
            result_df=result_df, feature_col_arr=feature_col_arr
        )

        result_df = self._clustering_step_2(result_df=result_df)

        meta_col_arr = ["Label", "Index"]
        if "Label" not in input_df.columns:
            meta_col_arr.remove("Label")
        sub_cluster_df = input_df[meta_col_arr].copy()
        sub_cluster_df["Cluster"] = result_df["Cluster"]

        if consider_val:
            temp_inp_df = input_df.copy()
            temp_inp_df["Cluster"] = result_df["Cluster"]
            sub_cluster_df = self._value_clustering_1(input_df=temp_inp_df, osc_detail_importance=osc_detail_importance)

        if osc_detail_importance:
            sub_cluster_df = self._value_clustering_2(input_df=temp_inp_df, sub_cluster_df=sub_cluster_df)

        return result_df, sub_cluster_df

    def _clustering_step_1(self, result_df: pd.DataFrame, feature_col_arr: np.array):
        temp_result_df = result_df.copy()

        if "Linear" in feature_col_arr:
            lnr_cluster_idx = temp_result_df[temp_result_df["Linear"] >= 0.999].index
            temp_result_df.loc[lnr_cluster_idx, "Cluster"] = "Linear"
            self.n_step1_cluster += 1

        if "Oscillation" in feature_col_arr:
            next_cluster_mask = temp_result_df["Cluster"].isna()
            osc_mask = (temp_result_df["Oscillation"] >= 2) & next_cluster_mask
            osc_cluster_idx = temp_result_df[osc_mask].index
            temp_result_df.loc[osc_cluster_idx, "Cluster"] = "Cyclic"
            self.n_step1_cluster += 1

        if "Growth_Decline" in feature_col_arr:
            next_cluster_mask = temp_result_df["Cluster"].isna()
            gad_mask = (temp_result_df["Growth_Decline"] == 1) & next_cluster_mask
            gad_cluster_idx = temp_result_df[gad_mask].index
            temp_result_df.loc[gad_cluster_idx, "Cluster"] = "Growth&Decline"
            self.n_step1_cluster += 1

        return temp_result_df

    def _clustering_step_2(self, result_df: pd.DataFrame):
        temp_result_df = result_df.copy()
        next_cluster_df = result_df.copy()
        next_cluster_mask = next_cluster_df["Cluster"].isna()
        next_cluster_df = next_cluster_df[next_cluster_mask]
        drop_col_arr = next_cluster_df.columns.isin(["Linear", "Oscillation", "Growth_Decline", "Cluster"])
        drop_col_arr = next_cluster_df.columns[drop_col_arr]
        next_cluster_df = next_cluster_df.drop(
            drop_col_arr, axis=1
        )

        next_cluster_df.loc[:, :] = (
            next_cluster_df - next_cluster_df.mean(axis=0)
        ) / next_cluster_df.std(axis=0)
        self.linkage = linkage(next_cluster_df, method="median")
        cluster_labels = fcluster(
            self.linkage, self.n_clusters - self.n_step1_cluster, criterion="maxclust"
        )
        temp_result_df.loc[next_cluster_df.index, "Cluster"] = np.astype(cluster_labels, str)

        return temp_result_df

    def _value_clustering_1(self, input_df: pd.DataFrame, osc_detail_importance: bool):
        temp_result_df = input_df.copy()
        meta_col_arr = ["Label", "Index", "Cluster", "Sub-cluster"]
        drop_col_mask = temp_result_df.columns.isin(meta_col_arr)
        value_agg_df = temp_result_df.loc[:, ~drop_col_mask]

        input_min_ser = value_agg_df.min(axis=1).values
        input_max_ser = value_agg_df.max(axis=1).values
        #input_mean_ser = value_agg_df.mean(axis=1).values
        input_value_df = pd.DataFrame(data={"min": input_min_ser, "max": input_max_ser}, index=temp_result_df.index)

        temp_result_df["Sub-cluster"] = temp_result_df["Cluster"].astype(str) + "_"
        for cluster in temp_result_df["Cluster"].unique():
            if (cluster == "Cyclic") and (osc_detail_importance):
                continue
            cluster_mask = temp_result_df["Cluster"] == cluster
            cluster_value_df = input_value_df[cluster_mask]
            cluster_value_df = (cluster_value_df - cluster_value_df.mean(axis=0)) / cluster_value_df.std(axis=0)

            cluster_kmeans = KMeans(n_clusters=self.n_sub_clusters, random_state=0)
            cluster_kmeans.fit(cluster_value_df)
            sub_cluster_arr = cluster_kmeans.labels_
            temp_result_df.loc[cluster_mask, "Sub-cluster"] = temp_result_df.loc[cluster_mask, "Sub-cluster"] + np.astype(sub_cluster_arr, str)

        if "Label" not in temp_result_df.columns:
            meta_col_arr.remove("Label")
        return temp_result_df[meta_col_arr]

    def _value_clustering_2(self, input_df: pd.DataFrame, sub_cluster_df: pd.DataFrame):
        osc_mask = sub_cluster_df["Cluster"] == "Cyclic"
        temp_cluster_df = sub_cluster_df.copy()
        if "Sub-cluster" not in sub_cluster_df.columns:
            temp_cluster_df["Sub-cluster"] = temp_cluster_df["Cluster"].astype(str) + "_"
        osc_cluster_df = sub_cluster_df.loc[osc_mask]
        drop_col_mask = input_df.columns.isin(["Label", "Index", "Cluster"])
        osc_value_df = input_df.loc[osc_mask, ~drop_col_mask]
        freq_col_arr = osc_value_df.columns.values
        linear_fit_arr = sm.add_constant(freq_col_arr.astype(float)[:, None])

        for idx in osc_value_df.index:
            trend_status = "No-Trend"
            iter_val_arr = osc_value_df.loc[idx, freq_col_arr].values

            model = sm.OLS(iter_val_arr, linear_fit_arr).fit()
            p_value = model.pvalues[1]
            if p_value < 1e-10:
                slope = model.params[1]
                trend_status = "Up-Trend" if slope > 0 else "Down-Trend"
            osc_cluster_df.loc[idx, "Sub-cluster"] = osc_cluster_df.loc[idx, "Sub-cluster"] + trend_status + "_"

        for idx in osc_value_df.index:
            stability_status = "Stable"
            iter_val_arr = osc_value_df.loc[idx, freq_col_arr].values
            iter_val_arr = detrend(iter_val_arr)

            peaks, _   = find_peaks(iter_val_arr, prominence=0.005)
            valleys, _ = find_peaks(-iter_val_arr, prominence=0.005)
            if len(peaks) < 2 or len(valleys) < 2:
                stability_status = "Indeterminate"
            else:
                x_peaks = peaks.reshape(-1, 1)
                y_peaks = iter_val_arr[peaks]
                model_p = LinearRegression().fit(x_peaks, y_peaks)
                slope_p = model_p.coef_[0]

                x_valleys = valleys.reshape(-1, 1)
                y_valleys = iter_val_arr[valleys]
                model_v = LinearRegression().fit(x_valleys, y_valleys)
                slope_v = model_v.coef_[0]

                if slope_p > 0.001 and slope_v < -0.001:
                    stability_status = "Diverging"
                elif slope_p < -0.001 and slope_v > 0.001:
                    stability_status = "Dampening"
            osc_cluster_df.loc[idx, "Sub-cluster"] = osc_cluster_df.loc[idx, "Sub-cluster"] + stability_status

        temp_cluster_df.loc[osc_mask, "Sub-cluster"] = osc_cluster_df["Sub-cluster"]
        return temp_cluster_df
