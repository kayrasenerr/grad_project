import os

import pandas as pd
import streamlit as st

from src.feature_extractor import FeatureExtractor
from src.clusterer import Clusterer
from src.reporter import Reporter
from src.utils import file_selector, importance_checkbox, image_question, input_check


def main():
    st.set_page_config(page_title="Behavior Clustering Server", layout="centered")

    st.title("Behavior Clustering")
    st.header("Input Data Parameters")

    input_data_dir = file_selector(
        folder_path=".",
        select_label=r"$\textsf{\Large Select the Input Directory}$",
        not_contains=".",
    )
    if input_data_dir is None:
        st.warning("Select the problem directory to continue!")
        st.stop()

    input_data_path = file_selector(
        folder_path=input_data_dir,
        select_label=r"$\textsf{\Large Select the Input File}$",
        contains=".csv",
    )

    st.header("Behavior Importance Parameters")
    lnr_importance = importance_checkbox(label=r"$\textsf{\Large Evaluate Linearity}$")
    lnr_importance = int(lnr_importance)
    osc_importance = importance_checkbox(label=r"$\textsf{\Large Evaluate Cyclicity}$")
    osc_importance = int(osc_importance)
    if osc_importance:
        osc_detail_importance = importance_checkbox(
            label=r"$\textsf{\Large Detailed Cyclicity Analysis}$"
        )
    else:
        osc_detail_importance = False

    question_img_dir = "Interface_Images"
    st.subheader("Divergence - Convergence")
    conv_div_img = os.path.join(question_img_dir, "diverging_converging.png")
    is_cd_together = image_question(
        caption="Divergence/Convergence",
        question=r"$\textsf{\Large Cluster together?}$",
        img_path=conv_div_img,
    )
    if not is_cd_together:
        cd_importance = 1
    else:
        cd_importance = 0

    st.subheader("Monotonicity")
    monotonicity_img = os.path.join(question_img_dir, "monotonicity.png")
    is_mntn_together = image_question(
        caption="Monotonicity",
        question=r"$\textsf{\Large Cluster together?}$",
        img_path=monotonicity_img,
    )
    if not is_mntn_together:
        gad_importance = 1
    else:
        gad_importance = 0

    st.subheader("Time Value")
    value_img = os.path.join(question_img_dir, "value_level.png")
    is_value_imp = image_question(
        caption="Time_value",
        question=r"$\textsf{\Large Cluster together?}$",
        img_path=value_img,
        init_val=True,
    )
    is_value_imp = ~is_value_imp
    n_sub_clusters = st.number_input(
        label="Enter the number of sub-clusters: ",
        value=1,
        min_value=1,
        max_value=10,
        step=1,
        disabled=~is_value_imp,
    )
    if not is_value_imp:
        n_sub_clusters = 1

    bhv_imp_dict = {
        "lnr": lnr_importance,
        "osc": osc_importance,
        "gad": gad_importance,
        "log": cd_importance,
        "exp": cd_importance,
        "nexp": cd_importance,
    }

    st.header("Output Parameters")
    output_dir = st.text_input(r"$\textsf{\Large Enter the output directory name}$")
    if output_dir == "":
        st.warning("Enter an output directory name to continue!")
        st.stop()

    run_clustering = st.button(label="Start Clustering!", type="primary")
    if run_clustering:
        input_df = pd.read_csv(input_data_path)
        input_check(input_df=input_df)

        feature_extractor = FeatureExtractor()
        trans_input_df = feature_extractor.extract_features(
            input_df=input_df, bhv_imp_dict=bhv_imp_dict
        )

        clusterer = Clusterer(n_sub_clusters=n_sub_clusters)
        clusterer.find_n_clusters(bhv_imp_dict)
        cluster_result_df, subcluster_result_df = clusterer.run_clustering(
            trans_input_df=trans_input_df,
            input_df=input_df,
            consider_val=is_value_imp,
            osc_detail_importance=osc_detail_importance,
        )

        reporting_df = input_df.copy()
        reporting_df["Cluster"] = cluster_result_df["Cluster"]
        if subcluster_result_df is not None:
            reporting_df["Sub-cluster"] = subcluster_result_df["Sub-cluster"]

        reporter = Reporter()
        reporter.report_results(reporting_df=reporting_df, output_dir=output_dir)


if __name__ == "__main__":
    main()
