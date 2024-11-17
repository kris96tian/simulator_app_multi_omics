import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
import os

# MultiOmics Simulator Class
class MultiOmicsSimulator:
    def __init__(self, n_samples_per_group=(100, 100), n_features=None, seed=42):
        np.random.seed(seed)
        self.n_samples_per_group = n_samples_per_group
        self.total_samples = sum(n_samples_per_group)
        self.n_features = n_features or {
            "transcriptomics": 3234,
            "proteomics": 2187,
            "metabolomics": 129,
            "methylation": 1110,
        }

    def _create_feature_names(self):
        feature_names = {}
        for omic, n_feat in self.n_features.items():
            if omic == "transcriptomics":
                feature_names[omic] = [f"ENSG{str(i).zfill(11)}" for i in range(1, n_feat + 1)]
            elif omic == "proteomics":
                feature_names[omic] = [f"PROT{str(i).zfill(6)}" for i in range(1, n_feat + 1)]
            elif omic == "metabolomics":
                feature_names[omic] = [f"HMDB{str(i).zfill(7)}" for i in range(1, n_feat + 1)]
            elif omic == "methylation":
                feature_names[omic] = [f"cg{str(i).zfill(8)}" for i in range(1, n_feat + 1)]
        return feature_names

    def simulate_data(self):
        feature_names = self._create_feature_names()
        sample_names = [f"Sample_{i}" for i in range(1, self.total_samples + 1)]
        data = {}

        for omic, n_feat in self.n_features.items():
            if omic == "transcriptomics":
                base_data = np.random.poisson(5, (n_feat, self.n_samples_per_group[0]))
                treatment_data = np.random.poisson(5.5, (n_feat, self.n_samples_per_group[1]))
                data[omic] = np.concatenate([base_data, treatment_data], axis=1)
            else:
                data[omic] = np.random.normal(size=(n_feat, self.total_samples))
                data[omic][:, self.n_samples_per_group[0]:] += 0.5

            data[omic] = pd.DataFrame(data[omic], index=feature_names[omic], columns=sample_names)

        metadata = pd.DataFrame({
            "sample_id": sample_names,
            "group": np.repeat(["Control", "Treatment"], self.n_samples_per_group),
            "batch": np.tile(["Batch1", "Batch2"], self.total_samples // 2),
        })

        return {"data": data, "metadata": metadata}

# Streamlit UI
def main():
    # Header
    st.title("Multi-Omics Data Simulator")
    st.write("""
        **Welcome!** This app allows you to simulate multi-omics data for analysis and exploration. 
        Customize the simulation by selecting omics types, sample sizes, and other parameters on the sidebar.
        You will be shown the first rows and columns of selected omic-data, as well as download links.
    """)
    st.sidebar.title("Simulation Configuration")

    # User input: sample size and features
    n_samples_per_group = st.sidebar.slider("Samples per group", 10, 500, (100, 100))
    n_transcriptomics = st.sidebar.slider("Transcriptomics Features", 100, 5000, 3234)
    n_proteomics = st.sidebar.slider("Proteomics Features", 100, 5000, 2187)
    n_metabolomics = st.sidebar.slider("Metabolomics Features", 10, 500, 129)
    n_methylation = st.sidebar.slider("Methylation Features", 100, 2000, 1110)
    seed = st.sidebar.number_input("Random Seed", min_value=0, value=42)

    # Simulate button
    simulate_btn = st.sidebar.button("Simulate Data")

    # Simulation logic
    if simulate_btn:
        st.write("Simulating data...")
        simulator = MultiOmicsSimulator(
            n_samples_per_group=n_samples_per_group,
            n_features={
                "transcriptomics": n_transcriptomics,
                "proteomics": n_proteomics,
                "metabolomics": n_metabolomics,
                "methylation": n_methylation,
            },
            seed=seed,
        )
        sim_data = simulator.simulate_data()
        
        st.write("**Metadata:**")
        st.dataframe(sim_data["metadata"].head())
        
        for omic, data in sim_data["data"].items():
            st.write(f"**{omic.capitalize()} Data:**")
            st.dataframe(data.head())

        # Download simulated data
        st.write("Download simulated data:")
        for omic, data in sim_data["data"].items():
            csv = data.to_csv().encode("utf-8")
            st.download_button(
                label=f"Download {omic.capitalize()} Data",
                data=csv,
                file_name=f"{omic}_data.csv",
                mime="text/csv",
            )

        metadata_csv = sim_data["metadata"].to_csv().encode("utf-8")
        st.download_button(
            label="Download Metadata",
            data=metadata_csv,
            file_name="metadata.csv",
            mime="text/csv",
        )
    # Footer
    st.markdown("""
        ---
        **Created by Kristian Alikaj**  
        For more, visit [My GitHub](https://github.com/kris96tian) or [My Portfolio Website](https://kris96tian.github.io/)
    """)


if __name__ == "__main__":
    main()
