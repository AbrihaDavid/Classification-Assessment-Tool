#!/usr/bin/env python
# coding: utf-8

# In[29]:


import streamlit as st
import pandas as pd
import numpy as np
from seed_generator import *
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import zipfile
import time
from io import BytesIO
from itertools import combinations
warnings.simplefilter(action='ignore', category=FutureWarning)


# In[78]:


def extractAccuracyRep(input_file, fraction=0.6, iteration=10, all_data=False, seed=42):
    df = pd.read_csv(input_file, na_values="NA")
    df = df.sort_values(df.columns[0])
    nrow = len(df.iloc[:,0].unique())
    ncol = 9
    data_list = []
    seeds =generate_seeds(seed,iteration)
    reference_categories = df.iloc[:, 0].unique()

    for it in range(iteration):
        metrics_all = pd.DataFrame()
        random_seed = seeds[it]
        random_indices = np.random.default_rng(seeds[it]).choice(len(df), size=int(fraction * len(df)), replace=False)
        df_fractioned = df.iloc[random_indices].sort_values(df.columns[0])
        for q in range(1, len(df.columns)):
            matrix_table = pd.crosstab(df_fractioned.iloc[:, 0], pd.Categorical(df_fractioned.iloc[:, q], categories=reference_categories),
                                       dropna=False)
            matrix_table = matrix_table.reindex(index=reference_categories, columns=reference_categories, fill_value=0)
            
            metrics = np.full((nrow, ncol), nrow*ncol)
            metrics = pd.DataFrame(metrics, columns=['Model', 'Class', 'Precision', 'Sensitivity', 'Specificity', 'Accuracy', 'MCC', 'F1', 'IOU'])
            for c in range(nrow):
                metrics.iloc[c, 0] = df.columns[q]
                metrics.iloc[c, 1] = "Class_" + str(df.iloc[:,0].unique()[c])
            for e in range(nrow):
                TP = matrix_table.iloc[e, e]
                TN = 0
                FP = 0
                FN = 0
                for p in range(nrow):
                    if e != p:
                        FP += matrix_table.iloc[p, e]
                        FN += matrix_table.iloc[e, p]
                    for k in range(nrow):
                        if e != p and e != k:
                            TN += matrix_table.iloc[p, k]

                if TP+FP == 0:
                    metrics.iloc[e,2] = 0
                else:           
                    metrics.iloc[e, 2] = TP / (TP + FP)

                if TP+FN == 0:
                    metrics.iloc[e,3] = 0
                else:      
                    metrics.iloc[e, 3] = TP / (TP + FN)
                metrics.iloc[e, 4] = TN / (TN + FP)
                metrics.iloc[e, 5] = (TP + TN) / (TP + TN + FP + FN)

                if (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) == 0:
                    metrics.iloc[e,6] = 0
                else:
                    metrics.iloc[e, 6] = (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
                    
                metrics.iloc[e, 7] = (2 * TP / (2 * TP + FP + FN))
                metrics.iloc[e, 8] = (TP / (TP + FP + FN))
            metrics_all = pd.concat([metrics_all, metrics])
            metrics_all.iloc[:, 2:] = metrics_all.iloc[:, 2:].apply(pd.to_numeric)
        data_list.append(metrics_all)
    combined_data = pd.concat(data_list)
    combined_sd = combined_data.groupby(['Model', 'Class']).std().reset_index()
    combined_sd.insert(0, "stat", "sd")
    combined_mean = combined_data.groupby(['Model', 'Class']).mean().reset_index()
    combined_mean.insert(0, "stat", "mean")

    
    result = pd.concat([combined_mean, combined_sd])
    result["iteration"] = iteration
    result["fraction"] = fraction
    result = result.to_csv(index=False).encode("utf-8")
    combined_result = combined_data.to_csv(index=False).encode("utf-8")
    
    zip_data = []
    zip_data.append(("data.csv",combined_result))


    if all_data:
        
        combined_data = pd.melt(combined_data, id_vars=["Model", "Class"], var_name="Metrics")
        unique_metrics = combined_data['Metrics'].unique()

        fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(20, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(unique_metrics):
            subset_df = combined_data[combined_data['Metrics'] == metric]
        
            ax = axes[i]
            sns.boxplot(x='Class', y='value', hue='Model', data=subset_df, ax=ax, showfliers=False, palette="Set3")
            
            ax.set_title(metric)
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_ylim(0.00, 1.00)
            ax.set_yticks([0.00, 0.25, 0.50, 0.75, 1.00])
        
            ax.axhline(y=0.9, linestyle='--', color='red', linewidth=1)
            ax.get_legend().remove()

        fig.delaxes(axes[-1])
        plt.legend(loc = "lower right", bbox_to_anchor = (1.2,0.5))
        plt.tight_layout(rect=[0, 0.05, 1, 1])  # Adjust the rectangle to make room for the common legend
        plt.figtext(0.006, 0.5, 'Accuracy', va='center', ha='center', rotation=90, fontsize=16)
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=300)
        buffer.seek(0)
        plt.close()
        zip_data.append(("accuracy.png",buffer.getvalue()))
         
        zip_buffer = BytesIO()
            
        with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
            for file_name, file_content in zip_data:
                print(file_name, file_content)
                zip_file.writestr(file_name,file_content)
        
        zip_buffer.seek(0)  
    
        return zip_buffer
    else:
        return result


# In[ ]:


def main():
    st.title("Extract Accuracy with reps")
    
    input_file_tab2 = st.file_uploader("Upload Input File", type=["csv"],key = 2)
    fraction_tab2 = st.text_input("Fraction (default: 0.6)", value="0.6", key = "tab2_frac")
    iteration_tab2 = st.text_input("Iteration (default: 10)", value="10", key = "tab2_iter")
    all_data_tab2 = st.checkbox("All Data", value=False)
    seed_tab2 = st.text_input("Seed (default: 42)", value="42", key = "tab2_seed")
    default_acc = ["Precision", "Sensitivity"]
    acc = st.multiselect("Select at least 2 accuracy metrics for plotting", ["Precision","Sensitivity","Specificity","Accuracy","MCC","F1","IOU"],
                        default = default_acc)
    
    if st.button("Run extractAccuracyRep"):
        if input_file_tab2 is not None:             
            loading_placeholder = st.empty()
            loading_placeholder.markdown("Please wait while we process your data...🚀")
            try:
                fraction_tab2 = float(fraction_tab2) if fraction_tab2 else 0.6
                iteration_tab2 = int(iteration_tab2) if iteration_tab2 else 10
                all_data_tab2 = bool(all_data_tab2)
                seed_tab2 = int(seed_tab2) if seed_tab2 else 42

                if all_data_tab2:
                    zip_data = []
                    res = extractAccuracyRep(input_file_tab2, fraction=fraction_tab2, iteration=iteration_tab2,seed=seed_tab2,
                                             all_data = all_data_tab2)
                    
                    with zipfile.ZipFile(res,"r") as zip_file:
                        with zip_file.open("data.csv") as csv_file:
                            scatter = pd.read_csv(csv_file)

                    with zipfile.ZipFile(res, "r") as zip_file:
                        combined_zip_buffer = BytesIO()
                        with zipfile.ZipFile(combined_zip_buffer, 'w') as combined_zip_file:
                            # Add all files from the original ZIP file to the new combined ZIP file
                            for file_info in zip_file.infolist():
                                print(file_info)
                                with zip_file.open(file_info.filename) as original_file:
                                    combined_zip_file.writestr(file_info.filename, original_file.read())
                            scatter = scatter.groupby(['Model', 'Class']).mean().reset_index()
                            scatter_folder = "scatter_plots"
                            scatter["Class"] = scatter["Class"].str.replace(r'^Class_', '', regex=True)
                            combs = list(combinations(acc, 2))
                            for pair in combs:
                                plt.figure(figsize=(8, 6))
                                p1 = sns.scatterplot(data=scatter, x=pair[0], y=pair[1], hue="Model", legend="auto", markers = True, style = "Class", s  = 70)
                                buffer = BytesIO()
                                plt.savefig(buffer,format="png",dpi = 300)
                                buffer.seek(0)
                                plt.close()
                                combined_zip_file.writestr(f'{scatter_folder}/{pair[0]}_{pair[1]}_scatter.png', buffer.getvalue())

                    combined_zip_buffer.seek(0)
                    loading_placeholder.empty()
                    st.write("Your results are ready! 📊")

            
                    st.download_button(label="Download data", data=combined_zip_buffer, file_name="data.zip", key="download_scatter",
                                      mime="application/zip")

                else:

                    res = extractAccuracyRep(input_file_tab2, fraction=fraction_tab2, iteration=iteration_tab2,seed=seed_tab2)
          
                    loading_placeholder.empty()
                    st.write("Your results are ready! 📊")
                    st.download_button(label="Download data", data=res, file_name="extracted_result_accuracy_rep.csv", key="download_button_reps",
                                      mime = "text/csv")

            
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("Please provide the input(s)")

if __name__ == "__main__":
    main()

