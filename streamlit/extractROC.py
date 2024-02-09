#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.utils import resample
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import io
import zipfile
from seed_generator import *
import time


# In[ ]:


def calculate_roc_stats(df, seeds, iteration,m, ea, n_samples):
    filt = [df.columns[0]] + df.columns[df.columns.str.startswith(m)].tolist()
    df_filtered = df[filt].copy()
    df_filtered.rename(columns={col: col.split('_')[1] for col in df_filtered.columns if col.startswith(m)}, inplace=True)
    all_tpr = []
    all_fpr = []

    for i in range(iteration):
        res = resample(df_filtered, random_state=seeds[i], replace=False, n_samples=n_samples)

        try:
            targets = res.columns[1:].astype(int)
        except:
            targets = res.columns[1:]


        class_of_interest = targets[ea]
        class_id = np.flatnonzero(targets == class_of_interest)[0]

        score = np.array(res.iloc[:, 1:])
        y_true = label_binarize(res['id'], classes=targets)

        fpr, tpr, _ = roc_curve(y_true[:, class_id], score[:, class_id])
        all_tpr.append(tpr)
        all_fpr.append(fpr)

    interp_fpr = np.linspace(0, 1, 1000)
    interp_tpr = np.zeros((len(all_fpr), len(interp_fpr)))

    for i in range(len(all_fpr)):
        interp_tpr[i, :] = np.interp(interp_fpr, all_fpr[i], all_tpr[i])

    median_tpr = np.median(interp_tpr, axis=0)
    std_tpr = np.std(interp_tpr, axis=0)

    median_auc = auc(interp_fpr, median_tpr)
    std_auc = np.std([auc(interp_fpr, curve) for curve in interp_tpr])

    return interp_fpr, median_tpr, std_tpr, median_auc, std_auc,targets


# In[ ]:


def calculate_roc_stats_binary(df, seeds, iteration, m, n_samples):
    filt = [df.columns[0]] + df.columns[df.columns.str.startswith(m)].tolist()
    df_filtered = df[filt].copy()
    df_filtered.rename(columns={col: col.split('_')[1] for col in df_filtered.columns if col.startswith(m)}, inplace=True)
    
    if df_filtered.columns[-1].isnumeric():
        pos_label = pd.to_numeric(df_filtered.columns[-1])
    else:
        pos_label = df_filtered.columns[-1]
    all_tpr = []
    all_fpr = []

    for i in range(iteration):
        res = resample(df_filtered, random_state=seeds[i], replace=False, n_samples=n_samples)
        targets = res.columns[1:]
        score = np.array(res.iloc[:, 1:])
        fpr, tpr, _ = roc_curve(np.asarray(res["id"]), score[:,1],pos_label=pos_label)
        all_tpr.append(tpr)
        all_fpr.append(fpr)
    
    interp_fpr = np.linspace(0, 1, 1000)
    interp_tpr = np.zeros((len(all_fpr), len(interp_fpr)))
    
    
    for i in range(len(all_fpr)):
        interp_tpr[i, :] = np.interp(interp_fpr, all_fpr[i], all_tpr[i])
    
    median_tpr = np.median(interp_tpr, axis=0)
    std_tpr = np.std(interp_tpr, axis=0)
    
    median_auc = auc(interp_fpr, median_tpr)
    std_auc = np.std([auc(interp_fpr, curve) for curve in interp_tpr])

    return interp_fpr, median_tpr, std_tpr, median_auc, std_auc,targets


# In[ ]:


def extractROC(input_file, fraction=0.6, iteration=10, seed=42):
    df = pd.read_csv(input_file)
    n_samples = int(len(df)*fraction)
    seeds=generate_seeds(seed,iteration)
    buffers = []
    models = np.unique([model.split("_")[0] for model in df.columns[1:]])
    classes = np.unique([a.split("_")[1] for a in df.columns[1:]])
    full_targets = []
    
    if len(classes) > 2:
        for m in models:
            for ea in range(len(classes)):
                interp_fpr, median_tpr, std_tpr, median_auc, std_auc, targets = calculate_roc_stats(df,seeds,iteration,m,ea,n_samples)
                plt.figure(figsize=(8, 6))
                plt.plot(interp_fpr, median_tpr, color='red',
                            label=f'{targets[ea]} vs the rest | Median Curve {m} (AUC = {median_auc:.2f} ± {std_auc:.2f})')
                plt.plot([interp_fpr[0], interp_fpr[0]], [0, median_tpr[0]], color = "red")
                plt.fill_between(interp_fpr, median_tpr - std_tpr, median_tpr + std_tpr, color='lightgrey', alpha=0.3,
                                    label=f' ± 1 std | Iterations = {iteration}')
                plt.plot([0, 1], [0, 1], 'k--', label='Chance level (AUC = 0.5)')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.legend(loc = "lower right")
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', dpi=300)
                buffer.seek(0)
                plt.close()
                buffers.append(buffer)
                plt.figure(figsize=(8, 6))
                fill_between_plotted = False     
            _ = list(targets)
            _ = [m+"_"+str(t) for t in _]
            full_targets.append(_)
    
        for m in models:
            for ea in range(len(classes)):
                interp_fpr, median_tpr, std_tpr, median_auc, std_auc, targets = calculate_roc_stats(df,seeds,iteration,m,ea,n_samples)
                color = plt.plot(interp_fpr, median_tpr, label=f'{targets[ea]} vs the rest | Median Curve {m} (AUC = {median_auc:.2f} ± {std_auc:.2f})')[0].get_color()
                plt.plot([interp_fpr[0], interp_fpr[0]], [0, median_tpr[0]], color = color)
                if ea == len(classes)-1:    
                    plt.fill_between(interp_fpr, median_tpr - std_tpr, median_tpr + std_tpr, color="lightgrey", alpha=0.3,
                         label=f' ± 1 std | Iterations = {iteration}')                        
                plt.fill_between(interp_fpr, median_tpr - std_tpr, median_tpr + std_tpr, color=color, alpha=0.3)         
            plt.plot([0, 1], [0, 1], 'k--', label='Chance level (AUC = 0.5)')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend(loc="lower right")
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=300)
            buffer.seek(0)
            plt.close()
            buffers.append(buffer)
        
    else:
    
        for m in models:
            interp_fpr, median_tpr, std_tpr, median_auc, std_auc, targets = calculate_roc_stats_binary(df,seeds,iteration,m,n_samples)
            plt.figure(figsize=(8, 6))
            plt.plot(interp_fpr, median_tpr, color='red',
                        label=f'Median Curve {m} (AUC = {median_auc:.2f} ± {std_auc:.2f})')
            plt.plot([interp_fpr[0], interp_fpr[0]], [0, median_tpr[0]], color = "red")
            
            plt.fill_between(interp_fpr, median_tpr - std_tpr, median_tpr + std_tpr, color='lightgrey', alpha=0.3,
                                label=f' ± 1 std | Iterations = {iteration}')       
            plt.plot([0, 1], [0, 1], 'k--', label='Chance level (AUC = 0.5)') 
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend(loc="lower right")
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=300)
            buffer.seek(0)
            plt.close()
            buffers.append(buffer)
            
        full_targets.append(list(models))
        if len(models) > 1:
            for m in models:
                interp_fpr, median_tpr, std_tpr, median_auc, std_auc, targets = calculate_roc_stats_binary(df,seeds,iteration,m,n_samples)         
                color = plt.plot(interp_fpr, median_tpr, label=f'Median Curve {m} (AUC = {median_auc:.2f} ± {std_auc:.2f})')[0].get_color()
                plt.plot([interp_fpr[0], interp_fpr[0]], [0, median_tpr[0]], color = color)            
                if m == models[-1]:  
                    plt.fill_between(interp_fpr, median_tpr - std_tpr, median_tpr + std_tpr, color="lightgrey", alpha=0.3,
                         label=f' ± 1 std | Iterations = {iteration}')                          
                plt.fill_between(interp_fpr, median_tpr - std_tpr, median_tpr + std_tpr, color=color, alpha=0.3)         
            plt.plot([0, 1], [0, 1], 'k--', label='Chance level (AUC = 0.5)')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend(loc="lower right")
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=300)
            buffer.seek(0)
            plt.close()
            buffers.append(buffer)

    if len(classes) > 2 :
        full_targets.append([p+"_stacked" for p in models])
    else:
        full_targets.append(["stacked"])
    full_targets = [element for innerList in full_targets for element in innerList]
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
        for idx, buffer in enumerate(buffers):
            zip_file.writestr(f'plot_{full_targets[idx]}.png', buffer.getvalue())
    zip_buffer.seek(0)
        
    return zip_buffer


# In[ ]:


def main():
    st.title("Extract ROC")
   
    input_file_tab4 = st.file_uploader("Upload Input File", type=["csv"], key = 3)
    fraction_tab4 = st.text_input("Fraction (default: 0.6)", value="0.6", key = "tab4_frac")
    iteration_tab4 = st.text_input("Iteration (default: 10)", value="10", key = "tab4_iter")
    seed_tab4 = st.text_input("Seed (default: 42)", value="42", key = "tab4_seed")

    if st.button("Run extractROC"):   
        if input_file_tab4 is not None:    
            loading_placeholder = st.empty()
            loading_placeholder.markdown("Please wait while we process your data...🚀")
            try:
                fraction_tab4 = float(fraction_tab4) if fraction_tab4 else 0.6
                iteration_tab4 = int(iteration_tab4) if iteration_tab4 else 10
                seed_tab4 = int(seed_tab4) if seed_tab4 else 42

                buffer_download = extractROC(input_file_tab4, fraction=fraction_tab4, iteration=iteration_tab4,
                           seed=seed_tab4)
                
                loading_placeholder.empty()
                st.write("Your results are ready! 📊")

                
                st.download_button(label="Download data", data=buffer_download, file_name="plots.zip", key="download_button_roc",
                                  mime = "application/zip")

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("Please provide the input(s)")
    

if __name__ == "__main__":
    main()

