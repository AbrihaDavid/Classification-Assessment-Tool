#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st
from st_pages import add_page_title
import pandas as pd
import numpy as np
import time


# In[ ]:


def extractAccuracy(input_file):
    df = pd.read_csv(input_file, na_values="NA")
    df = df.sort_values(df.columns[0])
    metrics_all = pd.DataFrame()
    nrow = len(df.iloc[:,0].unique())
    ncol = 9
    reference_categories = df.iloc[:, 0].unique()
    for q in range(1, len(df.columns)):
        matrix_table = pd.crosstab(df.iloc[:, 0], pd.Categorical(df.iloc[:, q], categories=reference_categories),
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

            if TP+FP ==0:
                metrics.iloc[e, 2] = 0
            else:
                metrics.iloc[e, 2] = TP / (TP + FP)
            metrics.iloc[e, 3] = TP / (TP + FN)
            metrics.iloc[e, 4] = TN / (TN + FP)
            metrics.iloc[e, 5] = (TP + TN) / (TP + TN + FP + FN)
            if (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) ==0:
                metrics.iloc[e,6] = 0
            else:
                metrics.iloc[e, 6] = (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
            metrics.iloc[e, 7] = (2 * TP / (2 * TP + FP + FN))
            metrics.iloc[e, 8] = (TP / (TP + FP + FN))  
        metrics_all = pd.concat([metrics_all, metrics])
    return metrics_all


# In[ ]:


def main():
    st.title("Extract Accuracy")
    input_file_tab1 = st.file_uploader("Upload Input File", type=["csv"], key =1)
    if st.button("Run extractAccuracy"):
        if input_file_tab1 is not None:
            loading_placeholder = st.empty()
            loading_placeholder.markdown("Please wait while we process your data...🚀")
            try:
                df = extractAccuracy(input_file_tab1)
                df = df.to_csv(index=False).encode("utf-8")

                
                loading_placeholder.empty()
                st.write("Your results are ready! 📊")
                st.download_button(label="Download data", data=df, file_name="extracted_result_accuracy.csv", key="download_button_accuracy")

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("Please provide the input(s)")

if __name__ == "__main__":
    main()

