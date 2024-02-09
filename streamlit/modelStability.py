#!/usr/bin/env python
# coding: utf-8

# In[37]:


from io import BytesIO
from itertools import product
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import tempfile
import os
import zipfile
import time
from extractAccuracyRep import *
warnings.simplefilter(action='ignore', category=FutureWarning)


# In[ ]:


def main():
    st.title("Extract model stability")
    input_file_tab5 = st.file_uploader("Upload Input File", type=["csv"],key = 5)
    seed_tab5 = st.text_input("Seed (default: 42)", value="42", key = "tab5_seed")
    iteration_values = st.text_input("Enter Iteration values (comma-separated):", "10,20,30")
    fraction_values = st.text_input("Enter Fraction values (comma-separated):", "0.6,0.7")
    acc = st.multiselect("Select accuracy metrics for plotting", ["Precision","Sensitivity","Specificity","Accuracy","MCC","F1","IOU"])
    
    iteration_values = [int(val.strip()) for val in iteration_values.split(",")]
    fraction_values = [float(val.strip()) for val in fraction_values.split(",")]
       
    parameters = {"iteration": iteration_values, "fraction": fraction_values}
    param_combinations = list(product(*parameters.values()))
    df = pd.DataFrame()
    zip_data = []

    if st.button("Run extractStability"):  
        if input_file_tab5 is not None:
            loading_placeholder = st.empty()
            loading_placeholder.markdown("Please wait while we process your data...🚀")
            try:
                seed_tab5 = int(seed_tab5) if seed_tab5 else 42                        
                for iteration, fraction in param_combinations:
                    input_file_tab5.seek(0)
                    result = extractAccuracyRep(input_file_tab5,iteration = iteration, fraction = fraction, seed = seed_tab5)
                    result = pd.read_csv(BytesIO(result))
                    df = pd.concat([df, result], ignore_index=True)

                plot_data = pd.DataFrame()
                for model in pd.unique(df["Model"]):   
                    filtered_data = df[(df['stat'] == 'mean') & (df['Model'] == model)]
                    plot_data = pd.concat([plot_data,filtered_data],ignore_index = True)
                    for metric in acc:
                        heatmap_data = filtered_data.pivot_table(index='iteration', columns='fraction', values=metric, aggfunc='mean')
                        plt.figure(figsize=(12, 8))
                        sns.heatmap(heatmap_data, cmap='viridis', annot=True, fmt=".3f", linewidths=.5)
                        plt.xlabel('Fraction')
                        plt.ylabel('Iteration')
                        buffer = BytesIO()
                        plt.savefig(buffer, format='png', dpi=300)
                        buffer.seek(0)
                        plt.close()
                        zip_data.append((f'{model}_{metric}_heatmap.png',buffer.getvalue()))
                zip_buffer = BytesIO()
                
                plot_data = plot_data.iloc[:,1:]
                plot_data = plot_data.to_csv(index=False).encode("utf-8")
                zip_data.append(("plot_data.csv",plot_data))
                
                with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                    for file_name, file_content in zip_data:
                        print(file_name, file_content)
                        zip_file.writestr(file_name,file_content)
                
                zip_buffer.seek(0)
                loading_placeholder.empty()
                st.write("Your results are ready! 📊")
                
                st.download_button(label="Download data", data=zip_buffer, file_name="data.zip", key="download_button_stab",
                                  mime = "application/zip")
            
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("Please provide the input(s)")

if __name__ == "__main__":
    main()

