#!/usr/bin/env python
# coding: utf-8

# In[3]:


import streamlit as st
from st_pages import Page, get_nav_from_toml, add_page_title
nav = get_nav_from_toml("streamlit/pages.toml")
pg = st.navigation(nav)
pg.run()

"""
show_pages(
    [
        Page("streamlit/main.py", "Introduction"),
        Page("streamlit/extractAccuracy.py", "Extract Accuracy"),
        Page("streamlit/extractAccuracyRep.py","Extract Accuracy with reps"),
        Page("streamlit/modelStability.py","Extract model stability"),
        Page("streamlit/extractROC.py","Extract ROC"),
        Page("streamlit/credits.py","Credits")
    ]  
)
"""

add_page_title(layout="wide")
st.write("Welcome to our Python tool, developed to enhance the interpretability and robustness of predictive models. This versatile tool has been designed with a focus on providing a comprehensive evaluation of model performance.")


st.markdown("## Key Features")

st.subheader("Accuracy Metric Extraction")
st.write("Evaluate model performance through the extraction of essential accuracy metrics, including Precision, Sensitivity, Specificity, Accuracy, Matthews Correlation Coefficient (MCC), F1-score, and Intersection over Union (IOU). This function works both with binary and multiclass datasets. In addition, an extended variant of this function introduces repetitions with user-defined parameters.")


st.subheader("Model Stability Assessment")
   
st.write("Gain insights into the stability of your models with a specialized function that conducts a grid search over different iteration and fraction values. The output includes a heatmap visualization based on selected accuracy metrics, offering an understanding of model stability.")

st.subheader("ROC Curve Analysis")
st.write("Explore the Receiver Operating Characteristic (ROC) curves. This function is capable of handling both binary and multiclass datasets (One-vs-Rest), providing insightful plots for model evaluation based on probabilities.")

## Usage
st.markdown("## Usage")

st.write("To get started, simply select the desired function based on your analytical goals. The tool allows you to customize parameters such as fraction, iteration, and seed, ensuring flexibility in your analyses. Whether you are focusing on accuracy metrics, model stability, or ROC curve analysis, our tool is here to streamline your evaluation process.")

st.write("For additional information, details on the tool, and access to sample data, please refer to our [GitHub repository](https://github.com/AbrihaDavid) and explore further insights in our article: [Link](https://www.sciencedirect.com/science/article/pii/S1568494624002424?via%3Dihub).")

