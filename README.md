# Welcome to our Python tool, developed to enhance the interpretability and robustness of predictive models.

[Link to the Streamlit application](https://classification-assessment-tool.streamlit.app/)
[Link to the executable file](https://zenodo.org/records/10646420)

## Key Features

### Accuracy Metric Extraction
Evaluate model performance through the extraction of essential accuracy metrics, including Precision, Sensitivity, Specificity, Accuracy, Matthews Correlation Coefficient (MCC), F1-score, and Intersection over Union (IOU). This function works both with binary and multiclass datasets. In addition, an extended variant of this function introduces repetitions with user-defined parameters.

### Model Stability Assessment
Gain insights into the stability of your models with a specialized function that conducts a grid search over different iteration and fraction values. The output includes a heatmap visualization based on selected accuracy metrics, offering an understanding of model stability.

### ROC Curve Analysis
Explore the Receiver Operating Characteristic (ROC) curves. This function is capable of handling both binary and multiclass datasets (One-vs-Rest), providing insightful plots for model evaluation based on probabilities.

## Usage

To get started, simply select the desired function based on your analytical goals. The tool allows you to customize parameters such as fraction, iteration, and seed, ensuring flexibility in your analyses. Whether you are focusing on accuracy metrics, model stability, or ROC curve analysis, our tool is here to streamline your evaluation process.