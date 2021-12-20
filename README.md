# Flight Delay Prediction
## MIDS W261: Machine Learning at Scale | Fall 2021 | Final Project
### Authors: Toby Petty, Zixi Wang, Yao Chen, Ferdous Alam

#### Introduction
This repo contains my group's final project for the _Machine Learning at Scale (W261)_ class in UC Berkeley's Masters in Data Science program, from the Fall 2021 semester. The problem involves predicting flight delays using a dataset of ~30m flights over a 5 year period, along with a supplementary dataset of more than 700m weather observations.

The project was implemented on a distributed computing cluster on the <a href="https://databricks.com/">Databricks</a> cloud platform, using PySpark for all data engineering, modeling, prediction, and evaluation.

#### Start here

For a full overview see the <a href="https://github.com/toby-p/pyspark-flight-delay-prediction/blob/master/notebooks/W261_AU21_FINAL_PROJECT_TEAM8.ipynb">final report</a> detailing the problem, our approach, and the full technical specifications of the solution pipeline.

#### Full Contents

* **code** - executable Python files to run the full pipeline from start to finish:
  * `full_data_pipeline.py` - data engineering pipeline to clean, transform, and augment the raw data to produce final train/test datasets suitable for supervised machine learning.
  * `gridsearch_cv.py` - custom class to grid-search model parameters in a cross-validation method designed for sequential time-series data.
  * `model_selection_ensemble.py` - ensemble of the best machine learning models found in the grid-searches to make final predictions and evaluate performance on the test dataset.
* **notebooks** - Databricks/Jupyter notebooks detailing each stage of the pipeline, along with full EDA, and final written report.
* **Flight Delay Prediction - Presentation Slides.pdf** - slide deck for the final presentation of our project work and findings.