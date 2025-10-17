#  School Activity Recommender

A full baseline **machine learning pipeline** that predicts whether a student will participate in extracurricular activities, based on features such as GPA, attendance rate, grade level, and club interests.

This project was built as part of a data science workflow demo — covering the **entire lifecycle**:  
**Data Cleaning → Outlier Handling → Feature Encoding → Model Training → Prediction.**

---


---

##  Setup & Installation

###  Install dependencies
You can either create a new environment or use your default one.

```bash
pip -r requirements.txt

### Verify data paths
data/processed/activities_outliers_winsorized_v1.csv


### run the full pipeline
Run all steps automatically:
python scripts/run_all.py
This executes:
clean_data.py → handle_outliers.py → scale_encode.py → train_model.py → predict_batch.py

###output:
File	Description
models/preprocessor_v1.joblib	Saved preprocessing pipeline
models/baseline_logreg_v1.joblib	Trained Logistic Regression model
reports/model_cv_report_v1.json	Cross-validation metrics
reports/model_holdout_report_v1.json	Holdout evaluation metrics
reports/feature_importance_v1.csv	Feature weights
data/predictions/predictions_v1.csv	Final predictions


