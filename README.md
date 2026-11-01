# Explainable Insider Threat Detection: Experiment Code

Code for the insider threat detection experiments by Kitoye Ebire Okonny (Ignatius Ajuru University of Education; PhD research, Centre for Information and Telecommunications Engineering, University of Port Harcourt). Supervisors: Prof. Eseosa Omorogiuwa and Dr. Matthew Ehikhamenle.

## Files
- `experiments_documented.py`: five-fold stratified cross-validation, pairwise Wilcoxon signed-rank tests, the AE-MLP ablation study, and confusion matrices for Random Forest, XGBoost, MLP and the autoencoder pipeline.
- `thesis_validation_pipeline.ipynb`: Google Colab validation pipeline (XGBoost and TensorFlow).
- `lime_explanation_mlp.ipynb`: LIME explanations for the MLP model.

## Data
The data is not included here. Download the Insider Threat Dataset for Classified Environments (Bdil, 2025) from Kaggle:
https://www.kaggle.com/datasets/efchbd1013/insider-threat-dataset-for-classified-environments

## Requirements
Python 3.12. Install with `pip install -r requirements.txt`.

## Citation
If you use this code, please cite the archived release (Zenodo DOI to be added) and the related papers.

## Licence
MIT
