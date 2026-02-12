# A Robust and Explainable IoT Intrusion Detection:
## An Ensemble GNN Approach Against Adversarial Attacks

This repository contains the implementation and experimental notebooks for the paper:
"A Robust and Explainable IoT Intrusion Detection: An Ensemble GNN Approach Against Adversarial Attacks"
Submitted to: Scientific Reports
---------------------------------------------------------------------
## 1. Project Overview
This project presents a robust and explainable intrusion detection framework
for IoT environments using an Ensemble Graph Neural Network (GNN) model.
The framework is designed to defend against adaptive adversarial attacks
including FGSM and DeepFool.
We introduce an adversarially extended dataset, AdvCICDDoS2019,
constructed by injecting four types of adversarial manipulations:
- Adversarial Perturbation (AP)
- Adversarial Outlier Injection (AOI)
- Adversarial Noise Injection (ANI)
- Adversarial Benign (AB)
The system integrates:
- Ensemble-based adversarial training
- Graph Neural Networks (GNN)
- DeepFool and FGSM attacks
- Explainability via SHAP and LIME
---------------------------------------------------------------------
## 2. Repository Structure
The repository currently contains the following notebooks:
- Adversarial Attacks DeepFool.ipynb
- Adversarial Attacks GNN.ipynb
- Adversarial Attacks-Ensemble Model.ipynb
- Ablation study.ipynb
Description:
1. Adversarial Attacks DeepFool.ipynb  
   Implements DeepFool attack generation and evaluation.
2. Adversarial Attacks GNN.ipynb  
   Training and evaluation of the GNN-based intrusion detection model.
3. Adversarial Attacks-Ensemble Model.ipynb  
   Implementation of the ensemble adversarial training framework.
4. Ablation study.ipynb  
   Ablation experiments evaluating individual components of the framework.
---------------------------------------------------------------------
## 3. Dataset Availability
The AdvCICDDoS2019 dataset (4 files, ~336MB total) is available upon reasonable request.
Google Drive Link:
https://drive.google.com/drive/folders/1INBP7Nfe-YGZ_5XB4qAapIzkRhlE5xDZ?usp=drive_link
If you use the dataset for research purposes, please cite the corresponding paper.
Note:
The original CICDDoS2019 dataset is publicly available from its official source.
---------------------------------------------------------------------
## 4. Implemented Attacks
### 4.1 Fast Gradient Sign Method (FGSM)
Single-step gradient-based adversarial attack.
### 4.2 DeepFool
Iterative attack that computes minimal perturbations required to change model prediction.
### 4.3 Ensemble Attack (DeepFool + FGSM)
Sequential combination of DeepFool and FGSM to enhance adversarial effectiveness.
---------------------------------------------------------------------
## 5. Requirements
- Python >= 3.8
- NumPy
- Pandas
- PyTorch
- PyTorch Geometric
- Scikit-learn
- SHAP
- LIME
- Matplotlib
- Jupyter Notebook

Install dependencies using:
pip install -r requirements.txt
---------------------------------------------------------------------
## 6. Experimental Pipeline
1. Preprocess dataset
2. Train baseline GNN model
3. Generate adversarial samples (FGSM / DeepFool)
4. Perform ensemble adversarial training
5. Evaluate detection performance
6. Conduct ablation study
7. Interpret results using SHAP and LIME
---------------------------------------------------------------------
## 7. Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score
- Detection Rate
- Attack Success Rate
- Robustness under adversarial settings
---------------------------------------------------------------------
## 8. License
This repository is provided for academic and research purposes.
---------------------------------------------------------------------
## 9. Citation
Available soon 
---------------------------------------------------------------------
