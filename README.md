# Infant Audio Classification  

This project classifies infant audio into three categories: **Cry, Scream, and Noncry** using **Wav2Vec2**, **YAMNet**, and **XGBoost**. It employs multiple **ensemble learning techniques** for improved accuracy and supports **real-time inference**.  

---

## 📌 Table of Contents  
- [Overview](#overview)  
- [Project Structure](#project-structure)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Models Used](#models-used)  
- [Evaluation Metrics](#evaluation-metrics)  
- [Ensemble Learning Methods](#ensemble-learning-methods)  
- [Results](#results)  
- [Future Work](#future-work)  
- [License](#license)  

---

## 🔹 Overview  
The project aims to classify **infant cries, screams, and other background noises** to assist in medical diagnostics and childcare.  

👉 **Preprocessing**: Augementation, silence trimming, resampling  
👉 **Feature Extraction**: YAMNet and Wav2Vec2  
👉 **Machine Learning**: XGBoost on YAMNet features  
👉 **Deep Learning**: Fine-tuned Wav2Vec2 model  
👉 **Ensemble Learning**: Combining multiple models for accuracy  
👉 **Real-time Inference**: Microphone-based classification  
👉 **GUI Interface**: Interactive Tkinter-based UI  

---

## 📝 Project Structure  
```
Infant_Audio_Classification/
│── data/                   # Raw and processed audio data  
│── models/                 # Saved models  
│   │── ensemble/           # Trained ensemble models  
│   │   │── majority_voting_ensemble.pkl  
│   │   │── platt_scaling_ensemble.pkl  
│   │   │── stacking_ensemble.pkl  
│   │   │── weighted_ensemble.pkl  
│   │── wav2vec2_finetuned/ # Fine-tuned Wav2Vec2  
│   │── xgboost_model.pkl   # Trained XGBoost model  
│── notebooks/              # Jupyter notebooks  
│── outputs/                # Evaluation results (ROC, Confusion Matrices)  
│── src/                    # Source code  
│   │── preprocessing/      # Data preprocessing scripts  
│   │── feature_extraction/ # Feature extraction scripts  
│   │── training/           # Model training scripts  
│   │── inference/          # Inference & real-time prediction  
│── README.md               # Project documentation  
│── train.py                # Model training script  
│── inference.py            # Real-time inference script  
│── evaluate.py             # Model evaluation script  
│── requirements.txt        # Dependencies  
```

---

## 🛠️ Installation  

### Prerequisites  
- Python **3.8+**  
- **GPU Recommended** for training  

### Steps  
Clone the repository:  
```bash
git clone https://github.com/your-username/audio-classification.git
cd audio-classification
```
Create a virtual environment:  
```bash
python -m venv venv  
source venv/bin/activate  # Windows: venv\Scripts\activate  
```
Install dependencies:  
```bash
pip install -r requirements.txt  
```
Prepare the dataset:  
```bash
mkdir -p data/raw_audio/Dataset
```
Place raw audio files in:  
```
data/raw_audio/Dataset/  
├── experimental/  
│   ├── cry/  
│   ├── scream/  
└── control/  
    ├── noncry/  
```
Run preprocessing:  
```bash
python src/preprocessing/audiopreprocessor.py  
```
Extract features:  
```bash
python src/feature_extraction/feature_extractor.py  
```
Train the models:  
```bash
python train.py  
```

---

## 🚀 Usage  

### **1. Single File Inference**  
```bash
python inference.py --file path/to/audio.wav  
```
### **2. Batch Processing**  
```bash
python inference.py --folder path/to/audio/folder  
```
### **3. Real-Time Inference**  
```bash
python inference.py --realtime  
```
### **4. GUI-Based Inference**  
```bash
python inference.py --gui  
```

---

## 📊 Models Used  
### **1. Wav2Vec2**  
- Transformer-based model, fine-tuned for classification.  

### **2. XGBoost**  
- Trained on **YAMNet features** extracted from audio.  

### **3. Ensemble Learning**  
- Combines multiple models to enhance accuracy.  

---

## 📊 Evaluation Metrics  
- **Accuracy**  
- **Precision**  
- **Recall**  
- **F1-score**  
- **Confusion Matrix**  
- **ROC-AUC Curve**  

---

## 🤖 Ensemble Learning Methods  
### **1. Weighted Ensemble (Used for Final Inference)**  
- Combines **Wav2Vec2 (70%)** and **XGBoost (30%)** for final predictions.  

### **2. Stacking Ensemble**  
- Uses a **Logistic Regression** meta-classifier for combination.  

### **3. Majority Voting**  
- Predicts the most frequently chosen class.  

### **4. Platt Scaling**  
- Uses **logistic regression calibration** for probability adjustment.  

---

## 📊 Results 
✅ XGBoost model and test data loaded.
🎯 XGBoost Test Accuracy: 0.8713
📙 Classification Report (XGBoost):
               precision    recall  f1-score   support

         cry       0.89      0.86      0.87       125
      scream       0.80      0.89      0.84       112
      noncry       0.94      0.87      0.90        97

    accuracy                           0.87       334
   macro avg       0.88      0.87      0.87       334
weighted avg       0.88      0.87      0.87       334

📊 Confusion Matrix saved at: outputs\XGBoost Confusion Matrix.png
📊 ROC Curve saved at: outputs\XGBoost_roc_curve.png
✅ Wav2Vec2 model loaded.
🎯 Wav2Vec2 Test Accuracy: 0.8358
📙 Classification Report (Wav2Vec2):
               precision    recall  f1-score   support

         cry       0.88      0.84      0.86       118
      scream       0.79      0.85      0.82       112
      noncry       0.85      0.82      0.83       105

    accuracy                           0.84       335
   macro avg       0.84      0.84      0.84       335
weighted avg       0.84      0.84      0.84       335

📊 Confusion Matrix saved at: outputs\Wav2Vec2 Confusion Matrix.png
📊 ROC Curve saved at: outputs\Wav2Vec2 ROC Curve_roc_curve.png
✅ weighted ensemble Model Saved at: models\ensemble\weighted ensemble.pkl
🎯 Ensemble Model Test Accuracy: 0.9373
📙 Classification Report (Ensemble Model):
               precision    recall  f1-score   support

         cry       0.98      0.93      0.96       118
      scream       0.90      0.95      0.92       112
      noncry       0.93      0.93      0.93       105

    accuracy                           0.94       335
   macro avg       0.94      0.94      0.94       335
weighted avg       0.94      0.94      0.94       335

📊 Confusion Matrix saved at: outputs\Ensemble Model Confusion Matrix.png
📊 ROC Curve saved at: outputs\Ensemble Model ROC Curve_roc_curve.png
✅ stacking_ensemble Model Saved at: models\ensemble\stacking_ensemble.pkl
🎯 Stacking Ensemble Test Accuracy: 0.9612
📙 Classification Report (Stacking Ensemble):
               precision    recall  f1-score   support

         cry       0.97      0.99      0.98       118
      scream       0.96      0.95      0.95       112
      noncry       0.94      0.94      0.94       105

    accuracy                           0.96       335
   macro avg       0.96      0.96      0.96       335
weighted avg       0.96      0.96      0.96       335

📊 Confusion Matrix saved at: outputs\Stacking Ensemble Confusion Matrix.png
📊 ROC Curve saved at: outputs\Stacking Ensemble ROC Curve_roc_curve.png
✅ majority_voting_ensemble Model Saved at: models\ensemble\majority_voting_ensemble.pkl
🎯 Majority Voting Ensemble Test Accuracy: 0.8955
📙 Classification Report (Majority Voting Ensemble):
               precision    recall  f1-score   support

         cry       0.88      1.00      0.94       118
      scream       0.85      0.89      0.87       112
      noncry       0.98      0.78      0.87       105

    accuracy                           0.90       335
   macro avg       0.90      0.89      0.89       335
weighted avg       0.90      0.90      0.89       335

📊 Confusion Matrix saved at: outputs\Majority Voting Ensemble Confusion Matrix.png
📊 ROC Curve saved at: outputs\Majority Voting Ensemble ROC Curve_roc_curve.png
✅ platt_scaling_ensemble Model Saved at: models\ensemble\platt_scaling_ensemble.pkl
🎯 Platt Scaling Ensemble Test Accuracy: 0.9612
📙 Classification Report (Platt Scaling Ensemble):
               precision    recall  f1-score   support

         cry       0.97      0.99      0.98       118
      scream       0.96      0.95      0.95       112
      noncry       0.95      0.94      0.95       105

    accuracy                           0.96       335
   macro avg       0.96      0.96      0.96       335
weighted avg       0.96      0.96      0.96       335

📊 Confusion Matrix saved at: outputs\Platt Scaling Ensemble Confusion Matrix.png
📊 ROC Curve saved at: outputs\Platt Scaling Ensemble ROC Curve_roc_curve.png



### **Confusion Matrices**  
![Weighted Ensemble Confusion Matrix](outputs/weighted_confusion_matrix.png)  

### **ROC Curves**  
![Weighted Ensemble ROC](outputs/weighted_roc_curve.png)  

---

## 💡 Future Work  
💡 **Expand dataset** for better generalization.  
💡 **Improve real-time inference** with optimized pipelines.  
💡 **Deploy as a web or mobile application.**  
💡 **Enhance ensemble models** with adaptive weighting.  
💡 **Test additional deep learning models like Whisper or Hubert.**  

---

## 📚 License  
This project is licensed under the **MIT License**.  
See the [LICENSE](LICENSE) file for details.  
