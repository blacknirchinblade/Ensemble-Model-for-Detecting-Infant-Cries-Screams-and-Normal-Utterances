# Ensemble-Model-for-Detecting-Infant-Cries-Screams-and-Normal-Utterances
# Audio Classification Project

This project focuses on classifying audio segments into three categories:
- **Cry**: Infant crying sounds.
- **Scream**: Human screaming sounds.
- **Noncry**: Background noise, speech, or other sounds.

We utilize a combination of **YAMNet**, **Wav2Vec2**, and ensemble models to improve classification accuracy. The project includes **data preprocessing, feature extraction, model training, evaluation, and real-time inference capabilities**.

---

## Features
### Data Processing:
- Audio trimming, silence removal, and resampling to 16kHz.
- Data augmentation (time-stretching, pitch-shifting, noise addition).

### Model Training:
- Fine-tuning **Wav2Vec2** for classification.
- Training **XGBoost** classifier using **YAMNet** embeddings.
- Implementing **ensemble models** combining multiple predictions.

### Ensemble Learning:
- **Platt Scaling**: Logistic regression-based calibration.
- **Stacking**: Meta-classifier combining Wav2Vec2 and XGBoost.
- **Majority Voting**: Selects the most common prediction.
- **Weighted Ensemble**: Wav2Vec2 (70\%) + XGBoost (30\%).

### Real-Time Inference:
- Live classification using microphone input.
- Processes 5-15 second audio chunks.

### GUI Interface:
- Tkinter-based GUI for selecting files, recording, and batch processing.

---

## Installation
### Prerequisites:
- **Python 3.8+**
- **GPU** (Recommended for faster training and inference)

### Steps:
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-username/audio-classification.git
   cd audio-classification
   ```

2. **Create a Virtual Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download and Organize the Dataset:**
   Structure your dataset as follows:
   ```
   data/raw_audio/Dataset/
   ├── experimental/
   │   ├── cry/
   │   └── scream/
   └── control/
       └── noncry/
   ```

5. **Run Preprocessing:**
   ```bash
   python audiopreprocessor.py
   ```

6. **Extract Features:**
   ```bash
   python feature_extractor.py
   ```

7. **Train the Models:**
   ```bash
   python train.py
   ```

---

## Usage
### Real-Time Inference (GUI):
Run the GUI for real-time audio classification:
```bash
python inference.py
```

#### GUI Features:
- **Choose File**: Select a .wav file for classification.
- **Record Audio**: Capture and classify audio using a microphone.
- **Batch Process Folder**: Process multiple audio files in a directory.

### Command-Line Inference:
You can also use the inference script programmatically:
```python
from inference import AudioInference

inferencer = AudioInference()

# Predict from a file
result = inferencer.predict_single("path/to/audio.wav")
print(result)

# Real-time inference
audio_chunk = ...  # Load or record a 5-15 second audio chunk
result = inferencer.real_time_inference(audio_chunk)
print(result)
```

---

## File Structure
```
audio-classification/
├── data/
│   ├── raw_audio/          # Raw audio dataset
│   ├── processed/          # Processed audio files
│   └── features/           # Extracted features (YAMNet, Wav2Vec2)
├── models/                 # Saved models (XGBoost, Wav2Vec2, Ensembles)
├── outputs/                # Evaluation outputs (ROC curves, confusion matrices)
├── src/
│   ├── training/           # Training scripts and dataset class
│   └── inference/          # Inference scripts
├── audiopreprocessor.py    # Audio preprocessing script
├── feature_extractor.py    # Feature extraction script
├── train.py                # Model training script
├── evaluate.py             # Model evaluation script
├── inference.py            # Real-time inference and GUI
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## Results
### Performance Metrics
| Model           | Accuracy | Precision | Recall | F1-Score |
|----------------|----------|------------|------------|------------|
| **XGBoost**     | X%       | X%         | X%         | X%         |
| **Wav2Vec2**    | X%       | X%         | X%         | X%         |
| **Stacking**    | X%       | X%         | X%         | X%         |
| **Weighted**    | X%       | X%         | X%         | X%         |
| **Voting**      | X%       | X%         | X%         | X%         |

### ROC Curves
![XGBoost ROC](outputs/XGBoost_ROC_Curve.png)
![Wav2Vec2 ROC](outputs/Wav2Vec2_ROC_Curve.png)
![Ensemble ROC](outputs/ensemble_roc_curve.png)

### Confusion Matrices
![XGBoost Confusion Matrix](outputs/XGBoost_Confusion_Matrix.png)
![Wav2Vec2 Confusion Matrix](outputs/Wav2Vec2_Confusion_Matrix.png)
![Ensemble Confusion Matrix](outputs/ensemble_confusion_matrix.png)

---

## Contributing
Contributions are welcome! Follow these steps:
1. **Fork the Repository.**
2. **Create a New Branch:**
   ```bash
   git checkout -b feature-branch
   ```
3. **Commit Changes:**
   ```bash
   git commit -m "Add new feature"
   ```
4. **Push to the Branch:**
   ```bash
   git push origin feature-branch
   ```
5. **Open a Pull Request.**

---

## License
This project is licensed under the MIT License. See the LICENSE file for details.

