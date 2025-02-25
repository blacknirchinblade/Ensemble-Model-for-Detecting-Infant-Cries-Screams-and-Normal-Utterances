import os
import pickle
import sys
import numpy as np
import torch
import librosa
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
from torch.utils.data import DataLoader, random_split

sys.path.append(os.path.abspath("src"))
from training.dataset import AudioDataset


class ModelEvaluator:
    def __init__(self, dataset_root):
        """
        Initialize the model evaluator.
        
        Args:
            dataset_root (str): Path to the processed dataset.
        """
        self.dataset_root = dataset_root
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.label_map = {"cry": 0, "scream": 1, "noncry": 2}
        self.labels = list(self.label_map.keys())

    def save_model(self, model, model_name):
        """
        Save trained ensemble models.
        
        Args:
            model: The trained model.
            model_name (str): Name of the model.
        """
        model_dir = os.path.join("models", "ensemble")
        os.makedirs(model_dir, exist_ok=True)  # Ensure the directory exists

        save_path = os.path.join("models", "ensemble", f"{model_name}.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(model, f)
        print(f"✅ {model_name} Model Saved at: {save_path}")

    def print_classification_report(self, y_true, y_pred, model_name):
        """
        Print classification accuracy and detailed report.
        
        Args:
            y_true (np.array): True labels.
            y_pred (np.array): Predicted labels.
            model_name (str): Name of the model.
        """
        print(f"\U0001F3AF {model_name} Test Accuracy: {accuracy_score(y_true, y_pred):.4f}")
        print(f"\U0001F4D9 Classification Report ({model_name}):\n", classification_report(y_true, y_pred, target_names=self.labels))

    def plot_roc_curve(self, y_true, y_probs, model_name):
        """
        Plot ROC curve and save the figure.
        
        Args:
            y_true (np.array): True labels.
            y_probs (np.array): Predicted probabilities.
            model_name (str): Name of the model.
        """
        y_true = np.array(y_true)
        y_probs = np.array(y_probs)

        if y_probs.shape[0] != len(y_true):
            print(f"⚠️ Warning: Shape mismatch! Reshaping y_probs from {y_probs.shape} to ({len(y_true)}, {y_probs.shape[1]})")
            y_probs = y_probs[:len(y_true), :]

        y_true_bin = label_binarize(y_true, classes=[0, 1, 2])

        plt.figure(figsize=(7, 6))
        for i, label in enumerate(self.labels):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
            plt.plot(fpr, tpr, label=f"{label} (AUC = {auc(fpr, tpr):.4f})")

        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{model_name} ROC Curve")
        plt.legend()

        # Ensure 'outputs/' directory exists
        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)

        save_path = os.path.join(output_dir, f"{model_name}_roc_curve.png")
        plt.savefig(save_path)
        print(f"📊 ROC Curve saved at: {save_path}")
        plt.close()

    def platt_scaling_ensemble(self, dataset):
        """
        Platt Scaling Ensemble: Calibrates model probabilities using Logistic Regression.
        
        Args:
            dataset: The dataset for evaluation.
        """
        self.wav2vec2_model.eval()  # Ensure Wav2Vec2 model is in inference mode
        all_labels, all_features = [], []

        with torch.no_grad():
            for inputs, label, audio_features in dataset:
                # XGBoost Predictions (Raw Probabilities)
                xgb_probs = self.xgb_model.predict_proba(audio_features.reshape(1, -1))

                # Wav2Vec2 Predictions (Raw Logits)
                inputs = {key: val.to(self.device) for key, val in inputs.items()}
                input_values = inputs["input_values"].to(self.device)

                # Shape Handling
                if input_values.dim() == 3:
                    input_values = input_values.squeeze(1)  # [batch, 1, seq_len] → [batch, seq_len]
                elif input_values.dim() == 1:
                    input_values = input_values.unsqueeze(0)  # [seq_len] → [1, seq_len]
                if input_values.dim() != 2:
                    raise ValueError(f"❌ Unexpected shape for input_values: {input_values.shape}")

                logits = self.wav2vec2_model(input_values).logits
                wav2vec2_probs = torch.softmax(logits, dim=1).cpu().numpy()

                # Stack Wav2Vec2 & XGBoost Features
                all_features.append(np.hstack((wav2vec2_probs, xgb_probs)).squeeze())
                all_labels.append(label)

        # Convert to NumPy Arrays
        all_features = np.array(all_features).reshape(len(all_features), -1)
        all_labels = np.array(all_labels)

        # Train the base model first
        base_model = LogisticRegression()
        base_model.fit(all_features, all_labels)  # Train Logistic Regression First

        # Apply Platt Scaling (Calibrated Classifier)
        calibrated_model = CalibratedClassifierCV(base_model, method="sigmoid", cv=5)  # No "prefit"

        # Train Calibration Model
        calibrated_model.fit(all_features, all_labels)

        self.save_model(calibrated_model, "platt_scaling_ensemble")
        
        # Get Calibrated Predictions
        preds = calibrated_model.predict(all_features)
        probs = calibrated_model.predict_proba(all_features)

        # Evaluate Platt Scaling Ensemble
        self.print_classification_report(all_labels, preds, "Platt Scaling Ensemble")
        self.plot_confusion_matrix(all_labels, preds, "Platt Scaling Ensemble Confusion Matrix","Platt Scaling Ensemble Confusion Matrix.png")
        self.plot_roc_curve(all_labels, probs, "Platt Scaling Ensemble ROC Curve")

    def stacking_ensemble(self, dataset):
        """
        Evaluate a stacking ensemble using Logistic Regression as the meta-classifier.
        
        Args:
            dataset: The dataset for evaluation.
        """
        self.wav2vec2_model.eval()  # Ensure model is in inference mode
        all_labels, stacked_features = [], []

        with torch.no_grad():
            for inputs, label, audio_features in dataset:
                # XGBoost Predictions (Probabilities)
                xgb_probs = self.xgb_model.predict_proba(audio_features.reshape(1, -1))

                # Wav2Vec2 Predictions (Probabilities)
                inputs = {key: val.to(self.device) for key, val in inputs.items()}
                input_values = inputs["input_values"].to(self.device)

                # Shape Handling (Ensure Proper Dimensions)
                if input_values.dim() == 3:
                    input_values = input_values.squeeze(1)  # Convert [batch, 1, seq_len] → [batch, seq_len]
                elif input_values.dim() == 1:
                    input_values = input_values.unsqueeze(0)  # Convert [seq_len] → [1, seq_len]

                if input_values.dim() != 2:
                    raise ValueError(f"❌ Unexpected shape for input_values: {input_values.shape}")

                logits = self.wav2vec2_model(input_values).logits
                wav2vec2_probs = torch.softmax(logits, dim=1).cpu().numpy()

                # Stacking Feature: Concatenation of Wav2Vec2 + XGBoost
                stacked_features.append(np.hstack((wav2vec2_probs, xgb_probs)).squeeze())
                all_labels.append(label)

        # Convert to NumPy Arrays
        stacked_features = np.array(stacked_features).reshape(len(stacked_features), -1)
        all_labels = np.array(all_labels)

        # Train Meta-Model (Logistic Regression)
        meta_model = LogisticRegression()
        meta_model.fit(stacked_features, all_labels)
        preds = meta_model.predict(stacked_features)
        pred_probs = meta_model.predict_proba(stacked_features)  # Get probabilities for ROC Curve
        self.save_model(meta_model, "stacking_ensemble")

        # Evaluate Stacking Ensemble
        self.print_classification_report(all_labels, preds, "Stacking Ensemble")
        self.plot_confusion_matrix(all_labels, preds, "Stacking Ensemble Confusion Matrix","Stacking Ensemble Confusion Matrix.png")
        self.plot_roc_curve(all_labels, pred_probs, "Stacking Ensemble ROC Curve")  # Added ROC Curve

    def majority_voting_ensemble(self, dataset):
        """
        Evaluate Majority Voting Ensemble (Chooses the most frequent class).
        
        Args:
            dataset: The dataset for evaluation.
        """
        self.wav2vec2_model.eval()  # Ensure model is in inference mode
        all_preds, all_labels, all_probs = [], [], []

        with torch.no_grad():
            for inputs, label, audio_features in dataset:
                # XGBoost Prediction (Probabilities & Class Prediction)
                xgb_probs = self.xgb_model.predict_proba(audio_features.reshape(1, -1))
                xgb_pred = np.argmax(xgb_probs)

                # Wav2Vec2 Prediction (Probabilities & Class Prediction)
                inputs = {key: val.to(self.device) for key, val in inputs.items()}
                input_values = inputs["input_values"].to(self.device)

                # Ensure Correct Shape
                if input_values.dim() == 3:
                    input_values = input_values.squeeze(1)
                elif input_values.dim() == 1:
                    input_values = input_values.unsqueeze(0)

                if input_values.dim() != 2:
                    raise ValueError(f"❌ Unexpected shape for input_values: {input_values.shape}")

                logits = self.wav2vec2_model(input_values).logits
                wav2vec2_probs = torch.softmax(logits, dim=1).cpu().numpy()

                # Ensure xgb_probs & wav2vec2_probs have same shape
                if xgb_probs.shape != wav2vec2_probs.shape:
                    print(f"⚠️ Shape Mismatch: xgb_probs {xgb_probs.shape} != wav2vec2_probs {wav2vec2_probs.shape}")
                    xgb_probs = xgb_probs.reshape(wav2vec2_probs.shape)  # Reshape XGBoost output

                wav2vec2_pred = np.argmax(wav2vec2_probs)

                # Majority Voting
                final_pred = np.bincount([xgb_pred, wav2vec2_pred]).argmax()

                # Store Results
                all_preds.append(final_pred)
                all_labels.append(label)
                all_probs.append((wav2vec2_probs + xgb_probs) / 2)  # Averaging Probabilities

        # Convert to NumPy Arrays
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs).squeeze()  # Ensure correct shape
        self.save_model(all_preds, "majority_voting_ensemble")

        # Evaluate Model
        self.print_classification_report(all_labels, all_preds, "Majority Voting Ensemble")
        self.plot_confusion_matrix(all_labels, all_preds, "Majority Voting Ensemble Confusion Matrix","Majority Voting Ensemble Confusion Matrix.png")
        self.plot_roc_curve(all_labels, all_probs, "Majority Voting Ensemble ROC Curve")

    def load_xgboost_model(self, model_path, feature_file):
        """
        Load trained XGBoost model and correctly split the test set.
        
        Args:
            model_path (str): Path to the XGBoost model.
            feature_file (str): Path to the feature file.
        """
        with open(model_path, "rb") as f:
            self.xgb_model = pickle.load(f)

        with open(feature_file, "rb") as f:
            features, labels = pickle.load(f)

        # Split Dataset: 70% Train, 15% Validation, 15% Test
        X_train, X_temp, y_train, y_temp = train_test_split(features, labels, test_size=0.3, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        # Use Only the Test Set for Evaluation
        self.xgb_features = np.array(X_test)
        self.xgb_labels = np.array(y_test)

        print("✅ XGBoost model and test data loaded.")

    def load_wav2vec2_model(self, model_path, processor_path):
        """
        Load trained Wav2Vec2 model.
        
        Args:
            model_path (str): Path to the Wav2Vec2 model.
            processor_path (str): Path to the Wav2Vec2 processor.
        """
        self.processor = Wav2Vec2Processor.from_pretrained(processor_path)
        self.wav2vec2_model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path).to(self.device)
        print("✅ Wav2Vec2 model loaded.")

    def plot_confusion_matrix(self, y_true, y_pred, title, filename):
        """
        Plot confusion matrix and save the figure.
        
        Args:
            y_true (np.array): True labels.
            y_pred (np.array): Predicted labels.
            title (str): Title of the plot.
            filename (str): Name of the file to save.
        """
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=self.labels, yticklabels=self.labels)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(title)

        # Ensure 'outputs/' directory exists
        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)

        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path)
        print(f"📊 Confusion Matrix saved at: {save_path}")
        plt.close()

    def evaluate_xgboost(self):
        """Evaluate the XGBoost model."""
        y_pred = self.xgb_model.predict(self.xgb_features)
        self.print_classification_report(self.xgb_labels, y_pred, "XGBoost")
        self.plot_confusion_matrix(self.xgb_labels, y_pred, "XGBoost Confusion Matrix","XGBoost Confusion Matrix.png")
        self.plot_roc_curve(self.xgb_labels, self.xgb_model.predict_proba(self.xgb_features), "XGBoost")

    def evaluate_wav2vec2(self, dataloader):
        """Evaluate the Wav2Vec2 model and plot ROC curve."""
        self.wav2vec2_model.eval()
        all_preds, all_labels, all_probs = [], [], []

        with torch.no_grad():
            for inputs, labels, _ in dataloader:
                inputs = {key: val.to(self.device) for key, val in inputs.items()}
                labels = labels.to(self.device)

                # Ensure proper shape
                input_values = inputs["input_values"].to(self.device)
                if input_values.dim() == 3:
                    input_values = input_values.squeeze(1)  # Convert [batch, 1, sequence_length] → [batch, sequence_length]
                elif input_values.dim() == 1:
                    input_values = input_values.unsqueeze(0)  # Convert [sequence_length] → [1, sequence_length]

                if input_values.dim() != 2:
                    raise ValueError(f"❌ Unexpected shape for input_values: {input_values.shape}")

                # Get model logits & probabilities
                logits = self.wav2vec2_model(input_values).logits
                probs = torch.softmax(logits, dim=1).cpu().numpy()

                preds = np.argmax(probs, axis=1)

                all_probs.extend(probs)
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

        # Convert to NumPy
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)

        # Evaluation
        self.print_classification_report(all_labels, all_preds, "Wav2Vec2")
        self.plot_confusion_matrix(all_labels, all_preds, "Wav2Vec2 Confusion Matrix","Wav2Vec2 Confusion Matrix.png")
        self.plot_roc_curve(all_labels, all_probs, "Wav2Vec2 ROC Curve")  # Fix

    def evaluate_ensemble(self, dataset):
        """Evaluate the ensemble model combining Wav2Vec2 and XGBoost."""
        self.wav2vec2_model.eval()
        all_preds, all_labels, all_probs = [], [], []

        with torch.no_grad():
            for inputs, label, audio_features in dataset:
                xgb_probs = self.xgb_model.predict_proba(audio_features.reshape(1, -1))

                inputs = {key: val.to(self.device) for key, val in inputs.items()}
                input_values = inputs["input_values"].to(self.device)

                # Ensure correct shape for model input
                if input_values.dim() == 3:  # Case: [batch, 1, sequence_length]
                    input_values = input_values.squeeze(1)  # → [batch, sequence_length]
                elif input_values.dim() == 1:  # Case: Missing batch dim
                    input_values = input_values.unsqueeze(0)  # → [1, sequence_length]

                if input_values.dim() != 2:
                    raise ValueError(f"Unexpected shape for input_values: {input_values.shape}")

                logits = self.wav2vec2_model(input_values).logits
                wav2vec2_probs = torch.softmax(logits, dim=1).cpu().numpy()

                # Weighted Ensemble: 70% Wav2Vec2 + 30% XGBoost
                ensemble_probs = (0.7 * wav2vec2_probs) + (0.3 * xgb_probs)
                final_pred = np.argmax(ensemble_probs)

                all_preds.append(final_pred)
                all_labels.append(label)
                all_probs.append(ensemble_probs.squeeze())
        # Convert to NumPy Arrays
        all_labels = np.array(all_labels)
        all_probs = np.vstack(all_probs)
        self.save_model(all_preds, "weighted ensemble")

        self.print_classification_report(all_labels, all_preds, "Ensemble Model")
        self.plot_confusion_matrix(all_labels, all_preds, "Ensemble Model Confusion Matrix","Ensemble Model Confusion Matrix.png")
        self.plot_roc_curve(all_labels, all_probs, "Ensemble Model ROC Curve")  # Now works!


# Run Model Evaluation
if __name__ == "__main__":
    dataset_root = os.path.join("data", "processed", "Dataset")
    evaluator = ModelEvaluator(dataset_root)

    # Load XGBoost Model
    evaluator.load_xgboost_model(
        model_path=os.path.join("models", "xgboost_model.pkl"),
        feature_file=os.path.join("data", "features", "yamnet_features.pkl")
    )
    evaluator.evaluate_xgboost()

    # Load Wav2Vec2 Model
    evaluator.load_wav2vec2_model(
        model_path=os.path.join("models", "wav2vec2_finetuned"),
        processor_path=os.path.join("models", "wav2vec2_processor")
    )

    # Load Full Dataset
    full_dataset = AudioDataset(
        dataset_root,
        evaluator.processor,
        feature_file=os.path.join("data", "features", "yamnet_features.pkl")
    )

    # Split: 70% Train, 15% Validation, 15% Test
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size

    _, _, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # Evaluate Models
    evaluator.evaluate_wav2vec2(test_loader)
    evaluator.evaluate_ensemble(test_dataset)
    evaluator.stacking_ensemble(test_dataset)
    evaluator.majority_voting_ensemble(test_dataset)
    evaluator.platt_scaling_ensemble(test_dataset)