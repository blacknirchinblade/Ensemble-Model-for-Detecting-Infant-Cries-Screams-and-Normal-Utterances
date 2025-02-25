import os
import pickle
import numpy as np
import librosa
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from xgboost import XGBClassifier
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from dataset import AudioDataset


class AudioClassifier:
    def __init__(self, dataset_root, num_epochs=8, batch_size=8, learning_rate=1e-5):
        """
        Initialize the classifier with dataset path and training parameters.
        
        Args:
            dataset_root (str): Path to the processed dataset.
            num_epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
            learning_rate (float): Learning rate for the optimizer.
        """
        self.dataset_root = dataset_root
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.label_map = {"cry": 0, "scream": 1, "noncry": 2}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 Using device: {self.device}")

    def load_features(self, feature_file):
        """
        Load extracted features from a pickle file.
        
        Args:
            feature_file (str): Path to the feature file.
        
        Returns:
            np.array: Features.
            np.array: Labels.
        """
        feature_path = os.path.join("data", "features", feature_file)
        with open(feature_path, "rb") as f:
            features, labels = pickle.load(f)
        return np.array(features), np.array(labels)

    def train_xgboost(self, feature_file):
        """
        Train an XGBoost classifier using YAMNet features.
        
        Args:
            feature_file (str): Path to the feature file.
        """
        features, labels = self.load_features(feature_file)
        X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.3, random_state=42)
        
        xgb_model = XGBClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=6,
            random_state=42,
            early_stopping_rounds=10,
            eval_metric="mlogloss"
        )
        xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True)
        
        y_pred = xgb_model.predict(X_val)
        print(f"🎯 XGBoost Test Accuracy: {accuracy_score(y_val, y_pred):.4f}")
        print("🔹 Classification Report:\n", classification_report(y_val, y_pred))
        
        # Save the XGBoost model
        os.makedirs("models", exist_ok=True)
        with open(os.path.join("models", "xgboost_model.pkl"), "wb") as f:
            pickle.dump(xgb_model, f)
        print("✅ XGBoost Model Saved!")

    def train_wav2vec2(self):
        """
        Fine-tune the Wav2Vec2 model for audio classification.
        """
        # Initialize the Wav2Vec2 processor and model
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
        model = Wav2Vec2ForSequenceClassification.from_pretrained(
            "facebook/wav2vec2-base", num_labels=3
        ).to(self.device)

        # Prepare the dataset
        dataset = AudioDataset(self.dataset_root, processor)
        train_size = int(0.7 * len(dataset))
        val_size = int(0.15 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, pin_memory=True)

        # Initialize optimizer, loss function, and gradient scaler
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate)
        criterion = torch.nn.CrossEntropyLoss()
        scaler = torch.cuda.amp.GradScaler()

        # Early stopping variables
        best_val_loss = float("inf")
        patience = 5  # Number of epochs to wait for improvement
        patience_counter = 0

        # Training loop
        for epoch in range(self.num_epochs):
            model.train()
            total_loss = 0.0

            for batch_idx, (inputs, labels, _) in enumerate(train_loader):
                inputs = {key: val.to(self.device, non_blocking=True) for key, val in inputs.items()}
                labels = labels.to(self.device, non_blocking=True)

                optimizer.zero_grad()

                # Mixed precision training
                with torch.cuda.amp.autocast():
                    outputs = model(**inputs)
                    loss = criterion(outputs.logits, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item()

                # Log training progress
                if batch_idx % 10 == 0:
                    print(f"Epoch [{epoch+1}/{self.num_epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")

            # Calculate average training loss for the epoch
            avg_train_loss = total_loss / len(train_loader)

            # Validate the model
            val_loss = self.validate_wav2vec2(model, val_loader, criterion)
            print(f"🔥 Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f} | Validation Loss: {val_loss:.4f}")

            # Early stopping and model saving
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0

                # Save the model and processor
                os.makedirs("models", exist_ok=True)
                model.save_pretrained(os.path.join("models", "wav2vec2_finetuned"))
                processor.save_pretrained(os.path.join("models", "wav2vec2_processor"))
                print("✅ Model Saved (Best Validation Loss)")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("⏹️ Early Stopping Triggered!")
                    break

    def validate_wav2vec2(self, model, val_loader, criterion):
        """
        Validate the Wav2Vec2 model on the validation set.
        
        Args:
            model: The Wav2Vec2 model.
            val_loader: DataLoader for the validation set.
            criterion: Loss function.
        
        Returns:
            float: Average validation loss.
        """
        model.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for inputs, labels, _ in val_loader:
                inputs = {key: val.to(self.device, non_blocking=True) for key, val in inputs.items()}
                labels = labels.to(self.device, non_blocking=True)

                outputs = model(**inputs)
                loss = criterion(outputs.logits, labels)
                total_val_loss += loss.item()

        return total_val_loss / len(val_loader)


if __name__ == "__main__":
    # Define paths
    dataset_root = "data/processed/Dataset"

    # Initialize the classifier
    classifier = AudioClassifier(dataset_root)

    # Train XGBoost
    classifier.train_xgboost("yamnet_features.pkl")

    # Train Wav2Vec2
    classifier.train_wav2vec2()






