import os
import numpy as np
import librosa
import soundfile as sf
import tensorflow_hub as hub
import torch
import pickle
from tqdm import tqdm
from transformers import Wav2Vec2Processor, Wav2Vec2Model

class AudioFeatureExtractor:
    def __init__(self, dataset_root, features_root, target_sample_rate=16000, model_name="facebook/wav2vec2-base"):
        """
        Initializes the feature extractor.

        Args:
            dataset_root (str): Path to the preprocessed dataset.
            features_root (str): Path to save extracted features.
            target_sample_rate (int): Sampling rate for feature extraction.
            model_name (str): Pretrained model name for Wav2Vec2.
        """
        self.dataset_root = dataset_root
        self.features_root = features_root
        self.target_sample_rate = target_sample_rate
        self.dataset_folders = ["experimental/cry", "experimental/scream", "control/noncry"]
        self.label_map = {"cry": 0, "scream": 1, "noncry": 2}
        
        # ✅ Load YAMNet model
        self.yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
        
        # ✅ Load Wav2Vec2 model
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.wav2vec2_model = Wav2Vec2Model.from_pretrained(model_name)
        self.wav2vec2_model.gradient_checkpointing_enable()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.wav2vec2_model.to(self.device)

    def extract_yamnet_features(self, file_path):
        """Extracts YAMNet embeddings from an audio file."""
        try:
            audio, sr = librosa.load(file_path, sr=self.target_sample_rate)
            scores, embeddings, _ = self.yamnet_model(audio)
            return np.mean(embeddings.numpy(), axis=0)  # Mean pooling
        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")
            return None

    def extract_wav2vec2_features(self, file_path):
        """Extracts Wav2Vec2 embeddings from an audio file."""
        try:
            audio, sr = librosa.load(file_path, sr=self.target_sample_rate)
            inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
            inputs = {key: val.to(self.device) for key, val in inputs.items()}
            
            with torch.no_grad():
                outputs = self.wav2vec2_model(**inputs)
            
            return outputs.last_hidden_state.mean(dim=1).cpu().numpy()  # Mean pooling
        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")
            return None
    
    def extract_realtime_wav2vec2_features(self, audio_np):
        """
        Extracts Wav2Vec2 embeddings from real-time audio input.
        
        Args:
            audio_np (numpy array): Raw audio waveform.
        
        Returns:
            numpy array: Extracted feature vector.
        """
        try:
            inputs = self.processor(audio_np, sampling_rate=16000, return_tensors="pt", padding=True)
            inputs = {key: val.to(self.device) for key, val in inputs.items()}
            
            with torch.no_grad():
                outputs = self.wav2vec2_model(**inputs)
            
            return outputs.last_hidden_state.mean(dim=1).cpu().numpy()  # Mean pooling
        except Exception as e:
            print(f"❌ Error processing real-time audio: {e}")
            return None
    
    def process_dataset(self, model_type="yamnet", real_time=False, audio_np=None):
        """
        Processes dataset to extract features for the specified model.
        If real_time=True, extracts features from a NumPy array instead of files.

        Args:
            model_type (str): Model type ("yamnet" or "wav2vec2").
            real_time (bool): If True, process audio from NumPy array.
            audio_np (numpy array): Raw audio input for real-time processing.
        """
        if real_time and audio_np is not None:
            if model_type == "wav2vec2":
                return self.extract_realtime_wav2vec2_features(audio_np)
            else:
                print("❌ Real-time YAMNet extraction is not implemented.")
                return None

        # ✅ Normal file-based processing (batch mode)
        features, labels = [], []
        feature_save_path = os.path.join(self.features_root, f"{model_type}_features.pkl")

        if os.path.exists(feature_save_path):
            print(f"🔄 Replacing existing {model_type} features: {feature_save_path}")
            os.remove(feature_save_path)  # ✅ Delete existing feature file

        for folder in self.dataset_folders:
            folder_path = os.path.join(self.dataset_root, folder)
            for file in tqdm(os.listdir(folder_path), desc=f"Processing {folder}"):
                if file.endswith(".wav"):
                    file_path = os.path.join(folder_path, file)
                    feature_vector = None

                    if model_type == "yamnet":
                        feature_vector = self.extract_yamnet_features(file_path)
                    elif model_type == "wav2vec2":
                        feature_vector = self.extract_wav2vec2_features(file_path)

                    if feature_vector is not None:
                        features.append(feature_vector)
                        labels.append(self.label_map[folder.split("/")[-1]])

        # ✅ Convert to NumPy Arrays
        features = np.array(features)
        labels = np.array(labels)

        # ✅ Save Features to File (Always replaces existing ones)
        os.makedirs(self.features_root, exist_ok=True)
        with open(feature_save_path, "wb") as f:
            pickle.dump((features, labels), f)

        print(f"✅ {model_type} feature extraction completed! Features saved as '{feature_save_path}'.")

# ✅ Usage
if __name__ == "__main__":
    dataset_root = r"data/processed/Dataset"
    features_root = r"data/features"

    extractor = AudioFeatureExtractor(dataset_root, features_root)
    
    # Extract features for YAMNet
    extractor.process_dataset(model_type="yamnet")
    
    # Extract features for Wav2Vec2
    extractor.process_dataset(model_type="wav2vec2")
