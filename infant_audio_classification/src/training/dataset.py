import os
import pickle
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset



class AudioDataset(Dataset):
    def __init__(self, dataset_root=None, processor=None, feature_file=None, real_time_audio=None, label=None, max_length=16000*5):
        """
        Supports both real-time and batch dataset loading.

        Args:
            dataset_root (str): Path to preprocessed dataset (if using files).
            processor: Wav2Vec2 processor.
            feature_file (str): Path to YAMNet features (optional).
            real_time_audio (numpy array): Real-time audio input for inference.
            label (int): Label for real-time input (optional).
            max_length (int): Maximum allowed audio length in samples.
        """
        self.processor = processor
        self.max_length = max_length
        self.label_map = {"cry": 0, "scream": 1, "noncry": 2}
        self.audio_paths = []
        self.labels = []
        self.yamnet_features = {}

        # ✅ Real-time mode: Use directly provided audio
        if real_time_audio is not None:
            self.real_time_audio = real_time_audio
            self.labels.append(label if label is not None else -1)  # Unknown label
            return

        # ✅ Normal mode: Load from dataset root
        if dataset_root:
            for label_name, label in self.label_map.items():
                folder_path = os.path.join(dataset_root, "experimental" if label_name in ["cry", "scream"] else "control", label_name)
                if os.path.exists(folder_path):
                    for file_name in os.listdir(folder_path):
                        if file_name.endswith(".wav"):
                            self.audio_paths.append(os.path.join(folder_path, file_name))
                            self.labels.append(label)

        # ✅ Load YAMNet features (for XGBoost)
        if feature_file:
            try:
                feature_file_path = os.path.join(feature_file)
                if not os.path.exists(feature_file_path):  
                    raise FileNotFoundError(f"❌ Feature file not found: {feature_file_path}")
                
                with open(feature_file_path, "rb") as f:
                    yamnet_data, _ = pickle.load(f)
                self.yamnet_features = {idx: feature for idx, feature in enumerate(yamnet_data)}
            except Exception as e:
                print(f"❌ Error loading YAMNet features: {e}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if hasattr(self, "real_time_audio"):  # ✅ Handle real-time input
            audio = self.real_time_audio
        else:
            file_path = self.audio_paths[idx]
            audio, _ = librosa.load(file_path, sr=16000)

        # ✅ Ensure correct length
        if len(audio) < self.max_length:
            audio = np.pad(audio, (0, self.max_length - len(audio)), mode="constant")
        else:
            audio = audio[:self.max_length]

        # ✅ Convert to tensors
        inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
        inputs["input_values"] = inputs["input_values"].squeeze(0).to(torch.float32)  

        label = torch.tensor(self.labels[idx], dtype=torch.long).to(torch.int64)
        yamnet_feature = self.yamnet_features.get(idx, np.zeros(1024))

        return inputs, label, yamnet_feature
