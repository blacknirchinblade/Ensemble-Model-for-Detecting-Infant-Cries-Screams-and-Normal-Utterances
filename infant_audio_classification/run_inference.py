import os
import pickle
import numpy as np
import librosa
import torch
import pyaudio
import wave
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
import tensorflow_hub as hub
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
from src.training.dataset import AudioDataset
from torch.utils.data import DataLoader

class AudioInference:
    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.label_map = {0: "cry", 1: "scream", 2: "noncry"}
        self.target_sample_rate = 16000
        self.max_length = 16000 * 5  # 5 seconds
        
        # Load models
        self.load_models()
        
        # Initialize YAMNet
        self.yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
        
    def load_models(self):
        """Load trained models from disk"""
        # Load XGBoost
        with open(os.path.join("models", "xgboost_model.pkl"), "rb") as f:
            self.xgb_model = pickle.load(f)
            
        # Load Wav2Vec2
        self.processor = Wav2Vec2Processor.from_pretrained(os.path.join("models", "wav2vec2_processor"))
        self.wav2vec2_model = Wav2Vec2ForSequenceClassification.from_pretrained(
            os.path.join("models", "wav2vec2_finetuned")
        ).to(self.device)
        self.wav2vec2_model.eval()

    def preprocess_audio(self, audio):
        """Preprocess raw audio waveform"""
        # Trim silence
        trimmed, _ = librosa.effects.trim(audio, top_db=20)
        
        # Pad/trim to 5 seconds
        if len(trimmed) < self.max_length:
            padded = np.pad(trimmed, (0, self.max_length - len(trimmed)), 
                            mode='constant')
        else:
            padded = trimmed[:self.max_length]
        return padded

    def extract_yamnet_features(self, audio):
        """Extract YAMNet features for XGBoost"""
        scores, embeddings, _ = self.yamnet_model(audio)
        return np.mean(embeddings.numpy(), axis=0)  # Mean pooling

    def predict_single(self, audio_input):
        """
        Make prediction on single audio input (file path or numpy array)
        Returns: Tuple (ensemble_pred, wav2vec2_pred, xgb_pred)
        """
        # Load and preprocess audio
        if isinstance(audio_input, str):
            audio, _ = librosa.load(audio_input, sr=self.target_sample_rate)
        else:
            audio = audio_input
            
        processed_audio = self.preprocess_audio(audio)

        # Create real-time dataset
        dataset = AudioDataset(
            real_time_audio=processed_audio,
            processor=self.processor,
            max_length=self.max_length
        )
        dataloader = DataLoader(dataset, batch_size=1)

        # Extract features
        with torch.no_grad():
            for inputs, _, _ in dataloader:
                # Wav2Vec2 prediction
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                logits = self.wav2vec2_model(**inputs).logits
                wav_probs = torch.softmax(logits, dim=1).cpu().numpy()
                wav_pred = np.argmax(wav_probs, axis=1)[0]

                # XGBoost prediction
                yamnet_feature = self.extract_yamnet_features(processed_audio)
                xgb_probs = self.xgb_model.predict_proba([yamnet_feature])
                xgb_pred = np.argmax(xgb_probs, axis=1)[0]

                # Ensemble prediction (weighted average)
                ensemble_probs = (0.7 * wav_probs) + (0.3 * xgb_probs)
                ensemble_pred = np.argmax(ensemble_probs, axis=1)[0]

        return (
            self.label_map[ensemble_pred],
            self.label_map[wav_pred],
            self.label_map[xgb_pred],
            {
                "ensemble_probs": ensemble_probs.squeeze(),
                "wav2vec2_probs": wav_probs.squeeze(),
                "xgb_probs": xgb_probs.squeeze()
            }
        )

    def real_time_inference(self, audio_chunk):
        """For real-time streaming audio (5-15 second chunks)"""
        if len(audio_chunk) < self.target_sample_rate * 5:
            raise ValueError("Audio chunk too short (min 5 seconds)")
            
        return self.predict_single(audio_chunk)

    def record_audio(self, output_file="realtime_audio.wav", duration=5):
        """Records audio from the microphone and saves it as a WAV file"""
        chunk = 1024  
        format = pyaudio.paInt16  
        channels = 1  
        rate = self.target_sample_rate  

        p = pyaudio.PyAudio()
        stream = p.open(format=format, channels=channels, rate=rate, input=True, frames_per_buffer=chunk)

        print("🎤 Recording...")
        frames = []
        for _ in range(0, int(rate / chunk * duration)):
            data = stream.read(chunk)
            frames.append(data)

        print("✅ Recording complete!")

        stream.stop_stream()
        stream.close()
        p.terminate()

        wf = wave.open(output_file, "wb")
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(format))
        wf.setframerate(rate)
        wf.writeframes(b"".join(frames))
        wf.close()

        return output_file

    def batch_inference(self, folder_path):
        """Process all audio files in a folder"""
        results = {}
        for filename in os.listdir(folder_path):
            if filename.endswith(".wav"):
                file_path = os.path.join(folder_path, filename)
                results[filename] = self.predict_single(file_path)
        return results


class AudioGUI:
    def __init__(self, root, inferencer):
        self.inferencer = inferencer
        self.root = root
        self.root.title("Audio Classification")
        
        tk.Label(root, text="Select an audio file:").pack()
        
        self.file_button = tk.Button(root, text="Choose File", command=self.select_file)
        self.file_button.pack()

        self.record_button = tk.Button(root, text="Record Audio", command=self.record_and_predict)
        self.record_button.pack()

        self.batch_button = tk.Button(root, text="Batch Process Folder", command=self.batch_process)
        self.batch_button.pack()

        self.result_label = tk.Label(root, text="Result: ", font=("Arial", 12, "bold"))
        self.result_label.pack()

    def select_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav")])
        if file_path:
            pred, wav_pred, xgb_pred, _ = self.inferencer.predict_single(file_path)
            self.result_label.config(text=f"Prediction: {pred} (Wav2Vec2: {wav_pred}, XGBoost: {xgb_pred})")

    def record_and_predict(self):
        output_file = self.inferencer.record_audio()
        pred, wav_pred, xgb_pred, _ = self.inferencer.predict_single(output_file)
        self.result_label.config(text=f"Prediction: {pred} (Wav2Vec2: {wav_pred}, XGBoost: {xgb_pred})")

    def batch_process(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            results = self.inferencer.batch_inference(folder_path)
            messagebox.showinfo("Batch Processing", f"Processed {len(results)} files successfully!")


if __name__ == "__main__":
    inferencer = AudioInference()
    
    root = tk.Tk()
    app = AudioGUI(root, inferencer)
    root.mainloop()
