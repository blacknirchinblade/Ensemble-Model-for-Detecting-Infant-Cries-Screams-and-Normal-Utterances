import os
import librosa
import soundfile as sf
import numpy as np
from pydub import AudioSegment
import random
from librosa.effects import time_stretch, pitch_shift
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class AudioPreprocessor:
    def __init__(self, dataset_root, processed_root, target_sample_rate=16000, min_length=5000, max_length=15000, augment=False):
        """
        Initializes the preprocessor with dataset paths and parameters.

        Args:
            dataset_root (str): Path to the raw audio dataset.
            processed_root (str): Path to save processed audio files.
            target_sample_rate (int): Target sample rate for resampling.
            min_length (int): Minimum allowed duration (in milliseconds).
            max_length (int): Maximum allowed duration (in milliseconds).
            augment (bool): Whether to apply data augmentation.
        """
        self.dataset_root = dataset_root
        self.processed_root = processed_root
        self.target_sample_rate = target_sample_rate
        self.min_length = min_length
        self.max_length = max_length
        self.dataset_folders = ["experimental/cry", "experimental/scream", "control/noncry"]
        self.augment = augment  # Whether to apply augmentation or not

    def preprocess_audio(self, input_path, output_path):
        """
        Normalizes, converts to mono, trims silence, applies augmentation, and saves processed audio.

        Args:
            input_path (str): Path to the raw audio file.
            output_path (str): Path to save the processed file.
        """
        if self._is_already_processed(output_path):
            logging.info(f"⏩ Skipping (Already Processed): {output_path}")
            return

        try:
            # Load and preprocess audio
            audio, sr = librosa.load(input_path, sr=self.target_sample_rate, mono=True)
            trimmed_audio, _ = librosa.effects.trim(audio, top_db=20)

            # Check duration
            duration_ms = (len(trimmed_audio) / self.target_sample_rate) * 1000  # Convert to ms
            if duration_ms < self.min_length:
                logging.warning(f"🗑️ Skipped (Too Short): {input_path}")
                return

            # Apply augmentation if enabled
            if self.augment:
                trimmed_audio = self.apply_augmentation(trimmed_audio)

            # Save processed audio
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            sf.write(output_path, trimmed_audio, self.target_sample_rate)
            logging.info(f"✅ Processed: {output_path}")

        except Exception as e:
            logging.error(f"❌ Error processing {input_path}: {e}")

    def apply_augmentation(self, audio):
        """
        Apply augmentation techniques to the audio (e.g., time-stretching, pitch-shifting, noise addition).

        Args:
            audio (np.array): The audio signal to augment.

        Returns:
            np.array: The augmented audio.
        """
        # Time-stretching
        if random.random() > 0.5:
            stretch_rate = random.uniform(0.8, 1.2)  # Time-stretch factor
            audio = time_stretch(audio, rate=stretch_rate)

        # Pitch-shifting
        if random.random() > 0.5:
            pitch_shift_steps = random.randint(-3, 3)  # Pitch shift range (in semitones)
            audio = pitch_shift(audio, sr=self.target_sample_rate, n_steps=pitch_shift_steps)

        # Add background noise
        if random.random() > 0.5:
            noise_factor = random.uniform(0.005, 0.05)  # Noise magnitude
            noise = np.random.randn(len(audio))
            audio = audio + noise_factor * noise
            audio = np.clip(audio, -1.0, 1.0)  # Clip to valid audio range

        return audio

    def process_audio_length(self, input_path, output_path):
        """
        Ensures audio files are within min-max duration; splits long ones.

        Args:
            input_path (str): Path to the raw or processed audio file.
            output_path (str): Path to save the processed or split audio.
        """
        if self._is_already_processed(output_path):
            logging.info(f"⏩ Skipping (Already Processed): {output_path}")
            return

        try:
            audio = AudioSegment.from_wav(input_path)
            duration = len(audio)  # Get duration in ms

            # Skip if too short
            if duration < self.min_length:
                logging.warning(f"🗑️ Skipped (Too Short): {input_path}")
                return

            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Split long audio files
            if duration > self.max_length:
                start = 0
                base_name = output_path.replace(".wav", "")

                while start < duration:
                    end = min(start + np.random.randint(self.min_length, self.max_length), duration)
                    chunk = audio[start:end]
                    chunk_filename = f"{base_name}_chunk{start}.wav"  # Unique name for each chunk

                    if not os.path.exists(chunk_filename):  # Avoid overwriting
                        chunk.export(chunk_filename, format="wav")
                        logging.info(f"✅ Saved Split: {chunk_filename}")

                    start = end
            else:
                # Save the file if it's within the allowed duration
                audio.export(output_path, format="wav")
                logging.info(f"✅ Processed: {output_path}")

        except Exception as e:
            logging.error(f"❌ Error processing {input_path}: {e}")

    def _is_already_processed(self, output_path):
        """
        Check if the output file already exists.

        Args:
            output_path (str): Path to the output file.

        Returns:
            bool: True if the file already exists, False otherwise.
        """
        return os.path.exists(output_path)

    def apply_preprocessing(self):
        """Run preprocessing on all dataset folders."""
        for folder in self.dataset_folders:
            raw_folder_path = os.path.join(self.dataset_root, folder)
            processed_folder_path = os.path.join(self.processed_root, folder)

            os.makedirs(processed_folder_path, exist_ok=True)  # Ensure processed folder exists

            for root, _, files in os.walk(raw_folder_path):
                for file in files:
                    if file.endswith(".wav"):
                        input_path = os.path.join(root, file)
                        output_path = os.path.join(processed_folder_path, file)

                        self.preprocess_audio(input_path, output_path)  # Normalize & Trim
                        self.process_audio_length(output_path, output_path)  # Check length

        logging.info("🎯 All audio preprocessing completed!")


# ✅ Usage
if __name__ == "__main__":
    dataset_root = r"data/raw_audio/Dataset"
    processed_root = r"data/processed/Dataset"

    # Set augment=True to apply augmentation
    preprocessor = AudioPreprocessor(dataset_root, processed_root, augment=True)
    preprocessor.apply_preprocessing()