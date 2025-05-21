import torch
import os
import ffmpeg
import torchaudio
import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import (
    VideoMAEImageProcessor, VideoMAEModel,
    ClapProcessor, ClapModel
)

class EpisodeEmbedder:
    def __init__(self, device=None):
        # Initialize the device (GPU if available, otherwise CPU)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load VideoMAE model and processor for video feature extraction
        self.video_processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")
        self.video_model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base").to(self.device).eval()

        # Load CLAP model and processor for audio feature extraction
        self.audio_processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
        self.audio_model = ClapModel.from_pretrained("laion/clap-htsat-unfused").to(self.device).eval()

    def extract_video_frames(self, video_path, num_frames=16):
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Select evenly spaced frame indices
        frame_idxs = np.linspace(0, total_frames - 1, num_frames).astype(int)

        frames = []
        idx_set = set(frame_idxs)
        for idx in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                continue
            # Collect frames at the selected indices
            if idx in idx_set:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        # Process frames into tensors for the VideoMAE model
        pixel_values = self.video_processor(frames, return_tensors="pt")["pixel_values"]
        return pixel_values.to(self.device)

    def extract_audio(self, audio_path):
        # Load the audio file and resample to 48kHz
        audio, sr = torchaudio.load(audio_path)
        audio = torchaudio.functional.resample(audio, sr, 48000)
        # Convert stereo audio to mono if necessary
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        # Limit audio to the first 10 seconds (480,000 samples at 48kHz)
        audio = audio[..., :480000]
        return audio

    def embed_episode(self, video_path, audio_path):
        with torch.no_grad():
            # Extract video frames and compute video features
            video_tensor = self.extract_video_frames(video_path)
            video_features = self.video_model(pixel_values=video_tensor).last_hidden_state[:, 0]

            # Extract audio and compute audio features
            audio_tensor = self.extract_audio(audio_path)
            # Convert to 1D numpy array for CLAP
            audio_np = audio_tensor.squeeze().cpu().numpy()
            audio_inputs = self.audio_processor(audios=audio_np, return_tensors="pt", sampling_rate=48000).to(self.device)
            audio_features = self.audio_model.get_audio_features(**audio_inputs)

            # Combine video and audio features and normalize
            combined_features = torch.cat([video_features, audio_features], dim=-1)
            combined_features = torch.nn.functional.normalize(combined_features, dim=-1)

        return combined_features.cpu().numpy()


# Function to extract video and audio components from a video file
def extract_audio_video_components(input_video_path, output_stem):
    import sys
    video_out = f"{output_stem}_video.mp4"
    audio_out = f"{output_stem}_audio.wav"

    # Extract video-only component
    try:
        ffmpeg.input(input_video_path).output(video_out, an=None).overwrite_output().run(quiet=True)
    except ffmpeg.Error as e:
        print(f"Error extracting video: {e.stderr.decode() if hasattr(e, 'stderr') else e}", file=sys.stderr)
        raise

    # Extract audio-only component at 48kHz for CLAP
    try:
        ffmpeg.input(input_video_path).output(audio_out, vn=None, acodec='pcm_s16le', ar=48000).overwrite_output().run(quiet=True)
    except ffmpeg.Error as e:
        print(f"Error extracting audio: {e.stderr.decode() if hasattr(e, 'stderr') else e}", file=sys.stderr)
        raise

    return video_out, audio_out

if __name__ == '__main__':
    embedder = EpisodeEmbedder()

    # Step 1: Extract video and audio components from input files
    lively_video, lively_audio = extract_audio_video_components("/home/nima/test-env/lively.mp4", "lively")
    calm_video, calm_audio = extract_audio_video_components("/home/nima/test-env/calm.mp4", "calm")

    # Step 2: Compute embeddings for both episodes
    lively_embedding = embedder.embed_episode(lively_video, lively_audio)
    calm_embedding = embedder.embed_episode(calm_video, calm_audio)

    # Step 3: Estimate energy levels from audio files
    def estimate_energy(wav_path):
        # Load audio and resample to 48kHz
        audio, sr = torchaudio.load(wav_path)
        audio = torchaudio.functional.resample(audio, sr, 48000)
        # Compute mean squared energy
        energy = (audio**2).mean().item()
        return energy

    lively_energy = estimate_energy(lively_audio)
    calm_energy = estimate_energy(calm_audio)

    # Step 4: Label scenes based on energy levels
    if lively_energy > calm_energy:
        print("✅ Lively.mp4 is the LIVELY scene")
        print("✅ Calm.mp4 is the CALM scene")
    else:
        print("✅ Calm.mp4 is the LIVELY scene")
        print("✅ Lively.mp4 is the CALM scene")

    # Step 5: Compute and display similarity score between the two episodes
    similarity_score = cosine_similarity(lively_embedding, calm_embedding)[0][0]
    print(f"\nSimilarity between the two: {similarity_score:.3f}")
