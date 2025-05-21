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
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load VideoMAE
        self.video_processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")
        self.video_model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base").to(self.device).eval()

        # Load CLAP
        self.audio_processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
        self.audio_model = ClapModel.from_pretrained("laion/clap-htsat-unfused").to(self.device).eval()

    def extract_video_frames(self, video_path, num_frames=16):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idxs = np.linspace(0, total_frames - 1, num_frames).astype(int)

        frames = []
        idx_set = set(frame_idxs)
        for idx in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                continue
            if idx in idx_set:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        pixel_values = self.video_processor(frames, return_tensors="pt")["pixel_values"]
        return pixel_values.to(self.device)

    def extract_audio(self, audio_path):
        audio, sr = torchaudio.load(audio_path)
        audio = torchaudio.functional.resample(audio, sr, 48000)
        # Convert to mono if stereo
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        # Limit to first 10 seconds (480,000 samples at 48kHz)
        audio = audio[..., :480000]
        return audio

    def embed_episode(self, video_path, audio_path):
        with torch.no_grad():
            video_tensor = self.extract_video_frames(video_path)
            video_features = self.video_model(pixel_values=video_tensor).last_hidden_state[:, 0]

            audio_tensor = self.extract_audio(audio_path)
            # Convert to 1D numpy array for CLAP
            audio_np = audio_tensor.squeeze().cpu().numpy()
            audio_inputs = self.audio_processor(audios=audio_np, return_tensors="pt", sampling_rate=48000).to(self.device)
            audio_features = self.audio_model.get_audio_features(**audio_inputs)

            combined_features = torch.cat([video_features, audio_features], dim=-1)
            combined_features = torch.nn.functional.normalize(combined_features, dim=-1)

        return combined_features.cpu().numpy()


def extract_audio_video_components(input_video_path, output_stem):
    import sys
    video_out = f"{output_stem}_video.mp4"
    audio_out = f"{output_stem}_audio.wav"

    # Extract video-only
    try:
        ffmpeg.input(input_video_path).output(video_out, an=None).overwrite_output().run(quiet=True)
    except ffmpeg.Error as e:
        print(f"Error extracting video: {e.stderr.decode() if hasattr(e, 'stderr') else e}", file=sys.stderr)
        raise

    # Extract audio-only at 48 kHz for CLAP
    try:
        ffmpeg.input(input_video_path).output(audio_out, vn=None, acodec='pcm_s16le', ar=48000).overwrite_output().run(quiet=True)
    except ffmpeg.Error as e:
        print(f"Error extracting audio: {e.stderr.decode() if hasattr(e, 'stderr') else e}", file=sys.stderr)
        raise

    return video_out, audio_out

if __name__ == '__main__':
    embedder = EpisodeEmbedder()

    # Step 1: extract components
    lively_video, lively_audio = extract_audio_video_components("/home/nima/test-env/lively.mp4", "lively")
    calm_video, calm_audio = extract_audio_video_components("/home/nima/test-env/calm.mp4", "calm")

    # Step 2: embed episodes
    lively_embedding = embedder.embed_episode(lively_video, lively_audio)
    calm_embedding = embedder.embed_episode(calm_video, calm_audio)

    # Step 3: naive energy estimation from audio
    def estimate_energy(wav_path):
        audio, sr = torchaudio.load(wav_path)
        audio = torchaudio.functional.resample(audio, sr, 48000)
        energy = (audio**2).mean().item()
        return energy

    lively_energy = estimate_energy(lively_audio)
    calm_energy = estimate_energy(calm_audio)

    # Step 4: label based on energy
    if lively_energy > calm_energy:
        print("✅ Lively.mp4 is the LIVELY scene")
        print("✅ Calm.mp4 is the CALM scene")
    else:
        print("✅ Calm.mp4 is the LIVELY scene")
        print("✅ Lively.mp4 is the CALM scene")

    # Step 5: similarity score (optional)
    similarity_score = cosine_similarity(lively_embedding, calm_embedding)[0][0]
    print(f"\nSimilarity between the two: {similarity_score:.3f}")
