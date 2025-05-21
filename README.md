# 🧠 Lively vs Calm Episode Classifier using VideoMAE + CLAP

This project uses **pretrained AI models** to analyze short video clips and determine which one is more socially active ("lively") and which is quieter or more passive ("calm"). It combines advanced **vision and audio understanding** models: [VideoMAE](https://huggingface.co/docs/transformers/en/model_doc/videomae) and [CLAP](https://huggingface.co/docs/transformers/v4.51.3//model_doc/clap).

---

## 📦 What this project does

Given two `.mp4` video files:

* It automatically splits them into separate **audio** and **video** components.
* It uses:

  * **VideoMAE** to extract a vector from the visual scene.
  * **CLAP** to extract a vector from the sound.
* It combines both into a rich 1536-dimensional **multimodal vector**.
* It compares both episodes and:

  * Prints which one is **lively** and which is **calm**.
  * Shows a similarity score (how close the two scenes are).

---

## ✅ How to use

### 1. Clone and Install Requirements

```bash
pip install torch torchvision torchaudio ffmpeg-python transformers opencv-python scikit-learn
```

Make sure you have `ffmpeg` installed on your system:

```bash
# Ubuntu
sudo apt install ffmpeg

# macOS (with brew)
brew install ffmpeg
```

### 2. Place Your Video Files

You need two short `.mp4` clips:

* One that is **lively** (e.g. people talking, laughing)
* One that is **calm** (e.g. a quiet desk scene)

Put them in a folder and update these two lines in the script:

```python
lively_video, lively_audio = extract_audio_video_components("/path/to/lively.mp4", "lively")
calm_video, calm_audio     = extract_audio_video_components("/path/to/calm.mp4", "calm")
```

### 3. Run the Code

```bash
python your_script_name.py
```

You’ll see output like:

```
👉 Lively.mp4 is the LIVELY scene
👉 Calm.mp4 is the CALM scene
Similarity between the two: 0.234
```

---

## 💠 How It Works (Simple View)

* **VideoMAE** summarizes what’s happening visually.
* **CLAP** summarizes what’s happening in the sound.
* Each episode becomes a "memory vector".
* The code compares those vectors and looks at **audio energy** to decide which is more active.

---

## 📘 Model Docs

* 🔎 **VideoMAE (Masked Autoencoders for Video):**
  [https://huggingface.co/docs/transformers/en/model\_doc/videomae](https://huggingface.co/docs/transformers/en/model_doc/videomae)

* 🎧 **CLAP (Contrastive Language-Audio Pretraining):**
  [https://huggingface.co/docs/transformers/v4.51.3//model\_doc/clap](https://huggingface.co/docs/transformers/v4.51.3//model_doc/clap)

---

## 🧪 Suggested Use Cases

* Test how different social environments look and sound to an AI.
* Simulate perception for social robots.
* Explore how multimodal models "understand" episodes.
* Build a foundation for self-organizing memory or episode labeling in robotics.

---

## ✉️ Questions or Help?

Open an issue or contact the author to ask for usage examples or custom extensions.

Enjoy exploring visual+audio cognition! 🚀
