# 🎬 Video Audio Embedding

This project uses multimodal AI models to automatically determine whether two video clips represent **lively** or **calm** scenes, by analyzing both the video and audio content. It combines:

- 🧠 [VideoMAE](https://huggingface.co/docs/transformers/en/model_doc/videomae) – for video understanding
- 🔊 [CLAP](https://huggingface.co/docs/transformers/v4.51.3//model_doc/clap) – for semantic audio representation

---

## ✅ What it does

- Splits video into separate **visual** and **audio** components
- Embeds video using **VideoMAE**
- Embeds audio using **CLAP**
- Combines both into a rich 1536-dimensional feature vector
- Compares two videos:
  - Determines which one is **LIVELY** and which is **CALM**
  - Outputs a similarity score between the two scenes

---

## 🧾 Requirements

Make sure your environment has these packages:

```txt
torch
ffmpeg-python
torchaudio
opencv-python-headless
numpy
scikit-learn
transformers
````

Install all dependencies using:

```bash
pip install -r requirements.txt
```

---

## 📂 File structure

```
Video-Audio-Embedding-test/
├── main.py                 # Main script
├── requirements.txt        # Dependency list
├── lively.mp4              # Example lively scene (group talking)
├── calm.mp4                # Example calm scene (quiet workspace)
```

---

## 🚀 How to run

1. Place your two input videos in the repo folder:

   * `lively.mp4` (lively, busy scene)
   * `calm.mp4` (quiet or still scene)

2. Run the main script:

```bash
python main.py
```

---

## 📈 Output

The script will:

* Extract and process video/audio from each clip
* Compute embeddings using pretrained models
* Print which video is LIVELY and which is CALM
* Show a similarity score (0 = different, 1 = very similar)

Example output:

```
✅ Lively.mp4 is the LIVELY scene
✅ Calm.mp4 is the CALM scene

Similarity between the two: 0.778
```

---

## 📸 Recording tips (if you create your own videos)

* Resolution: **640×480 (VGA)**
* Length: **5–15 seconds**
* Use a static camera (like a tripod)
* Lively video: people talking, laughter, movement
* Calm video: stillness, light ambient sounds

---

## 📚 Model Documentation

* [VideoMAE on Hugging Face](https://huggingface.co/docs/transformers/en/model_doc/videomae)
* [CLAP on Hugging Face](https://huggingface.co/docs/transformers/v4.51.3//model_doc/clap)

---

## 🙋‍♂️ Got questions?

Open an issue or reach out via GitHub! Contributions are welcome.
