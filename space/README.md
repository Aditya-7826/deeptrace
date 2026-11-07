---
title: DeepTrace
emoji: 🔍
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Deepfake face detector with Grad-CAM heatmaps
---

# DeepTrace demo

Upload a photo. DeepTrace finds each face and estimates how likely it is to be a deepfake. A Grad-CAM heatmap then shows which regions drove each verdict.

- **Scoring:** the ONNX model, INT8 when quantization passed the accuracy gate, otherwise FP32. The model in use is shown at the bottom of the page.
- **Heatmaps:** the FP32 PyTorch model, loaded the first time a heatmap is drawn.
- **Hardware:** runs on the free CPU tier.
- **Weights:** downloaded from the GitHub Release at startup and verified with SHA-256.

> **Note:** this is a probabilistic estimate. It may be wrong on unseen generators, heavy compression or screenshots. Don't use it as the only evidence about a real person.

**Privacy:** uploads are processed for the request and are not stored by the app. Usage analytics are switched off.

Code, metrics, limitations and the Chrome extension: <https://github.com/Aditya-7826/deeptrace>

The example images are real, public-domain photos (U.S. government works). Their sources are listed in `space/examples/SOURCES.md` in the repository.
