# CIFAR-10-C Robustness Evaluation: ResNet-18 and ViT-Small

This repository contains the evaluation suite used to compare the performance of a distilled ResNet-18 against a Vision Transformer (ViT) on the CIFAR-10-C benchmark. 

## The Pivot from ViT-Tiny to ViT-Small
Our original project plan involved a comparison using ViT-Tiny to match the parameter count of ResNet-18. However, during testing, we found that the available weights for ViT-Tiny were not fine-tuned for CIFAR-10, leading to poor baseline results. We pivoted to a ViT-Small ($224 \times 224$ resolution) to ensure a valid scientific comparison. This change increased the inference time significantly but provided a stable baseline for our robustness tests.

## Key Technical Details
* **Parameter Disparity:** We compared a 0.27M parameter ResNet-18 (optimized for $32 \times 32$) against a 21.67M parameter ViT-Small.
* **Up-sampling:** To accommodate the ViT's requirement for $224 \times 224$ inputs, we implemented a bilinear interpolation pipeline in `data_loader.py`.
* **Compute:** The ResNet-18 inference took approximately 1 hour, while the ViT-Small required a 12-hour compute cycle for the full 75-test benchmark.

## Repository Layout
* `main.py`: Main entry point for the benchmark suite.
* `src/data_loader.py`: Handles dataset loading, severity slicing, and image resizing.
* `src/models.py`: Model initialization using `transformers` and `safetensors`.
* `src/evaluate.py`: The logic for running evaluations across all 15 corruptions and 5 severity levels.

## Setup Instructions

### 1. Requirements
Install the necessary libraries via pip:
```bash
pip install torch torchvision transformers safetensors numpy matplotlib
```

## 2. Manual Dataset Download (Required)
The CIFAR-10-C dataset files are approximately 2.9GB and are excluded from this repository via .gitignore. You must set them up manually to run the experiments:

Download: Access the CIFAR-10-C Zenodo page and download the dataset.

Directory Structure: Create a folder named data/ in the root of this repository.

Extraction: Extract the .npy files into the data/ folder.

Verification: Ensure your directory matches the following structure:

Evaluating-Vision-Transformer-Robustness/
├── data/CIFAR-10-C
│   ├── labels.npy
│   ├── fog.npy
│   ├── snow.npy
│   ├── glass_blur.npy
│   └── ... (all 15 corruption files)
├── src/
├── main.py
└── README.md


## 3. Running the Experiment
Once the data is in place, you can run the full benchmark for both models. The script iterates through the dataset, calculates accuracy for every corruption/severity combination, and saves the results to JSON files for later visualization.

Run the execution engine:

```bash
python main.py
```

# Results Summary
The ViT-Small demonstrated a significant robustness advantage in the Blur and Digital categories. Our analysis suggests that the ViT's global attention mechanism allows it to maintain shape recognition when local pixel data is corrupted, whereas the CNN's local filters fail once fine textures are obscured.