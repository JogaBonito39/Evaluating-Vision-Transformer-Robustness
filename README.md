# CIFAR-10-C Robustness Benchmark: ResNet-18 vs. ViT-Tiny

This project evaluates and compares the robustness of Convolutional Neural Networks (ResNet-18) and Vision Transformers (ViT-Tiny) against 15 common image corruptions across 5 levels of severity.

## Getting Started

### 1. Clone the Repository
```bash
git clone [https://github.com/](https://github.com/)[YOUR_USERNAME]/CIFAR10-C_Robustness.git
cd CIFAR10-C_Robustness
```

### 2. Install Dependencies
```
pip install -r requirements.txt
```

### 3. Data Setup
```The CIFAR-10-C dataset files are too large for Git. You must set them up manually:

Download the dataset (approx 2.9GB) from Zenodo.

Create a folder named data/ in the root directory.

Extract the .npy files into data/.

Your structure should look like this:

data/labels.npy

data/fog.npy

data/snow.npy

... (all 15 corruption files)
```

### Project Structure
```src/data_loader.py: Handles severity slicing (1-5) and image resizing to 224x224.

src/models.py: Initializes pretrained ResNet-18 and ViT-Tiny.

src/evaluate.py: The core engine that runs the 75-test benchmark loop.

main.py: The entry point to run the full experiment.
```

### Running ther Experiment
```
To run the full benchmark for both models and generate the results JSON files, simply run:
python main.py
```

### Next Tasks

[ ] Run main.py to ensure paths and GPU/CPU detection work.

[ ] Develop visualization scripts in notebooks/ to process the generated .json files.

[ ] Calculate final Mean Corruption Error (mCE) scores.

---

### 3. How to add this to Git
Once you've created the file, use your branching workflow to upload it:

```bash
# 1. Stay on your feature branch or create a new one
git checkout -b update-readme

# 2. Add and commit
git add README.md
git commit -m "Add documentation and setup instructions for the team"

# 3. Merge to main
git checkout main
git merge update-readme

# 4. Push to GitHub
git push origin main

