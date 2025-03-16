# 3D-Human-Image-Construction-from-Video

```markdown
# Data Acquisition and Preprocessing

This guide outlines the steps required to prepare the dataset and run the preprocessing pipeline for the project.

## Step 1: Download the Dataset
Download the dataset from the following link:  
[People Snapshot Dataset](https://graphics.tu-bs.de/people-snapshot)

## Step 2: Extract Images
Run the following command to extract images from the dataset:

```bash
python image_extraction.py
```

## Step 3: Apply Laplacian Filter
To obtain unblurred images using the Laplacian filter, run:

```bash
python laplacian_filter.py
```

## Step 4: Clone U-2-Net Repository
Clone the U-2-Net repository by running:

```bash
git clone https://github.com/NathanUA/U-2-Net.git
```

## Step 5: Download the U-2-Net Model
Download the `u2net.pth` (173.6 MB) file from the [U-2-Net repository](https://github.com/NathanUA/U-2-Net).

## Step 6: Place the Model File
Move the downloaded `u2net.pth` file into the `saved_model` folder inside the U-2-Net repository:

```bash
mv u2net.pth U-2-Net/saved_models/
```

## Step 7: Replace and Run `u2net_test.py`
Replace the existing `u2net_test.py` file in the U-2-Net repository with the customized version from this repository. Then run the following command:

```bash
python u2net_test.py
```

---

Following these steps will complete the data acquisition and preprocessing pipeline.
```
