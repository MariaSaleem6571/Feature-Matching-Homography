# Feature Matching and Homography Estimation

This repository provides feature matching and homography estimation for consecutive images using three different methods: **AKAZE** (traditional feature detector), **LoFTR** (transformer-based matching) and **Lightglue with Superpoint**.

## Features

- **Method Support**: Choose between AKAZE, LoFTR and Lightglue feature matching
- **Homography Estimation**: Compute 4x4 homography matrices between image pairs
- **CSV Export**: Export matches with keypoints and homography data
- **Visualization**: Optional match visualization
- **Batch Processing**: Process all consecutive image pairs in a directory

## Requirements

### Dependencies

```bash
pip install opencv-python
pip install numpy
pip install torch torchvision torchaudio  # CPU version
pip install kornia
pip install natsort
```

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd <repo-directory>
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python main.py --dir /path/to/your/images
```

### Method Selection

#### AKAZE Method (Default)
```bash
python main.py --dir /path/to/images --method akaze
```

#### LoFTR Method
```bash
python main.py --dir /path/to/images --method loftr
```

#### Lightglue Method
```bash
python main.py --dir /path/to/images --method lightglue
```

### Complete Example with All Options
```bash
python main.py --dir ./images --method loftr --threshold 25.0 --csv_dir ./results --visualize
```

## Command Line Arguments

| Argument | Type | Required | Default | Description                                              |
|----------|------|----------|---------|----------------------------------------------------------|
| `--dir` | str | ✅ | - | Directory containing input images                        |
| `--method` | str | ❌ | `akaze` | Feature matching method (`akaze`, `loftr` or `lightglue`) |
| `--threshold` | float | ❌ | `30.0` | Pixel distance threshold for filtering matches           |
| `--csv_dir` | str | ❌ | `matches_csv` | Directory to save output CSV files                       |
| `--visualize` | flag | ❌ | `False` | Show match visualizations (press any key to continue)    |


## Output Format

### CSV Structure

#### AKAZE Output
```
uuid, image1_index, image2_index, pair_id, method, threshold, x1, y1, x2, y2, confidence, desc1_0, desc1_1, ..., desc2_0, desc2_1, ..., r11, r12, r13, tx, r21, r22, r23, ty, r31, r32, r33, tz, h41, h42, h43, h44
```

#### LoFTR Output  
```
uuid, image1_index, image2_index, pair_id, method, threshold, x1, y1, x2, y2, confidence, r11, r12, r13, tx, r21, r22, r23, ty, r31, r32, r33, tz, h41, h42, h43, h44
```

#### Lightglue Output
```
uuid, image1_index, image2_index, pair_id, method, threshold, x1, y1, x2, y2, confidence, desc1_0, desc1_1, ..., desc2_0, desc2_1, ..., r11, r12, r13, tx, r21, r22, r23, ty, r31, r32, r33, tz, h41, h42, h43, h44
```

### Column Descriptions
- `uuid`: Unique identifier for each match
- `image1_index`, `image2_index`: Image pair indices
- `pair_id`: Concatenated image names
- `method`: Akaze, Loftr, Lightglue
- `threshold`: In px.
- `x1, y1`: Keypoint coordinates in first image
- `x2, y2`: Keypoint coordinates in second image 
- `confidence`
- `desc1_*`, `desc2_*`: Feature descriptors (AKAZE and Lightglue only)
- `r11-r33, tx, ty, tz`: Homography matrix elements
- `h41-h44`: Additional homography matrix elements

## File Structure

```
project/
├── main.py                 # Main execution script
├── registration_utils.py   # Utility functions
├── requirements.txt        # Dependencies
├── README.md              
├── images/                # Input images directory
└── matches_csv/           # Output CSV files
```

## Supported Image Formats

- `.png`
- `.jpg` 
- `.jpeg`

