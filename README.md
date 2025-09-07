# Feature Matching and Homography Estimation

This repository provides feature matching and homography estimation for images using three different methods: **AKAZE** (traditional feature detector), **LoFTR** (transformer-based matching), and **LightGlue with SuperPoint** (state-of-the-art deep learning approach).

## Features

- **Triple Method Support**: Choose between AKAZE, LoFTR, and LightGlue feature matching
- **Flexible Input**: Process consecutive images from directory OR specific image pairs from CSV
- **Confidence-based Selection**: Select top/bottom keypoints based on confidence scores
- **Homography Estimation**: Compute 4x4 homography matrices between image pairs
- **CSV Export**: Export matches with keypoints and homography data to organized directories
- **Visualization**: Automatic match visualizations with color-coded confidence levels
- **Batch Processing**: Process multiple image pairs efficiently

## Requirements

### Dependencies

```bash
pip install opencv-python
pip install numpy
pip install torch torchvision torchaudio  # CPU version
pip install kornia
pip install natsort
```

### For GPU acceleration (optional)
```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1  
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### For LightGlue
You'll also need the LightGlue repository:
```bash
git clone https://github.com/cvg/LightGlue.git
# Add LightGlue to your Python path or install it
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

### Processing Directory (Consecutive Images)

Process all consecutive image pairs in a directory:

```bash
python main.py --dir /path/to/your/images
```

### Processing Specific Pairs (CSV Input)

Create a CSV file with columns `image1,image2`:
```csv
image1,image2
/path/to/img1.jpg,/path/to/img2.jpg
/path/to/img3.jpg,/path/to/img4.jpg
```

Then run:
```bash
python main.py --pairs_csv /path/to/pairs.csv
```

### Method Selection

#### AKAZE Method (Default)
```bash
python main.py --dir /path/to/images --method akaze
```

#### LoFTR Method
```bash
python main.py --pairs_csv /path/to/pairs.csv --method loftr
```

#### LightGlue Method
```bash
python main.py --dir /path/to/images --method lightglue
```

### Complete Example with All Options
```bash
python main.py --dir ./images --method lightglue --num_top_kp 20 --num_bottom_kp 10 --logs_dir ./logs --vis_dir ./visualizations --keypoints_dir ./keypoints
```

## Command Line Arguments

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--dir` | str | ✅ (if no `--pairs_csv`) | - | Directory containing input images for consecutive matching |
| `--pairs_csv` | str | ✅ (if no `--dir`) | - | CSV file with columns `image1,image2` for pairwise matching |
| `--method` | str | ❌ | `akaze` | Feature matching method (`akaze`, `loftr`, or `lightglue`) |
| `--logs_dir` | str | ❌ | `experiment_logs` | Directory to save experiment log CSVs |
| `--keypoints_dir` | str | ❌ | `keypoints_csvs` | Directory to save individual keypoint CSVs |
| `--vis_dir` | str | ❌ | `visualizations` | Directory to save match visualizations |
| `--num_top_kp` | int | ❌ | `None` | Number of highest confidence keypoints to keep (shown in green) |
| `--num_bottom_kp` | int | ❌ | `None` | Number of lowest confidence keypoints to keep (shown in red) |

## Methods Comparison

### AKAZE
- **Type**: Traditional feature detector/descriptor
- **Speed**: Fast
- **Memory**: Low
- **Best for**: Quick processing, traditional computer vision pipelines

### LoFTR
- **Type**: Transformer-based detector-free matching
- **Speed**: Slower (especially on CPU)
- **Memory**: Higher
- **Best for**: High-quality matches, challenging scenarios

### LightGlue + SuperPoint
- **Type**: State-of-the-art deep learning approach
- **Speed**: Fast (optimized for real-time)
- **Memory**: Moderate
- **Best for**: Best balance of speed and accuracy, works well both indoor and outdoor

## Output Structure

The tool creates three types of outputs:

### 1. Experiment Log CSV
Located in `logs_dir/experiment_log.csv`:
```
uuid, image1_name, image2_name, keypoints_csv_path, visualization_path, r11, r12, r13, tx, r21, r22, r23, ty, r31, r32, r33, tz, h41, h42, h43, h44
```

### 2. Individual Keypoint CSVs
Located in `keypoints_dir/keypoints_{uuid}.csv`:
```
uuid, image1_index, image2_index, pair_id, method, x1, y1, x2, y2, confidence
```

### 3. Visualizations
Located in `vis_dir/vis_{uuid}.png`:
- **Green lines**: Top confidence matches
- **Red lines**: Bottom confidence matches

## Column Descriptions

### Experiment Log
- `uuid`: Unique identifier for each image pair
- `image1_name`, `image2_name`: Original image filenames
- `keypoints_csv_path`: Path to detailed keypoints CSV
- `visualization_path`: Path to match visualization image
- `r11-r33, tx, ty, tz, h41-h44`: 4x4 homography matrix elements

### Keypoint CSVs
- `uuid`: Unique identifier for each match
- `image1_index`, `image2_index`: Image pair indices
- `pair_id`: Concatenated image names
- `method`: Feature matching method used
- `x1, y1`: Keypoint coordinates in first image
- `x2, y2`: Keypoint coordinates in second image
- `confidence`: Match confidence score (higher = better)

## File Structure

```
project/
├── main.py                 # Main execution script
├── registration_utils.py   # Utility functions
├── requirements.txt        # Dependencies
├── README.md              # This file
├── experiment_logs/        # Experiment summary CSVs
├── keypoints_csvs/        # Individual keypoint CSVs
└── visualizations/        # Match visualization images
```

## Supported Image Formats

- `.png`
- `.jpg` 
- `.jpeg`

## Examples

### Process consecutive images with confidence filtering
```bash
python main.py --dir ./dataset/images --method lightglue --num_top_kp 15 --num_bottom_kp 5
```

### Process specific pairs with LoFTR
```bash
python main.py --pairs_csv ./pairs.csv --method loftr --logs_dir ./loftr_results
```

### Compare methods on same dataset
```bash
python main.py --dir ./images --method akaze --logs_dir ./akaze_results
python main.py --dir ./images --method loftr --logs_dir ./loftr_results  
python main.py --dir ./images --method lightglue --logs_dir ./lightglue_results
```


## Confidence-Based Selection

The tool allows you to filter matches based on confidence scores:

- `--num_top_kp N`: Keep only the N most confident matches (green in visualization)
- `--num_bottom_kp M`: Keep only the M least confident matches (red in visualization)
- If both specified: You'll get N+M total matches, sorted by confidence

This is useful for:
- **Quality control**: Focus on high-confidence matches
- **Outlier analysis**: Examine low-confidence matches
- **Dataset balancing**: Create balanced datasets with good and poor matches
