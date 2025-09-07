import os
import argparse
import csv
from natsort import natsorted
import cv2
from registration_utils import *


def main():
    parser = argparse.ArgumentParser(description="Feature matching for images")
    parser.add_argument("--dir", type=str, help="Directory containing images (consecutive matching)")
    parser.add_argument("--pairs_csv", type=str, help="CSV file with columns image1,image2 for pairwise matching")
    parser.add_argument("--logs_dir", type=str, default="experiment_logs", help="Directory to save experiment log CSVs")
    parser.add_argument("--keypoints_dir", type=str, default="keypoints_csvs", help="Directory to save keypoints CSVs")
    parser.add_argument("--vis_dir", type=str, default="visualizations", help="Directory to save visualizations")
    parser.add_argument("--method", type=str, choices=['akaze', 'loftr', 'lightglue'], default='akaze',
                        help="Feature matching method: 'akaze', 'loftr', or 'lightglue'")
    parser.add_argument("--num_top_kp", type=int, default=None, help="Number of keypoints with highest confidence to keep (green)")
    parser.add_argument("--num_bottom_kp", type=int, default=None, help="Number of keypoints with lowest confidence to keep (red)")

    args = parser.parse_args()

    os.makedirs(args.logs_dir, exist_ok=True)
    os.makedirs(args.keypoints_dir, exist_ok=True)
    os.makedirs(args.vis_dir, exist_ok=True)

    if args.pairs_csv:
        print(f"Processing image pairs from CSV: {args.pairs_csv}")
        image_pairs = []
        with open(args.pairs_csv, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                image_pairs.append((row['image1'], row['image2']))

        process_image_pairs_and_save_logs(
            image_pairs=image_pairs,
            logs_dir=args.logs_dir,
            vis_dir=args.vis_dir,
            keypoints_dir=args.keypoints_dir,
            method=args.method,
            num_top_kp=args.num_top_kp,
            num_bottom_kp=args.num_bottom_kp
        )
        return

    if not args.dir:
        print("Please provide either --dir or --pairs_csv")
        return

    image_files = [os.path.join(args.dir, f) for f in os.listdir(args.dir)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_files = natsorted(image_files)

    if len(image_files) < 2:
        print("Need at least 2 images in the directory.")
        return

    image_pairs = [(image_files[i], image_files[i + 1]) for i in range(len(image_files) - 1)]
    print(f"Using {args.method.upper()} method for feature matching (directory mode)")
    process_image_pairs_and_save_logs(
        image_pairs=image_pairs,
        logs_dir=args.logs_dir,
        vis_dir=args.vis_dir,
        keypoints_dir=args.keypoints_dir,
        method=args.method,
        num_top_kp=args.num_top_kp,
        num_bottom_kp=args.num_bottom_kp
    )


if __name__ == "__main__":
    main()
