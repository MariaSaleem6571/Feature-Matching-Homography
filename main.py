import os
import argparse
import csv
from natsort import natsorted
import cv2
from registration_utils import *


def main():
    parser = argparse.ArgumentParser(description="Feature matching for images")
    parser.add_argument("--dir", type=str, help="Directory containing images (consecutive matching)")
    parser.add_argument("--pairs_csv", type=str, help="CSV file with image pairs")
    parser.add_argument("--image_base_dir", type=str, help="Base directory for images when using pairs_csv")
    parser.add_argument("--logs_dir", type=str, default="experiment_logs", help="Directory to save experiment log CSVs")
    parser.add_argument("--keypoints_dir", type=str, default="keypoints_csvs", help="Directory to save keypoints CSVs")
    parser.add_argument("--vis_dir", type=str, default="visualizations", help="Directory to save visualizations")
    parser.add_argument("--method", type=str, choices=['akaze', 'loftr', 'lightglue'], default='akaze',
                        help="Feature matching method: 'akaze', 'loftr', or 'lightglue'")
    parser.add_argument("--num_top_kp", type=int, default=None,
                        help="Number of keypoints with highest confidence to keep (green)")
    parser.add_argument("--num_bottom_kp", type=int, default=None,
                        help="Number of keypoints with lowest confidence to keep (red)")
    parser.add_argument("--process_all_pairs", action='store_true',
                        help="Process all possible image pairs (not just consecutive)")

    args = parser.parse_args()

    os.makedirs(args.logs_dir, exist_ok=True)
    os.makedirs(args.keypoints_dir, exist_ok=True)
    os.makedirs(args.vis_dir, exist_ok=True)

    if args.pairs_csv:
        print(f"Processing image pairs from CSV: {args.pairs_csv}")

        if args.image_base_dir:
            all_image_files = [f for f in os.listdir(args.image_base_dir)
                               if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            all_image_files = natsorted(all_image_files)
            print(f"Found {len(all_image_files)} images in directory: {all_image_files}")

        if args.process_all_pairs and args.image_base_dir:
            print("Generating all sequential image pairs...")
            image_pairs = []

            for i in range(len(all_image_files)):
                for j in range(i + 1, len(all_image_files)):
                    img1_path = os.path.join(args.image_base_dir, all_image_files[i])
                    img2_path = os.path.join(args.image_base_dir, all_image_files[j])

                    if os.path.exists(img1_path) and os.path.exists(img2_path):
                        image_pairs.append((img1_path, img2_path))

            print(f"Generated {len(image_pairs)} image pairs for processing")

        else:
            image_pairs = []

            with open(args.pairs_csv, 'r', newline='') as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames

                if 'img_name_1' in fieldnames and 'img_name_2' in fieldnames:
                    print("Detected sonar CSV format with img_name_1, img_name_2 columns")
                    for row in reader:
                        img1_name = row['img_name_1']
                        img2_name = row['img_name_2']

                        if args.image_base_dir:
                            img1_path = os.path.join(args.image_base_dir, img1_name)
                            img2_path = os.path.join(args.image_base_dir, img2_name)
                        else:
                            img1_path = img1_name
                            img2_path = img2_name

                        if os.path.exists(img1_path) and os.path.exists(img2_path):
                            image_pairs.append((img1_path, img2_path))
                        else:
                            print(f"Warning: Skipping pair - files not found: {img1_path}, {img2_path}")

                elif 'image1' in fieldnames and 'image2' in fieldnames:
                    print("Detected simple CSV format with image1, image2 columns")
                    for row in reader:
                        img1_path = row['image1']
                        img2_path = row['image2']

                        if args.image_base_dir:
                            img1_path = os.path.join(args.image_base_dir, os.path.basename(img1_path))
                            img2_path = os.path.join(args.image_base_dir, os.path.basename(img2_path))

                        if os.path.exists(img1_path) and os.path.exists(img2_path):
                            image_pairs.append((img1_path, img2_path))
                        else:
                            print(f"Warning: Skipping pair - files not found: {img1_path}, {img2_path}")
                else:
                    print("Error: CSV must contain either 'img_name_1,img_name_2' or 'image1,image2' columns")
                    return

            print(f"Loaded {len(image_pairs)} image pairs from CSV")

        if len(image_pairs) == 0:
            print("No image pairs found")
            return

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

    if args.process_all_pairs:
        image_pairs = []
        for i in range(len(image_files)):
            for j in range(i + 1, len(image_files)):
                image_pairs.append((image_files[i], image_files[j]))
        print(f"Processing all {len(image_pairs)} possible image pairs")
    else:
        image_pairs = [(image_files[i], image_files[i + 1]) for i in range(len(image_files) - 1)]
        print(f"Processing {len(image_pairs)} consecutive image pairs")

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