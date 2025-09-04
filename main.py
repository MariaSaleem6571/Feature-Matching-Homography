import os
import argparse
from natsort import natsorted
from registration_utils import *


def main():
    parser = argparse.ArgumentParser(description="Feature matching for images")
    parser.add_argument("--dir", type=str, help="Directory containing images (consecutive matching)")
    parser.add_argument("--pairs_csv", type=str, help="CSV file with columns image1_path,image2_path for pairwise matching")
    parser.add_argument("--csv_dir", type=str, default="matches_csv", help="Directory to save CSVs")
    parser.add_argument("--vis_dir", type=str, default="visualizations", help="Directory to save match visualizations")
    parser.add_argument("--method", type=str, choices=['akaze', 'loftr', 'lightglue'], default='akaze',
                        help="Feature matching method: 'akaze', 'loftr', or 'lightglue' (default: akaze)")
    parser.add_argument("--num_top_kp", type=int, default=None, help="Number of keypoints with highest confidence to keep")
    parser.add_argument("--num_bottom_kp", type=int, default=None, help="Number of keypoints with lowest confidence to keep")

    args = parser.parse_args()
    os.makedirs(args.csv_dir, exist_ok=True)
    os.makedirs(args.vis_dir, exist_ok=True)

    if args.pairs_csv:
        print(f"Processing image pairs from CSV: {args.pairs_csv}")
        process_image_pairs_from_csv(
            csv_file=args.pairs_csv,
            output_csv_dir=args.csv_dir,
            output_vis_dir=args.vis_dir,
            method=args.method,
            num_top_kp=args.num_top_kp,
            num_bottom_kp=args.num_bottom_kp
        )
        return

    if not args.dir:
        print("Please provide either --dir or --pairs_csv")
        return

    image_files = [f for f in os.listdir(args.dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_files = natsorted(image_files)

    if len(image_files) < 2:
        print("Need at least 2 images in the directory.")
        return

    print(f"Using {args.method.upper()} method for feature matching (folder mode)")

    for i in range(len(image_files) - 1):
        img1_path = os.path.join(args.dir, image_files[i])
        img2_path = os.path.join(args.dir, image_files[i + 1])

        img1 = load_image(img1_path)
        img2 = load_image(img2_path)

        if args.method == 'akaze':
            kp1 = detect_and_compute(img1)
            kp2 = detect_and_compute(img2)
            detector = cv2.AKAZE_create()
            _, desc1 = detector.compute(img1, kp1)
            _, desc2 = detector.compute(img2, kp2)
            matches, confidences = match_keypoints(desc1, desc2)
            mkpts0 = [kp1[m.queryIdx].pt for m in matches]
            mkpts1 = [kp2[m.trainIdx].pt for m in matches]

        elif args.method == 'loftr':
            kp1, kp2, matches, mkpts0, mkpts1, confidences = detect_and_match_loftr(img1, img2)

        elif args.method == 'lightglue':
            kp1, kp2, matches, mkpts0, mkpts1, confidences = detect_and_match_lightglue_superpoint(img1, img2)
            if len(matches) == 0:
                print(f"Warning: No matches found for pair {i + 1}")
                continue

        print(f"Pair {i + 1}: {image_files[i]} -> {image_files[i + 1]} | Matches: {len(matches)}")

        H4 = compute_4x4_homography_from_matches(mkpts0, mkpts1)

        rows = create_csv_rows_with_confidence_selection(
            kp1, kp2, matches, confidences, H4, image_files, i, args.method,
            num_top_kp=args.num_top_kp,
            num_bottom_kp=args.num_bottom_kp
        )

        base1 = os.path.splitext(image_files[i])[0]
        base2 = os.path.splitext(image_files[i + 1])[0]
        pair_id = f"{base1}_{base2}"
        csv_name = os.path.join(args.csv_dir, f"{pair_id}_{args.method.lower()}.csv")
        save_csv(csv_name, rows)
        print(f"Saved results for pair {pair_id} to {csv_name}")

        img_matches = draw_matches(img1, img2, kp1, kp2, matches)
        vis_path = os.path.join(args.vis_dir, f"{pair_id}_{args.method.lower()}.png")
        cv2.imwrite(vis_path, img_matches)
        print(f"Saved visualization for pair {pair_id} to {vis_path}")


if __name__ == "__main__":
    main()
