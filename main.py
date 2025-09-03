import os
import argparse
from natsort import natsorted
from registration_utils import *


def main():
    parser = argparse.ArgumentParser(description="Feature matching for consecutive images in a directory")
    parser.add_argument("--dir", type=str, required=True, help="Directory containing images")
    parser.add_argument("--threshold", type=float, default=30.0, help="Pixel distance threshold for matches")
    parser.add_argument("--csv_dir", type=str, default="matches_csv", help="Directory to save CSVs")
    parser.add_argument("--visualize", action="store_true", help="Visualize matches if flag is set")
    parser.add_argument("--method", type=str, choices=['akaze', 'loftr'], default='akaze',
                        help="Feature matching method: 'akaze' or 'loftr' (default: akaze)")
    args = parser.parse_args()

    os.makedirs(args.csv_dir, exist_ok=True)
    image_files = [f for f in os.listdir(args.dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_files = natsorted(image_files)

    if len(image_files) < 2:
        print("Need at least 2 images in the directory.")
        return

    print(f"Using {args.method.upper()} method for feature matching")
    all_rows = []

    for i in range(len(image_files) - 1):
        img1_path = os.path.join(args.dir, image_files[i])
        img2_path = os.path.join(args.dir, image_files[i + 1])
        img1 = load_image(img1_path)
        img2 = load_image(img2_path)

        if args.method == 'akaze':
            # AKAZE method
            kp1, desc1 = detect_and_compute(img1)
            kp2, desc2 = detect_and_compute(img2)
            matches = match_keypoints(desc1, desc2)
            matches = filter_matches_by_pixel_distance(kp1, kp2, matches, args.threshold)

            # Extract matched points for homography
            mkpts0 = [kp1[m.queryIdx].pt for m in matches]
            mkpts1 = [kp2[m.trainIdx].pt for m in matches]

        elif args.method == 'loftr':
            # LoFTR method
            kp1, kp2, matches, mkpts0, mkpts1 = detect_and_match_loftr(img1, img2)

            # Apply pixel distance filtering for LoFTR as well
            if args.threshold > 0:
                matches = filter_matches_by_pixel_distance(kp1, kp2, matches, args.threshold)
                # Update matched points after filtering
                mkpts0 = [kp1[m.queryIdx].pt for m in matches]
                mkpts1 = [kp2[m.trainIdx].pt for m in matches]

            # For LoFTR, we need to create dummy descriptors since LoFTR doesn't provide them
            # You can modify this based on your CSV requirements
            desc1 = np.zeros((len(kp1), 61), dtype=np.uint8)  # AKAZE descriptor size is 61
            desc2 = np.zeros((len(kp2), 61), dtype=np.uint8)

        print(f"Pair {i + 1}: {image_files[i]} -> {image_files[i + 1]} | Matches: {len(matches)}")

        # Compute homography
        H4 = compute_4x4_homography_from_matches(mkpts0, mkpts1)

        # Create CSV rows (with or without descriptors based on method)
        if args.method == 'akaze':
            rows = create_csv_rows_with_descriptors(kp1, kp2, desc1, desc2, matches, H4, image_files, i)
        else:  # loftr
            rows = create_csv_rows_without_descriptors(kp1, kp2, matches, H4, image_files, i)
        all_rows.extend(rows)

        # Visualization
        if args.visualize:
            img_matches = draw_matches(img1, img2, kp1, kp2, matches)
            cv2.imshow(f"Matches - {args.method.upper()}", img_matches)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    # Save results
    method_suffix = args.method.lower()
    csv_name = os.path.join(args.csv_dir, f"matches_homography_descriptors_{method_suffix}.csv")
    save_csv(csv_name, all_rows)
    print(f"Saved all results to {csv_name}")
    print(f"Total matches across all pairs: {len(all_rows)}")


if __name__ == "__main__":
    main()