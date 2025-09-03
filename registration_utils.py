import cv2
import numpy as np
import csv
import uuid
from kornia.feature import LoFTR
import torch

def load_image(path, grayscale=True):
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    img = cv2.imread(path, flag)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return img


def detect_and_compute(img):
    detector = cv2.AKAZE_create()
    keypoints, descriptors = detector.detectAndCompute(img, None)
    return keypoints, descriptors


def match_keypoints(desc1, desc2):
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(desc1, desc2)
    return matches


def detect_and_match_loftr(img1, img2):
    """
    LoFTR feature detection and matching
    """
    # Initialize LoFTR model
    matcher = LoFTR(pretrained='outdoor')

    # Convert images to torch tensors
    img1_tensor = torch.from_numpy(img1).float() / 255.0
    img2_tensor = torch.from_numpy(img2).float() / 255.0

    # Add batch dimension and channel dimension
    img1_tensor = img1_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    img2_tensor = img2_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

    # Perform matching
    with torch.no_grad():
        input_dict = {
            "image0": img1_tensor,
            "image1": img2_tensor
        }
        correspondences = matcher(input_dict)

    # Extract matched points
    mkpts0 = correspondences['keypoints0'].cpu().numpy()
    mkpts1 = correspondences['keypoints1'].cpu().numpy()
    confidence = correspondences['confidence'].cpu().numpy()

    # Convert to keypoints and matches format similar to AKAZE
    kp1 = [cv2.KeyPoint(x=pt[0], y=pt[1], size=1) for pt in mkpts0]
    kp2 = [cv2.KeyPoint(x=pt[0], y=pt[1], size=1) for pt in mkpts1]

    # Create matches
    matches = []
    for i in range(len(mkpts0)):
        match = cv2.DMatch()
        match.queryIdx = i
        match.trainIdx = i
        match.distance = 1.0 - confidence[i]  # Convert confidence to distance
        matches.append(match)

    return kp1, kp2, matches, mkpts0, mkpts1

def filter_matches_by_pixel_distance(kp1, kp2, matches, pixel_threshold):
    filtered = []
    for m in matches:
        pt1 = np.array(kp1[m.queryIdx].pt)
        pt2 = np.array(kp2[m.trainIdx].pt)
        dist = np.linalg.norm(pt1 - pt2)
        if dist <= pixel_threshold:
            filtered.append(m)
    return filtered


def compute_4x4_homography_from_matches(mkpts0, mkpts1):
    """
    Compute 4x4 homography in the CSV-friendly format.
    """
    H4 = np.eye(4)

    if len(mkpts0) >= 4:
        src_pts = np.array(mkpts1, dtype=np.float32).reshape(-1, 1, 2)
        dst_pts = np.array(mkpts0, dtype=np.float32).reshape(-1, 1, 2)
        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if H is not None:
            H4[0, 0] = H[0, 0]
            H4[0, 1] = H[0, 1]
            H4[0, 2] = H[0, 2]
            H4[0, 3] = H[0, 2]

            H4[1, 0] = H[1, 0]
            H4[1, 1] = H[1, 1]
            H4[1, 2] = H[1, 2]
            H4[1, 3] = H[1, 2]

            H4[2, 0] = H[2, 0]
            H4[2, 1] = H[2, 1]
            H4[2, 2] = 1.0
            H4[2, 3] = 1.0

            H4[3, 0] = 0.0
            H4[3, 1] = 0.0
            H4[3, 2] = 0.0
            H4[3, 3] = 1.0
    return H4


def create_csv_rows_without_descriptors(kp1, kp2, matches, H4, image_files, i):
    """
    Create CSV rows without descriptor columns (for LoFTR)
    """
    rows = []
    pair_id = f"{image_files[i].replace('.png', '')}_{image_files[i + 1].replace('.png', '')}"

    for m in matches:
        row = {
            "uuid": str(uuid.uuid4()),
            "image1_index": i + 1,
            "image2_index": i + 2,
            "pair_id": pair_id,
            "x1": kp1[m.queryIdx].pt[0],
            "y1": kp1[m.queryIdx].pt[1],
            "x2": kp2[m.trainIdx].pt[0],
            "y2": kp2[m.trainIdx].pt[1],
        }

        # Add homography matrix values
        homography_vals = {
            "r11": H4[0, 0], "r12": H4[0, 1], "r13": H4[0, 2], "tx": H4[0, 3],
            "r21": H4[1, 0], "r22": H4[1, 1], "r23": H4[1, 2], "ty": H4[1, 3],
            "r31": H4[2, 0], "r32": H4[2, 1], "r33": H4[2, 2], "tz": H4[2, 3],
            "h41": H4[3, 0], "h42": H4[3, 1], "h43": H4[3, 2], "h44": H4[3, 3],
        }
        row.update(homography_vals)
        rows.append(row)

    return rows

def create_csv_rows_with_descriptors(kp1, kp2, desc1, desc2, matches, H4, image_files, i):
    rows = []
    pair_id = f"{image_files[i].replace('.png', '')}_{image_files[i + 1].replace('.png', '')}"

    for m in matches:
        row = {
            "uuid": str(uuid.uuid4()),
            "image1_index": i + 1,
            "image2_index": i + 2,
            "pair_id": pair_id,
            "x1": kp1[m.queryIdx].pt[0],
            "y1": kp1[m.queryIdx].pt[1],
            "x2": kp2[m.trainIdx].pt[0],
            "y2": kp2[m.trainIdx].pt[1],
        }

        desc1_vals = {f"desc1_{k}": float(val) for k, val in enumerate(desc1[m.queryIdx])}
        desc2_vals = {f"desc2_{k}": float(val) for k, val in enumerate(desc2[m.trainIdx])}
        row.update(desc1_vals)
        row.update(desc2_vals)

        homography_vals = {
            "r11": H4[0, 0], "r12": H4[0, 1], "r13": H4[0, 2], "tx": H4[0, 3],
            "r21": H4[1, 0], "r22": H4[1, 1], "r23": H4[1, 2], "ty": H4[1, 3],
            "r31": H4[2, 0], "r32": H4[2, 1], "r33": H4[2, 2], "tz": H4[2, 3],
            "h41": H4[3, 0], "h42": H4[3, 1], "h43": H4[3, 2], "h44": H4[3, 3],
        }
        row.update(homography_vals)
        rows.append(row)
    return rows


def save_csv(csv_path, rows):
    if not rows:
        return
    keys = rows[0].keys()
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def draw_matches(img1, img2, kp1, kp2, matches, mask=None):
    matchesMask = mask.ravel().tolist() if mask is not None else None
    return cv2.drawMatches(img1, kp1, img2, kp2, matches, None, matchesMask=matchesMask,
                           flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

