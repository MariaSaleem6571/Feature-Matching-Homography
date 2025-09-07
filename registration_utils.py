import cv2
import numpy as np
import csv
import uuid
from kornia.feature import LoFTR
import torch
from LightGlue.lightglue import LightGlue
from LightGlue.lightglue.superpoint import SuperPoint
import os
import json


def load_image(path, grayscale=True):
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    img = cv2.imread(path, flag)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return img


def detect_and_compute(img):
    detector = cv2.AKAZE_create()
    keypoints, _ = detector.detectAndCompute(img, None)
    return keypoints


def match_keypoints(desc1, desc2):
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(desc1, desc2)
    max_dist = max(m.distance for m in matches) if matches else 1.0
    confidences = [1.0 - (m.distance / max_dist) for m in matches]
    return matches, confidences


def detect_and_match_loftr(img1, img2):
    matcher = LoFTR(pretrained='outdoor')
    img1_tensor = torch.from_numpy(img1).float() / 255.0
    img2_tensor = torch.from_numpy(img2).float() / 255.0
    img1_tensor = img1_tensor.unsqueeze(0).unsqueeze(0)
    img2_tensor = img2_tensor.unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        input_dict = {"image0": img1_tensor, "image1": img2_tensor}
        correspondences = matcher(input_dict)

    mkpts0 = correspondences['keypoints0'].cpu().numpy()
    mkpts1 = correspondences['keypoints1'].cpu().numpy()
    confidence = correspondences['confidence'].cpu().numpy()

    kp1 = [cv2.KeyPoint(x=pt[0], y=pt[1], size=1) for pt in mkpts0]
    kp2 = [cv2.KeyPoint(x=pt[0], y=pt[1], size=1) for pt in mkpts1]

    matches, confidences = [], []
    for i in range(len(mkpts0)):
        match = cv2.DMatch()
        match.queryIdx = i
        match.trainIdx = i
        match.distance = 1.0 - confidence[i]
        matches.append(match)
        confidences.append(float(confidence[i]))

    return kp1, kp2, matches, mkpts0, mkpts1, confidences


def detect_and_match_lightglue_superpoint(img1, img2):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    superpoint = SuperPoint(pretrained=True).eval().to(device)
    lightglue = LightGlue(features='superpoint', pretrained=True).eval().to(device)

    img1_tensor = torch.from_numpy(img1).float() / 255.0
    img2_tensor = torch.from_numpy(img2).float() / 255.0
    img1_tensor = img1_tensor.unsqueeze(0).unsqueeze(0).to(device)
    img2_tensor = img2_tensor.unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        feats0 = superpoint({'image': img1_tensor})
        feats1 = superpoint({'image': img2_tensor})
        matches01 = lightglue({'image0': feats0, 'image1': feats1})

        feats0 = rbd(feats0)
        feats1 = rbd(feats1)
        matches01 = rbd(matches01)

        matches_tensor = matches01['matches']

        if isinstance(matches_tensor, list):
            matches_tensor = np.array(matches_tensor)
            if matches_tensor.ndim > 2:
                matches_tensor = matches_tensor[0]
            matches_tensor = torch.as_tensor(matches_tensor, dtype=torch.long, device=device)
        elif isinstance(matches_tensor, np.ndarray):
            matches_tensor = torch.as_tensor(matches_tensor, dtype=torch.long, device=device)

        valid_mask = (matches_tensor >= 0).all(dim=1)
        valid_matches = matches_tensor[valid_mask]

        if len(valid_matches) == 0:
            return [], [], [], [], [], []

        kp0_all = feats0['keypoints'].cpu().numpy()
        kp1_all = feats1['keypoints'].cpu().numpy()
        match_indices = valid_matches.cpu().numpy()
        mkpts0 = kp0_all[match_indices[:, 0]]
        mkpts1 = kp1_all[match_indices[:, 1]]

    kp1 = [cv2.KeyPoint(x=pt[0], y=pt[1], size=1) for pt in kp0_all]
    kp2 = [cv2.KeyPoint(x=pt[0], y=pt[1], size=1) for pt in kp1_all]

    confidences = []
    cv_matches = []
    for idx0, idx1 in match_indices:
        match = cv2.DMatch()
        match.queryIdx = int(idx0)
        match.trainIdx = int(idx1)
        cv_matches.append(match)
        pt0 = kp0_all[idx0]
        pt1 = kp1_all[idx1]
        dist = np.linalg.norm(pt0 - pt1)
        confidences.append(1.0 / (1.0 + dist))

    return kp1, kp2, cv_matches, mkpts0, mkpts1, confidences


def rbd(data: dict) -> dict:
    return {k: v[0] if isinstance(v, torch.Tensor) else v for k, v in data.items()}


def compute_4x4_homography_from_matches(mkpts0, mkpts1):
    H4 = np.eye(4)
    if len(mkpts0) >= 4:
        src_pts = np.array(mkpts1, dtype=np.float32).reshape(-1, 1, 2)
        dst_pts = np.array(mkpts0, dtype=np.float32).reshape(-1, 1, 2)
        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H is not None:
            H4[:3, :3] = H
            H4[:3, 3] = H[:, 2]
    return H4


def create_csv_rows_with_confidence_selection(kp1, kp2, matches, confidences, H4, image_files, i, method, num_top_kp=None, num_bottom_kp=None):
    confidences = np.array(confidences)
    sorted_indices = np.argsort(confidences)[::-1]

    top_indices, bottom_indices = [], []
    if num_top_kp is not None:
        top_indices = sorted_indices[:num_top_kp].tolist()
    if num_bottom_kp is not None:
        bottom_indices = sorted_indices[-num_bottom_kp:].tolist()

    selected_indices = np.unique(top_indices + bottom_indices)
    selected_confidences = confidences[selected_indices]
    final_order = selected_indices[np.argsort(selected_confidences)[::-1]]

    rows = []
    pair_id = f"{os.path.basename(image_files[0])}_{os.path.basename(image_files[1])}"
    for idx in final_order:
        m = matches[idx]
        row = {
            "uuid": str(uuid.uuid4()),
            "image1_index": i + 1,
            "image2_index": i + 2,
            "pair_id": pair_id,
            "method": method,
            "x1": kp1[m.queryIdx].pt[0],
            "y1": kp1[m.queryIdx].pt[1],
            "x2": kp2[m.trainIdx].pt[0],
            "y2": kp2[m.trainIdx].pt[1],
            "confidence": confidences[idx]
        }
        homography_vals = {
            "r11": H4[0, 0], "r12": H4[0, 1], "r13": H4[0, 2], "tx": H4[0, 3],
            "r21": H4[1, 0], "r22": H4[1, 1], "r23": H4[1, 2], "ty": H4[1, 3],
            "r31": H4[2, 0], "r32": H4[2, 1], "r33": H4[2, 2], "tz": H4[2, 3],
            "h41": H4[3, 0], "h42": H4[3, 1], "h43": H4[3, 2], "h44": H4[3, 3],
        }
        row.update(homography_vals)
        rows.append(row)

    return rows, top_indices, bottom_indices


def save_csv(csv_path, rows):
    if not rows:
        return
    keys = rows[0].keys()
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def draw_selected_matches(img1, img2, kp1, kp2, matches, top_indices, bottom_indices):
    vis = cv2.hconcat([
        cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    ])
    h, w = img1.shape[:2]

    for idx in top_indices:
        m = matches[idx]
        pt1 = tuple(map(int, kp1[m.queryIdx].pt))
        pt2 = tuple(map(int, kp2[m.trainIdx].pt))
        pt2_shifted = (int(pt2[0] + w), int(pt2[1]))
        cv2.line(vis, pt1, pt2_shifted, (0, 255, 0), 2)

    for idx in bottom_indices:
        m = matches[idx]
        pt1 = tuple(map(int, kp1[m.queryIdx].pt))
        pt2 = tuple(map(int, kp2[m.trainIdx].pt))
        pt2_shifted = (int(pt2[0] + w), int(pt2[1]))
        cv2.line(vis, pt1, pt2_shifted, (0, 0, 255), 2)

    return vis


def process_image_pairs_and_save_logs(
    image_pairs, logs_dir, vis_dir, keypoints_dir,
    method='akaze', num_top_kp=None, num_bottom_kp=None
):
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(keypoints_dir, exist_ok=True)

    experiment_rows = []

    for i, (img1_path, img2_path) in enumerate(image_pairs):
        img1 = load_image(img1_path)
        img2 = load_image(img2_path)

        if method == 'akaze':
            kp1 = detect_and_compute(img1)
            kp2 = detect_and_compute(img2)
            detector = cv2.AKAZE_create()
            _, desc1 = detector.compute(img1, kp1)
            _, desc2 = detector.compute(img2, kp2)
            matches, confidences = match_keypoints(desc1, desc2)
            mkpts0 = [kp1[m.queryIdx].pt for m in matches]
            mkpts1 = [kp2[m.trainIdx].pt for m in matches]

        elif method == 'loftr':
            kp1, kp2, matches, mkpts0, mkpts1, confidences = detect_and_match_loftr(img1, img2)

        elif method == 'lightglue':
            kp1, kp2, matches, mkpts0, mkpts1, confidences = detect_and_match_lightglue_superpoint(img1, img2)
            if len(matches) == 0:
                print(f"Warning: No matches found for pair {i + 1}")
                continue

        H4 = compute_4x4_homography_from_matches(mkpts0, mkpts1)

        pair_uuid = str(uuid.uuid4())

        rows, top_indices, bottom_indices = create_csv_rows_with_confidence_selection(
            kp1, kp2, matches, confidences, H4,
            [img1_path, img2_path], 0, method,
            num_top_kp=num_top_kp,
            num_bottom_kp=num_bottom_kp
        )
        for row in rows:
            for h in ['r11','r12','r13','tx','r21','r22','r23','ty','r31','r32','r33','tz','h41','h42','h43','h44']:
                row.pop(h, None)

        base_kp_csv = f"keypoints_{pair_uuid}.csv"
        kp_csv_path = os.path.join(keypoints_dir, base_kp_csv)
        save_csv(kp_csv_path, rows)

        img_matches = draw_selected_matches(img1, img2, kp1, kp2, matches, top_indices, bottom_indices)
        vis_path = os.path.join(vis_dir, f"vis_{pair_uuid}.png")
        cv2.imwrite(vis_path, img_matches)

        experiment_row = {
            "uuid": pair_uuid,
            "image1_name": os.path.basename(img1_path),
            "image2_name": os.path.basename(img2_path),
            "keypoints_csv_path": kp_csv_path,
            "visualization_path": vis_path,
            **{
                "r11": H4[0,0], "r12": H4[0,1], "r13": H4[0,2], "tx": H4[0,3],
                "r21": H4[1,0], "r22": H4[1,1], "r23": H4[1,2], "ty": H4[1,3],
                "r31": H4[2,0], "r32": H4[2,1], "r33": H4[2,2], "tz": H4[2,3],
                "h41": H4[3,0], "h42": H4[3,1], "h43": H4[3,2], "h44": H4[3,3],
            }
        }
        experiment_rows.append(experiment_row)

        print(f"Processed pair {img1_path} -> {img2_path} | {len(matches)} matches")

    if experiment_rows:
        exp_log_path = os.path.join(logs_dir, f"experiment_log.csv")
        save_csv(exp_log_path, experiment_rows)
        print(f"Saved experiment log to {exp_log_path}")
