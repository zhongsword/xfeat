import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from modules.xfeat import XFeat
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'

# 假设 X-Feat 是一个自定义的特征提取方法
extractor = XFeat('/DATA/jupyter/personal/xfeat/weights/2/xfeat_default_last.pth', top_k=2000)

def x_feat_extract(image):
    """
    使用 X-Feat 提取特征点和描述子
    """
    global extractor
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    if torch.cuda.is_available():
        image_tensor = image_tensor.cuda()
    with torch.no_grad():
        res = extractor.detectAndCompute(image_tensor)
        keypoints, descriptors = res[0]['keypoints'].cpu().numpy(), res[0]['descriptors'].cpu().numpy()
    return keypoints, descriptors

def match_features(desc1, desc2):
    """
    使用 BFMatcher 进行特征匹配
    """
    bf = cv2.BFMatcher(crossCheck=True)
    matches = bf.match(desc1, desc2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches

def compute_ransac_inliers(kp1, kp2, matches):
    """
    使用 RANSAC 估计基础矩阵并计算内点
    """
    src_pts = np.float32([kp1[m.queryIdx] for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx] for m in matches]).reshape(-1, 1, 2)
    F, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_RANSAC)
    inliers = np.sum(mask)
    return inliers

def compute_angle_overlap_union(matches):
    """
    计算匹配角度 AOU
    """
    angles = []
    for match in matches:
        angle = np.arctan2(match.trainIdx - match.queryIdx, match.distance)
        angles.append(angle)
    angles = np.array(angles)
    angle_mean = np.mean(angles)
    angle_std = np.std(angles)
    return angle_mean, angle_std

def visualize_matches(img1, img2, kp1, kp2, matches):
    """
    可视化匹配结果
    """
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:30], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.figure(figsize=(12, 6))
    plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
    plt.title("Feature Matches")
    plt.show()

def main():
    # KITTI Odometry 数据集路径
    dataset_path = "/tmp/alluxio/kitti-odometry/fuse"
    sequence = "00"  # 选择一个序列
    image_dir_left = os.path.join(dataset_path, sequence, "image_2")
    image_dir_right = os.path.join(dataset_path, sequence, "image_3")

    # 获取图像列表
    image_files_left = sorted([os.path.join(image_dir_left, f) for f in os.listdir(image_dir_left) if f.endswith(".png")])
    image_files_right = sorted([os.path.join(image_dir_right, f) for f in os.listdir(image_dir_right) if f.endswith(".png")])

    # 初始化统计变量
    total_matches = 0
    total_inliers = 0
    total_angle_mean = 0
    total_angle_std = 0
    num_images = len(image_files_left)

    # 遍历所有图像对
    for img1_path_left, img1_path_right in tqdm(zip(image_files_left, image_files_right), desc="Processing images", total=num_images):
        # 读取图像
        img1_left = cv2.imread(img1_path_left, cv2.IMREAD_COLOR)
        img1_right = cv2.imread(img1_path_right, cv2.IMREAD_COLOR)

        # 提取特征
        kp1_left, desc1_left = x_feat_extract(img1_left)
        kp1_right, desc1_right = x_feat_extract(img1_right)

        # 匹配特征
        matches = match_features(desc1_left, desc1_right)

        # 计算匹配点总数
        num_matches = len(matches)
        total_matches += num_matches

        # 计算 RANSAC 内点数
        inliers = compute_ransac_inliers(kp1_left, kp1_right, matches)
        total_inliers += inliers

        # 计算匹配角度 AOU
        angle_mean, angle_std = compute_angle_overlap_union(matches)
        total_angle_mean += angle_mean
        total_angle_std += angle_std

    # 计算平均值
    avg_matches = total_matches / num_images
    avg_inliers = total_inliers / num_images
    avg_angle_mean = total_angle_mean / num_images
    avg_angle_std = total_angle_std / num_images

    # 打印结果
    print(f"Average Number of Matches: {avg_matches:.2f}")
    print(f"Average Number of RANSAC Inliers: {avg_inliers:.2f}")
    print(f"Average Angle Mean: {avg_angle_mean:.2f} radians")
    print(f"Average Angle Std Dev: {avg_angle_std:.2f} radians")

if __name__ == "__main__":
    main()
