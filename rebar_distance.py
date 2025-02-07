import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import open3d as o3d
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "12"

# 1. XYZRGB 형식의 PTS 파일 읽기
def read_xyzrgb_pts(file_path):
    try:
        data = np.loadtxt(file_path)
        if data.shape[1] < 6:
            raise ValueError("PTS file does not contain RGB information.")
        points = data[:, :3]  # XYZ 좌표
        colors = data[:, 3:6] / 255.0  # RGB 값 (0-1로 정규화)
        return points, colors
    except Exception as e:
        print(f"Error reading PTS file: {e}")
        return None, None

# 2. 파일 경로 지정
file_path = "rebar.pts" 

# 3. 데이터 로드
points, colors = read_xyzrgb_pts(file_path)

if points is not None and colors is not None:
    # 4. Open3D 포인트 클라우드 생성
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # 5. Z축 제거 (2D 변환)
    points_2d = points.copy()
    points_2d[:, 2] = 0  # Z축 제거

    # 6. K-Means 클러스터링 (클러스터 개수 K 설정)
    K = 28  # 철근 클러스터 개수
    kmeans = KMeans(n_clusters=K, random_state=0, n_init=10).fit(points_2d[:, :2])
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    # 7. PCA로 주축 계산 및 회전
    pca = PCA(n_components=2)
    pca.fit(centroids)
    principal_axis = pca.components_[0]  # 첫 번째 주축
    angle = np.arctan2(principal_axis[1], principal_axis[0])  # 회전 각도 계산

    rotation_matrix = np.array([
        [np.cos(-angle), -np.sin(-angle)],
        [np.sin(-angle),  np.cos(-angle)]
    ])
    rotated_centroids = (rotation_matrix @ centroids.T).T  # 회전 적용

    # ---- 시각화: 회전된 좌표 ----
    plt.figure(figsize=(8, 8))
    plt.scatter(rotated_centroids[:, 0], rotated_centroids[:, 1], c='red', s=50, label="Rotated Centroids")
    plt.title("Rotated Centroids (Aligned with X-Axis)", fontsize=14)
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend(loc="best")
    plt.grid(True)
    plt.axis("equal")
    plt.show()

    # 8. 그룹화: Y축 기준으로 철근을 그룹화
    y_threshold = 0.05  # Y축 값이 같은 그룹으로 간주되는 임계값
    groups = {}
    for i, centroid in enumerate(rotated_centroids):
        y_val = np.round(centroid[1] / y_threshold) * y_threshold  # Y값을 그룹화
        if y_val not in groups:
            groups[y_val] = []
        groups[y_val].append(centroid)

    # 9. 각 그룹 내에서 X축 방향 거리 계산
    lines = []
    text_annotations = []

    for y_val, group in groups.items():
        group = np.array(group)
        group = group[np.argsort(group[:, 0])]  # X축 기준 정렬

        for i in range(len(group) - 1):
            point1 = group[i]
            point2 = group[i + 1]

            # 수평 방향 거리 계산
            distance = np.linalg.norm(point1 - point2)

            # 선 연결
            lines.append((point1, point2))

            # 거리 텍스트 데이터
            mid_point = (point1 + point2) / 2
            text_annotations.append((mid_point, f"{distance:.2f} m"))

    # ---- 시각화: 수평 방향 거리 ----
    plt.figure(figsize=(8, 8))

    # 선 연결
    for line in lines:
        point1, point2 = line
        plt.plot([point1[0], point2[0]], [point1[1], point2[1]], 'r-', alpha=0.8)  # 빨간 실선

    # 클러스터 중심 표시
    plt.scatter(rotated_centroids[:, 0], rotated_centroids[:, 1], c='red', s=50, label="Rotated Centroids")

    # 거리 텍스트 표시
    for text_point, label in text_annotations:
        plt.text(text_point[0], text_point[1], label, fontsize=7, color="blue")

    plt.title("Horizontal Rebar Distances (Grouped by Y-Coordinate)", fontsize=14)
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend(loc="best")
    plt.grid(True)
    plt.axis("equal")
    plt.show()
else:
    print("Failed to load PTS file.")
