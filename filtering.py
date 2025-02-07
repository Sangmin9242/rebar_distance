import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

# 1. .pts 파일 로드 (XYZ + RGB 포함)
def load_pts_file_with_rgb(file_path):
    points = []
    colors = []
    with open(file_path, 'r') as file:
        for line in file:
            values = line.split()
            if len(values) >= 6:
                x, y, z, r, g, b = map(float, values[:6])
                points.append([x, y, z])
                colors.append([r / 255.0, g / 255.0, b / 255.0])
    return np.array(points), np.array(colors)

# 2. 평면 및 바닥 뒤쪽 제거 함수
def remove_planes(points, colors, distance_threshold=0.05, num_iterations=6000):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # 평면 감지 (RANSAC)
    plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold,
                                             ransac_n=3,
                                             num_iterations=num_iterations)
    
    # 평면에 속하는 포인트 (벽) 제거
    remaining_cloud = pcd.select_by_index(inliers, invert=True)
    return remaining_cloud

# 3. 바닥 기준 2.5m 이상의 포인트 제거
def filter_points_above_height(pcd, height_threshold=1.9):
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    # Z축 기준으로 2.5m 이상의 포인트 제거
    mask = points[:, 2] <= height_threshold
    filtered_points = points[mask]
    filtered_colors = colors[mask]
    
    return filtered_points, filtered_colors

# 4. 최적 eps 값 찾기 위한 K-거리 그래프
def plot_k_distance(points, k=4):
    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors_fit = neighbors.fit(points)
    distances, indices = neighbors_fit.kneighbors(points)
    distances = np.sort(distances[:, k-1], axis=0)

    plt.plot(distances)
    plt.ylabel(f"{k}-거리")
    plt.xlabel("데이터 포인트")
    plt.title(f"{k}-거리 그래프")
    plt.show()

# 5. 노이즈 제거 함수 (업데이트)
def remove_noise(points, colors, nb_neighbors=20, std_ratio=1.5):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    clean_pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    filtered_points = np.asarray(clean_pcd.points)
    filtered_colors = np.asarray(clean_pcd.colors)
    
    return filtered_points, filtered_colors

# 6. 클러스터링 함수 (DBSCAN)
def dbscan_clustering(points, eps, min_samples):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(points)
    return labels

# 메인 함수
def main(file_path, output_file, point_size=3):
    # .pts 파일 로드
    points, colors = load_pts_file_with_rgb(file_path)

    # 평면 및 바닥 뒤쪽 포인트 제거
    remaining_cloud = remove_planes(points, colors)

    # 바닥 기준 2.5m 이상의 포인트 제거
    filtered_points, filtered_colors = filter_points_above_height(remaining_cloud, height_threshold=1.9)

    # 노이즈 제거 적용
    final_points, final_colors = remove_noise(filtered_points, filtered_colors, nb_neighbors=50, std_ratio=1.5)

    # 최적 eps 값 찾기 위한 K-거리 그래프 시각화
    plot_k_distance(final_points, k=4)

    # eps와 min_samples 값 설정 후 DBSCAN 클러스터링
    eps = 0.05  # K-거리 그래프에서 얻은 최적의 eps 값으로 설정
    min_samples = 10  # 최소 샘플 수 설정
    labels = dbscan_clustering(final_points, eps, min_samples)

    # 클러스터 색상 설정
    unique_labels = set(labels)
    colors = plt.get_cmap("tab10").colors
    colored_points = []

    for i, label in enumerate(labels):
        if label == -1:  # 노이즈 포인트
            color = [0, 0, 0]
        else:
            color = colors[label % len(colors)]
        colored_points.append((*final_points[i], *color))

    # 포인트 클라우드 시각화
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(final_points)
    pcd.colors = o3d.utility.Vector3dVector(np.array(colored_points)[:, 3:])

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)

    render_option = vis.get_render_option()
    render_option.point_size = point_size
    vis.run()
    vis.destroy_window()

# 실행 설정
input_file = '/home/lsm/rebar_distance/105(ROOF).pts'
output_file = '/home/lsm/rebar_distance/output.pts'
main(input_file, output_file, point_size=3)
