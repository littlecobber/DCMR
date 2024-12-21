import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

# Ground Truth Geometry Center Points for 10 Fish
first_frame_points = {
    'sam2_cam0': [(1071, 725), (1378, 604), (1705, 570), (710, 582), (1087, 559), (1307, 425), (548, 390), (834, 355), (1547, 332), (1109, 261)],
    'sam2_cam1': [(1148, 795), (938, 716), (1140, 671), (1273, 588), (795, 550), (1448, 534), (1148, 528), (949, 455), (1385, 420), (1230, 425)],
    'sam2_cam2': [(1204, 791), (1020, 737), (1330, 745), (1252, 659), (984, 588), (1166, 556), (1235, 507), (1031, 441), (1194, 437), (1157, 350)]
}

# Predicted Geometry Center Points for 10 Fish (Predict + Correction by nearest gt points)
predicted_first_frame_points = [
    {'sam2_cam0': [(1071, 725), (1378, 604), (1705, 570), (710, 582), (1087, 559), (1307, 425), (548, 390), (834, 355), (1547, 332), (1109, 261)],
     'sam2_cam1': [(1150, 819), (945, 705), (1130, 680), (1259, 564), (800, 543), (1428, 510), (1139, 539), (969, 555), (1485, 450), (1221, 389)],
     'sam2_cam2': [(1150, 760), (1010, 760), (1300, 785), (1220, 691), (940, 600), (1210, 541), (1200, 530), (1100, 460), (1220, 487), (1166, 340)]}
]


fig, axes = plt.subplots(1, 3, figsize=(19.2, 10.8))

# Set fixed x and y range
# Adjust by dataset resolution
x_min, x_max = 0, 1920
y_min, y_max = -1080, 0

"""
1. Plot Initialization 2D Geometry Centers for 10 Fish (3 Fish case doesn't need prediction)
2. 3D Trajectories for 3 Fish and 10 Fish
3. Animation for 3D trajectories for 3 Fish and 10 Fish
"""

# Initialization comparison: Ground Truth vs Predicted
cam_keys = ['sam2_cam0', 'sam2_cam1', 'sam2_cam2']
for i, ax in enumerate(axes):
    # obtain cam names
    cam_key = cam_keys[i]

    # ground truth 2D points
    x1, y1 = zip(*first_frame_points[cam_key])
    y1 = [-y for y in y1]  # 将 y 值取反
    ax.scatter(x1, y1, label='Ground Truth ', color='blue', alpha=1, s=200)

    # Predicted 2D points
    x1_trans, y1_trans = zip(*predicted_first_frame_points[0][cam_key])
    y1_trans = [-y for y in y1_trans]  # 将 y 值取反
    ax.scatter(x1_trans, y1_trans, label='Predicted ' , color='red', alpha=1, s=200)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    ax.legend(prop={'size': 30})
    ax.set_title(f'Scatter Plot for {cam_key}')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.grid()

plt.tight_layout()
plt.show()


def load_fish_positions_from_npy(file_name):
    """
    从.npy文件加载鱼的3D位置。
    :param file_name: .npy文件名。
    :return: 鱼的3D位置字典。
    """
    fish_3d_positions = np.load(file_name, allow_pickle=True).item()
    print(f"Loaded 3D positions from {file_name}.")
    return fish_3d_positions


def plot_fish_trajectories(fish_3d_positions):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for fish_label, positions in fish_3d_positions.items():
        positions = np.array(positions)
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], label=fish_label)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.legend()
    plt.show()


# Set file name
# filename = "fish_3d_positions.npy"
filename = "fish_3d_positions_10_test.npy"

# 3D Trajectories
loaded_positions = load_fish_positions_from_npy(filename)
plot_fish_trajectories(loaded_positions)

# Animation for 3D Trajectories
# data_dict = np.load("fish_3d_positions.npy", allow_pickle=True).item()
data_dict = np.load("fish_3d_positions_10_test.npy", allow_pickle=True).item()

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

all_data = np.concatenate([np.array(data) for data in data_dict.values()])
ax.set_xlim(np.min(all_data[:, 0]), np.max(all_data[:, 0]))
ax.set_ylim(np.min(all_data[:, 1]), np.max(all_data[:, 1]))
ax.set_zlim(np.min(all_data[:, 2]), np.max(all_data[:, 2]))

lines = {}
for fish_key in data_dict.keys():

    line, = ax.plot([], [], [], marker="o", markersize=5, label=fish_key)
    lines[fish_key] = line


# Update Frames
def update(frame):
    for fish_key, line in lines.items():
        trajectory_data = np.array(data_dict[fish_key])
        line.set_data(trajectory_data[:frame, 0], trajectory_data[:frame, 1])
        line.set_3d_properties(trajectory_data[:frame, 2])
    return lines.values()


ani = FuncAnimation(fig, update, frames=len(next(iter(data_dict.values()))), interval=50, blit=True)

# Save as mp4
# output_file = "fish_trajectories.mp4"
# ani.save(output_file, writer="ffmpeg", fps=20)
# print(f"Animation saved as {output_file}")

plt.legend(loc="upper right")
plt.show()
