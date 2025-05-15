import os
import numpy as np


def dfa_to_colmap(c2w):
    c2w[2, :] *= -1  # flip whole world upside down
    # change deformation
    c2w = c2w[[1, 0, 2, 3], :]
    c2w = c2w[:, [1, 2, 0, 3]]

    w2c = np.linalg.inv(c2w)
    return w2c


def _load_extrinsics(data_dir):
    extrinsics_path = os.path.join(data_dir, "Campose.inf")
    extrinsics_list = []
    with open(extrinsics_path, "r") as f:
        lines = [line.strip() for line in f.readlines() if line.strip() != ""]
    for line in lines:
        parts = line.split()
        if len(parts) != 12:
            raise Exception(f"Line in CamPose.inf does not contain 12 numbers: {line}")
        nums = [float(x) for x in parts]
        mat_4x3 = np.array(nums).reshape(4, 3)
        mat_4x4 = np.zeros((4, 4))
        mat_4x4[:3, :3] = mat_4x3[:3, :3].T
        mat_4x4[:3, 3] = mat_4x3[3, :]
        mat_4x4[3, :] = np.array([0, 0, 0, 1])
        extrinsics_list.append(mat_4x4)
    
    return extrinsics_list


def load_extrinsics(data_dir):
    extrinsics_list = _load_extrinsics(data_dir)
    n_cameras = len(extrinsics_list)
    
    w2c_list = []
    file_name_list = []
    for view in range(n_cameras):
        file_name = f"img_{view:04d}_rgba.png"
        image_path = os.path.join(data_dir, "images", file_name)
        if not os.path.exists(image_path):
            print(f"Warning: {image_path} does not exist.")
            continue
        transform = extrinsics_list[view]
        w2c = dfa_to_colmap(transform)
        w2c_list.append(w2c)
        file_name_list.append(file_name)
    
    return w2c_list, file_name_list


def load_intrinsics(data_dir):
    intrinsics_path = os.path.join(data_dir, "Intrinsic.inf")
    intrinsic_list = []
    with open(intrinsics_path, "r") as f:
        lines = [line.strip() for line in f.readlines() if line.strip() != ""]

    i = 0
    while i < len(lines):
        cam_index = int(lines[i])
        row1 = [float(x) for x in lines[i + 1].split()]
        row2 = [float(x) for x in lines[i + 2].split()]
        row3 = [float(x) for x in lines[i + 3].split()]
        
        # Intrinsics matrix:
        # [ fx    0   cx ]
        # [  0   fy   cy ]
        # [  0    0    1 ]
        fx = row1[0]
        cx = row1[2]
        fy = row2[1]
        cy = row2[2]
        intrinsic_list.append((fx, fy, cx, cy))
        i += 4
    
    return intrinsic_list


if __name__ == "__main__":
    dir_from = "/data2/wlsgur4011/GESI/SC-GS/data/DFA_processed/beagle_dog(s1)/0"
    w2c_list1, file_name_list1 = load_extrinsics(dir_from)
    intrinsic_list1 = load_intrinsics(dir_from)

    dir_to = "/data2/wlsgur4011/GESI/SC-GS/data/DFA_processed/beagle_dog(s1)/5"
    w2c_list2, file_name_list2 = load_extrinsics(dir_to)
    intrinsic_list2 = load_intrinsics(dir_to)

    cam_idx = 0
    w2d_list_train = [w2c_list1 + [w2c_list2[cam_idx]]]
    w2d_list_test = [w2c for (i, w2c) in enumerate(w2c_list2) if i != cam_idx]
    breakpoint()
    
