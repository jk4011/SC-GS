import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from utils.graphics_utils import focal2fov
from scene.colmap_loader import qvec2rotmat
from scene.dataset_readers import CameraInfo
# from scene.neural_3D_dataset_NDC import get_spiral
from torchvision import transforms as T
import json
from scipy.spatial.transform import Rotation
import torch


def diva360_to_colmap(c2w):
    c2w[2, :] *= -1  # flip whole world upside down
    c2w = c2w[[1, 0, 2, 3], :]
    c2w[0:3, 1] *= -1  # flip the y and z axis
    c2w[0:3, 2] *= -1

    w2c = np.linalg.inv(c2w)
    rot = w2c[:3, :3]
    tvec = w2c[:3, -1]

    rotation = Rotation.from_matrix(rot)
    qvec = rotation.as_quat()  # Returns [x, y, z, w]
    qvec = np.array(qvec)[[3, 0, 1, 2]]

    return qvec, tvec

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
        2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
        2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
        1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
        2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
        2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
        1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])


class Diva360_dataset(Dataset):
    def __init__(
        self,
        cam_folder,
        split,
        frame_from=None,
        frame_to=None,
        cam_idx=None,
        white_background=True
    ):

        self.white_background = white_background
        self.transform = T.ToTensor()

        # with open(os.path.join(cam_folder, f"transforms_merged.json"), "r") as f:
        #     meta = json.load(f)
        # frames = meta["frames"] # transforms_train or transforms_test

        # self.focal = [frames[0]["fl_y"], frames[0]["fl_x"]]
        # height = frames[0]["h"]
        # width = frames[0]["w"]
        # self.FovY = focal2fov(self.focal[0], height)
        # self.FovX = focal2fov(self.focal[1], width)
        
        # 이미지 경로, 포즈, 시간 로드
        image_info_all = self.load_images_path(cam_folder, None, None, split, frame_from=frame_from, frame_to=frame_to, cam_idx=cam_idx)
        self.image_paths, self.image_poses, self.image_times, self.cxs, self.cys, self.fxs, self.fys = image_info_all

        # 비디오 카메라 정보 초기화
        # if split == "test":
        #     self.video_cam_infos = self.get_video_cam_infos(cam_folder)
    
    def load_images_path(self, cam_folder, cam_extrinsics, cam_intrinsics, split, frame_from, frame_to, cam_idx=None):
        # JSON 파일 로드
        from scipy.spatial.transform import Rotation as Rot
        
        with open(os.path.join(cam_folder, f"transforms_merged.json"), "r") as f:
            meta = json.load(f)
        
        image_paths = []
        image_poses = []
        image_times = []
        cx_px = []
        cy_px = []
        fxs = []
        fys = []

        # image_length = len(os.listdir(os.path.join(cam_folder, "cam00"))) # change cam__ if dir does not exist
        image_length = 1

        if cam_idx is None: # original logic
            for i, frame in enumerate(meta["frames"]): ## every element in "frames"            
                # transform_matrix에서 카메라 포즈 가져오기, Blender/OpenGL c2w
                c2w = np.array(frame["transform_matrix"])
                cx = frame["cx"]
                cy = frame["cy"]
                fx = frame["fl_x"]
                fy = frame["fl_y"]

                # OpenGL에서 COLMAP 좌표계로 변환
                c2w[:3, 1:3] *= -1
                
                # world-to-camera 변환
                w2c = np.linalg.inv(c2w)
                R = np.transpose(w2c[:3, :3])
                T = w2c[:3, 3]
                # breakpoint()

                file_path = frame["file_path"]
                cam_num = os.path.dirname(file_path)
                images_folder = os.path.join(cam_folder, cam_num) # .../cam01

                for frame_idx in range(image_length):
                    # 리스트에 추가
                    image_path = os.path.join(images_folder, f"frame_{str(frame_idx).zfill(5)}.png") # start from frame_00000.png 
                    image_paths.append(image_path)
                    image_poses.append((R, T))
                    image_times.append(float((frame_idx+1)/image_length))
                    cx_px.append(cx)
                    cy_px.append(cy)
                    fxs.append(fx)
                    fys.append(fy)
                # breakpoint()
            # breakpoint()
            return image_paths, image_poses, image_times, cx_px, cy_px, fxs, fys
        elif split == "train": # Gaussian recon stage, all image of frame_from + one image of frame_to in cam_idx

            last_element = {}
            cam_idx = f"cam{cam_idx}" # cam01
            for i, frame in enumerate(meta["frames"]): ## every element in "frames"            
                c2w = np.array(frame["transform_matrix"])
                cx = frame["cx"]
                cy = frame["cy"]
                fx = frame["fl_x"]
                fy = frame["fl_y"]

                # OpenGL에서 COLMAP 좌표계로 변환
                c2w[:3, 1:3] *= -1
                
                # world-to-camera 변환
                w2c = np.linalg.inv(c2w)
                R = np.transpose(w2c[:3, :3])
                T = w2c[:3, 3]
                # breakpoint()

                file_path = frame["file_path"]
                cam_num = os.path.dirname(file_path)
                # breakpoint()
                if cam_num == cam_idx:
                    # breakpoint()
                    last_element['image_path'] = os.path.join(cam_folder, cam_num, f"frame_{str(frame_to).zfill(5)}.png")
                    last_element['pose'] = (R, T)
                    last_element['cx'] = cx
                    last_element['cy'] = cy
                    last_element['fx'] = fx
                    last_element['fy'] = fy
                    
                images_folder = os.path.join(cam_folder, cam_num) # .../cam01
                image_path = os.path.join(images_folder, f"frame_{str(frame_from).zfill(5)}.png")
                image_paths.append(image_path)
                image_poses.append((R, T))
                image_times.append(0.5)
                cx_px.append(cx)
                cy_px.append(cy)
                fxs.append(fx)
                fys.append(fy)

            image_paths.append(last_element['image_path'])
            image_poses.append(last_element['pose'])
            image_times.append(1)
            cx_px.append(last_element['cx'])
            cy_px.append(last_element['cy'])
            fxs.append(last_element['fx'])
            fys.append(last_element['fy'])

            # breakpoint()
            return image_paths, image_poses, image_times, cx_px, cy_px, fxs, fys
        
        elif split == "test": # deform stage, N-1 image of frame_to not in cam_idx
            cam_idx = f"cam{cam_idx}" # cam01
            for i, frame in enumerate(meta["frames"]):           
                c2w = np.array(frame["transform_matrix"])
                cx = frame["cx"]
                cy = frame["cy"]
                fx = frame["fl_x"]
                fy = frame["fl_y"]

                # OpenGL에서 COLMAP 좌표계로 변환
                c2w[:3, 1:3] *= -1
                
                # world-to-camera 변환
                w2c = np.linalg.inv(c2w)
                R = np.transpose(w2c[:3, :3])
                T = w2c[:3, 3]
                # breakpoint()

                file_path = frame["file_path"]
                cam_num = os.path.dirname(file_path)
                
                if cam_num != cam_idx: # N-1 images
                    images_folder = os.path.join(cam_folder, cam_num) # .../cam01

                    image_path = os.path.join(images_folder, f"frame_{str(frame_to).zfill(5)}.png") # start from frame_00000.png 
                    image_paths.append(image_path)
                    image_poses.append((R, T))
                    image_times.append(1)
                    cx_px.append(cx)
                    cy_px.append(cy)
                    fxs.append(fx)
                    fys.append(fy)
                
            # breakpoint()
            return image_paths, image_poses, image_times, cx_px, cy_px, fxs, fys
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        img = Image.open(self.image_paths[index]) #.convert("RGB")
        im_data = np.array(img.convert("RGBA"))
        
        bg = np.array([1,1,1]) if self.white_background else np.array([0, 0, 0])
        norm_data = im_data / 255.0
        arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])

        img = torch.from_numpy(arr).permute(2, 0, 1)  # [3, H, W]
        # image_width = img.shape[2]
        # image_height = img.shape[1]
        
        return img, self.image_poses[index], self.image_times[index], \
            self.cxs[index], self.cys[index], self.fxs[index], self.fys[index]
    
    def load_pose(self, index):
        return self.image_poses[index]
