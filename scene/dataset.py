from torch.utils.data import Dataset
from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal, focal2fov
import torch
from utils.camera_utils import loadCam
from utils.graphics_utils import focal2fov


class FourDGSdataset(Dataset):
    def __init__(
        self,
        dataset,
        args,
        dataset_type
    ):
        self.dataset = dataset
        self.args = args
        self.dataset_type=dataset_type
    def __getitem__(self, index):
        # breakpoint()

        if self.dataset_type != "PanopticSports":
            try: # Diva360/DFA
                image, w2c, time, cx_px, cy_px, fx, fy = self.dataset[index]
                R,T = w2c
                image_width = image.shape[2]
                image_height = image.shape[1]
                FovX = focal2fov(fx, image_width)
                FovY = focal2fov(fy, image_height)
                
                return Camera(colmap_id=index,R=R,T=T,FoVx=FovX,FoVy=FovY,image=image,gt_alpha_mask=None,
                              image_name=f"{index}",uid=index,data_device=torch.device("cuda"),fid=time,
                              cx_px=cx_px, cy_px=cy_px)
            except: # single image per camera, ex) dnerf
                caminfo = self.dataset[index]
                image = caminfo.image
                R = caminfo.R
                T = caminfo.T
                FovX = caminfo.FovX
                FovY = caminfo.FovY
                time = caminfo.time
                mask = caminfo.mask
                
                return Camera(colmap_id=index,R=R,T=T,FoVx=FovX,FoVy=FovY,image=image,gt_alpha_mask=None,
                              image_name=f"{index}",uid=index,data_device=torch.device("cuda"),time=time, mask=mask)
        else:
            return self.dataset[index]
    def __len__(self):
        
        return len(self.dataset)


def get_cameraList_diva360(dataset, args, dataset_type):
    camera_list = []
    dataset = FourDGSdataset(dataset, args, dataset_type)
    for camera in dataset:
        camera_list.append(camera)
    return camera_list