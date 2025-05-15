#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
# os.environ["WANDB_API_KEY"] = "23301237f7961e638441b5ffc9c9d869f6799254"
import wandb

# os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
import time
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui, render_flow
import sys
from scene import Scene, GaussianModel, DeformModel
from utils.general_utils import safe_state, get_linear_noise_func
import uuid
import tqdm
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from train import training_report
import math
from cam_utils import OrbitCamera
import numpy as np
import dearpygui.dearpygui as dpg
import imageio
import datetime
from PIL import Image
from train_gui_utils import DeformKeypoints
from scipy.spatial.transform import Rotation as R

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

from train_gaussian import train_step
from train_node import train_node_rendering_step


def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 1 / tanHalfFovX
    P[1, 1] = 1 / tanHalfFovY
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def landmark_interpolate(landmarks, steps, step, interpolation='log'):
    stage = (step >= np.array(steps)).sum()
    if stage == len(steps):
        return max(0, landmarks[-1])
    elif stage == 0:
        return 0
    else:
        ldm1, ldm2 = landmarks[stage-1], landmarks[stage]
        if ldm2 <= 0:
            return 0
        step1, step2 = steps[stage-1], steps[stage]
        ratio = (step - step1) / (step2 - step1)
        if interpolation == 'log':
            return np.exp(np.log(ldm1) * (1 - ratio) + np.log(ldm2) * ratio)
        elif interpolation == 'linear':
            return ldm1 * (1 - ratio) + ldm2 * ratio
        else:
            print(f'Unknown interpolation type: {interpolation}')
            raise NotImplementedError

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)


class MiniCam:
    def __init__(self, c2w, width, height, fovy, fovx, znear, zfar, fid):
        # c2w (pose) should be in NeRF convention.

        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.fid = fid
        self.c2w = c2w

        w2c = np.linalg.inv(c2w)

        # rectify...
        w2c[1:3, :3] *= -1
        w2c[:3, 3] *= -1

        self.world_view_transform = torch.tensor(w2c).transpose(0, 1).cuda().float()
        self.projection_matrix = (
            getProjectionMatrix(
                znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy
            )
            .transpose(0, 1)
            .cuda().float()
        )
        self.full_proj_transform = self.world_view_transform @ self.projection_matrix
        self.camera_center = -torch.tensor(c2w[:3, 3]).cuda()

    def reset_extrinsic(self, R, T):
        self.world_view_transform = torch.tensor(getWorld2View2(R, T)).transpose(0, 1).cuda()
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]


class GUI:
    def __init__(self, args, dataset, opt, pipe, testing_iterations, saving_iterations) -> None:
        from jhutil import color_log; color_log(1111, "prepare training")
        self.dataset = dataset
        self.args = args
        self.opt = opt
        self.pipe = pipe
        self.testing_iterations = testing_iterations
        self.saving_iterations = saving_iterations

        if self.opt.progressive_train:
            self.opt.iterations_node_sampling = max(self.opt.iterations_node_sampling, int(self.opt.progressive_stage_steps / self.opt.progressive_stage_ratio))
            self.opt.iterations_node_rendering = max(self.opt.iterations_node_rendering, self.opt.iterations_node_sampling + 2000)
            print(f'Progressive trian is on. Adjusting the iterations node sampling to {self.opt.iterations_node_sampling} and iterations node rendering {self.opt.iterations_node_rendering}')

        self.tb_writer = prepare_output_and_logger(dataset)
        self.deform = DeformModel(K=self.dataset.K, deform_type=self.dataset.deform_type, is_blender=self.dataset.is_blender, skinning=self.args.skinning, hyper_dim=self.dataset.hyper_dim, node_num=self.dataset.node_num, pred_opacity=self.dataset.pred_opacity, pred_color=self.dataset.pred_color, use_hash=self.dataset.use_hash, hash_time=self.dataset.hash_time, d_rot_as_res=self.dataset.d_rot_as_res and not self.dataset.d_rot_as_rotmat, local_frame=self.dataset.local_frame, progressive_brand_time=self.dataset.progressive_brand_time, with_arap_loss=not self.opt.no_arap_loss, max_d_scale=self.dataset.max_d_scale, enable_densify_prune=self.opt.node_enable_densify_prune, is_scene_static=dataset.is_scene_static)

        deform_loaded = self.deform.load_weights(dataset.model_path, iteration=-1)
        deform_loaded = False
        self.deform.train_setting(opt)

        gs_fea_dim = self.deform.deform.node_num if args.skinning and self.deform.name == 'node' else self.dataset.hyper_dim
        self.gaussians = GaussianModel(dataset.sh_degree, fea_dim=gs_fea_dim, with_motion_mask=self.dataset.gs_with_motion_mask)

        from jhutil import color_log; color_log("aaaa", "load secne")
        self.scene = Scene(dataset, self.gaussians, load_iteration=-1, is_diva360=args.is_diva360)
        self.gaussians.training_setup(opt)
        if self.deform.name == 'node' and not deform_loaded:
            if not self.dataset.is_blender:
                if self.opt.random_init_deform_gs:
                    num_pts = 100_000
                    print(f"Generating random point cloud ({num_pts})...")
                    xyz = torch.rand((num_pts, 3)).float().cuda() * 2 - 1
                    mean, scale = self.gaussians.get_xyz.mean(dim=0), self.gaussians.get_xyz.std(dim=0).mean() * 3
                    xyz = xyz * scale + mean
                    self.deform.deform.init(init_pcl=xyz, force_init=True, opt=self.opt, as_gs_force_with_motion_mask=self.dataset.as_gs_force_with_motion_mask, force_gs_keep_all=True)
                else:
                    print('Initialize nodes with COLMAP point cloud.')
                    self.deform.deform.init(init_pcl=self.gaussians.get_xyz, force_init=True, opt=self.opt, as_gs_force_with_motion_mask=self.dataset.as_gs_force_with_motion_mask, force_gs_keep_all=self.dataset.init_isotropic_gs_with_all_colmap_pcl)
            else:
                print('Initialize nodes with Random point cloud.')
                self.deform.deform.init(init_pcl=self.gaussians.get_xyz, force_init=True, opt=self.opt, as_gs_force_with_motion_mask=False, force_gs_keep_all=args.skinning)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        self.iter_start = torch.cuda.Event(enable_timing=True)
        self.iter_end = torch.cuda.Event(enable_timing=True)
        self.iteration = 1 if self.scene.loaded_iter is None else self.scene.loaded_iter
        self.iteration_node_rendering = 1 if self.scene.loaded_iter is None else self.opt.iterations_node_rendering

        self.viewpoint_stack = None
        self.ema_loss_for_log = 0.0
        self.best_psnr = 0.0
        self.best_ssim = 0.0
        self.best_ms_ssim = 0.0
        self.best_lpips = np.inf
        self.best_alex_lpips = np.inf
        self.best_iteration = 0
        self.progress_bar = tqdm.tqdm(range(opt.iterations), desc="Training progress")
        self.smooth_term = get_linear_noise_func(lr_init=0.1, lr_final=1e-15, lr_delay_mult=0.01, max_steps=20000)

        # For UI
        self.visualization_mode = 'RGB'

        self.gui = args.gui # enable gui
        self.W = args.W
        self.H = args.H
        self.cam = OrbitCamera(args.W, args.H, r=args.radius, fovy=args.fovy)
        self.vis_scale_const = None
        self.mode = "render"
        self.seed = "random"
        self.buffer_image = np.ones((self.W, self.H, 3), dtype=np.float32)
        self.training = False
        self.video_speed = 1.

        # For Animation
        self.animation_time = 0.
        self.is_animation = False
        self.need_update_overlay = False
        self.buffer_overlay = None
        self.animation_trans_bias = None
        self.animation_rot_bias = None
        self.animation_scaling_bias = None
        self.animate_tool = None
        self.is_training_animation_weight = False
        self.is_training_motion_analysis = False
        self.motion_genmodel = None
        self.motion_animation_d_values = None
        self.showing_overlay = True
        self.should_save_screenshot = False
        self.should_vis_trajectory = False
        self.screenshot_id = 0
        self.screenshot_sv_path = f'./screenshot/' + datetime.datetime.now().strftime('%Y-%m-%d')
        self.traj_overlay = None
        self.vis_traj_realtime = False
        self.last_traj_overlay_type = None
        self.view_animation = True
        self.n_rings_N = 2
        # Use ARAP or Generative Model to Deform
        self.deform_mode = "arap_iterative"
        self.should_render_customized_trajectory = False
        self.should_render_customized_trajectory_spiral = False

        if self.gui:
            dpg.create_context()
            self.register_dpg()
            self.test_step()

    def animation_initialize(self, use_traj=True):
        from lap_deform import LapDeform
        gaussians = self.deform.deform.as_gaussians
        fid = torch.tensor(self.animation_time).cuda().float()
        time_input = fid.unsqueeze(0).expand(gaussians.get_xyz.shape[0], -1)
        values = self.deform.deform.node_deform(t=time_input)
        trans = values['d_xyz']
        pcl = gaussians.get_xyz + trans

        if use_traj:
            print('Trajectory analysis!')
            t_samp_num = 16
            t_samp = torch.linspace(0, 1, t_samp_num).cuda().float()
            time_input = t_samp[None, :, None].expand(gaussians.get_xyz.shape[0], -1, 1)
            trajectory = self.deform.deform.node_deform(t=time_input)['d_xyz'] + gaussians.get_xyz[:, None]
        else:
            trajectory = None

        self.animate_init_values = values
        self.animate_tool = LapDeform(init_pcl=pcl, K=4, trajectory=trajectory, node_radius=self.deform.deform.node_radius.detach())
        self.keypoint_idxs = []
        self.keypoint_3ds = []
        self.keypoint_labels = []
        self.keypoint_3ds_delta = []
        self.keypoint_idxs_to_drag = []
        self.deform_keypoints = DeformKeypoints()
        self.animation_trans_bias = None
        self.animation_rot_bias = None
        self.buffer_overlay = None
        print('Initialize Animation Model with %d control nodes' % len(pcl))

    def animation_reset(self):
        self.animate_tool.reset()
        self.keypoint_idxs = []
        self.keypoint_3ds = []
        self.keypoint_labels = []
        self.keypoint_3ds_delta = []
        self.keypoint_idxs_to_drag = []
        self.deform_keypoints = DeformKeypoints()
        self.animation_trans_bias = None
        self.animation_rot_bias = None
        self.buffer_overlay = None
        self.motion_animation_d_values = None
        print('Reset Animation Model ...')

    def __del__(self):
        if self.gui:
            dpg.destroy_context()

    def register_dpg(self):
        ### register texture
        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(
                self.W,
                self.H,
                self.buffer_image,
                format=dpg.mvFormat_Float_rgb,
                tag="_texture",
            )

        ### register window
        # the rendered image, as the primary window
        with dpg.window(
            tag="_primary_window",
            width=self.W,
            height=self.H,
            pos=[0, 0],
            no_move=True,
            no_title_bar=True,
            no_scrollbar=True,
        ):
            # add the texture
            dpg.add_image("_texture")

        # dpg.set_primary_window("_primary_window", True)

        # control window
        with dpg.window(
            label="Control",
            tag="_control_window",
            width=600,
            height=self.H,
            pos=[self.W, 0],
            no_move=True,
            no_title_bar=True,
        ):
            # button theme
            with dpg.theme() as theme_button:
                with dpg.theme_component(dpg.mvButton):
                    dpg.add_theme_color(dpg.mvThemeCol_Button, (23, 3, 18))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (51, 3, 47))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (83, 18, 83))
                    dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
                    dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 3, 3)

            # timer stuff
            with dpg.group(horizontal=True):
                dpg.add_text("Infer time: ")
                dpg.add_text("no data", tag="_log_infer_time")

            def callback_setattr(sender, app_data, user_data):
                setattr(self, user_data, app_data)

            # init stuff
            with dpg.collapsing_header(label="Initialize", default_open=True):

                # seed stuff
                def callback_set_seed(sender, app_data):
                    self.seed = app_data
                    self.seed_everything()

                dpg.add_input_text(
                    label="seed",
                    default_value=self.seed,
                    on_enter=True,
                    callback=callback_set_seed,
                )

                # input stuff
                def callback_select_input(sender, app_data):
                    # only one item
                    for k, v in app_data["selections"].items():
                        dpg.set_value("_log_input", k)
                        self.load_input(v)

                    self.need_update = True

                with dpg.file_dialog(
                    directory_selector=False,
                    show=False,
                    callback=callback_select_input,
                    file_count=1,
                    tag="file_dialog_tag",
                    width=700,
                    height=400,
                ):
                    dpg.add_file_extension("Images{.jpg,.jpeg,.png}")

                with dpg.group(horizontal=True):
                    dpg.add_button(
                        label="input",
                        callback=lambda: dpg.show_item("file_dialog_tag"),
                    )
                    dpg.add_text("", tag="_log_input")

                # save current model
                with dpg.group(horizontal=True):
                    dpg.add_text("Visualization: ")

                    def callback_vismode(sender, app_data, user_data):
                        self.visualization_mode = user_data

                    dpg.add_button(
                        label="RGB",
                        tag="_button_vis_rgb",
                        callback=callback_vismode,
                        user_data='RGB',
                    )
                    dpg.bind_item_theme("_button_vis_rgb", theme_button)

                    def callback_vis_traj_realtime():
                        self.vis_traj_realtime = not self.vis_traj_realtime
                        if not self.vis_traj_realtime:
                            self.traj_coor = None
                        print('Visualize trajectory: ', self.vis_traj_realtime)
                    dpg.add_button(
                        label="Traj",
                        tag="_button_vis_traj",
                        callback=callback_vis_traj_realtime,
                    )
                    dpg.bind_item_theme("_button_vis_traj", theme_button)

                    dpg.add_button(
                        label="MotionMask",
                        tag="_button_vis_motion_mask",
                        callback=callback_vismode,
                        user_data='MotionMask',
                    )
                    dpg.bind_item_theme("_button_vis_motion_mask", theme_button)

                    dpg.add_button(
                        label="NodeMotion",
                        tag="_button_vis_node_motion",
                        callback=callback_vismode,
                        user_data='MotionMask_Node',
                    )
                    dpg.bind_item_theme("_button_vis_node_motion", theme_button)

                    dpg.add_button(
                        label="Node",
                        tag="_button_vis_node",
                        callback=callback_vismode,
                        user_data='Node',
                    )
                    dpg.bind_item_theme("_button_vis_node", theme_button)

                    dpg.add_button(
                        label="Dynamic",
                        tag="_button_vis_Dynamic",
                        callback=callback_vismode,
                        user_data='Dynamic',
                    )
                    dpg.bind_item_theme("_button_vis_Dynamic", theme_button)

                    dpg.add_button(
                        label="Static",
                        tag="_button_vis_Static",
                        callback=callback_vismode,
                        user_data='Static',
                    )
                    dpg.bind_item_theme("_button_vis_Static", theme_button)

                with dpg.group(horizontal=True):
                    dpg.add_text("Scale Const: ")
                    def callback_vis_scale_const(sender):
                        self.vis_scale_const = 10 ** dpg.get_value(sender)
                        self.need_update = True
                    dpg.add_slider_float(
                        label="Log vis_scale_const (For debugging)",
                        default_value=-3,
                        max_value=-.5,
                        min_value=-5,
                        callback=callback_vis_scale_const,
                    )

                # save current model
                with dpg.group(horizontal=True):
                    dpg.add_text("Temporal Speed: ")
                    self.video_speed = 1.
                    def callback_speed_control(sender):
                        self.video_speed = 10 ** dpg.get_value(sender)
                        self.need_update = True
                    dpg.add_slider_float(
                        label="Play speed",
                        default_value=0.,
                        max_value=3.,
                        min_value=-3.,
                        callback=callback_speed_control,
                    )
                
                # save current model
                with dpg.group(horizontal=True):
                    dpg.add_text("Save: ")

                    def callback_save(sender, app_data, user_data):
                        print("\n[ITER {}] Saving Gaussians".format(self.iteration))
                        self.scene.save(self.iteration)
                        self.deform.save_weights(self.args.model_path, self.iteration)
                    dpg.add_button(
                        label="model",
                        tag="_button_save_model",
                        callback=callback_save,
                        user_data='model',
                    )
                    dpg.bind_item_theme("_button_save_model", theme_button)

                    def callback_screenshot(sender, app_data):
                        self.should_save_screenshot = True
                    dpg.add_button(
                        label="screenshot", tag="_button_screenshot", callback=callback_screenshot
                    )
                    dpg.bind_item_theme("_button_screenshot", theme_button)

                    def callback_render_traj(sender, app_data):
                        self.should_render_customized_trajectory = True
                    dpg.add_button(
                        label="render_traj", tag="_button_render_traj", callback=callback_render_traj
                    )
                    dpg.bind_item_theme("_button_render_traj", theme_button)

                    def callback_render_traj(sender, app_data):
                        self.should_render_customized_trajectory_spiral = not self.should_render_customized_trajectory_spiral
                        if self.should_render_customized_trajectory_spiral:
                            dpg.configure_item("_button_render_traj_spiral", label="camera")
                        else:
                            dpg.configure_item("_button_render_traj_spiral", label="spiral")
                    dpg.add_button(
                        label="spiral", tag="_button_render_traj_spiral", callback=callback_render_traj
                    )
                    dpg.bind_item_theme("_button_render_traj_spiral", theme_button)
                    
                    def callback_cache_nn(sender, app_data):
                        self.deform.deform.cached_nn_weight = not self.deform.deform.cached_nn_weight
                        print(f'Cached nn weight for higher rendering speed: {self.deform.deform.cached_nn_weight}')
                    dpg.add_button(
                        label="cache_nn", tag="_button_cache_nn", callback=callback_cache_nn
                    )
                    dpg.bind_item_theme("_button_cache_nn", theme_button)

            # training stuff
            with dpg.collapsing_header(label="Train", default_open=True):
                # lr and train button
                with dpg.group(horizontal=True):
                    dpg.add_text("Train: ")

                    def callback_train(sender, app_data):
                        if self.training:
                            self.training = False
                            dpg.configure_item("_button_train", label="start")
                        else:
                            # self.prepare_train()
                            self.training = True
                            dpg.configure_item("_button_train", label="stop")

                    dpg.add_button(
                        label="start", tag="_button_train", callback=callback_train
                    )
                    dpg.bind_item_theme("_button_train", theme_button)

                    def callback_save_deform_kpt(sender, app_data):
                        from utils.pickle_utils import save_obj
                        self.deform_keypoints.t = self.animation_time
                        save_obj(path=self.args.model_path+'/deform_kpt.pickle', obj=self.deform_keypoints)
                        print('Save kpt done!')
                    dpg.add_button(
                        label="save_deform_kpt", tag="_button_save_deform_kpt", callback=callback_save_deform_kpt
                    )
                    dpg.bind_item_theme("_button_save_deform_kpt", theme_button)

                    def callback_load_deform_kpt(sender, app_data):
                        from utils.pickle_utils import load_obj
                        self.deform_keypoints = load_obj(path=self.args.model_path+'/deform_kpt.pickle')
                        self.animation_time = self.deform_keypoints.t
                        with torch.no_grad():
                            animated_pcl, quat, ani_d_scaling = self.animate_tool.deform_arap(handle_idx=self.deform_keypoints.get_kpt_idx(), handle_pos=self.deform_keypoints.get_deformed_kpt_np(), return_R=True)
                            self.animation_trans_bias = animated_pcl - self.animate_tool.init_pcl
                            self.animation_rot_bias = quat
                            self.animation_scaling_bias = ani_d_scaling
                        self.need_update_overlay = True
                        print('Load kpt done!')
                    dpg.add_button(
                        label="load_deform_kpt", tag="_button_load_deform_kpt", callback=callback_load_deform_kpt
                    )
                    dpg.bind_item_theme("_button_load_deform_kpt", theme_button)

                with dpg.group(horizontal=True):
                    dpg.add_text("", tag="_log_train_psnr")
                with dpg.group(horizontal=True):
                    dpg.add_text("", tag="_log_train_log")

            # rendering options
            with dpg.collapsing_header(label="Rendering", default_open=True):
                # mode combo
                def callback_change_mode(sender, app_data):
                    self.mode = app_data
                    self.need_update = True

                dpg.add_combo(
                    ("render", "depth", "alpha", "normal_dep"),
                    label="mode",
                    default_value=self.mode,
                    callback=callback_change_mode,
                )

                # fov slider
                def callback_set_fovy(sender, app_data):
                    self.cam.fovy = np.deg2rad(app_data)
                    self.need_update = True

                dpg.add_slider_int(
                    label="FoV (vertical)",
                    min_value=1,
                    max_value=120,
                    format="%d deg",
                    default_value=np.rad2deg(self.cam.fovy),
                    callback=callback_set_fovy,
                )
            
            # animation options
            with dpg.collapsing_header(label="Motion Editing", default_open=True):
                # save current model
                with dpg.group(horizontal=True):
                    dpg.add_text("Freeze Time: ")
                    def callback_animation_time(sender):
                        self.animation_time = dpg.get_value(sender)
                        self.is_animation = True
                        self.need_update = True
                        # self.animation_initialize()
                    dpg.add_slider_float(
                        label="",
                        default_value=0.,
                        max_value=1.,
                        min_value=0.,
                        callback=callback_animation_time,
                    )

                with dpg.group(horizontal=True):
                    def callback_animation_mode(sender, app_data):
                        with torch.no_grad():
                            self.is_animation = not self.is_animation
                            if self.is_animation:
                                if not hasattr(self, 'animate_tool') or self.animate_tool is None:
                                    self.animation_initialize()
                    dpg.add_button(
                        label="Play",
                        tag="_button_vis_animation",
                        callback=callback_animation_mode,
                        user_data='Animation',
                    )
                    dpg.bind_item_theme("_button_vis_animation", theme_button)

                    def callback_animation_initialize(sender, app_data):
                        with torch.no_grad():
                            self.is_animation = True
                            self.animation_initialize()
                    dpg.add_button(
                        label="Init Graph",
                        tag="_button_init_graph",
                        callback=callback_animation_initialize,
                    )
                    dpg.bind_item_theme("_button_init_graph", theme_button)

                    def callback_clear_animation(sender, app_data):
                        with torch.no_grad():
                            self.is_animation = True
                            self.animation_reset()
                    dpg.add_button(
                        label="Clear Graph",
                        tag="_button_clc_animation",
                        callback=callback_clear_animation,
                    )
                    dpg.bind_item_theme("_button_clc_animation", theme_button)

                    def callback_overlay(sender, app_data):
                        if self.showing_overlay:
                            self.showing_overlay = False
                            dpg.configure_item("_button_train_motion_gen", label="show overlay")
                        else:
                            self.showing_overlay = True
                            dpg.configure_item("_button_train_motion_gen", label="close overlay")                    
                    dpg.add_button(
                        label="close overlay", tag="_button_overlay", callback=callback_overlay
                    )
                    dpg.bind_item_theme("_button_overlay", theme_button)

                    def callback_save_ckpt(sender, app_data):
                        from utils.pickle_utils import save_obj
                        if not self.is_animation:
                            print('Please switch to animation mode!')
                        deform_keypoint_files = sorted([file for file in os.listdir(os.path.join(self.args.model_path)) if file.startswith('deform_keypoints') and file.endswith('.pickle')])
                        if len(deform_keypoint_files) > 0:
                            newest_id = int(deform_keypoint_files[-1].split('.')[0].split('_')[-1])
                        else:
                            newest_id = -1
                        save_obj(os.path.join(self.args.model_path, f'deform_keypoints_{newest_id+1}.pickle'), [self.deform_keypoints, self.animation_time])
                    dpg.add_button(
                        label="sv_kpt", tag="_button_save_kpt", callback=callback_save_ckpt
                    )
                    dpg.bind_item_theme("_button_save_kpt", theme_button)

                with dpg.group(horizontal=True):
                    def callback_change_deform_mode(sender, app_data):
                        self.deform_mode = app_data
                        self.need_update = True
                    dpg.add_combo(
                        ("arap_iterative", "arap_from_init"),
                        label="Editing Mode",
                        default_value=self.deform_mode,
                        callback=callback_change_deform_mode,
                    )

                with dpg.group(horizontal=True):
                    def callback_change_n_rings_N(sender, app_data):
                        self.n_rings_N = int(app_data)
                    dpg.add_combo(
                        ("0", "1", "2", "3", "4"),
                        label="n_rings",
                        default_value="2",
                        callback=callback_change_n_rings_N,
                    )
                    

        def callback_set_mouse_loc(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return
            self.mouse_loc = np.array(app_data)

        def callback_keypoint_drag(sender, app_data):
            if not self.is_animation:
                print("Please switch to animation mode!")
                return
            if not dpg.is_item_focused("_primary_window"):
                return
            if len(self.deform_keypoints.get_kpt()) == 0:
                return
            if self.animate_tool is None:
                self.animation_initialize()
            # 2D to 3D delta
            dx = app_data[1]
            dy = app_data[2]
            if dpg.is_key_down(dpg.mvKey_R):
                side = self.cam.rot.as_matrix()[:3, 0]
                up = self.cam.rot.as_matrix()[:3, 1]
                forward = self.cam.rot.as_matrix()[:3, 2]
                rotvec_z = forward * np.radians(-0.05 * dx)
                rot_mat = (R.from_rotvec(rotvec_z)).as_matrix()
                self.deform_keypoints.set_rotation_delta(rot_mat)
            else:
                delta = 0.00010 * self.cam.rot.as_matrix()[:3, :3] @ np.array([dx, -dy, 0])
                self.deform_keypoints.update_delta(delta)
                self.need_update_overlay = True

            if self.deform_mode.startswith("arap"):
                with torch.no_grad():
                    if self.deform_mode == "arap_from_init" or self.animation_trans_bias is None:
                        init_verts = None
                    else:
                        init_verts = self.animation_trans_bias + self.animate_tool.init_pcl
                    animated_pcl, quat, ani_d_scaling = self.animate_tool.deform_arap(handle_idx=self.deform_keypoints.get_kpt_idx(), handle_pos=self.deform_keypoints.get_deformed_kpt_np(), init_verts=init_verts, return_R=True)
                    self.animation_trans_bias = animated_pcl - self.animate_tool.init_pcl
                    self.animation_rot_bias = quat
                    self.animation_scaling_bias = ani_d_scaling

        def callback_keypoint_add(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return
            ##### select keypoints by shift + click
            if dpg.is_key_down(dpg.mvKey_S) or dpg.is_key_down(dpg.mvKey_D) or dpg.is_key_down(dpg.mvKey_F) or dpg.is_key_down(dpg.mvKey_A) or dpg.is_key_down(dpg.mvKey_Q):
                if not self.is_animation:
                    print("Please switch to animation mode!")
                    return
                # Rendering the image with node gaussians to select nodes as keypoints
                fid = torch.tensor(self.animation_time).cuda().float()
                cur_cam = MiniCam(
                    self.cam.pose,
                    self.W,
                    self.H,
                    self.cam.fovy,
                    self.cam.fovx,
                    self.cam.near,
                    self.cam.far,
                    fid = fid
                )
                with torch.no_grad():
                    time_input = self.deform.deform.expand_time(fid)
                    d_values = self.deform.step(self.gaussians.get_xyz.detach(), time_input, feature=self.gaussians.feature, is_training=False, motion_mask=self.gaussians.motion_mask, camera_center=cur_cam.camera_center, node_trans_bias=self.animation_trans_bias, node_rot_bias=self.animation_rot_bias, node_scaling_bias=self.animation_scaling_bias)
                    gaussians = self.gaussians
                    d_xyz, d_rotation, d_scaling, d_opacity, d_color = d_values['d_xyz'], d_values['d_rotation'], d_values['d_scaling'], d_values['d_opacity'], d_values['d_color']

                    out = render(viewpoint_camera=cur_cam, pc=gaussians, pipe=self.pipe, bg_color=self.background, d_xyz=d_xyz, d_rotation=d_rotation, d_scaling=d_scaling, d_opacity=d_opacity, d_color=d_color, d_rot_as_res=self.deform.d_rot_as_res)

                    # Project mouse_loc to points_3d
                    pw, ph = int(self.mouse_loc[0]), int(self.mouse_loc[1])

                    d = out['depth'][0][ph, pw]
                    z = cur_cam.zfar / (cur_cam.zfar - cur_cam.znear) * d - cur_cam.zfar * cur_cam.znear / (cur_cam.zfar - cur_cam.znear)
                    uvz = torch.tensor([((pw-.5)/self.W * 2 - 1) * d, ((ph-.5)/self.H*2-1) * d, z, d]).cuda().float().view(1, 4)
                    p3d = (uvz @ torch.inverse(cur_cam.full_proj_transform))[0, :3]

                    # Pick the closest node as the keypoint
                    node_trans = self.deform.deform.node_deform(time_input)['d_xyz']
                    if self.animation_trans_bias is not None:
                        node_trans = node_trans + self.animation_trans_bias
                    nodes = self.deform.deform.nodes[..., :3] + node_trans
                    keypoint_idxs = torch.tensor([(p3d - nodes).norm(dim=-1).argmin()]).cuda()

                if dpg.is_key_down(dpg.mvKey_A):
                    if True:
                        keypoint_idxs = self.animate_tool.add_n_ring_nbs(keypoint_idxs, n=self.n_rings_N)
                    keypoint_3ds = nodes[keypoint_idxs]
                    self.deform_keypoints.add_kpts(keypoint_3ds, keypoint_idxs)
                    print(f'Add kpt: {self.deform_keypoints.selective_keypoints_idx_list}')

                elif dpg.is_key_down(dpg.mvKey_S):
                    self.deform_keypoints.select_kpt(keypoint_idxs.item())

                elif dpg.is_key_down(dpg.mvKey_D):
                    if True:
                        keypoint_idxs = self.animate_tool.add_n_ring_nbs(keypoint_idxs, n=self.n_rings_N)
                    keypoint_3ds = nodes[keypoint_idxs]
                    self.deform_keypoints.add_kpts(keypoint_3ds, keypoint_idxs, expand=True)
                    print(f'Expand kpt: {self.deform_keypoints.selective_keypoints_idx_list}')

                elif dpg.is_key_down(dpg.mvKey_F):
                    keypoint_idxs = torch.arange(nodes.shape[0]).cuda()
                    keypoint_3ds = nodes[keypoint_idxs]
                    self.deform_keypoints.add_kpts(keypoint_3ds, keypoint_idxs, expand=True)
                    print(f'Add all the control points as kpt: {self.deform_keypoints.selective_keypoints_idx_list}')

                elif dpg.is_key_down(dpg.mvKey_Q):
                    self.deform_keypoints.select_rotation_kpt(keypoint_idxs.item())
                    print(f"select rotation control points: {keypoint_idxs.item()}")

                self.need_update_overlay = True

        self.callback_keypoint_add = callback_keypoint_add
        self.callback_keypoint_drag = callback_keypoint_drag

        ### register camera handler

        def callback_camera_drag_rotate_or_draw_mask(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.orbit(dx, dy)
            self.need_update = True

        def callback_camera_wheel_scale(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            delta = app_data

            self.cam.scale(delta)
            self.need_update = True

        def callback_camera_drag_pan(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.pan(dx, dy)
            self.need_update = True
                
        with dpg.handler_registry():
            # for camera moving
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Left,
                callback=callback_camera_drag_rotate_or_draw_mask,
            )
            dpg.add_mouse_wheel_handler(callback=callback_camera_wheel_scale)
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Middle, callback=callback_camera_drag_pan
            )

            dpg.add_mouse_move_handler(callback=callback_set_mouse_loc)
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Right, callback=callback_keypoint_drag)
            dpg.add_mouse_click_handler(button=dpg.mvMouseButton_Left, callback=callback_keypoint_add)

        dpg.create_viewport(
            title="Gaussian3D",
            width=self.W + 600,
            height=self.H + (45 if os.name == "nt" else 0),
            resizable=False,
        )

        ### global theme
        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                # set all padding to 0 to avoid scroll bar
                dpg.add_theme_style(
                    dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core
                )

        dpg.bind_item_theme("_primary_window", theme_no_padding)

        dpg.setup_dearpygui()

        if os.path.exists("LXGWWenKai-Regular.ttf"):
            with dpg.font_registry():
                with dpg.font("LXGWWenKai-Regular.ttf", 18) as default_font:
                    dpg.bind_font(default_font)

        dpg.show_viewport()
    
    @torch.no_grad()
    def draw_gs_trajectory(self, time_gap=0.3, samp_num=512, gs_num=512, thickness=1):
        fid = torch.tensor(self.animation_time).cuda().float() if self.is_animation else torch.remainder(torch.tensor((time.time()-self.t0) * self.fps_of_fid).float().cuda() / len(self.scene.getTrainCameras()) * self.video_speed, 1.)
        from utils.pickle_utils import load_obj, save_obj
        if os.path.exists(os.path.join(self.args.model_path, 'trajectory_camera.pickle')):
            print('Use fixed camera for screenshot: ', os.path.join(self.args.model_path, 'trajectory_camera.pickle'))
            cur_cam = load_obj(os.path.join(self.args.model_path, 'trajectory_camera.pickle'))
        else:
            cur_cam = MiniCam(
                self.cam.pose,
                self.W,
                self.H,
                self.cam.fovy,
                self.cam.fovx,
                self.cam.near,
                self.cam.far,
                fid = fid
            )
            save_obj(os.path.join(self.args.model_path, 'trajectory_camera.pickle'), cur_cam)
        fid = cur_cam.fid
        
        # Calculate the gs position at t0
        t = fid
        time_input = t.unsqueeze(0).expand(self.gaussians.get_xyz.shape[0], -1) if self.deform.name == 'mlp' else self.deform.deform.expand_time(t)
        d_values = self.deform.step(self.gaussians.get_xyz.detach(), time_input, feature=self.gaussians.feature, is_training=False, motion_mask=self.gaussians.motion_mask)
        cur_pts = self.gaussians.get_xyz + d_values['d_xyz']
    
        if not os.path.exists(os.path.join(self.args.model_path, 'trajectory_keypoints.pickle')):
            from utils.time_utils import farthest_point_sample
            pts_idx = farthest_point_sample(cur_pts[None], gs_num)[0]
            save_obj(os.path.join(self.args.model_path, 'trajectory_keypoints.pickle'), cur_pts[pts_idx].detach().cpu().numpy())
        else:
            print('Load keypoints from ', os.path.join(self.args.model_path, 'trajectory_keypoints.pickle'))
            kpts = torch.from_numpy(load_obj(os.path.join(self.args.model_path, 'trajectory_keypoints.pickle'))).cuda()
            import pytorch3d.ops
            _, idxs, _ = pytorch3d.ops.knn_points(kpts[None], cur_pts[None], None, None, K=1)
            pts_idx = idxs[0,:,0]
        delta_ts = torch.linspace(0, time_gap, samp_num)
        traj_pts = []
        for i in range(samp_num):
            t = fid + delta_ts[i]
            time_input = t.unsqueeze(0).expand(gs_num, -1) if self.deform.name == 'mlp' else self.deform.deform.expand_time(t)
            d_values = self.deform.step(self.gaussians.get_xyz[pts_idx].detach(), time_input, feature=self.gaussians.feature[pts_idx], is_training=False, motion_mask=self.gaussians.motion_mask[pts_idx])
            cur_pts = self.gaussians.get_xyz[pts_idx] + d_values['d_xyz']
            cur_pts = torch.cat([cur_pts, torch.ones_like(cur_pts[..., :1])], dim=-1)
            cur_pts2d = cur_pts @ cur_cam.full_proj_transform
            cur_pts2d = cur_pts2d[..., :2] / cur_pts2d[..., -1:]
            cur_pts2d = (cur_pts2d + 1) / 2 * torch.tensor([cur_cam.image_height, cur_cam.image_width]).cuda()
            traj_pts.append(cur_pts2d)
        traj_pts = torch.stack(traj_pts, dim=1).detach().cpu().numpy()  # N, T, 2

        import cv2
        from matplotlib import cm
        color_map = cm.get_cmap("jet")
        colors = np.array([np.array(color_map(i/max(1, float(gs_num - 1)))[:3]) * 255 for i in range(gs_num)], dtype=np.int32)
        alpha_img = np.zeros([cur_cam.image_height, cur_cam.image_width, 3])
        traj_img = np.zeros([cur_cam.image_height, cur_cam.image_width, 3])
        for i in range(gs_num):            
            alpha_img = cv2.polylines(img=alpha_img, pts=[traj_pts[i].astype(np.int32)], isClosed=False, color=[1, 1, 1], thickness=thickness)
            color = colors[i] / 255
            traj_img = cv2.polylines(img=traj_img, pts=[traj_pts[i].astype(np.int32)], isClosed=False, color=[float(color[0]), float(color[1]), float(color[2])], thickness=thickness)
        traj_img = np.concatenate([traj_img, alpha_img[..., :1]], axis=-1) * 255
        Image.fromarray(traj_img.astype('uint8')).save(os.path.join(self.args.model_path, 'trajectory.png'))
        
        from utils.vis_utils import render_cur_cam
        img_begin = render_cur_cam(self=self, cur_cam=cur_cam)
        cur_cam.fid = cur_cam.fid + delta_ts[-1]
        img_end = render_cur_cam(self=self, cur_cam=cur_cam)
        img_begin = (img_begin.permute(1,2,0).clamp(0, 1).detach().cpu().numpy() * 255).astype('uint8')
        img_end = (img_end.permute(1,2,0).clamp(0, 1).detach().cpu().numpy() * 255).astype('uint8')
        Image.fromarray(img_begin).save(os.path.join(self.args.model_path, 'traj_start.png'))
        Image.fromarray(img_end).save(os.path.join(self.args.model_path, 'traj_end.png'))

    # gui mode
    def render(self):
        assert self.gui
        while dpg.is_dearpygui_running():
            # update texture every frame
            if self.training:
                if self.deform.name == 'node' and self.iteration_node_rendering < self.opt.iterations_node_rendering:
                    self.train_node_rendering_step()
                else:
                    self.train_step()
            if self.should_vis_trajectory:
                self.draw_gs_trajectory()
                self.should_vis_trajectory = False
            if self.should_render_customized_trajectory:
                self.render_customized_trajectory(use_spiral=self.should_render_customized_trajectory_spiral)
            self.test_step()

            dpg.render_dearpygui_frame()
    
    # no gui mode
    def train(self, iters=5000):
        
        from jhutil import color_log; color_log(5555, "training start")
        if iters > 0:
            for step in tqdm.trange(iters):
                if self.deform.name == 'node' and self.iteration_node_rendering < self.opt.iterations_node_rendering:
                    self.train_node_rendering_step(step)
                else:
                    self.train_step(step)
    
    train_step=train_step
    train_node_rendering_step=train_node_rendering_step

    @torch.no_grad()
    def test_step(self, specified_cam=None):

        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()

        if not hasattr(self, 't0'):
            self.t0 = time.time()
            self.fps_of_fid = 10
        
        if self.is_animation:
            if not self.showing_overlay:
                self.buffer_overlay = None
            else:
                self.update_control_point_overlay()
            fid = torch.tensor(self.animation_time).cuda().float()
        else:
            fid = torch.remainder(torch.tensor((time.time()-self.t0) * self.fps_of_fid).float().cuda() / len(self.scene.getTrainCameras()) * self.video_speed, 1.)

        if self.should_save_screenshot and os.path.exists(os.path.join(self.args.model_path, 'screenshot_camera.pickle')):
            print('Use fixed camera for screenshot: ', os.path.join(self.args.model_path, 'screenshot_camera.pickle'))
            from utils.pickle_utils import load_obj
            cur_cam = load_obj(os.path.join(self.args.model_path, 'screenshot_camera.pickle'))
        elif specified_cam is not None:
            cur_cam = specified_cam
        else:
            cur_cam = MiniCam(
                self.cam.pose,
                self.W,
                self.H,
                self.cam.fovy,
                self.cam.fovx,
                self.cam.near,
                self.cam.far,
                fid = fid
            )
        fid = cur_cam.fid

        if self.deform.name == 'node':
            if 'Node' in self.visualization_mode:
                d_rotation_bias = None
                gaussians = self.deform.deform.as_gaussians
                time_input = fid.unsqueeze(0).expand(gaussians.get_xyz.shape[0], -1)
                d_values = self.deform.deform.query_network(x=gaussians.get_xyz.detach(), t=time_input)
                if self.motion_animation_d_values is not None:
                    for key in self.motion_animation_d_values:
                        d_values[key] = self.motion_animation_d_values[key]
                d_xyz, d_opacity, d_color = d_values['d_xyz'] * gaussians.motion_mask, d_values['d_opacity'] * gaussians.motion_mask if d_values['d_opacity'] is not None else None, d_values['d_color'] * gaussians.motion_mask if d_values['d_color'] is not None else None
                d_rotation, d_scaling = 0., 0.
                if self.view_animation and self.animation_trans_bias is not None:
                    d_xyz = d_xyz + self.animation_trans_bias
                vis_scale_const = self.vis_scale_const
            else:
                if self.view_animation:
                    node_trans_bias = self.animation_trans_bias
                else:
                    node_trans_bias = None
                time_input = self.deform.deform.expand_time(fid)
                d_values = self.deform.step(self.gaussians.get_xyz.detach(), time_input, feature=self.gaussians.feature, is_training=False, node_trans_bias=node_trans_bias, motion_mask=self.gaussians.motion_mask, camera_center=cur_cam.camera_center, animation_d_values=self.motion_animation_d_values)
                gaussians = self.gaussians
                d_xyz, d_rotation, d_scaling, d_opacity, d_color = d_values['d_xyz'], d_values['d_rotation'], d_values['d_scaling'], d_values['d_opacity'], d_values['d_color']
                vis_scale_const = None
                d_rotation_bias = d_values['d_rotation_bias'] if 'd_rotation_bias' in d_values.keys() else None
        else:
            vis_scale_const = None
            d_rotation_bias = None
            if self.iteration < self.opt.warm_up:
                d_xyz, d_rotation, d_scaling, d_opacity, d_color = 0.0, 0.0, 0.0, 0.0, 0.0
                gaussians = self.gaussians
            else:
                N = self.gaussians.get_xyz.shape[0]
                time_input = fid.unsqueeze(0).expand(N, -1)
                gaussians = self.gaussians
                d_values = self.deform.step(self.gaussians.get_xyz.detach(), time_input, feature=self.gaussians.feature, camera_center=cur_cam.camera_center)
                d_xyz, d_rotation, d_scaling, d_opacity, d_color = d_values['d_xyz'], d_values['d_rotation'], d_values['d_scaling'], d_values['d_opacity'], d_values['d_color']
        
        if self.vis_traj_realtime:
            if 'Node' in self.visualization_mode:
                if self.last_traj_overlay_type != 'node':
                    self.traj_coor = None
                self.update_trajectory_overlay(gs_xyz=gaussians.get_xyz+d_xyz, camera=cur_cam, gs_num=512)
                self.last_traj_overlay_type = 'node'
            else:
                if self.last_traj_overlay_type != 'gs':
                    self.traj_coor = None
                self.update_trajectory_overlay(gs_xyz=gaussians.get_xyz+d_xyz, camera=cur_cam)
                self.last_traj_overlay_type = 'gs'
        
        if self.visualization_mode == 'Dynamic' or self.visualization_mode == 'Static':
            d_opacity = torch.zeros_like(self.gaussians.motion_mask)
            if self.visualization_mode == 'Dynamic':
                d_opacity[self.gaussians.motion_mask < .9] = - 1e3
            else:
                d_opacity[self.gaussians.motion_mask > .1] = - 1e3
        
        render_motion = "Motion" in self.visualization_mode
        if render_motion:
            vis_scale_const = self.vis_scale_const
        if type(d_rotation) is not float and gaussians._rotation.shape[0] != d_rotation.shape[0]:
            d_xyz, d_rotation, d_scaling = 0, 0, 0
            print('Async in Gaussian Switching')
        out = render(viewpoint_camera=cur_cam, pc=gaussians, pipe=self.pipe, bg_color=self.background, d_xyz=d_xyz, d_rotation=d_rotation, d_scaling=d_scaling, render_motion=render_motion, d_opacity=d_opacity, d_color=d_color, d_rot_as_res=self.deform.d_rot_as_res, scale_const=vis_scale_const, d_rotation_bias=d_rotation_bias)

        if self.mode == "normal_dep":
            from utils.other_utils import depth2normal
            normal = depth2normal(out["depth"])
            out["normal_dep"] = (normal + 1) / 2

        buffer_image = out[self.mode]  # [3, H, W]

        if self.should_save_screenshot:
            alpha = out['alpha']
            sv_image = torch.cat([buffer_image, alpha], dim=0).clamp(0,1).permute(1,2,0).detach().cpu().numpy()
            def save_image(image, image_dir):
                os.makedirs(image_dir, exist_ok=True)
                idx = len(os.listdir(image_dir))
                print('>>> Saving image to %s' % os.path.join(image_dir, '%05d.png'%idx))
                Image.fromarray((image * 255).astype('uint8')).save(os.path.join(image_dir, '%05d.png'%idx))
                # Save the camera of screenshot
                from utils.pickle_utils import save_obj
                save_obj(os.path.join(image_dir, '%05d_cam.pickle'% idx), cur_cam)
            save_image(sv_image, self.screenshot_sv_path)
            self.should_save_screenshot = False

        if self.mode in ['depth', 'alpha']:
            buffer_image = buffer_image.repeat(3, 1, 1)
            if self.mode == 'depth':
                buffer_image = (buffer_image - buffer_image.min()) / (buffer_image.max() - buffer_image.min() + 1e-20)

        buffer_image = torch.nn.functional.interpolate(
            buffer_image.unsqueeze(0),
            size=(self.H, self.W),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        self.buffer_image = (
            buffer_image.permute(1, 2, 0)
            .contiguous()
            .clamp(0, 1)
            .contiguous()
            .detach()
            .cpu()
            .numpy()
        )

        self.need_update = True

        ender.record()
        torch.cuda.synchronize()
        t = starter.elapsed_time(ender)

        if self.is_animation and self.buffer_overlay is not None:
            overlay_mask = self.buffer_overlay.sum(axis=-1, keepdims=True) == 0
            try:
                buffer_image = self.buffer_image * overlay_mask + self.buffer_overlay
            except:
                buffer_image = self.buffer_image
        else:
            buffer_image = self.buffer_image

        if self.vis_traj_realtime:
            buffer_image = buffer_image * (1 - self.traj_overlay[..., 3:]) + self.traj_overlay[..., :3] * self.traj_overlay[..., 3:]

        if self.gui:
            dpg.set_value("_log_infer_time", f"{t:.4f}ms ({int(1000/t)} FPS FID: {fid.item()})")
            dpg.set_value(
                "_texture", buffer_image
            )  # buffer must be contiguous, else seg fault!
        return buffer_image

    def update_control_point_overlay(self):
        from skimage.draw import line_aa
        # should update overlay
        # if self.need_update_overlay and len(self.keypoint_3ds) > 0:
        if self.need_update_overlay and len(self.deform_keypoints.get_kpt()) > 0:
            try:
                buffer_overlay = np.zeros_like(self.buffer_image)
                mv = self.cam.view # [4, 4]
                proj = self.cam.perspective # [4, 4]
                mvp = proj @ mv
                # do mvp transform for keypoints
                # source_points = np.array(self.keypoint_3ds)
                source_points = np.array(self.deform_keypoints.get_kpt())
                # target_points = source_points + np.array(self.keypoint_3ds_delta)
                target_points = self.deform_keypoints.get_deformed_kpt_np()
                points_indices = np.arange(len(source_points))

                source_points_clip = np.matmul(np.pad(source_points, ((0, 0), (0, 1)), constant_values=1.0), mvp.T)  # [N, 4]
                target_points_clip = np.matmul(np.pad(target_points, ((0, 0), (0, 1)), constant_values=1.0), mvp.T)  # [N, 4]
                source_points_clip[:, :3] /= source_points_clip[:, 3:] # perspective division
                target_points_clip[:, :3] /= target_points_clip[:, 3:] # perspective division

                source_points_2d = (((source_points_clip[:, :2] + 1) / 2) * np.array([self.H, self.W])).round().astype(np.int32)
                target_points_2d = (((target_points_clip[:, :2] + 1) / 2) * np.array([self.H, self.W])).round().astype(np.int32)

                radius = int((self.H + self.W) / 2 * 0.005)
                keypoint_idxs_to_drag = self.deform_keypoints.selective_keypoints_idx_list
                for i in range(len(source_points_clip)):
                    point_idx = points_indices[i]
                    # draw source point
                    if source_points_2d[i, 0] >= radius and source_points_2d[i, 0] < self.W - radius and source_points_2d[i, 1] >= radius and source_points_2d[i, 1] < self.H - radius:
                        buffer_overlay[source_points_2d[i, 1]-radius:source_points_2d[i, 1]+radius, source_points_2d[i, 0]-radius:source_points_2d[i, 0]+radius] += np.array([1,0,0]) if not point_idx in keypoint_idxs_to_drag else np.array([1,0.87,0])
                        # draw target point
                        if target_points_2d[i, 0] >= radius and target_points_2d[i, 0] < self.W - radius and target_points_2d[i, 1] >= radius and target_points_2d[i, 1] < self.H - radius:
                            buffer_overlay[target_points_2d[i, 1]-radius:target_points_2d[i, 1]+radius, target_points_2d[i, 0]-radius:target_points_2d[i, 0]+radius] += np.array([0,0,1]) if not point_idx in keypoint_idxs_to_drag else np.array([0.5,0.5,1])
                        # draw line
                        rr, cc, val = line_aa(source_points_2d[i, 1], source_points_2d[i, 0], target_points_2d[i, 1], target_points_2d[i, 0])
                        in_canvas_mask = (rr >= 0) & (rr < self.H) & (cc >= 0) & (cc < self.W)
                        buffer_overlay[rr[in_canvas_mask], cc[in_canvas_mask]] += val[in_canvas_mask, None] * np.array([0,1,0]) if not point_idx in keypoint_idxs_to_drag else np.array([0.5,1,0])
                self.buffer_overlay = buffer_overlay
            except:
                print('Async Fault in Overlay Drawing!')
                self.buffer_overlay = None

    def update_trajectory_overlay(self, gs_xyz, camera, samp_num=32, gs_num=512, thickness=1):
        if not hasattr(self, 'traj_coor') or self.traj_coor is None:
            from utils.time_utils import farthest_point_sample
            self.traj_coor = torch.zeros([0, gs_num, 4], dtype=torch.float32).cuda()
            opacity_mask = self.gaussians.get_opacity[..., 0] > .1 if self.gaussians.get_xyz.shape[0] == gs_xyz.shape[0] else torch.ones_like(gs_xyz[:, 0], dtype=torch.bool)
            masked_idx = torch.arange(0, opacity_mask.shape[0], device=opacity_mask.device)[opacity_mask]
            self.traj_idx = masked_idx[farthest_point_sample(gs_xyz[None, opacity_mask], gs_num)[0]]
            from matplotlib import cm
            self.traj_color_map = cm.get_cmap("jet")
        pts = gs_xyz[None, self.traj_idx]
        pts = torch.cat([pts, torch.ones_like(pts[..., :1])], dim=-1)
        self.traj_coor = torch.cat([self.traj_coor, pts], axis=0)
        if self.traj_coor.shape[0] > samp_num:
            self.traj_coor = self.traj_coor[-samp_num:]
        traj_uv = self.traj_coor @ camera.full_proj_transform
        traj_uv = traj_uv[..., :2] / traj_uv[..., -1:]
        traj_uv = (traj_uv + 1) / 2 * torch.tensor([camera.image_height, camera.image_width]).cuda()
        traj_uv = traj_uv.detach().cpu().numpy()

        import cv2
        colors = np.array([np.array(self.traj_color_map(i/max(1, float(gs_num - 1)))[:3]) * 255 for i in range(gs_num)], dtype=np.int32)
        alpha_img = np.zeros([camera.image_height, camera.image_width, 3], dtype=np.float32)
        traj_img = np.zeros([camera.image_height, camera.image_width, 3], dtype=np.float32)
        for i in range(gs_num):            
            alpha_img = cv2.polylines(img=alpha_img, pts=[traj_uv[:, i].astype(np.int32)], isClosed=False, color=[1, 1, 1], thickness=thickness)
            color = colors[i] / 255
            traj_img = cv2.polylines(img=traj_img, pts=[traj_uv[:, i].astype(np.int32)], isClosed=False, color=[float(color[0]), float(color[1]), float(color[2])], thickness=thickness)
        traj_img = np.concatenate([traj_img, alpha_img[..., :1]], axis=-1)
        self.traj_overlay = traj_img
        
    def test_speed(self, round=500):
        self.deform.deform.cached_nn_weight = True
        self.test_step()
        t0 = time.time()
        for i in range(round):
            self.test_step()
        t1 = time.time()
        fps = round / (t1 - t0)
        print(f'FPS: {fps}')
        return fps
    
    def render_customized_trajectory(self, use_spiral=False, traj_dir=None, fps=30, motion_repeat=1):
        from utils.pickle_utils import load_obj
        # Remove history trajectory
        if self.vis_traj_realtime:
            self.traj_coor = None
            self.traj_overlay = None
        # Default trajectory path
        if traj_dir is None:
            traj_dir = os.path.join(self.args.model_path, 'trajectory')
        # Read deformation files for animation presentation
        deform_keypoint_files = [None] + sorted([file for file in os.listdir(os.path.join(self.args.model_path)) if file.startswith('deform_keypoints') and file.endswith('.pickle')])
        rendering_animation = len(deform_keypoint_files) > 0
        if rendering_animation:
            deform_keypoints, self.animation_time = load_obj(os.path.join(self.args.model_path, deform_keypoint_files[1]))
            self.animation_initialize()
        # Read camera trajectory files
        if os.path.exists(traj_dir):
            cameras = sorted([cam for cam in os.listdir(traj_dir) if cam.endswith('.pickle')])
            cameras = [load_obj(os.path.join(traj_dir, cam)) for cam in cameras]
            if len(cameras) < 2:
                print('No trajectory cameras found')
                self.should_render_customized_trajectory = False
                return
            if os.path.exists(os.path.join(traj_dir, 'time.txt')):
                with open(os.path.join(traj_dir, 'time.txt'), 'r') as file:
                    time = file.readline()
                    time = time.split(' ')
                    timesteps = np.array([float(t) for t in time])
            else:
                timesteps = np.array([3] * len(cameras))  # three seconds by default
        elif use_spiral:
            from utils.pose_utils import render_path_spiral
            from copy import deepcopy
            c2ws = []
            for camera in self.scene.getTrainCameras():
                c2w = np.eye(4)
                c2w[:3, :3] = camera.R
                c2w[:3, 3] = camera.T
                c2ws.append(c2w)
            c2ws = np.stack(c2ws, axis=0)
            poses = render_path_spiral(c2ws=c2ws, focal=self.cam.fovx*200, rots=3, N=30*12)
            print(f'Use spiral camera poses with {poses.shape[0]} cameras!')
            cameras_ = []
            for i in range(len(poses)):
                cam = MiniCam(
                    self.cam.pose,
                    self.W,
                    self.H,
                    self.cam.fovy,
                    self.cam.fovx,
                    self.cam.near,
                    self.cam.far,
                    0
                )
                cam.reset_extrinsic(R=poses[i, :3, :3], T=poses[i, :3, 3])
                cameras_.append(cam)
            cameras = cameras_
        else:
            if self.is_animation:
                if not self.showing_overlay:
                    self.buffer_overlay = None
                else:
                    self.update_control_point_overlay()
                fid = torch.tensor(self.animation_time).cuda().float()
            else:
                fid = torch.tensor(0).float().cuda()
            cur_cam = MiniCam(
                self.cam.pose,
                self.W,
                self.H,
                self.cam.fovy,
                self.cam.fovx,
                self.cam.near,
                self.cam.far,
                fid = fid
            )
            cameras = [cur_cam, cur_cam]
            timesteps = np.array([3] * len(cameras))  # three seconds by default
        
        def min_line_dist_center(rays_o, rays_d):
            try:
                if len(np.shape(rays_d)) == 2:
                    rays_o = rays_o[..., np.newaxis]
                    rays_d = rays_d[..., np.newaxis]
                A_i = np.eye(3) - rays_d * np.transpose(rays_d, [0, 2, 1])
                b_i = -A_i @ rays_o
                pt_mindist = np.squeeze(-np.linalg.inv((A_i @ A_i).mean(0)) @ (b_i).mean(0))
            except:
                pt_mindist = None
            return pt_mindist

        # Define camera pose keypoints
        vis_cams = []
        c2ws = np.stack([cam.c2w for cam in cameras], axis=0)
        rs = c2ws[:, :3, :3]
        from scipy.spatial.transform import Slerp
        slerp = Slerp(times=np.arange(len(c2ws)), rotations=R.from_matrix(rs))
        from scipy.spatial import geometric_slerp
        
        if rendering_animation:
            from utils.bezier import BezierCurve, PieceWiseLinear
            points = []
            for deform_keypoint_file in deform_keypoint_files:
                if deform_keypoint_file is None:
                    points.append(self.animate_tool.init_pcl.detach().cpu().numpy())
                else:
                    deform_keypoints = load_obj(os.path.join(self.args.model_path, deform_keypoint_file))[0]
                    animated_pcl, _, _ = self.animate_tool.deform_arap(handle_idx=deform_keypoints.get_kpt_idx(), handle_pos=deform_keypoints.get_deformed_kpt_np(), return_R=True)
                    points.append(animated_pcl.detach().cpu().numpy())
            points = np.stack(points, axis=1)
            bezier = PieceWiseLinear(points=points)
        
        # Save path
        sv_dir = os.path.join(self.args.model_path, 'render_trajectory')
        os.makedirs(sv_dir, exist_ok=True)
        import cv2
        video = cv2.VideoWriter(sv_dir + f'/{self.mode}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (self.W, self.H))

        # Camera loop
        for i in range(len(cameras)-1):
            if use_spiral:
                total_rate = i / (len(cameras) - 1)
                cam = cameras[i]
                if rendering_animation:
                    cam.fid = torch.tensor(self.animation_time).cuda().float()
                    animated_pcl = bezier(t=total_rate)
                    animated_pcl = torch.from_numpy(animated_pcl).cuda()
                    self.animation_trans_bias = animated_pcl - self.animate_tool.init_pcl
                else:
                    cam.fid = torch.tensor(total_rate).cuda().float()
                image = self.test_step(specified_cam=cam)
                image = (image * 255).astype('uint8')
                video.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                vis_cams = poses
            else:
                cam0, cam1 = cameras[i], cameras[i+1]
                frame_num = int(timesteps[i] * fps)
                avg_center = min_line_dist_center(c2ws[i:i+2, :3, 3], c2ws[i:i+2, :3, 2])
                if avg_center is not None:
                    vec1_norm1, vec2_norm = np.linalg.norm(c2ws[i, :3, 3] - avg_center), np.linalg.norm(c2ws[i+1, :3, 3] - avg_center)
                    slerp_t = geometric_slerp(start=(c2ws[i, :3, 3]-avg_center)/vec1_norm1, end=(c2ws[i+1, :3, 3]-avg_center)/vec2_norm, t=np.linspace(0, 1, frame_num))
                else:
                    print('avg_center is None. Move along a line.')
                
                for j in range(frame_num):
                    rate = j / frame_num
                    total_rate = (i + rate) / (len(cameras) - 1)
                    if rendering_animation:
                        animated_pcl = bezier(t=total_rate)
                        animated_pcl = torch.from_numpy(animated_pcl).cuda()
                        self.animation_trans_bias = animated_pcl - self.animate_tool.init_pcl

                    rot = slerp(i+rate).as_matrix()
                    if avg_center is not None:
                        trans = slerp_t[j] * (vec1_norm1 + (vec2_norm - vec1_norm1) * rate) + avg_center
                    else:
                        trans = c2ws[i, :3, 3] + (c2ws[i+1, :3, 3] - c2ws[i, :3, 3]) * rate
                    c2w = np.eye(4)
                    c2w[:3, :3] = rot
                    c2w[:3, 3] = trans
                    c2w = np.array(c2w, dtype=np.float32)
                    vis_cams.append(c2w)
                    fid = cam0.fid + (cam1.fid - cam0.fid) * rate if not rendering_animation else torch.tensor(self.animation_time).cuda().float()
                    cam = MiniCam(c2w=c2w, width=cam0.image_width, height=cam0.image_height, fovy=cam0.FoVy, fovx=cam0.FoVx, znear=cam0.znear, zfar=cam0.zfar, fid=fid)
                    image = self.test_step(specified_cam=cam)
                    image = (image * 255).astype('uint8')
                    video.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        video.release()

        print('Trajectory rendered done!')
        self.should_render_customized_trajectory = False


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    
    parser.add_argument('--gui', action='store_true', help="start a GUI")
    parser.add_argument('--W', type=int, default=800, help="GUI width")
    parser.add_argument('--H', type=int, default=800, help="GUI height")
    parser.add_argument('--elevation', type=float, default=0, help="default GUI camera elevation")
    parser.add_argument('--radius', type=float, default=5, help="default GUI camera radius from center")
    parser.add_argument('--fovy', type=float, default=50, help="default GUI camera fovy")

    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int,
                        default=[5000, 6000, 7_000] + list(range(8000, 100_0001, 1000)))
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 10_000, 20_000, 30_000, 40000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--deform-type", type=str, default='mlp')
    parser.add_argument("--wandb_group", type=str, default='tmp')
    parser.add_argument("--is_diva360", action='store_true', default=False)
    

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    name = args.source_path.split('/')[-1]
    wandb.init(project=f"SC-GS", dir="./wandb", name=name, group=args.wandb_group)

    if not args.model_path.endswith(args.deform_type):
        args.model_path = os.path.join(os.path.dirname(os.path.normpath(args.model_path)), os.path.basename(os.path.normpath(args.model_path)) + f'_{args.deform_type}')
    
    if os.path.exists(args.model_path):
        import shutil
        shutil.rmtree(args.model_path)
    
    print("Optimizing " + args.model_path)
    safe_state(args.quiet)

    # lp = ModelParams(parser)
    # if "DFA" in args.source_path:
    #     lp._white_background = True

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    gui = GUI(args=args, dataset=lp.extract(args), opt=op.extract(args), pipe=pp.extract(args),testing_iterations=args.test_iterations, saving_iterations=args.save_iterations)

    if args.gui:
        gui.render()
    else:
        gui.train(args.iterations)
    
    # All done
    print("\nTraining complete.")
