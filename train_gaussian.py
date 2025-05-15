
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



def train_step(self, step):
    from jhutil import color_log; color_log(7777, "train_step", repeat=False)

    if network_gui.conn == None:
        network_gui.try_connect()
    while network_gui.conn != None:
        try:
            net_image_bytes = None
            custom_cam, do_training, self.pipe.do_shs_python, self.pipe.do_cov_python, keep_alive, scaling_modifer = network_gui.receive()
            if custom_cam != None:
                net_image = render(custom_cam, self.gaussians, self.pipe, self.background, scaling_modifer)["render"]
                net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
            network_gui.send(net_image_bytes, self.dataset.source_path)
            if do_training and ((self.iteration < int(self.opt.iterations)) or not keep_alive):
                break
        except Exception as e:
            network_gui.conn = None

    self.iter_start.record()

    # Every 1000 its we increase the levels of SH up to a maximum degree
    if self.iteration % self.opt.oneupSHdegree_step == 0:
        self.gaussians.oneupSHdegree()

    # Pick a random Camera
    if not self.viewpoint_stack:
        if self.opt.progressive_train and self.iteration < int(self.opt.progressive_stage_steps / self.opt.progressive_stage_ratio):
            cameras_to_train_idx = int(min(((self.iteration) / self.opt.progressive_stage_steps + 1) * self.opt.progressive_stage_ratio, 1.) * len(self.scene.getTrainCameras()))
            cameras_to_train_idx = max(cameras_to_train_idx, 1)
            interval_len = int(len(self.scene.getTrainCameras()) * self.opt.progressive_stage_ratio)
            min_idx = max(0, cameras_to_train_idx - interval_len)
            sorted_train_cams = sorted(self.scene.getTrainCameras().copy(), key=lambda x: x.fid)
            viewpoint_stack = sorted_train_cams[min_idx: cameras_to_train_idx]
            out_domain_idx = np.arange(min_idx)
            if len(out_domain_idx) >= interval_len:
                out_domain_idx = np.random.choice(out_domain_idx, [interval_len], replace=False)
                out_domain_stack = [sorted_train_cams[idx] for idx in out_domain_idx]
                viewpoint_stack = viewpoint_stack + out_domain_stack
        else:
            viewpoint_stack = self.scene.getTrainCameras().copy()
        self.viewpoint_stack = viewpoint_stack
    
    total_frame = len(self.scene.getTrainCameras())
    time_interval = 1 / total_frame

    from jhutil import color_log; color_log("aaaa", "pick random_camera", repeat=False)
    viewpoint_cam = self.viewpoint_stack.pop(randint(0, len(self.viewpoint_stack) - 1))
    if self.dataset.load2gpu_on_the_fly:
        viewpoint_cam.load2device()
    fid = viewpoint_cam.fid


    from jhutil import color_log; color_log("bbbb", "get deformation", repeat=False)
    if self.deform.name == 'mlp' or self.deform.name == 'static':
        if self.iteration < self.opt.warm_up:
            d_xyz, d_rotation, d_scaling, d_opacity, d_color = 0.0, 0.0, 0.0, 0.0, 0.0
        else:
            N = self.gaussians.get_xyz.shape[0]
            time_input = fid.unsqueeze(0).expand(N, -1)
            ast_noise = 0 if self.dataset.is_blender else torch.randn(1, 1, device='cuda').expand(N, -1) * time_interval * self.smooth_term(self.iteration)
            d_values = self.deform.step(self.gaussians.get_xyz.detach(), time_input + ast_noise, iteration=self.iteration, feature=self.gaussians.feature, camera_center=viewpoint_cam.camera_center)
            d_xyz, d_rotation, d_scaling, d_opacity, d_color = d_values['d_xyz'], d_values['d_rotation'], d_values['d_scaling'], d_values['d_opacity'], d_values['d_color']
    elif self.deform.name == 'node':
        if not self.deform.deform.inited:
            print('Notice that warping nodes are initialized with Gaussians!!!')
            self.deform.deform.init(self.opt, self.gaussians.get_xyz.detach(), feature=self.gaussians.feature)
        time_input = self.deform.deform.expand_time(fid)
        N = time_input.shape[0]
        ast_noise = 0 if self.dataset.is_blender else torch.randn(1, 1, device='cuda').expand(N, -1) * time_interval * self.smooth_term(self.iteration)
        d_values = self.deform.step(self.gaussians.get_xyz.detach(), time_input + ast_noise, iteration=self.iteration, feature=self.gaussians.feature, motion_mask=self.gaussians.motion_mask, camera_center=viewpoint_cam.camera_center, time_interval=time_interval)
        d_xyz, d_rotation, d_scaling, d_opacity, d_color = d_values['d_xyz'], d_values['d_rotation'], d_values['d_scaling'], d_values['d_opacity'], d_values['d_color']
        if self.iteration < self.opt.warm_up:
            d_xyz, d_rotation, d_scaling, d_opacity, d_color = d_xyz.detach(), d_rotation.detach(), d_scaling.detach(), d_opacity.detach() if d_opacity is not None else None, d_color.detach() if d_color is not None else None
        elif self.iteration < self.opt.dynamic_color_warm_up:
            d_color = d_color.detach() if d_color is not None else None

    from jhutil import color_log; color_log("cccc", "render", repeat=False)
    # Render
    random_bg_color = (not self.dataset.white_background and self.opt.random_bg_color) and self.opt.gt_alpha_mask_as_scene_mask and viewpoint_cam.gt_alpha_mask is not None
    render_pkg_re = render(viewpoint_cam, self.gaussians, self.pipe, self.background, d_xyz, d_rotation, d_scaling, random_bg_color=random_bg_color, d_opacity=d_opacity, d_color=d_color, d_rot_as_res=self.deform.d_rot_as_res)
    image, viewspace_point_tensor, visibility_filter, radii = render_pkg_re["render"], render_pkg_re["viewspace_points"], render_pkg_re["visibility_filter"], render_pkg_re["radii"]

    from jhutil import color_log; color_log("dddd", "calc loss", repeat=False)
    # Loss
    gt_image = viewpoint_cam.original_image.cuda()
    if random_bg_color:
        gt_alpha_mask = viewpoint_cam.gt_alpha_mask.cuda()
        gt_image = gt_alpha_mask * gt_image + (1 - gt_alpha_mask) * render_pkg_re['bg_color'][:, None, None]
    elif self.dataset.white_background and viewpoint_cam.gt_alpha_mask is not None and self.opt.gt_alpha_mask_as_scene_mask:
        gt_alpha_mask = viewpoint_cam.gt_alpha_mask.cuda()
        gt_image = gt_alpha_mask * gt_image + (1 - gt_alpha_mask) * self.background[:, None, None]

    Ll1 = l1_loss(image, gt_image)
    loss_img = (1.0 - self.opt.lambda_dssim) * Ll1 + self.opt.lambda_dssim * (1.0 - ssim(image, gt_image))
    loss = loss_img


    if step % 1000 == 0:
        from jhutil import get_img_diff
        wandb.log({
            "train_diff_img(gaussian)": wandb.Image(get_img_diff(image, gt_image))
        }, commit=True)
    
    n_gaussian = self.gaussians.get_xyz.detach()
    # from jhutil import color_log; color_log(0000, f'n_gaussian: {n_gaussian}   loss: {loss:.3f}', update=True)

    if self.iteration > self.opt.warm_up:
        loss = loss + self.deform.reg_loss

    # Flow loss
    flow_id2_candidates = viewpoint_cam.flow_dirs
    lambda_optical = landmark_interpolate(self.opt.lambda_optical_landmarks, self.opt.lambda_optical_steps, self.iteration)
    if flow_id2_candidates != [] and lambda_optical > 0 and self.iteration >= self.opt.warm_up:
        # Pick flow file and read it
        flow_id2_dir = np.random.choice(flow_id2_candidates)
        flow = np.load(flow_id2_dir)
        mask_id2_dir = flow_id2_dir.replace('raft_neighbouring', 'raft_masks').replace('.npy', '.png')
        masks = imageio.imread(mask_id2_dir) / 255.
        flow = torch.from_numpy(flow).float().cuda()
        masks = torch.from_numpy(masks).float().cuda()
        if flow.shape[0] != image.shape[1] or flow.shape[1] != image.shape[2]:
            flow = torch.nn.functional.interpolate(flow.permute([2, 0, 1])[None], (image.shape[1], image.shape[2]))[0].permute(1, 2, 0)
            masks = torch.nn.functional.interpolate(masks.permute([2, 0, 1])[None], (image.shape[1], image.shape[2]))[0].permute(1, 2, 0)
        fid1 = viewpoint_cam.fid
        cam2_id = os.path.basename(flow_id2_dir).split('_')[-1].split('.')[0]
        if not hasattr(self, 'img2cam'):
            self.img2cam = {cam.image_name: idx for idx, cam in enumerate(self.scene.getTrainCameras().copy())}
        if cam2_id in self.img2cam:  # Only considering the case with existing files
            cam2_id = self.img2cam[cam2_id]
            viewpoint_cam2 = self.scene.getTrainCameras().copy()[cam2_id]
            fid2 = viewpoint_cam2.fid
            # Calculate the GT flow, weight, and mask
            coor1to2_flow = flow / torch.tensor(flow.shape[:2][::-1], dtype=torch.float32).cuda() * 2
            cycle_consistency_mask = masks[..., 0] > 0
            occlusion_mask = masks[..., 1] > 0
            mask_flow = cycle_consistency_mask | occlusion_mask
            pair_weight = torch.clamp(torch.cos((fid1 - fid2).abs() * np.pi / 2), 0.2, 1)
            # Calculate the motion at t2
            time_input2 = self.deform.deform.expand_time(fid2)
            ast_noise = 0 if self.dataset.is_blender else torch.randn(1, 1, device='cuda').expand(N, -1) * time_interval * self.smooth_term(self.iteration)
            d_xyz2 = self.deform.step(self.gaussians.get_xyz.detach(), time_input2 + ast_noise, iteration=self.iteration, feature=self.gaussians.feature, motion_mask=self.gaussians.motion_mask, camera_center=viewpoint_cam2.camera_center)['d_xyz']
            # Render the flow image
            render_pkg2 = render_flow(pc=self.gaussians, viewpoint_camera1=viewpoint_cam, viewpoint_camera2=viewpoint_cam2, d_xyz1=d_xyz, d_xyz2=d_xyz2, d_rotation1=d_rotation, d_scaling1=d_scaling, scale_const=None)
            coor1to2_motion = render_pkg2["render"][:2].permute(1, 2, 0)
            mask_motion = (render_pkg2['alpha'][0] > .9).detach()  # Only optimizing the space with solid points to avoid dilation
            mask = (mask_motion & mask_flow)[..., None] * pair_weight
            # Flow loss based on pixel rgb loss
            l1_loss_weight = (image.detach() - gt_image).abs().mean(dim=0)
            l1_loss_weight = torch.cos(l1_loss_weight * torch.pi / 2)
            mask = mask * l1_loss_weight[..., None]
            # Flow mask
            optical_flow_loss = l1_loss(mask * coor1to2_flow, mask * coor1to2_motion)
            loss = loss + lambda_optical * optical_flow_loss

    # Motion Mask Loss
    lambda_motion_mask = landmark_interpolate(self.opt.lambda_motion_mask_landmarks, self.opt.lambda_motion_mask_steps, self.iteration)
    if not self.opt.no_motion_mask_loss and self.deform.name == 'node' and self.opt.gt_alpha_mask_as_dynamic_mask and viewpoint_cam.gt_alpha_mask is not None and lambda_motion_mask > 0:
        gt_alpha_mask = viewpoint_cam.gt_alpha_mask.cuda()
        render_pkg_motion = render(viewpoint_cam, self.gaussians, self.pipe, self.background, d_xyz, d_rotation, d_scaling, random_bg_color=random_bg_color, render_motion=True, detach_xyz=True, detach_rot=True, detach_scale=True, detach_opacity=True, d_opacity=d_opacity, d_color=d_color, d_rot_as_res=self.deform.d_rot_as_res)
        motion_image = render_pkg_motion["render"][0]
        L_motion = l1_loss(gt_alpha_mask, motion_image)
        loss = loss + L_motion * lambda_motion_mask

    loss.backward()

    self.iter_end.record()

    if self.dataset.load2gpu_on_the_fly:
        viewpoint_cam.load2device('cpu')

    with torch.no_grad():
        # Progress bar
        self.ema_loss_for_log = 0.4 * loss.item() + 0.6 * self.ema_loss_for_log
        if self.iteration % 10 == 0:
            self.progress_bar.set_postfix({"Loss": f"{self.ema_loss_for_log:.{7}f}"})
            self.progress_bar.update(10)
        if self.iteration == self.opt.iterations:
            self.progress_bar.close()

        # Keep track of max radii in image-space for pruning
        if self.gaussians.max_radii2D.shape[0] == 0:
            self.gaussians.max_radii2D = torch.zeros_like(radii)
        self.gaussians.max_radii2D[visibility_filter] = torch.max(self.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])

        # Log and save
        cur_psnr, cur_ssim, cur_lpips, cur_ms_ssim, cur_alex_lpips = training_report(self.tb_writer, self.iteration, Ll1, loss, l1_loss, self.iter_start.elapsed_time(self.iter_end), self.testing_iterations, self.scene, render, (self.pipe, self.background), self.deform, self.dataset.load2gpu_on_the_fly, self.progress_bar)
        if self.iteration in self.testing_iterations:
            if cur_psnr.item() > self.best_psnr:
                self.best_psnr = cur_psnr.item()
                self.best_iteration = self.iteration
                self.best_ssim = cur_ssim.item()
                self.best_ms_ssim = cur_ms_ssim.item()
                self.best_lpips = cur_lpips.item()
                self.best_alex_lpips = cur_alex_lpips.item()

        if self.iteration in self.saving_iterations or self.iteration == self.best_iteration or self.iteration == self.opt.warm_up-1:
            print("\n[ITER {}] Saving Gaussians".format(self.iteration))
            self.scene.save(self.iteration)
            self.deform.save_weights(self.args.model_path, self.iteration)

        # Densification
        if self.iteration < self.opt.densify_until_iter:
            self.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

            if self.iteration > self.opt.node_densify_from_iter and self.iteration % self.opt.node_densification_interval == 0 and self.iteration < self.opt.node_densify_until_iter and self.iteration > self.opt.warm_up or self.iteration == self.opt.node_force_densify_prune_step:
                # Nodes densify
                self.deform.densify(max_grad=self.opt.densify_grad_threshold, x=self.gaussians.get_xyz, x_grad=self.gaussians.xyz_gradient_accum / self.gaussians.denom, feature=self.gaussians.feature, force_dp=(self.iteration == self.opt.node_force_densify_prune_step))

            if self.iteration > self.opt.densify_from_iter and self.iteration % self.opt.densification_interval == 0:
                size_threshold = 20 if self.iteration > self.opt.opacity_reset_interval else None
                self.gaussians.densify_and_prune(self.opt.densify_grad_threshold, 0.005, self.scene.cameras_extent, size_threshold)

            if self.iteration % self.opt.opacity_reset_interval == 0 or (
                    self.dataset.white_background and self.iteration == self.opt.densify_from_iter):
                self.gaussians.reset_opacity()

        # Optimizer step
        if self.iteration < self.opt.iterations:
            self.gaussians.optimizer.step()
            self.gaussians.update_learning_rate(self.iteration)
            self.gaussians.optimizer.zero_grad(set_to_none=True)
            self.deform.optimizer.step()
            self.deform.optimizer.zero_grad()
            self.deform.update_learning_rate(self.iteration)
            
    self.deform.update(max(0, self.iteration - self.opt.warm_up))

    if self.gui:
        dpg.set_value(
            "_log_train_psnr",
            "Best PSNR={} in Iteration {}, SSIM={}, LPIPS={},\n MS-SSIM={}, Alex-LPIPS={}".format('%.5f' % self.best_psnr, self.best_iteration, '%.5f' % self.best_ssim, '%.5f' % self.best_lpips, '%.5f' % self.best_ms_ssim, '%.5f' % self.best_alex_lpips)
        )
    else:
        self.progress_bar.set_description("Best PSNR={} in Iteration {}, SSIM={}, LPIPS={}, MS-SSIM={}, ALex-LPIPS={}".format('%.5f' % self.best_psnr, self.best_iteration, '%.5f' % self.best_ssim, '%.5f' % self.best_lpips, '%.5f' % self.best_ms_ssim, '%.5f' % self.best_alex_lpips))
    self.iteration += 1

    if self.gui:
        dpg.set_value(
            "_log_train_log",
            f"step = {self.iteration: 5d} loss = {loss.item():.4f}",
        )



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