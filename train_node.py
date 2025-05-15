
import wandb
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from scene import Scene, GaussianModel, DeformModel
import numpy as np
import dearpygui.dearpygui as dpg
from scipy.spatial.transform import Rotation as R
from gaussian_renderer import render


def train_node_rendering_step(self, step):
    
    from jhutil import color_log; color_log(6666, "train_node", repeat=False)
    # Pick a random Camera
    from jhutil import color_log; color_log("aaaa", "pick random camera", repeat=False)
    if not self.viewpoint_stack:
        if self.opt.progressive_train_node and self.iteration_node_rendering < int(self.opt.progressive_stage_steps / self.opt.progressive_stage_ratio) + self.opt.node_warm_up:
            if self.iteration_node_rendering < self.opt.node_warm_up:
                sorted_train_cams = sorted(self.scene.getTrainCameras().copy(), key=lambda x: x.fid)
                max_cam_num = max(30, int(0.01 * len(sorted_train_cams)))
                viewpoint_stack = sorted_train_cams[0: max_cam_num]
            else:
                cameras_to_train_idx = int(min(((self.iteration_node_rendering - self.opt.node_warm_up) / self.opt.progressive_stage_steps + 1) * self.opt.progressive_stage_ratio, 1.) * len(self.scene.getTrainCameras()))
                cameras_to_train_idx = max(cameras_to_train_idx, 1)
                interval_len = int(len(self.scene.getTrainCameras()) * self.opt.progressive_stage_ratio)
                min_idx = max(0, cameras_to_train_idx - interval_len)
                sorted_train_cams = sorted(self.scene.getTrainCameras().copy(), key=lambda x: x.fid)
                viewpoint_stack = sorted_train_cams[min_idx: cameras_to_train_idx]
                out_domain_idx = np.concatenate([np.arange(min_idx), np.arange(cameras_to_train_idx, min(len(self.scene.getTrainCameras()), cameras_to_train_idx+interval_len))])
                if len(out_domain_idx) >= interval_len:
                    out_domain_len = min(interval_len*5, len(out_domain_idx))
                    out_domain_idx = np.random.choice(out_domain_idx, [out_domain_len], replace=False)
                    out_domain_stack = [sorted_train_cams[idx] for idx in out_domain_idx]
                    viewpoint_stack = viewpoint_stack + out_domain_stack
        else:
            viewpoint_stack = self.scene.getTrainCameras().copy()
        self.viewpoint_stack = viewpoint_stack

    viewpoint_cam = self.viewpoint_stack.pop(randint(0, len(self.viewpoint_stack) - 1))
    if self.dataset.load2gpu_on_the_fly:
        viewpoint_cam.load2device()
    fid = viewpoint_cam.fid
    
    time_input = fid.unsqueeze(0).expand(self.deform.deform.as_gaussians.get_xyz.shape[0], -1)
    N = time_input.shape[0]

    total_frame = len(self.scene.getTrainCameras())
    time_interval = 1 / total_frame

    ast_noise = 0 if self.dataset.is_blender else torch.randn(1, 1, device='cuda').expand(N, -1) * time_interval * self.smooth_term(self.iteration_node_rendering)
    d_values = self.deform.deform.query_network(x=self.deform.deform.as_gaussians.get_xyz.detach(), t=time_input + ast_noise)
    d_xyz, d_opacity, d_color = d_values['d_xyz'] * self.deform.deform.as_gaussians.motion_mask, d_values['d_opacity'] * self.deform.deform.as_gaussians.motion_mask if d_values['d_opacity'] is not None else None, d_values['d_color'] * self.deform.deform.as_gaussians.motion_mask if d_values['d_color'] is not None else None

    d_rot, d_scale = 0., 0.
    if self.iteration_node_rendering < self.opt.node_warm_up:
        d_xyz = d_xyz.detach()
    d_color = d_color.detach() if d_color is not None else None
    d_opacity = d_opacity.detach() if d_opacity is not None else None

    from jhutil import color_log; color_log("bbbb", "render gaussian", repeat=False)
    # Render
    random_bg_color = (self.opt.gt_alpha_mask_as_scene_mask or (self.opt.gt_alpha_mask_as_dynamic_mask and not self.deform.deform.as_gaussians.with_motion_mask)) and viewpoint_cam.gt_alpha_mask is not None
    render_pkg_re = render(viewpoint_cam, self.deform.deform.as_gaussians, self.pipe, self.background, d_xyz, d_rot, d_scale, random_bg_color=random_bg_color, d_opacity=d_opacity, d_color=d_color, d_rot_as_res=self.deform.d_rot_as_res)
    image, viewspace_point_tensor, visibility_filter, radii = render_pkg_re["render"], render_pkg_re["viewspace_points"], render_pkg_re["visibility_filter"], render_pkg_re["radii"]

    from jhutil import color_log; color_log("cccc", "calc loss", repeat=False)
    # Loss
    gt_image = viewpoint_cam.original_image.cuda()
    if random_bg_color:
        gt_alpha_mask = viewpoint_cam.gt_alpha_mask.cuda()
        gt_image = gt_image * gt_alpha_mask + render_pkg_re['bg_color'][:, None, None] * (1 - gt_alpha_mask)
    Ll1 = l1_loss(image, gt_image)
    if step % 10 == 0:
        # torch.save(self.deform.deform.as_gaussians.get_xyz.detach(), "/tmp/.cache/xyz3.pt")
        # xyz = torch.load("/tmp/.cache/xyz3.pt")
        xyz_mean = self.deform.deform.as_gaussians.get_xyz.detach().mean()
        opacity_mean = self.deform.deform.as_gaussians.get_opacity.detach().mean()
        scale_mean = self.deform.deform.as_gaussians.get_scaling.detach().mean()
        logging_data = {
            "xyz_mean": xyz_mean,
            "opacity_mean": opacity_mean,
            "scale_mean": scale_mean,
        }
        if step % 1000 == 0:
            from jhutil import get_img_diff
            logging_data["train_diff_img(node)"]= wandb.Image(get_img_diff(image, gt_image))
            wandb.log(logging_data, commit=True)
        else:
            wandb.log(logging_data)



    loss_img = (1.0 - self.opt.lambda_dssim) * Ll1 + self.opt.lambda_dssim * (1.0 - ssim(image, gt_image))
    loss = loss_img

    if self.iteration_node_rendering > self.opt.node_warm_up:
        if not self.deform.deform.use_hash:
            elastic_loss = 1e-3 * self.deform.deform.elastic_loss(t=fid, delta_t=time_interval)
            loss_acc = 1e-5 * self.deform.deform.acc_loss(t=fid, delta_t=3*time_interval)
            loss = loss + elastic_loss + loss_acc
        if not self.opt.no_arap_loss:
            loss_opt_trans = 1e-2 * self.deform.deform.arap_loss()
            loss = loss + loss_opt_trans

    # Motion Mask Loss
    if self.opt.gt_alpha_mask_as_dynamic_mask and self.deform.deform.as_gaussians.with_motion_mask and viewpoint_cam.gt_alpha_mask is not None and self.iteration_node_rendering > self.opt.node_warm_up:
        gt_alpha_mask = viewpoint_cam.gt_alpha_mask.cuda()[0]
        render_pkg_motion = render(viewpoint_cam, self.deform.deform.as_gaussians, self.pipe, self.background, d_xyz, d_rot, d_scale, render_motion=True, detach_xyz=True, detach_rot=True, detach_scale=True, detach_opacity=self.deform.deform.as_gaussians.with_motion_mask, d_opacity=d_opacity, d_color=d_color, d_rot_as_res=self.deform.d_rot_as_res)
        motion_image = render_pkg_motion["render"][0]
        L_motion = l1_loss(gt_alpha_mask, motion_image)
        loss = loss + L_motion

    n_node = len(self.deform.deform.as_gaussians.get_xyz)
    from jhutil import color_log; color_log(0000, f'n_node: {n_node}   loss: {loss:.3f}', update=True)
    
    loss.backward()
    with torch.no_grad():
        if self.iteration_node_rendering < self.opt.iterations_node_sampling:
            # Densification
            self.deform.deform.as_gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
            if self.iteration_node_rendering % self.opt.densification_interval == 0 or self.iteration_node_rendering == self.opt.node_warm_up - 1:
                size_threshold = 20 if self.iteration_node_rendering > self.opt.opacity_reset_interval else None
                if self.dataset.is_blender:
                    grad_max = self.opt.densify_grad_threshold
                else:
                    if self.deform.deform.as_gaussians.get_xyz.shape[0] > self.deform.deform.node_num * self.opt.node_max_num_ratio_during_init:
                        grad_max = torch.inf
                    else:
                        grad_max = self.opt.densify_grad_threshold
                self.deform.deform.as_gaussians.densify_and_prune(grad_max, 0.005, self.scene.cameras_extent, size_threshold)
            if self.iteration_node_rendering % self.opt.opacity_reset_interval == 0 or (
                    self.dataset.white_background and self.iteration_node_rendering == self.opt.densify_from_iter):
                self.deform.deform.as_gaussians.reset_opacity()
        elif self.iteration_node_rendering == self.opt.iterations_node_sampling:
            # Downsampling nodes for sparse control
            # Strategy 1: Directly use the original gs as nodes
            # Strategy 2: Sampling in the hyper space across times
            strategy = self.opt.deform_downsamp_strategy
            if strategy == 'direct':
                original_gaussians: GaussianModel = self.deform.deform.as_gaussians
                self.deform.deform.init(opt=self.opt, init_pcl=original_gaussians.get_xyz, keep_all=True, force_init=True, reset_bbox=False, feature=self.gaussians.feature)
                gaussians: GaussianModel = self.deform.deform.as_gaussians
                gaussians._features_dc = torch.nn.Parameter(original_gaussians._features_dc)
                gaussians._features_rest = torch.nn.Parameter(original_gaussians._features_rest)
                gaussians._scaling = torch.nn.Parameter(original_gaussians._scaling)
                gaussians._opacity = torch.nn.Parameter(original_gaussians._opacity)
                gaussians._rotation = torch.nn.Parameter(original_gaussians._rotation)
                if gaussians.fea_dim > 0:
                    gaussians.feature = torch.nn.Parameter(original_gaussians.feature)
                print('Reset the optimizer of the deform model.')
                self.deform.train_setting(self.opt)
            elif strategy == 'samp_hyper':
                original_gaussians: GaussianModel = self.deform.deform.as_gaussians
                time_num = 16
                t_samp = torch.linspace(0, 1, time_num).cuda()
                x = original_gaussians.get_xyz.detach()
                trans_samp = []
                for i in range(time_num):
                    time_input = t_samp[i:i+1, None].expand_as(x[..., :1])
                    trans_samp.append(self.deform.deform.query_network(x=x, t=time_input)['d_xyz'] * original_gaussians.motion_mask)
                trans_samp = torch.stack(trans_samp, dim=1)
                hyper_pcl = (trans_samp + original_gaussians.get_xyz[:, None]).reshape([original_gaussians.get_xyz.shape[0], -1])
                dynamic_mask = self.deform.deform.as_gaussians.motion_mask[..., 0] > .5
                if not self.opt.deform_downsamp_with_dynamic_mask:
                    dynamic_mask = torch.ones_like(dynamic_mask)
                idx = self.deform.deform.init(init_pcl=original_gaussians.get_xyz[dynamic_mask], hyper_pcl=hyper_pcl[dynamic_mask], force_init=True, opt=self.opt, reset_bbox=False, feature=self.gaussians.feature)
                gaussians: GaussianModel = self.deform.deform.as_gaussians
                gaussians._features_dc = torch.nn.Parameter(original_gaussians._features_dc[dynamic_mask][idx])
                gaussians._features_rest = torch.nn.Parameter(original_gaussians._features_rest[dynamic_mask][idx])
                gaussians._scaling = torch.nn.Parameter(original_gaussians._scaling[dynamic_mask][idx])
                gaussians._opacity = torch.nn.Parameter(original_gaussians._opacity[dynamic_mask][idx])
                gaussians._rotation = torch.nn.Parameter(original_gaussians._rotation[dynamic_mask][idx])
                if gaussians.fea_dim > 0:
                    gaussians.feature = torch.nn.Parameter(original_gaussians.feature[dynamic_mask][idx])
                gaussians.training_setup(self.opt)
            # No update at the step
            self.deform.deform.as_gaussians.optimizer.zero_grad(set_to_none=True)
            self.deform.optimizer.zero_grad()

        if self.iteration_node_rendering == self.opt.iterations_node_rendering - 1 and self.iteration_node_rendering > self.opt.iterations_node_sampling:
            # Just finish node training and has down sampled control nodes
            self.deform.deform.nodes.data[..., :3] = self.deform.deform.as_gaussians._xyz

        if not self.iteration_node_rendering == self.opt.iterations_node_sampling and not self.iteration_node_rendering == self.opt.iterations_node_rendering - 1:
            # Optimizer step
            self.deform.deform.as_gaussians.optimizer.step()
            self.deform.deform.as_gaussians.update_learning_rate(self.iteration_node_rendering)
            self.deform.deform.as_gaussians.optimizer.zero_grad(set_to_none=True)
            self.deform.update_learning_rate(self.iteration_node_rendering)
            self.deform.optimizer.step()
            self.deform.optimizer.zero_grad()
    
    self.deform.update(max(0, self.iteration_node_rendering - self.opt.node_warm_up))

    if self.dataset.load2gpu_on_the_fly:
        viewpoint_cam.load2device('cpu')

    self.iteration_node_rendering += 1

    if self.gui:
        dpg.set_value(
            "_log_train_log",
            f"step = {self.iteration_node_rendering: 5d} loss = {loss.item():.4f}",
        )

