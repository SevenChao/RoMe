import json
import copy
import numpy as np
import os
from hat.core.nus_box3d_utils import adjust_coords, get_min_max_coords

CAMERAS = ['front', 'left', 'back', 'right']


def gen_3d_points(z_range, bev_size, grid_size):
    """Generate 3D points in the Bird's Eye View (BEV) space.

    Args:
        z_range: The range of z-coordinates.

    Returns:
        coords (ndarray): The generated 3D points in BEV space. Shape: (H, W, Z, 4)
    """
    # Get the minimum and maximum x and y coordinates in the BEV space
    bev_min_x, bev_max_x, bev_min_y, bev_max_y = get_min_max_coords(bev_size)

    W = int(grid_size[0])
    H = int(grid_size[1])
    Z = int(z_range[1] - z_range[0])

    # x: (1, W, 1) -> (H, W, Z)
    x = np.linspace(bev_min_x, bev_max_x, W, dtype=np.float64).reshape(1, W, 1)
    x = np.repeat(x, H, axis=0)
    x = np.repeat(x, Z, axis=2)

    # y: (H, 1, 1) -> (H, W, Z)
    y = np.linspace(bev_min_y, bev_max_y, H, dtype=np.float64).reshape(H, 1, 1)
    y = np.repeat(y, W, axis=1)
    y = np.repeat(y, Z, axis=2)

    # z: (1, 1, Z) -> (H, W, Z)
    z = np.linspace(z_range[0], z_range[1], Z, dtype=np.float64).reshape(1, 1, Z)
    z = np.repeat(z, H, axis=0)
    z = np.repeat(z, W, axis=1)

    ones = np.ones((H, W, Z), dtype=np.float64)
    coords = np.stack([x, y, z, ones], axis=-1)
    return coords


def create_refpoint_bevlss(config):

 z_range = config['z_range']
 num_points = config['num_points']
 calib_path = config['calib_path']
 load_cylinder = config['load_cylinder']
 raw_imgs_shape = config['raw_imgs_shape']
 num_views = config['num_views']
 depth = config['depth']
 grid_size = config['grid_size']
 save_dir = config['save_dir']
 bev_size = config['bev_size']
 feat_hw = config['feat_hw']

    coords = gen_3d_points(z_range, bev_size, grid_size)
    H, W, Z = coords.shape[:3]

    new_coords = []
    for idx in range(4):  ###
        cam_pose = np.array(
            json.load(open(calib_path, 'r'))['camera_settings'][CAMERAS[idx]]['extrinsics']['pose'],
            dtype=np.float64,
        )
        cam2ego = np.linalg.inv(cam_pose)
        print(cam2ego)

        if load_cylinder:
            cam2ego = np.linalg.inv(cam2ego)

        # (H, W, Z, 4) x (4, 4)^T -> (H, W, Z, 4)
        new_coord = np.matmul(coords.astype(np.float32), cam2ego.T.astype(np.float32))

        if not load_cylinder:
            # perspective divide
            z_nonzero = new_coord[..., 2] != 0
            new_coord[..., 0][z_nonzero] = new_coord[..., 0][z_nonzero] / new_coord[..., 2][z_nonzero]
            new_coord[..., 1][z_nonzero] = new_coord[..., 1][z_nonzero] / new_coord[..., 2][z_nonzero]

        if not load_cylinder:
            r = np.sqrt(new_coord[..., 0] * new_coord[..., 0] + new_coord[..., 1] * new_coord[..., 1])
            st = np.arctan(r)
            t_i = copy.deepcopy(st)
            std0 = copy.deepcopy(st)

            tmp_dist = np.array(
                json.load(open(calib_path, 'r'))['camera_settings'][CAMERAS[idx]]['distort'],
                dtype=np.float32,
            )
            # expect 4 coefficients
            distcoeff = [np.array(t, dtype=np.float32) for t in tmp_dist]
            print(distcoeff)

            for i in range(4):
                t_i = t_i * (st * st)
                std0 = std0 + t_i * distcoeff[i]

            pos_mask = r > 0
            new_coord[..., 0][pos_mask] = (std0[pos_mask] / r[pos_mask]) * new_coord[..., 0][pos_mask]
            new_coord[..., 1][pos_mask] = (std0[pos_mask] / r[pos_mask]) * new_coord[..., 1][pos_mask]

        tmp_K = np.array(
            json.load(open(calib_path, 'r'))['camera_settings'][CAMERAS[idx]]['intrinsics'],
            dtype=np.float32,
        )
        camera_intrinsic = np.eye(4, dtype=np.float32)
        camera_intrinsic[: tmp_K.shape[0], : tmp_K.shape[1]] = tmp_K
        print(camera_intrinsic)

        new_coord_tmp = new_coord.copy()

        if not load_cylinder:
            # depth must be 1 to multiply with intrinsic
            new_coord_tmp[..., 2] = 1.0
            proj = np.matmul(new_coord_tmp, camera_intrinsic.T)
            new_coord[..., :2] = proj[..., :2]
        else:  # camera coord -> pic coord: cylinder different
            x_ang = np.arctan2(new_coord_tmp[..., 0], new_coord_tmp[..., 2])
            y_rat = new_coord_tmp[..., 1] / np.sqrt(
                new_coord_tmp[..., 0] * new_coord_tmp[..., 0] + new_coord_tmp[..., 2] * new_coord_tmp[..., 2]
            )
            new_coord[..., 0] = camera_intrinsic[0, 0] * x_ang + camera_intrinsic[0, 1] * y_rat + camera_intrinsic[0, 2]
            new_coord[..., 1] = camera_intrinsic[1, 0] * x_ang + camera_intrinsic[1, 1] * y_rat + camera_intrinsic[1, 2]

        ori_hw = raw_imgs_shape
        scales = (feat_hw[0] / ori_hw[1], feat_hw[1] / ori_hw[2])

        new_coord[..., 0] = new_coord[..., 0] * scales[1]
        new_coord[..., 1] = new_coord[..., 1] * scales[0]

        # permute to (Z, H, W, 4)
        new_coord = np.transpose(new_coord, (2, 0, 1, 3))
        new_coords.append(new_coord)

    new_coords = np.stack(new_coords, axis=1)  # (Z, num_cams, H, W, 4)
    B = new_coords.shape[1] // num_views

    # reshape to (Z, num_views, B, H, W, 4)
    Zdim = new_coords.shape[0]
    new_coords = new_coords.reshape(Zdim, B, num_views, H, W, 4).transpose(0, 2, 1, 3, 4, 5)

    #-----------------------------------------#
    # cast like torch long (truncate toward zero)
    X = np.trunc(new_coords[..., 0]).astype(np.int64)
    Y = np.trunc(new_coords[..., 1]).astype(np.int64)
    D = np.trunc(new_coords[..., 2]).astype(np.int64)
    #-----------------------------------------#

    idx = np.arange(num_views, dtype=np.int64).reshape(1, num_views, 1, 1, 1)
    idx = np.repeat(idx, Z, axis=0)
    idx = np.repeat(idx, B, axis=2)
    idx = np.repeat(idx, H, axis=3)
    idx = np.repeat(idx, W, axis=4)
    new_coords = np.stack([X, Y, D, idx], axis=-1)

    feat_h, feat_w = feat_hw
    invalid = (
        (new_coords[..., 0] < 0)
        | (new_coords[..., 0] >= feat_w)
        | (new_coords[..., 1] < 0)
        | (new_coords[..., 1] >= feat_h)
        | (new_coords[..., 2] < 0)
        | (new_coords[..., 2] >= depth)
    )

    safe = np.array((feat_w - 1, feat_h - 1, depth, num_views - 1), dtype=np.int64)
    new_coords[invalid] = safe
    new_coords = new_coords.reshape(-1, B, H, W, 4)

    rank = (
        new_coords[..., 2] * feat_h * feat_w * num_views
        + new_coords[..., 1] * feat_w * num_views
        + new_coords[..., 0] * num_views
        + new_coords[..., 3]
    )

    # take k smallest along axis=0 (the Z*views combined dimension after reshape)
    # current rank shape: (Z, num_views, B, H, W) flattened to (-1, B, H, W)
    # already flattened first dim above; rank shares same shape as new_coords[...,0]
    # select top-k smallest along axis=0
    part_idx = np.argpartition(rank, num_points - 1, axis=0)[:num_points]
    # gather the values for consistency with torch.topk output order (ascending)
    gathered = np.take_along_axis(rank, part_idx, axis=0)
    order = np.argsort(gathered, axis=0)
    rank = np.take_along_axis(gathered, order, axis=0)

    D = rank // (feat_h * feat_w * num_views)
    rank = rank % (feat_h * feat_w * num_views)

    Y = rank // (feat_w * num_views)
    rank = rank % (feat_w * num_views)

    X = rank // num_views
    idx = rank % num_views

    idx_Y = idx * feat_h + Y
    feat_coords = np.stack((X, idx_Y), axis=-1)

    feat_points = adjust_coords(feat_coords, grid_size)

    X_Y = Y * feat_w + X
    idx_D = idx * depth + D
    depth_coords = np.stack((X_Y, idx_D), axis=-1)
    depth_points = adjust_coords(depth_coords, grid_size)
    feat_points = feat_points.reshape(-1, H, W, 2)
    depth_points = depth_points.reshape(-1, H, W, 2)

    print(feat_points.shape)
    print(depth_points.shape)
    print(feat_points.dtype)
    print(depth_points.dtype)
    np.asarray(feat_points).astype(np.float32).ravel().tofile(os.path.join(save_dir, 'feat_points.bin'))
    np.asarray(depth_points).astype(np.float32).ravel().tofile(os.path.join(save_dir, 'depth_points.bin'))

if __name__ == "__main__":
 config = {}
 config['z_range'] = (-10.0, 10.0)
 config['num_points'] = 10
 config['calib_path'] = '/mnt/workspace/gac_luoxianghong/unified_perception_training/data/fisheye_cam_param.json'
 config['load_cylinder'] = False
 config['raw_imgs_shape'] = (3, 512, 768)
 config['num_views'] = 4
 config['depth'] = 60
 config['grid_size'] = (80, 80)
 config['save_dir'] = '/mnt/workspace/gac_luoxianghong/unified_perception_training/data'
 config['bev_size'] = (25.6, 25.6, 0.64)
 config['feat_hw'] = (32, 48)
 create_refpoint_bevlss(config)