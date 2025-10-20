# Standard library imports
import argparse
import yaml
from tqdm import tqdm
import random
import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Third-party imports
import numpy as np
import torch
import os
import cv2
from os.path import join

from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

from pytorch3d.renderer import PerspectiveCameras
from pytorch3d.loss import mesh_laplacian_smoothing

# Local imports - Model components
from models.loss import L1MaskedLoss, CELossWithMask
from models.pose_model import ExtrinsicModel

# Local imports - Utility functions
from utils.geometry import fps_by_distance
from utils.renderer import Renderer
from utils.visualizer import Visualizer, loss2color, depth2color, save_cut_mesh, save_cut_label_mesh
from utils.wandb_loggers import WandbLogger
from utils.image import render_semantic

# Local imports - Evaluation
from scripts.eval import eval


def set_randomness(args):
    """
    Set random seeds for reproducibility across different libraries.
    
    This function ensures that all random operations (Python, NumPy, PyTorch)
    produce the same results across different runs, which is crucial for
    reproducible experiments and debugging.
    
    Args:
        args (dict): Configuration dictionary containing 'rand_seed' key
    """
    # Set Python random seed
    random.seed(args["rand_seed"])
    
    # Set NumPy random seed
    np.random.seed(args["rand_seed"])
    
    # Set PyTorch random seed
    torch.manual_seed(args["rand_seed"])
    
    # Ensure deterministic behavior in CUDA operations
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior, ensure the same results for the same input
    torch.backends.cudnn.benchmark = False     # Disable optimization for reproducibility


def get_configs():
    """
    Parse command line arguments and load configuration from YAML file.
    
    This function handles command line argument parsing and loads the training
    configuration from a YAML file. The configuration contains all necessary
    parameters for training including model settings, optimization parameters,
    and dataset specifications.
    
    Returns:
        dict: Configuration dictionary loaded from YAML file containing:
            - Model architecture parameters
            - Training hyperparameters
            - Dataset configuration
            - Optimization settings
    """
    # Create argument parser for command line interface
    parser = argparse.ArgumentParser(description='RoMe training configuration')
    parser.add_argument(
        '--config',
        default="configs/local_carla.yaml",
        help='Path to configuration YAML file')
    args = parser.parse_args()
    
    # Load configuration from YAML file
    with open(args.config) as file:
        configs = yaml.safe_load(file)
    return configs


def train(configs):
    """
    Main training function for 3D scene reconstruction and semantic segmentation.
    
    This function implements the complete training pipeline for the RoMe system,
    including:
    - Dataset loading and preprocessing
    - Model initialization and configuration
    - Multi-task optimization (RGB, semantic, geometry)
    - Incremental training with waypoint sampling
    - Visualization and logging
    
    Args:
        configs (dict): Configuration dictionary containing all training parameters:
            - Model architecture settings
            - Training hyperparameters
            - Dataset configuration
            - Optimization parameters
            - Visualization settings
    """
    # ==================== INITIALIZATION ====================
    # Set random seeds for reproducibility
    set_randomness(configs)
    
    # Configure wandb for cluster environments
    if configs["cluster"]:
        os.environ['WANDB_MODE'] = 'offline'
    
    # Set device (GPU)
    device = torch.device("cuda:0")

    # ==================== DATASET AND COMPONENT INITIALIZATION ====================
    # Import dataset class based on configuration
    if configs["dataset"] == "NuscDataset":
        from datasets.nusc import NuscDataset as Dataset
    elif configs["dataset"] == "KittiDataset":
        from datasets.kitti import KittiDataset as Dataset
    else:
        raise NotImplementedError("Dataset not implemented")

    # Initialize core components
    logger = WandbLogger(configs)           # For experiment logging and monitoring
    visualizer = Visualizer(device, configs) # For BEV visualization and mesh rendering
    renderer = Renderer().to(device)        # For 3D mesh rendering to 2D images
    dataset = Dataset(configs)              # Load and configure dataset
    
    # List of datasets that support depth supervision
    # These datasets provide ground truth depth maps for supervision
    supervise_depth_list = ["FsdDataset", "CarlaDataset"]
    # supervise_depth_list = ["CarlaDataset"]  # Alternative configuration

    # ==================== CAMERA POSE PROCESSING ====================
    # Extract camera positions and compute offset for grid alignment
    pose_xy = np.array(dataset.ref_camera2world_all)[:, :2, 3]  # Extract x,y coordinates from camera poses
    offset_pose_xy = pose_xy - np.asarray([configs["center_point"]["x"], configs["center_point"]["y"]])
    print(f"Get {len(dataset.ref_camera2world_all)} images for mapping")

    # ==================== OPTIMIZATION CONFIGURATION ====================
    # Determine which components to optimize based on learning rates
    # This allows flexible configuration of what to optimize (RGB, labels, geometry, poses)
    optim_dict = dict()
    for optim_option in ["vertices_rgb", "vertices_label", "vertices_z", "rotations", "translations"]:
        if configs["lr"].get(optim_option, 0) != 0:
            print("{} optimization is ON".format(optim_option))
            optim_dict[optim_option] = True
        else:
            optim_dict[optim_option] = False
            print("{} optimization is OFF".format(optim_option))

    # ==================== GRID MODEL SELECTION ====================
    # Select appropriate grid generator based on optimization configuration
    # Different combinations of RGB, label, and depth optimization require different grid types
    # This dynamic selection allows for flexible model architectures based on training objectives
    if optim_dict["vertices_rgb"] and optim_dict["vertices_label"] and (not optim_dict["vertices_z"]):
        from models.voxel import SquareFlatGridRGBLabel as SquareFlatGrid
    elif optim_dict["vertices_rgb"] and (not optim_dict["vertices_label"]) and optim_dict["vertices_z"]:
        from models.voxel import SquareFlatGridRGBZ as SquareFlatGrid
    elif optim_dict["vertices_rgb"] and (not optim_dict["vertices_label"]) and (not optim_dict["vertices_z"]):
        from models.voxel import SquareFlatGridRGB as SquareFlatGrid
    elif (not optim_dict["vertices_rgb"]) and optim_dict["vertices_label"] and (not optim_dict["vertices_z"]):
        from models.voxel import SquareFlatGridLabel as SquareFlatGrid
    elif (not optim_dict["vertices_rgb"]) and optim_dict["vertices_label"] and optim_dict["vertices_z"]:
        from models.voxel import SquareFlatGridLabelZ as SquareFlatGrid
    elif optim_dict["vertices_rgb"] and optim_dict["vertices_label"] and optim_dict["vertices_z"]:
        from models.voxel import SquareFlatGridRGBLabelZ as SquareFlatGrid
    else:
        raise NotImplementedError("No such grid generator, please check your config[\"lr\"]")

    # ==================== GRID MODEL INITIALIZATION ====================
    # Initialize the grid model based on whether depth optimization is enabled
    if optim_dict["vertices_z"]:
        # Grid with depth optimization requires positional encoding for neural network
        grid = SquareFlatGrid(configs["bev_x_length"], configs["bev_y_length"], offset_pose_xy,
                              configs["bev_resolution"], dataset.num_class, configs["pos_enc"], configs["cut_range"])
    else:
        # Grid without depth optimization (fixed height)
        grid = SquareFlatGrid(configs["bev_x_length"], configs["bev_y_length"], offset_pose_xy,
                              configs["bev_resolution"], dataset.num_class, configs["cut_range"])
    
    # Move grid to GPU and initialize vertex heights
    grid = grid.to(device)
    grid.init_vertices_z()  # Initialize z-coordinates of vertices

    # ==================== PARAMETER ORGANIZATION ====================
    # Organize trainable parameters into different groups with specific learning rates
    # This allows for different learning rates for different types of parameters
    parameters = []      # RGB and label parameters
    z_parameters = []    # Depth parameters (MLP weights)
    pose_parameters = [] # Camera pose parameters
    
    # Group grid parameters by type for different optimizers
    for param_key, param in grid.named_parameters():
        if "vertices_rgb" in param_key or "vertices_label" in param_key:
            # RGB and label parameters use their specific learning rates
            parameters.append({"params": param, "lr": float(configs["lr"][param_key.split('.')[-1]])})
        else:
            # Depth parameters (MLP weights) use vertices_z learning rate
            z_parameters.append({"params": param, "lr": float(configs["lr"]["vertices_z"])})

    # ==================== CAMERA POSE MODEL INITIALIZATION ====================
    # Initialize camera pose model for joint optimization of camera extrinsics
    poses = ExtrinsicModel(configs, optim_dict["rotations"], optim_dict["translations"], 
                          num_camera=len(dataset.camera_extrinsics)).to(device)
    
    # Group pose parameters for separate optimization
    for param_key, param in poses.named_parameters():
        pose_parameters.append({"params": param, "lr": float(configs["lr"][param_key])})

    # ==================== OPTIMIZER AND LOSS FUNCTION INITIALIZATION ====================
    # Initialize optimizers and loss functions for multi-task learning
    optimizer = torch.optim.Adam(parameters)  # Main optimizer for RGB and labels
    scheduler = MultiStepLR(optimizer, milestones=configs["lr_milestones"], gamma=configs["lr_gamma"])
    
    # Additional optimizers for specific parameter groups
    if optim_dict["vertices_z"]:
        z_optimizer = torch.optim.Adam(z_parameters)  # Depth optimizer for MLP parameters
    if optim_dict["translations"] or optim_dict["rotations"]:
        pose_optimizer = torch.optim.Adam(pose_parameters)  # Pose optimizer for camera extrinsics
    
    # Loss functions for different tasks
    loss_fuction = L1MaskedLoss()        # For RGB rendering (photometric loss)
    depth_loss_fuction = L1MaskedLoss()  # For depth supervision (geometric loss)
    CE_loss_with_mask = CELossWithMask() # For semantic segmentation (classification loss)

    # ==================== TRAINING CONFIGURATION ====================
    radius = configs["waypoint_radius"]  # Radius for waypoint-based local training
    
    # ==================== MAIN TRAINING LOOP ====================
    # Main training loop with progress bar
    loop = tqdm(range(1, configs["epochs"]+1))
    for epoch in loop:
        # ==================== WAYPOINT SAMPLING ====================
        # Sample waypoints using Farthest Point Sampling for efficient training
        # FPS ensures good coverage of the scene while minimizing computational cost
        waypoints = fps_by_distance(pose_xy, min_distance=radius*2, return_idx=False)
        print(f"epoch-{epoch}: get {waypoints.shape[0]} waypoints")
        
        # ==================== LOSS INITIALIZATION ====================
        # Initialize loss tracking dictionary for multi-task learning
        loss_dict = dict()
        if optim_dict["vertices_rgb"]:
            loss_dict["render_loss"] = 0      # RGB rendering loss
        if optim_dict["vertices_label"]:
            loss_dict["seg_loss"] = 0         # Semantic segmentation loss
        if optim_dict["vertices_z"]:
            loss_dict["laplacian_loss"] = 0   # Mesh smoothness loss
            if configs["dataset"] in supervise_depth_list:
                loss_dict["depth_loss"] = 0   # Depth supervision loss
        loss_dict["total_loss"] = 0           # Combined total loss

        # ==================== WAYPOINT PROCESSING LOOP ====================
        # Process each waypoint for local training to optimize memory usage
        num_frames = 0
        for waypoint in waypoints:
            # ==================== LOCAL ACTIVATION SETUP ====================
            # Convert waypoint to grid coordinates for local activation
            vertice_waypoint = waypoint + dataset.world2bev[:2, 3]
            
            # Get activation indices for local grid optimization (if depth is optimized)
            # This determines which vertices to optimize in the current waypoint region
            if optim_dict["vertices_z"]:
                activation_idx = grid.get_activation_idx(vertice_waypoint, radius)
            
            # ==================== LOCAL DATASET CREATION ====================
            # Set current waypoint and create local dataset
            # The dataset is filtered to only include images near the current waypoint
            dataset.set_waypoint(waypoint, radius * 1.1)
            num_frames += len(dataset)
            print(f"+ {len(dataset)}, num_frames = {num_frames}")
            
            # Create data loader for current waypoint
            dataloader = DataLoader(dataset, batch_size=configs["batch_size"],
                                    num_workers=configs["num_workers"],
                                    shuffle=True,
                                    drop_last=True)

            # ==================== BATCH PROCESSING LOOP ====================
            # Process each batch in the current waypoint
            for sample in dataloader:
                # ==================== DATA PREPARATION ====================
                # Move data to GPU for efficient computation
                for key, ipt in sample.items():
                    if key != "image_path":  # Skip string paths
                        sample[key] = ipt.clone().detach().to(device)
                
                # ==================== MESH GENERATION ====================
                # Generate mesh from grid with optional local activation
                if optim_dict["vertices_z"]:
                    # Local activation: only compute heights for activated vertices
                    mesh = grid(activation_idx, configs["batch_size"])
                else:
                    # Full mesh: compute all vertices (no height optimization)
                    mesh = grid(configs["batch_size"])
                
                # ==================== CAMERA POSE PROCESSING ====================
                # Get camera poses from the pose model
                pose = poses(sample["camera_idx"])
                
                # Apply pose optimization after specified epoch
                # Early epochs use fixed poses for stability, later epochs optimize poses
                if epoch >= configs["extrinsic"]["start_epoch"]:
                    # Joint optimization: combine learned poses with initial poses
                    transform = pose @ sample["Transform_pytorch3d"]
                else:
                    # Fixed poses: use only initial camera poses
                    transform = sample["Transform_pytorch3d"]

                # ==================== CAMERA PARAMETER EXTRACTION ====================
                # Extract camera parameters for PyTorch3D rendering
                R_pytorch3d = transform[:, :3, :3]  # Rotation matrix (3x3)
                T_pytorch3d = transform[:, :3, 3]   # Translation vector (3x1)
                focal_pytorch3d = sample["focal_pytorch3d"]  # Focal length
                p0_pytorch3d = sample["p0_pytorch3d"]        # Principal point
                image_shape = sample["image_shape"]          # Image dimensions
                
                # Create PyTorch3D camera objects for rendering
                cameras = PerspectiveCameras(
                    R=R_pytorch3d,              # Camera rotation
                    T=T_pytorch3d,              # Camera translation
                    focal_length=focal_pytorch3d, # Focal length
                    principal_point=p0_pytorch3d, # Principal point
                    image_size=image_shape,      # Image size
                    device=device                # GPU device
                )
                # ==================== GROUND TRUTH EXTRACTION ====================
                # Extract ground truth data for loss computation
                gt_image = sample["image"]  # Ground truth RGB images
                if optim_dict["vertices_z"] and (configs["dataset"] in supervise_depth_list):
                    gt_depth = sample["depth"]  # Ground truth depth maps (if available)
                gt_seg = sample["static_label"]  # Ground truth semantic segmentation

                # ==================== MESH RENDERING ====================
                # Render mesh to get images and depth using PyTorch3D
                images_feature, depth = renderer({"mesh": mesh, "cameras": cameras})
                
                # ==================== MASK CREATION ====================
                # Create silhouette mask from alpha channel to focus on rendered regions
                silhouette = images_feature[:, :, :, -1]  # Extract alpha channel
                silhouette[silhouette > 0] = 1           # Binary mask: 1 for rendered, 0 for background
                silhouette = torch.unsqueeze(silhouette, -1)  # Add channel dimension
                mask = silhouette
                
                # Apply static mask if available to exclude dynamic objects
                if "static_mask" in sample:
                    static_mask = torch.unsqueeze(sample["static_mask"], -1)
                    mask *= static_mask  # Combine silhouette and static masks

                # ==================== FEATURE EXTRACTION ====================
                # Extract RGB and segmentation channels from rendered features
                images = images_feature[:, :, :, :3]  # RGB channels (first 3 channels)
                if optim_dict["vertices_rgb"]:
                    # When RGB is optimized, segmentation channels are after RGB
                    images_seg = images_feature[:, :, :, 3:-1]  # Segmentation channels (excluding alpha)
                else:
                    # When RGB is not optimized, all channels except alpha are segmentation
                    images_seg = images_feature[:, :, :, :-1]  # All channels except alpha

                # ==================== GRADIENT CLEARING ====================
                # Clear gradients for all optimizers before backward pass
                optimizer.zero_grad()  # Clear gradients for RGB and label parameters
                if optim_dict["vertices_z"]:
                    z_optimizer.zero_grad()  # Clear gradients for depth parameters
                if optim_dict["translations"] or optim_dict["rotations"]:
                    pose_optimizer.zero_grad()  # Clear gradients for pose parameters
                
                # Initialize total loss accumulator
                total_loss = 0
                # ==================== LOSS COMPUTATION ====================
                # Compute different loss components for multi-task learning
                if optim_dict["vertices_rgb"]:
                    # RGB rendering loss: photometric consistency
                    render_loss = loss_fuction(images, gt_image, mask)
                    total_loss += render_loss.mean()
                
                if optim_dict["vertices_label"]:
                    # Semantic segmentation loss: classification accuracy
                    seg_loss = CE_loss_with_mask(images_seg.reshape(-1, images_seg.shape[-1]),
                                                 gt_seg.reshape(-1), mask.reshape(-1)) * configs["seg_loss_weight"]
                    total_loss += seg_loss
                
                if optim_dict["vertices_z"]:
                    # Depth supervision loss (if dataset supports it): geometric consistency
                    if configs["dataset"] in supervise_depth_list:
                        mask_depth = gt_depth > 0  # Valid depth mask
                        depth_loss = depth_loss_fuction(depth, gt_depth, mask * mask_depth) * configs["depth_loss_weight"]
                        total_loss += depth_loss.mean()
                    
                    # Laplacian smoothing loss for mesh regularization: surface smoothness
                    laplacian_loss = mesh_laplacian_smoothing(mesh) * configs["laplacian_loss_weight"]
                    total_loss += laplacian_loss

                # ==================== BACKWARD PASS AND OPTIMIZATION ====================
                # Backward pass and parameter updates for all optimizers
                total_loss.backward()  # Compute gradients
                optimizer.step()       # Update RGB and label parameters
                z_optimizer.step() if z_parameters else None      # Update depth parameters
                pose_optimizer.step() if pose_parameters else None  # Update pose parameters
                # ==================== LOSS ACCUMULATION ====================
                # Accumulate losses for logging and monitoring
                if optim_dict["vertices_rgb"]:
                    loss_dict["render_loss"] += render_loss.mean().detach().cpu().numpy()
                if optim_dict["vertices_label"]:
                    loss_dict["seg_loss"] += seg_loss.detach().cpu().numpy()
                if optim_dict["vertices_z"]:
                    loss_dict["laplacian_loss"] += laplacian_loss.detach().cpu().numpy()
                    if configs["dataset"] in supervise_depth_list:
                        loss_dict["depth_loss"] += depth_loss.mean().detach().cpu().numpy()

                loss_dict["total_loss"] += total_loss.detach().cpu().numpy()
        # ==================== EPOCH COMPLETION ====================
        # Update learning rate scheduler
        scheduler.step()
        
        # Generate full mesh for visualization (no gradients needed)
        with torch.no_grad():
            if optim_dict["vertices_z"]:
                # Generate full mesh with all vertices for visualization
                mesh = grid(None, configs["batch_size"])
            else:
                # Generate full mesh without height optimization
                mesh = grid(configs["batch_size"])

        # ==================== VISUALIZATION AND LOGGING ====================
        if not configs["cluster"]:
            # ==================== LOSS LOGGING ====================
            # Average losses over all batches and log to wandb
            for key, value in loss_dict.items():
                loss_dict[key] = value / len(dataloader)
            logger.log(loss_dict, epoch)
            
            # ==================== BEV VISUALIZATION ====================
            # Generate Bird's Eye View visualization of the mesh
            bev_features, bev_depth = visualizer(mesh[0])
            # ==================== BEV FEATURE EXTRACTION ====================
            # Extract segmentation features from BEV visualization
            if optim_dict["vertices_rgb"]:
                # When RGB is optimized, segmentation channels are after RGB
                bev_seg = bev_features[0, :, :, 3:-1].detach().cpu().numpy()
            else:
                # When RGB is not optimized, all channels except alpha are segmentation
                bev_seg = bev_features[0, :, :, :-1].detach().cpu().numpy()
            
            # ==================== GROUND TRUTH PREPARATION ====================
            # Prepare ground truth image for visualization
            gt_image_0 = gt_image[0].detach().cpu().numpy()
            gt_image_0 = (gt_image_0 * 255).astype(np.uint8)  # Convert to uint8 for display
            # ==================== RGB VISUALIZATION AND LOGGING ====================
            if optim_dict["vertices_rgb"]:
                # ==================== RENDER LOSS VISUALIZATION ====================
                # Render loss visualization for debugging
                render_loss = render_loss[0].detach().cpu().numpy()
                vis_render_loss = loss2color(render_loss)
                logger.log_image("vis_loss", vis_render_loss, epoch)
                
                # ==================== BEV RGB VISUALIZATION ====================
                # BEV RGB visualization from top-down view
                bev_rgb = bev_features[0, :, :, :3].detach().cpu().numpy()
                bev_rgb = np.clip(bev_rgb, 0, 1)  # Clamp values for wandb compatibility
                bev_rgb = bev_rgb[::-1, ::-1, :]  # Flip for proper orientation
                
                # ==================== RENDERED IMAGE VISUALIZATION ====================
                # Rendered image from camera view
                render_image = np.clip(images[0].detach().cpu().numpy(), 0, 1)
                
                # ==================== IMAGE LOGGING ====================
                # Log images to wandb for monitoring
                logger.log_image("bev_rgb", bev_rgb, epoch)
                logger.log_image("render_image", render_image, epoch)
                logger.log_image("gt_image", gt_image_0, epoch)
            # ==================== SEMANTIC SEGMENTATION VISUALIZATION ====================
            if optim_dict["vertices_label"]:
                # ==================== BEV SEGMENTATION VISUALIZATION ====================
                # BEV segmentation from top-down view
                bev_seg = np.argmax(bev_seg, axis=-1)  # Convert probabilities to class indices
                bev_seg = render_semantic(bev_seg, dataset.filted_color_map)  # Convert to RGB
                bev_seg = bev_seg[::-1, ::-1, :]  # Flip for proper orientation
                
                # ==================== RENDERED SEGMENTATION VISUALIZATION ====================
                # Rendered segmentation from camera view
                render_seg = images_seg[0].detach().cpu().numpy()
                render_seg = np.argmax(render_seg, axis=-1)  # Convert probabilities to class indices
                render_seg = render_semantic(render_seg, dataset.filted_color_map)  # Convert to RGB
                
                # ==================== GROUND TRUTH SEGMENTATION VISUALIZATION ====================
                # Ground truth segmentation for comparison
                render_gt_seg = render_semantic(gt_seg[0].detach().cpu().numpy(), dataset.filted_color_map)
                
                # ==================== MASK VISUALIZATION ====================
                # Render mask showing valid regions
                render_mask = (mask[0].detach().cpu().numpy() * 255).astype(np.uint8)
                
                # ==================== BLENDED VISUALIZATION ====================
                # Blend ground truth and rendered segmentation for comparison
                blend_image = cv2.addWeighted(gt_image_0, 0.5, render_seg, 0.5, 0)
                
                # ==================== SEGMENTATION LOGGING ====================
                # Log segmentation visualizations to wandb
                logger.log_image("render_mask", render_mask, epoch)
                logger.log_image("bev_seg", bev_seg, epoch)
                logger.log_image("render_seg", render_seg, epoch)
                logger.log_image("render_gt_seg", render_gt_seg, epoch)
                logger.log_image("blend_image", blend_image, epoch)

            # ==================== DEPTH VISUALIZATION AND LOGGING ====================
            if optim_dict["vertices_z"]:
                # ==================== BEV DEPTH VISUALIZATION ====================
                # BEV depth visualization from top-down view
                vis_bev_depth = depth2color(bev_depth[0, :, :, 0].detach().cpu().numpy(), min=0.8, max=1.2)
                vis_bev_depth = vis_bev_depth[::-1, ::-1, :]  # Flip for proper orientation
                
                # ==================== RENDERED DEPTH VISUALIZATION ====================
                # Rendered depth visualization from camera view
                vis_render_depth = depth2color(depth[0, :, :, 0].detach().cpu().numpy(), min=0, max=100, rescale=True)
                
                # ==================== DEPTH LOGGING ====================
                # Log depth visualizations to wandb
                logger.log_image("vis_bev_depth", vis_bev_depth, epoch)
                logger.log_image("vis_render_depth", vis_render_depth, epoch)
                
                # ==================== GROUND TRUTH DEPTH VISUALIZATION ====================
                # Ground truth depth visualization (if available)
                if configs["dataset"] in supervise_depth_list:
                    # ==================== GT DEPTH PROCESSING ====================
                    kernel = np.ones((10, 10), np.uint8)
                    vis_gt_depth = depth2color(gt_depth[0, :, :, 0].detach().cpu().numpy(), min=0, max=100, rescale=True)
                    vis_gt_depth = cv2.dilate(vis_gt_depth, kernel, iterations=1)  # Dilate for better visibility
                    logger.log_image("vis_gt_depth", vis_gt_depth, epoch)

                    # ==================== DEPTH LOSS VISUALIZATION ====================
                    # Depth loss visualization for debugging
                    vis_depth_loss = depth2color(depth_loss[0].detach().cpu().numpy(), min=0, max=20, rescale=False)
                    kernel = np.ones((10, 10), np.uint8)
                    vis_depth_loss = cv2.dilate(vis_depth_loss, kernel, iterations=1)
                    depth_loss_mask = vis_depth_loss.sum(axis=-1) > 0
                    rgb = (gt_image[0].detach().cpu().numpy() * 255).astype(np.uint8)
                    rgb[depth_loss_mask] = vis_depth_loss[depth_loss_mask]  # Overlay loss on RGB
                    logger.log_image("vis_depth_loss", rgb, epoch)

            # ==================== PROGRESS BAR UPDATE ====================
            # Update progress bar description with current loss
            description = "Epoch: {}, loss = {:.2}".format(
                epoch,
                loss_dict["total_loss"])
            loop.set_description(description)

    # ==================== MODEL SAVING ====================
    # Save 3D mesh as .obj files for visualization and analysis
    save_cut_mesh(mesh[0], join(logger.dir, f"bev_mesh_epoch_{epoch}.obj"))
    save_cut_label_mesh(mesh[0], join(logger.dir, f"bev_label_mesh_epoch_{epoch}.obj"), dataset.filted_color_map)

    # ==================== MODEL CHECKPOINT SAVING ====================
    # Save trained models for future use
    grid.eval()  # Set to evaluation mode
    poses.eval()  # Set to evaluation mode
    torch.save(grid, join(logger.dir, "grid_baseline.pt"))
    torch.save(poses, join(logger.dir, "pose_baseline.pt"))
    print(f"Saved model to {logger.dir}")

    # ==================== EVALUATION ====================
    # Run evaluation if specified in config
    if configs["eval"]:
        eval(grid, poses, dataset, renderer, configs, device)


if __name__ == "__main__":
    """
    Main entry point for training script.
    
    This is the entry point when the script is run directly.
    It loads the configuration and starts the training process.
    """
    # Load configuration from command line arguments and YAML file
    configs = get_configs()
    
    # Start the training process
    train(configs)
