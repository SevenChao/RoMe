# Standard library imports
import numpy as np
import torch
from torch import nn

# Local imports
from utils.geometry import createHiveFlatMesh, cutHiveMeshWithPoses

# PyTorch3D imports for 3D mesh handling
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex


def clean_nan(grad):
    """
    Clean NaN values from gradients to prevent training instability.
    
    Args:
        grad: Input gradient tensor that may contain NaN values
        
    Returns:
        torch.Tensor: Gradient tensor with NaN values replaced by zeros
    """
    grad = torch.nan_to_num_(grad)
    return grad


class HeightMLP(nn.Module):
    """
    Multi-Layer Perceptron for predicting vertex heights (z-coordinates).
    
    This MLP takes 2D normalized coordinates as input and predicts the
    corresponding height values for 3D mesh vertices. It uses positional
    encoding to improve the network's ability to learn high-frequency details.
    """
    def __init__(self, num_encoding, num_width):
        """
        Initialize the HeightMLP.
        
        Args:
            num_encoding (int): Number of encoding levels for positional encoding
            num_width (int): Width of hidden layers in the network
        """
        super().__init__()
        self.num_encoding = num_encoding
        self.D = num_width
        # Calculate position encoding channel size: 2D coordinates * (2*levels + 1)
        self.pos_channel = 2 * (2 * self.num_encoding + 1)
        
        # First MLP block: processes encoded position
        self.height_layer_0 = nn.Sequential(
            nn.Linear(self.pos_channel, self.D), nn.ReLU(),
            nn.Linear(self.D, self.D), nn.ReLU(),
            nn.Linear(self.D, self.D), nn.ReLU(),
            nn.Linear(self.D, self.D), nn.ReLU(),
        )
        
        # Second MLP block: combines features with encoded position
        self.height_layer_1 = nn.Sequential(
            nn.Linear(self.D + self.pos_channel, self.D), nn.ReLU(),
            nn.Linear(self.D, self.D), nn.ReLU(),
            nn.Linear(self.D, self.D), nn.ReLU(),
            nn.Linear(self.D, 1),  # Output single height value
        )

    def encode_position(self, input, levels, include_input=True):
        """
        Apply positional encoding to input coordinates using sinusoidal functions.
        
        This implements the positional encoding from "Attention is All You Need" (Vaswani et al., 2017).
        It helps neural networks better represent high-frequency details by encoding
        coordinates using sine and cosine functions at different frequencies.
        
        Args:
            input: Input coordinates with shape (..., C) where C is the number of coordinate dimensions
            levels: Number of encoding levels (frequency scales)
            include_input: Whether to include the original input in the encoding
            
        Returns:
            torch.Tensor: Encoded coordinates with shape (..., C*(2L+1)) where L=levels
        """
        # Start with original input if requested
        result_list = [input] if include_input else []
        
        # Apply sinusoidal encoding at different frequency scales
        for i in range(levels):
            temp = 2.0**i * input  # Scale by powers of 2 for different frequencies
            result_list.append(torch.sin(temp))  # Sine component
            result_list.append(torch.cos(temp))  # Cosine component

        # Concatenate all encoding components
        result_list = torch.cat(result_list, dim=-1)  # (..., C*(2L+1))
        return result_list

    def forward(self, norm_xy):
        """
        Forward pass to predict vertex heights from normalized 2D coordinates.
        
        Args:
            norm_xy: Normalized 2D coordinates with shape (N, 2)
            
        Returns:
            torch.Tensor: Predicted height values with shape (N, 1)
        """
        # Encode input coordinates
        encoded_norm_xy = self.encode_position(norm_xy, levels=self.num_encoding)
        
        # Process through first MLP block
        feature_z = self.height_layer_0(encoded_norm_xy)
        
        # Combine features with encoded coordinates and predict height
        vertices_z = self.height_layer_1(torch.cat([feature_z, encoded_norm_xy], dim=-1))
        return vertices_z


class FeatureMLP(nn.Module):
    """
    Multi-Layer Perceptron for processing features with positional encoding.
    
    This MLP is similar to HeightMLP but can also process additional features
    along with positional coordinates. It's designed for more complex feature
    processing tasks.
    """
    def __init__(self, num_encoding, num_width, num_feature):
        """
        Initialize the FeatureMLP.
        
        Args:
            num_encoding (int): Number of encoding levels for positional encoding
            num_width (int): Width of hidden layers in the network
            num_feature (int): Number of additional feature channels
        """
        super().__init__()
        self.num_encoding = num_encoding
        self.num_feature = num_feature
        self.D = num_width
        self.pos_channel = 2 * (2 * self.num_encoding + 1)
        
        # First MLP block: processes encoded position + features
        self.height_layer_0 = nn.Sequential(
            nn.Linear(self.pos_channel + self.num_feature, self.D), nn.ReLU(),
            nn.Linear(self.D, self.D), nn.ReLU(),
            nn.Linear(self.D, self.D), nn.ReLU(),
            nn.Linear(self.D, self.D), nn.ReLU(),
        )
        
        # Second MLP block: combines processed features with original inputs
        self.height_layer_1 = nn.Sequential(
            nn.Linear(self.D + self.pos_channel + self.num_feature, self.D), nn.ReLU(),
            nn.Linear(self.D, self.D), nn.ReLU(),
            nn.Linear(self.D, self.D), nn.ReLU(),
            nn.Linear(self.D, 1),  # Output single value
        )

    def encode_position(self, input, levels, include_input=True):
        """
        Apply positional encoding to input coordinates (same as HeightMLP).
        
        Args:
            input: Input coordinates with shape (..., C)
            levels: Number of encoding levels
            include_input: Whether to include the original input
            
        Returns:
            torch.Tensor: Encoded coordinates with shape (..., C*(2L+1))
        """
        # Start with original input if requested
        result_list = [input] if include_input else []
        
        # Apply sinusoidal encoding at different frequency scales
        for i in range(levels):
            temp = 2.0**i * input  # Scale by powers of 2
            result_list.append(torch.sin(temp))  # Sine component
            result_list.append(torch.cos(temp))  # Cosine component

        # Concatenate all encoding components
        result_list = torch.cat(result_list, dim=-1)  # (..., C*(2L+1))
        return result_list

    def forward(self, norm_xy, feature):
        """
        Forward pass to process features with positional coordinates.
        
        Args:
            norm_xy: Normalized 2D coordinates with shape (N, 2)
            feature: Additional features with shape (N, num_feature)
            
        Returns:
            torch.Tensor: Processed output with shape (N, 1)
        """
        # Encode input coordinates
        encoded_norm_xy = self.encode_position(norm_xy, levels=self.num_encoding)
        
        # Combine encoded coordinates with additional features
        encoded_xy_feature = torch.cat([encoded_norm_xy, feature], dim=-1)
        
        # Process through first MLP block
        feature_z = self.height_layer_0(encoded_xy_feature)
        
        # Combine processed features with original inputs and predict output
        vertices_z = self.height_layer_1(torch.cat([feature_z, encoded_xy_feature], dim=-1))
        return vertices_z


class SquareFlatGridBase(nn.Module):
    """
    Base class for flat grid-based 3D mesh representation.
    
    This class creates a hexagonal flat mesh and cuts it based on camera poses
    to optimize memory usage and rendering efficiency. It provides the foundation
    for various grid-based 3D reconstruction approaches.
    """
    def __init__(self, bev_x_length, bev_y_length, pose_xy, resolution, cut_range):
        """
        Initialize the base flat grid.
        
        Args:
            bev_x_length (float): Length of the grid along x-axis in meters
            bev_y_length (float): Length of the grid along y-axis in meters
            pose_xy (torch.Tensor): Camera positions for mesh cutting
            resolution (float): Grid resolution in meters
            cut_range (float): Range around poses to keep in meters
        """
        super().__init__()
        self.bev_x_length = bev_x_length
        self.bev_y_length = bev_y_length
        self.resolution = resolution
        
        # Create hexagonal flat mesh
        vertices, faces, self.bev_size_pixel = createHiveFlatMesh(bev_x_length, bev_y_length, resolution)
        print(f"Before cutting: {vertices.shape[0]} vertices, {faces.shape[0]} faces")
        
        # Cut mesh based on camera poses to optimize memory usage
        vertices, faces, self.bev_size_pixel = cutHiveMeshWithPoses(vertices, faces, self.bev_size_pixel,
                                                                    bev_x_length, bev_y_length, pose_xy,
                                                                    resolution, cut_range)
        print(f"After cutting: {vertices.shape[0]} vertices, {faces.shape[0]} faces")
        
        # Initialize mesh components
        self.texture = None
        self.mesh = None
        
        # Normalize coordinates to [-1, 1] range for neural network processing
        norm_x = vertices[:, 0]/self.bev_x_length * 2 - 1
        norm_y = vertices[:, 1]/self.bev_y_length * 2 - 1
        norm_xy = torch.cat([norm_x[:, None], norm_y[:, None]], dim=1)
        
        # Register as buffers (non-trainable parameters)
        self.register_buffer('norm_xy', norm_xy)
        self.register_buffer('vertices', vertices)
        self.register_buffer('faces', faces)

    def init_vertices_z(self):
        """
        Initialize vertex z-coordinates (heights) to zero.
        
        This method sets up the initial height values for all vertices.
        The commented code shows alternative initialization strategies.
        """
        with torch.no_grad():
            # Initialize all heights to zero
            self.vertices_z = torch.zeros((self.norm_xy.shape[0], 1), device=self.norm_xy.device)
            
            # Alternative initialization strategies (commented out):
            # norm_y = self.norm_xy[:, 0]
            # norm_y[norm_y < 0.1] = 0
            # vertices_z = torch.pow(norm_y, 0.5) * 0.2
            # vertices_z = torch.clamp(vertices_z, 0, 1).unsqueeze(1)
            # vertices_z *= -1
            
            # Update vertices with initialized z-coordinates
            self.vertices = torch.cat((self.vertices[:, :2], self.vertices_z), dim=1)

    def init_vertices_rgb(self):
        """
        Initialize RGB color parameters for vertices.
        
        Creates trainable parameters for vertex colors, initialized to zero.
        """
        self.vertices_rgb = nn.Parameter(torch.zeros_like(self.vertices)[None])

    def freeze_vertices_z(self, z):
        """
        Set vertex z-coordinates from external data and freeze them.
        
        Args:
            z: Numpy array containing height values for vertices
        """
        with torch.no_grad():
            self.vertices_z = torch.from_numpy(z).to(self.norm_xy.device)
            self.vertices = torch.cat((self.vertices[:, :2], self.vertices_z), dim=1)

    def freeze_vertices_rgb(self, rgb):
        """
        Set vertex RGB colors from external data and freeze them.
        
        Args:
            rgb: Numpy array containing RGB values for vertices
        """
        del self.vertices_rgb
        with torch.no_grad():
            self.vertices_rgb = nn.Parameter(torch.from_numpy(rgb)[None].to(self.norm_xy.device))

class SquareFlatGridRGB(SquareFlatGridBase):
    """
    Flat grid that optimizes RGB colors for 3D scene reconstruction.
    
    This grid variant focuses on learning RGB colors for each vertex,
    suitable for photorealistic rendering without semantic information.
    """
    def __init__(self, bev_x_length, bev_y_length, pose_xy, resolution, cut_range, num_classes=None):
        """
        Initialize RGB grid.
        
        Args:
            bev_x_length, bev_y_length, pose_xy, resolution, cut_range: Base grid parameters
            num_classes: Not used in this variant (for compatibility)
        """
        super().__init__(bev_x_length, bev_y_length, pose_xy, resolution, cut_range)
        # Initialize RGB parameters for all vertices
        self.vertices_rgb = nn.Parameter(torch.zeros_like(self.vertices)[None])

    def forward(self, batch_size=1):
        """
        Forward pass to create mesh with RGB colors.
        
        Args:
            batch_size: Number of meshes to return (for batch processing)
            
        Returns:
            Meshes: PyTorch3D mesh object with RGB textures
        """
        # Constrain RGB values to [0, 1] range using tanh activation
        constrained_vertices_rgb = (torch.tanh(self.vertices_rgb) + 1)/2
        
        # Create vertex texture from RGB values
        self.texture = TexturesVertex(verts_features=constrained_vertices_rgb)
        
        # Create mesh with vertices, faces, and textures
        self.mesh = Meshes(verts=[self.vertices], faces=[self.faces], textures=self.texture)
        return self.mesh.extend(batch_size)


class SquareFlatGridLabel(SquareFlatGridBase):
    """
    Flat grid that optimizes semantic labels for 3D scene reconstruction.
    
    This grid variant focuses on learning semantic segmentation labels
    for each vertex, suitable for semantic scene understanding.
    """
    def __init__(self, bev_x_length, bev_y_length, pose_xy, resolution, num_classes=None, cut_range=30):
        """
        Initialize label grid.
        
        Args:
            bev_x_length, bev_y_length, pose_xy, resolution, cut_range: Base grid parameters
            num_classes: Number of semantic classes
        """
        super().__init__(bev_x_length, bev_y_length, pose_xy, resolution, cut_range)
        num_vertices = self.vertices.shape[0]
        # Initialize label logits for all vertices and classes
        self.vertices_label = nn.Parameter(torch.zeros((1, num_vertices, num_classes), dtype=torch.float32))

    def forward(self, batch_size=1):
        """
        Forward pass to create mesh with semantic labels.
        
        Args:
            batch_size: Number of meshes to return
            
        Returns:
            Meshes: PyTorch3D mesh object with label textures
        """
        # Convert logits to probabilities using softmax
        softmax_vertices_label = torch.softmax(self.vertices_label, dim=-1)
        
        # Create vertex texture from label probabilities
        self.texture = TexturesVertex(verts_features=softmax_vertices_label)
        
        # Create mesh with vertices, faces, and textures
        self.mesh = Meshes(verts=[self.vertices], faces=[self.faces], textures=self.texture)
        return self.mesh.extend(batch_size)


class SquareFlatGridRGBLabel(SquareFlatGridBase):
    """
    Flat grid that optimizes both RGB colors and semantic labels.
    
    This grid variant learns both photorealistic colors and semantic
    segmentation labels simultaneously, providing comprehensive scene representation.
    """
    def __init__(self, bev_x_length, bev_y_length, pose_xy, resolution, num_classes=None, cut_range=30):
        """
        Initialize RGB+Label grid.
        
        Args:
            bev_x_length, bev_y_length, pose_xy, resolution, cut_range: Base grid parameters
            num_classes: Number of semantic classes
        """
        super().__init__(bev_x_length, bev_y_length, pose_xy, resolution, cut_range)
        num_vertices = self.vertices.shape[0]
        # Initialize both RGB and label parameters
        self.vertices_rgb = nn.Parameter(torch.zeros_like(self.vertices)[None])
        self.vertices_label = nn.Parameter(torch.zeros((1, num_vertices, num_classes), dtype=torch.float32))

    def forward(self, batch_size=1):
        """
        Forward pass to create mesh with both RGB colors and semantic labels.
        
        Args:
            batch_size: Number of meshes to return
            
        Returns:
            Meshes: PyTorch3D mesh object with combined RGB+label textures
        """
        # Use RGB values directly (no tanh constraint in this variant)
        constrained_vertices_rgb = self.vertices_rgb
        
        # Alternative RGB initialization strategies (commented out):
        # constrained_vertices_rgb = (torch.tanh(self.vertices_rgb) + 1)/2
        # norm_xy = self.norm_xy.clone()
        # norm_x = norm_xy[:, 0].unsqueeze(0)
        # norm_x = torch.clamp((norm_x + 1) / 2, 0, 1)
        # constrained_vertices_rgb[:, :, 0] = torch.pow((1 - norm_x), 0.5)
        # constrained_vertices_rgb[:, :, 1] = torch.pow((1 - norm_x), 0.5)
        # constrained_vertices_rgb[:, :, 2] = torch.pow((1 - norm_x), 0.5)

        # Convert label logits to probabilities
        softmax_vertices_label = torch.softmax(self.vertices_label, dim=-1)
        
        # Concatenate RGB and label features
        features = torch.cat((constrained_vertices_rgb, softmax_vertices_label), dim=-1)
        
        # Create vertex texture from combined features
        self.texture = TexturesVertex(verts_features=features)
        
        # Create mesh with vertices, faces, and textures
        self.mesh = Meshes(verts=[self.vertices], faces=[self.faces], textures=self.texture)
        return self.mesh.extend(batch_size)


class SquareFlatGridBaseZ(nn.Module):
    def __init__(self, bev_x_length, bev_y_length, pose_xy, resolution, num_encoding=2, cut_range=30):
        super().__init__()
        self.bev_x_length = bev_x_length
        self.bev_y_length = bev_y_length
        self.resolution = resolution
        vertices, faces, self.bev_size_pixel = createHiveFlatMesh(bev_x_length, bev_y_length, resolution)
        print(f"Before cutting,  {vertices.shape[0]} vertices, {faces.shape[0]} faces")
        vertices, faces, self.bev_size_pixel = cutHiveMeshWithPoses(vertices, faces, self.bev_size_pixel,
                                                                    bev_x_length, bev_y_length, pose_xy,
                                                                    resolution, cut_range)
        print(f"After cutting,  {vertices.shape[0]} vertices, {faces.shape[0]} faces")
        self.texture = None
        self.mesh = None
        self.register_buffer('faces', faces)
        self.mlp = HeightMLP(num_encoding=num_encoding, num_width=128)
        norm_x = vertices[:, 0]/self.bev_x_length * 2 - 1
        norm_y = vertices[:, 1]/self.bev_y_length * 2 - 1
        norm_xy = torch.cat([norm_x[:, None], norm_y[:, None]], dim=1)
        self.register_buffer('norm_xy', norm_xy)
        self.register_buffer('vertices_xy', vertices[:, :2])

    def get_activation_idx(self, center_xy, radius):
        distance = np.linalg.norm(self.vertices_xy.detach().cpu().numpy() - center_xy, ord=np.inf, axis=1)
        activation_idx = list(np.where(distance <= radius)[0])
        return activation_idx

    def init_vertices_z(self):
        with torch.no_grad():
            self.vertices_z = torch.zeros((self.norm_xy.shape[0], 1), device=self.norm_xy.device)
            for i in range(0, self.norm_xy.shape[0], 10000):
                activation_idx = torch.arange(i, min(i+10000, self.norm_xy.shape[0]))
                activation_idx = activation_idx.to(self.norm_xy.device)
                activation_norm_xy = self.norm_xy[activation_idx]
                activation_vertices_z = self.mlp(activation_norm_xy)
                self.vertices_z[activation_idx] = activation_vertices_z


class SquareFlatGridRGBZ(SquareFlatGridBaseZ):
    def __init__(self, bev_x_length, bev_y_length, pose_xy, resolution, num_classes=None, num_encoding=2, cut_range=30):
        super().__init__(bev_x_length, bev_y_length, pose_xy, resolution, num_encoding, cut_range)
        num_vertices = self.vertices_xy.shape[0]
        self.vertices_rgb = nn.Parameter(torch.zeros(num_vertices, 3)[None])

    def forward(self, activated_idx=None, batch_size=1):
        constrained_vertices_rgb = (torch.tanh(self.vertices_rgb) + 1)/2
        if activated_idx is None:
            vertices_z = self.mlp(self.norm_xy)
        else:
            activtated_norm_xy = self.norm_xy[activated_idx]
            activated_vertices_z = self.mlp(activtated_norm_xy)
            if activated_vertices_z.requires_grad:
                activated_vertices_z.register_hook(clean_nan)
            with torch.no_grad():
                self.vertices_z[activated_idx] = activated_vertices_z
                vertices_z = self.vertices_z.detach()
            vertices_z[activated_idx] = activated_vertices_z
        vertices = torch.cat((self.vertices_xy, vertices_z), dim=1)
        self.texture = TexturesVertex(verts_features=constrained_vertices_rgb)
        self.mesh = Meshes(verts=[vertices], faces=[self.faces], textures=self.texture)
        return self.mesh.extend(batch_size)


class SquareFlatGridLabelZ(SquareFlatGridBaseZ):
    def __init__(self, bev_x_length, bev_y_length, pose_xy, resolution, num_classes, num_encoding=2, cut_range=30):
        super().__init__(bev_x_length, bev_y_length, pose_xy, resolution, num_encoding, cut_range)
        num_vertices = self.vertices_xy.shape[0]
        self.vertices_label = nn.Parameter(torch.zeros((1, num_vertices, num_classes), dtype=torch.float32))

    def forward(self, activated_idx=None, batch_size=1):
        softmax_vertices_label = torch.softmax(self.vertices_label, dim=-1)
        if activated_idx is None:
            vertices_z = self.mlp(self.norm_xy)
        else:
            activtated_norm_xy = self.norm_xy[activated_idx]
            activated_vertices_z = self.mlp(activtated_norm_xy)
            if activated_vertices_z.requires_grad:
                activated_vertices_z.register_hook(clean_nan)
            with torch.no_grad():
                self.vertices_z[activated_idx] = activated_vertices_z
                vertices_z = self.vertices_z.detach()
            vertices_z[activated_idx] = activated_vertices_z
        vertices = torch.cat((self.vertices_xy, vertices_z), dim=1)
        self.texture = TexturesVertex(verts_features=softmax_vertices_label)
        self.mesh = Meshes(verts=[vertices], faces=[self.faces], textures=self.texture)
        return self.mesh.extend(batch_size)


class SquareFlatGridRGBLabelZ(SquareFlatGridBaseZ):
    def __init__(self, bev_x_length, bev_y_length, pose_xy, resolution, num_classes, num_encoding=2, cut_range=30):
        super().__init__(bev_x_length, bev_y_length, pose_xy, resolution, num_encoding, cut_range)
        num_vertices = self.vertices_xy.shape[0]
        self.vertices_rgb = nn.Parameter(torch.zeros(num_vertices, 3)[None])
        self.vertices_label = nn.Parameter(torch.zeros((1, num_vertices, num_classes), dtype=torch.float32))

    def forward(self, activated_idx=None, batch_size=1):
        constrained_vertices_rgb = (torch.tanh(self.vertices_rgb) + 1)/2
        softmax_vertices_label = torch.softmax(self.vertices_label, dim=-1)
        features = torch.cat((constrained_vertices_rgb, softmax_vertices_label), dim=-1)
        if activated_idx is None:
            vertices_z = self.vertices_z
        else:
            activtated_norm_xy = self.norm_xy[activated_idx]
            activated_vertices_z = self.mlp(activtated_norm_xy)
            if activated_vertices_z.requires_grad:
                activated_vertices_z.register_hook(clean_nan)
            with torch.no_grad():
                self.vertices_z[activated_idx] = activated_vertices_z
                vertices_z = self.vertices_z.detach()
            vertices_z[activated_idx] = activated_vertices_z
        vertices = torch.cat((self.vertices_xy, vertices_z), dim=1)
        self.texture = TexturesVertex(verts_features=features)
        self.mesh = Meshes(verts=[vertices], faces=[self.faces], textures=self.texture)
        return self.mesh.extend(batch_size)
