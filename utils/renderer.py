# Standard library imports
import torch
from torch import nn

# PyTorch3D imports for 3D rendering
from pytorch3d.renderer import (
    RasterizationSettings,
    MeshRasterizer,
)


def hard_channel_blend(
    colors: torch.Tensor, fragments,
) -> torch.Tensor:
    """
    Perform hard blending of top K faces to return an RGBA image.
    
    This function implements a simple blending strategy where only the closest face
    to the camera is used for each pixel, with background pixels set to white.
    
    Args:
        colors: (N, H, W, K, C) RGB color for each of the top K faces per pixel.
        fragments: The outputs of rasterization containing:
            - pix_to_face: LongTensor of shape (N, H, W, K) specifying the indices
              of the faces (in the packed representation) which overlap each pixel
              in the image. This is used to determine the output shape.
    Returns:
        RGBA pixel_channels: (N, H, W, C+1) where the last channel is alpha
    """
    # Extract dimensions from fragments
    N, H, W, K = fragments.pix_to_face.shape
    device = fragments.pix_to_face.device

    # Create mask for background pixels (negative face indices indicate background)
    is_background = fragments.pix_to_face[..., 0] < 0  # (N, H, W)

    # Set background color to white (all channels = 1)
    background_color = torch.ones(colors.shape[-1], dtype=colors.dtype, device=colors.device)

    # Count number of background pixels for efficient tensor operations
    num_background_pixels = is_background.sum()

    # Set background color for background pixels, keep foreground colors for others
    pixel_colors = colors[..., 0, :].masked_scatter(
        is_background[..., None],
        background_color[None, :].expand(num_background_pixels, -1),
    )  # (N, H, W, C)

    # Create alpha channel: 1 for foreground pixels, 0 for background
    alpha = (~is_background).type_as(pixel_colors)[..., None]

    # Concatenate RGB channels with alpha channel
    return torch.cat([pixel_colors, alpha], dim=-1)  # (N, H, W, C+1)


class SimpleShader(nn.Module):
    """
    Simple shader that performs texture sampling and hard blending.
    
    This shader samples textures from the mesh and applies hard blending
    to generate the final rendered image with alpha channel.
    """
    def __init__(self):
        super().__init__()

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        """
        Forward pass of the shader.
        
        Args:
            fragments: Rasterization fragments containing pixel-to-face mappings
            meshes: 3D meshes to be rendered
            **kwargs: Additional keyword arguments (unused)
            
        Returns:
            torch.Tensor: Rendered images with alpha channel (N, H, W, C+1)
        """
        # Sample textures from mesh faces
        texels = meshes.sample_textures(fragments)
        
        # Apply hard blending to generate final image
        images = hard_channel_blend(texels, fragments)
        return images


class MeshRendererWithDepth(nn.Module):
    """
    Mesh renderer that outputs both rendered images and depth information.
    
    This renderer combines rasterization and shading to produce both
    the final rendered image and the corresponding depth buffer.
    """
    def __init__(self, rasterizer, shader):
        """
        Initialize the renderer with rasterizer and shader components.
        
        Args:
            rasterizer: MeshRasterizer instance for 3D-to-2D projection
            shader: Shader instance for texture sampling and blending
        """
        super().__init__()
        self.rasterizer = rasterizer
        self.shader = shader

    def forward(self, meshes_world, **kwargs) -> torch.Tensor:
        """
        Render meshes and return both images and depth.
        
        Args:
            meshes_world: 3D meshes in world coordinates
            **kwargs: Additional arguments passed to rasterizer and shader
n            
        Returns:
            tuple: (rendered_images, depth_buffer)
                - rendered_images: (N, H, W, C+1) RGBA images
                - depth_buffer: (N, H, W, K) depth values for each pixel
        """
        # Rasterize meshes to get fragments
        fragments = self.rasterizer(meshes_world, **kwargs)
        
        # Apply shading to generate final images
        images = self.shader(fragments, meshes_world, **kwargs)
        
        # Return both images and depth buffer
        return images, fragments.zbuf


class Renderer(nn.Module):
    """
    Main renderer class that dynamically configures rendering based on camera parameters.
    
    This renderer automatically sets up rasterization settings based on the
    camera's image size and creates the appropriate mesh renderer.
    """
    def __init__(self):
        super().__init__()
        self.raster_settings = None  # Will be set dynamically based on camera

    def set_rasterization(self, cameras):
        """
        Configure rasterization settings based on camera image size.
        
        Args:
            cameras: PyTorch3D camera objects containing image size information
        """
        # Extract image size from camera parameters
        image_size = tuple(cameras.image_size[0].detach().cpu().numpy())
        
        # Configure rasterization settings
        self.raster_settings = RasterizationSettings(
            image_size=(int(image_size[0]), int(image_size[1])),  # (height, width)
            blur_radius=0.0,      # No anti-aliasing blur
            faces_per_pixel=1,    # Only render the closest face per pixel
        )

    def forward(self, input):
        """
        Render meshes using the provided cameras.
        
        Args:
            input: Dictionary containing:
                - "mesh": 3D meshes to render
                - "cameras": Camera objects for rendering
                
        Returns:
            tuple: (rendered_images, depth_buffer)
                - rendered_images: (N, H, W, C+1) RGBA images
                - depth_buffer: (N, H, W, K) depth values
        """
        mesh = input["mesh"]
        cameras = input["cameras"]
        
        # Set up rasterization settings if not already configured
        if self.raster_settings is None:
            self.set_rasterization(cameras)

        # Create mesh renderer with depth output
        mesh_renderer = MeshRendererWithDepth(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=self.raster_settings
            ),
            shader=SimpleShader()
        )
        
        # Render and return both images and depth
        images, depth = mesh_renderer(mesh)
        return images, depth


class RendererBev(nn.Module):
    """
    Specialized renderer for Bird's Eye View (BEV) rendering.
    
    This renderer uses a fixed image size since FOV cameras typically
    do not have explicit image_size parameters. It's optimized for
    top-down or aerial view rendering scenarios.
    """
    def __init__(self):
        super().__init__()
        self.raster_settings = None
        # Fixed image size for BEV rendering (height, width)
        self.image_size = tuple((640, 1024))  # FOV cameras do not have image_size

    def set_rasterization(self):
        """
        Configure rasterization settings with fixed image size for BEV rendering.
        """
        image_size = self.image_size
        self.raster_settings = RasterizationSettings(
            image_size=(int(image_size[0]), int(image_size[1])),  # (640, 1024)
            blur_radius=0.0,      # No anti-aliasing blur
            faces_per_pixel=1,    # Only render the closest face per pixel
        )

    def forward(self, input):
        """
        Render meshes for Bird's Eye View using fixed image size.
        
        Args:
            input: Dictionary containing:
                - "mesh": 3D meshes to render
                - "cameras": Camera objects for BEV rendering
                
        Returns:
            tuple: (rendered_images, depth_buffer)
                - rendered_images: (N, H, W, C+1) RGBA images with fixed size
                - depth_buffer: (N, H, W, K) depth values
        """
        mesh = input["mesh"]
        cameras = input["cameras"]
        
        # Set up rasterization settings if not already configured
        if self.raster_settings is None:
            self.set_rasterization()

        # Create mesh renderer with depth output
        mesh_renderer = MeshRendererWithDepth(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=self.raster_settings
            ),
            shader=SimpleShader()
        )
        
        # Render and return both images and depth
        images, depth = mesh_renderer(mesh)
        return images, depth
