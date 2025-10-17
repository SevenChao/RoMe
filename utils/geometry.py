# Standard library imports
import cv2
import torch
import numpy as np


def createFlatMesh(x_length, y_length, resolution=0.1):
    """
    Create a regular flat mesh with rectangular grid structure.
    
    This function generates a 2D grid mesh that can be used for testing or
    as a base for more complex mesh structures. The mesh is created by
    generating vertices in a regular grid pattern and connecting them
    with triangular faces.

    Args:
        x_length (float): Length along x-axis of the mesh in meters.
        y_length (float): Length along y-axis of the mesh in meters.
        resolution (float): Distance between adjacent vertices in meters.

    Returns:
        tuple: (vertices, faces, grid_shape)
            - vertices: torch.Tensor of shape (N, 3) containing vertex coordinates
            - faces: torch.Tensor of shape (M, 3) containing face indices
            - grid_shape: tuple (num_vertices_x, num_vertices_y) grid dimensions
    """
    # Calculate number of vertices in each dimension
    num_vertices_x = int(x_length / resolution) + 1
    num_vertices_y = int(y_length / resolution) + 1
    assert num_vertices_x > 0 and num_vertices_y > 0, "Mesh resolution too high."

    # Create vertex grid with x, y coordinates (z=0 for flat mesh)
    vertices = torch.zeros((num_vertices_x, num_vertices_y, 3), dtype=torch.float32)
    
    # Set x-coordinates: evenly spaced from 0 to x_length
    vertices[:, :, 0] = torch.unsqueeze(torch.linspace(0, x_length, num_vertices_x), dim=0).T
    
    # Set y-coordinates: evenly spaced from 0 to y_length
    vertices[:, :, 1] = torch.unsqueeze(torch.linspace(0, y_length, num_vertices_y), dim=0)
    
    # Flatten to (N, 3) format
    vertices = vertices.reshape(-1, 3)

    # Create triangular faces by connecting vertices in a grid pattern
    # Each grid cell is divided into 2 triangles (top-right and bottom-left)
    # Each triangle has 3 vertices
    faces = torch.zeros((num_vertices_x - 1, num_vertices_y - 1, 2, 3), dtype=torch.int64)
    
    # Create index mapping for all vertices
    all_indices = torch.arange(0, num_vertices_x * num_vertices_y, 1, dtype=torch.int64).reshape((num_vertices_x, num_vertices_y))
    
    # Define first triangle (top-right): bottom-left, top-left, top-right
    faces[:, :, 0, 0] = all_indices[:-1, :-1]  # bottom-left
    faces[:, :, 0, 1] = all_indices[:-1, 1:]   # top-left
    faces[:, :, 0, 2] = all_indices[1:, 1:]    # top-right
    
    # Define second triangle (bottom-left): bottom-left, top-right, bottom-right
    faces[:, :, 1, 0] = all_indices[:-1, :-1]  # bottom-left
    faces[:, :, 1, 1] = all_indices[1:, 1:]    # top-right
    faces[:, :, 1, 2] = all_indices[1:, :-1]   # bottom-right
    
    # Flatten to (M, 3) format
    faces = faces.reshape(-1, 3)
    return vertices, faces, (num_vertices_x, num_vertices_y)


def createHiveFlatMesh(x_length, y_length, resolution=0.1):
    """
    Create a flat mesh with hexagonal (hive) pattern for more efficient coverage.
    
    This function generates a mesh with a hexagonal grid pattern, which provides
    better coverage efficiency compared to regular rectangular grids. The hexagonal
    pattern is achieved by offsetting every other row of vertices.

    Args:
        x_length (float): Length along x-axis of the mesh in meters.
        y_length (float): Length along y-axis of the mesh in meters.
        resolution (float): Base resolution of the mesh in meters.
    
    Returns:
        tuple: (vertices, faces, grid_shape)
            - vertices: torch.Tensor of shape (N, 3) containing vertex coordinates
            - faces: torch.Tensor of shape (M, 3) containing face indices
            - grid_shape: tuple (num_vertices_x, num_vertices_y) grid dimensions
    """
    # Calculate hexagonal grid resolutions
    x_resolution = resolution
    y_resolution = x_resolution * 2 / 1.7320508075688772  # sqrt(3) for hexagonal spacing
    
    # Calculate number of vertices in each dimension
    num_vertices_x = int(x_length / x_resolution) + 1
    num_vertices_y = int(y_length / y_resolution) + 1
    assert num_vertices_x > 0 and num_vertices_y > 0, "Mesh resolution too high."
    
    # Create vertex grid
    vertices = torch.zeros((num_vertices_x, num_vertices_y, 3), dtype=torch.float32)
    
    # Set x-coordinates: evenly spaced from 0 to x_length
    vertices[:, :, 0] = torch.unsqueeze(torch.linspace(0, x_length, num_vertices_x), dim=0).T
    
    # Set y-coordinates with hexagonal offset pattern
    for i in range(num_vertices_x):
        if i % 2 == 0:
            # Even columns: standard spacing
            vertices[i, :, 1] = torch.linspace(0, y_length + y_resolution / 2, num_vertices_y)
        else:
            # Odd columns: offset by half resolution for hexagonal pattern
            vertices[i, :, 1] = torch.linspace(-y_resolution / 2, y_length, num_vertices_y)
    
    # Flatten to (N, 3) format
    vertices = vertices.reshape(-1, 3)

    # Create triangular faces for hexagonal grid
    # Each grid cell is divided into 2 triangles
    faces = torch.zeros((num_vertices_x - 1, num_vertices_y - 1, 2, 3), dtype=torch.int64)
    
    # Create index mapping for all vertices
    all_indices = torch.arange(0, num_vertices_x * num_vertices_y, 1, dtype=torch.int64).reshape((num_vertices_x, num_vertices_y))
    
    # Define basic triangular faces (same as regular grid initially)
    faces[:, :, 0, 0] = all_indices[:-1, :-1]  # bottom-left
    faces[:, :, 0, 1] = all_indices[:-1, 1:]   # top-left
    faces[:, :, 0, 2] = all_indices[1:, 1:]    # top-right
    faces[:, :, 1, 0] = all_indices[:-1, :-1]  # bottom-left
    faces[:, :, 1, 1] = all_indices[1:, 1:]    # top-right
    faces[:, :, 1, 2] = all_indices[1:, :-1]   # bottom-right

    # Adjust face connections for hexagonal pattern in odd rows
    # This creates the proper hexagonal connectivity
    faces[1::2, :, 0, 0] = faces[1::2, :, 1, 2]  # face 0 vertex 0 down
    faces[1::2, :, 1, 1] = faces[1::2, :, 0, 1]  # face 1 vertex 1 up
    
    # Flatten to (M, 3) format
    faces = faces.reshape(-1, 3)
    return vertices, faces, (num_vertices_x, num_vertices_y)


def cutHiveMeshWithPoses(vertices, faces, bev_size_pixel, x_length, y_length, poses_xy, resolution=0.1, cut_range=30):
    """
    Cut mesh to keep only regions around camera poses using morphological operations.
    
    This function creates a mask based on camera positions and uses morphological
    dilation to expand the region of interest. It then removes faces outside this
    region to create a more efficient mesh representation.

    Args:
        vertices (torch.Tensor): A tensor of shape (N, 3) containing the vertices of the mesh.
        faces (torch.Tensor): A tensor of shape (M, 3) containing the faces of the mesh.
        bev_size_pixel (tuple): The size of the BEV grid in pixels (width, height).
        x_length (float): Length along x-axis of the mesh in meters.
        y_length (float): Length along y-axis of the mesh in meters.
        poses_xy (torch.Tensor): A tensor of shape (N, 2) containing camera positions.
        resolution (float): Resolution of the mesh in meters.
        cut_range (float): Range around poses to keep in meters.
    
    Returns:
        tuple: (filtered_vertices, filtered_faces, grid_shape)
            - filtered_vertices: torch.Tensor of shape (K, 3) containing remaining vertices
            - filtered_faces: torch.Tensor of shape (L, 3) containing remaining faces
            - grid_shape: tuple (num_vertices_x, num_vertices_y) grid dimensions
    """
    import pymeshlab  # Import here to avoid dependency issues
    
    # Calculate hexagonal grid resolutions
    x_resolution = resolution
    y_resolution = x_resolution * 2 / 1.7320508075688772  # sqrt(3) for hexagonal spacing
    (num_vertices_x, num_vertices_y) = bev_size_pixel
    
    # Convert world coordinates to pixel coordinates
    # Transform from world space to grid space
    pixel_xy = np.zeros_like(poses_xy)
    pixel_xy[:, 0] = (x_length / 2 - poses_xy[:, 0]) / x_resolution
    pixel_xy[:, 1] = (y_length / 2 - poses_xy[:, 1]) / y_resolution
    
    # Remove duplicate positions and round to nearest pixel
    pixel_xy = np.unique(pixel_xy.round(), axis=0)

    # Construct binary mask for regions of interest
    mask = np.zeros((num_vertices_x - 1, num_vertices_y - 1), dtype=np.uint8)
    
    # Clamp pixel coordinates to valid range
    pixel_xy[:, 0] = np.clip(pixel_xy[:, 0], 0, num_vertices_x - 2)
    pixel_xy[:, 1] = np.clip(pixel_xy[:, 1], 0, num_vertices_y - 2)
    pixel_xy = pixel_xy.astype(np.long)
    
    # Mark camera positions in mask
    mask[pixel_xy[:, 0], pixel_xy[:, 1]] = 1
    
    # Rotate mask 180 degrees to match coordinate system
    mask = mask[::-1, ::-1]
    # cv2.imwrite('mask.png', mask.astype(np.uint8) * 255)  # Debug: save mask

    # Dilate the mask to expand regions around camera positions
    kernel_size = int(cut_range / resolution)  # Kernel size to cover cut_range meters
    kernel = np.ones((kernel_size, kernel_size), dtype=np.long)
    
    # Apply morphological dilation to expand the mask
    mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=2)
    # cv2.imwrite('mask_dilate.png', mask.astype(np.uint8) * 255)  # Debug: save dilated mask

    # Assign quality values to faces based on mask
    # Faces in masked regions get quality=1, others get quality=0
    face_quality = np.ones((num_vertices_x - 1, num_vertices_y - 1, 2, 1), dtype=np.float64)
    face_quality[mask == 0, :, 0] = 0.0  # Set quality to 0 for faces outside mask
    face_quality = face_quality.reshape(-1, 1)
    
    # Create mesh with face quality information
    source_mesh = pymeshlab.Mesh(vertex_matrix=vertices.numpy(), face_matrix=faces.numpy(), f_quality_array=face_quality)
    
    # Use pymeshlab to filter mesh
    ms = pymeshlab.MeshSet()
    ms.add_mesh(source_mesh, "source_mesh")
    m = ms.current_mesh()
    
    # Select faces with quality < 1 (i.e., faces outside the mask)
    ms.conditional_face_selection(condselect="fq < 1")  # fq = face quality
    
    # Delete selected faces and clean up unreferenced vertices
    ms.delete_selected_faces()
    ms.remove_unreferenced_vertices()
    m = ms.current_mesh()

    # Extract filtered vertices and faces from the processed mesh
    v_matrix = torch.from_numpy(m.vertex_matrix().astype(np.float32))
    f_matrix = torch.from_numpy(m.face_matrix().astype(np.int64))
    # ms.save_current_mesh("filted.ply")  # Debug: save filtered mesh

    return v_matrix, f_matrix, (num_vertices_x, num_vertices_y)


def fps_by_distance(pointcloud, min_distance, return_idx=True, allow_same_gps=False):
    """
    Subsample pointcloud using Farthest Point Sampling (FPS) algorithm.
    
    The FPS algorithm iteratively selects points that are farthest from all
    previously selected points, ensuring good coverage of the pointcloud.
    This is particularly useful for selecting representative viewpoints
    or waypoints for training.

    Args:
        pointcloud (ndarray): Input pointcloud with shape [N, 2] or [N, 3].
        min_distance (float): Minimum distance in meters between selected points.
        return_idx (bool, optional): If True, return boolean indices of selected points.
            If False, return the subsampled pointcloud. Defaults to True.
        allow_same_gps (bool, optional): If True, allow points at the same GPS location.
            Defaults to False.
    
    Returns:
        ndarray: Either boolean indices (if return_idx=True) or subsampled pointcloud
    """
    # Validate input dimensions
    assert 2 <= pointcloud.shape[1] <= 3, "Pointcloud must be 2D or 3D"
    num_points = pointcloud.shape[0]
    
    # Initialize sampling
    sample_idx = np.zeros(num_points, dtype=bool)  # Boolean mask for selected points
    start_idx = np.random.randint(0, num_points)   # Random starting point
    sample_idx[start_idx] = True
    sampled_min_distance = 1e9  # Initialize with large value

    # Iteratively select points using FPS algorithm
    while np.any(~sample_idx) and sampled_min_distance > min_distance:
        # Get currently selected points
        sampled_points = pointcloud[sample_idx]
        local_min_list = []
        
        # For each point, find minimum distance to already selected points
        for point in pointcloud:
            # Calculate distances to all selected points using L-infinity norm
            distance = np.linalg.norm(point - sampled_points, ord=np.inf, axis=1)
            local_min = np.min(distance)  # Minimum distance to any selected point
            
            # Handle special case for same GPS coordinates
            if allow_same_gps and local_min == 0:
                local_min = min_distance + 1  # Force selection if same GPS allowed
            local_min_list.append(local_min)
        
        # Convert to array and set already selected points to 0
        local_min_array = np.array(local_min_list)
        local_min_array[sample_idx] = 0
        
        # Find point with maximum minimum distance (farthest point)
        furthest_point_idx = np.argmax(local_min_array)
        sampled_min_distance = local_min_array[furthest_point_idx]
        
        # Add point to selection if it meets distance criteria
        sample_idx[furthest_point_idx] = sampled_min_distance > min_distance
    # Return results based on return_idx parameter
    if return_idx:
        return sample_idx  # Return boolean indices
    else:
        return pointcloud[sample_idx]  # Return subsampled pointcloud


if __name__ == '__main__':
    """
    Test function to demonstrate mesh creation.
    """
    # Create a hexagonal mesh for testing
    vertices, faces, bev_size_pixel = createHiveFlatMesh(1.0, 1.0)
    print(f"Created mesh with {len(vertices)} vertices and {len(faces)} faces")
    print(f"Grid shape: {bev_size_pixel}")
