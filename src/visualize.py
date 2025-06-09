import matplotlib.pyplot as plt
import trimesh
import numpy as np

def visualize_graycode(horizontal_mask, vertical_mask, object_mask, combined_mask,
                       horizontal_code, vertical_code, combined_code, masked_code):
    """
    Visualize the gray code and masks for horizontal and vertical scans.

    Parameters
    ----------
    horizontal_mask : 2D numpy.array (dtype=bool)
        Mask for horizontal scan.

    vertical_mask : 2D numpy.array (dtype=bool)
        Mask for vertical scan.

    object_mask : 2D numpy.array (dtype=bool)
        Mask for the object.

    combined_mask : 2D numpy.array (dtype=bool)
        Combined mask of horizontal, vertical, and object masks.

    horizontal_code : 2D numpy.array (dtype=float)
        Gray code values for horizontal scan.

    vertical_code : 2D numpy.array (dtype=float)
        Gray code values for vertical scan.

    combined_code : 2D numpy.array (dtype=float)
        Combined gray code values from horizontal and vertical scans.

    masked_code : 2D numpy.array (dtype=float)
        Combined gray code values masked by the combined mask.

    """

    fig, ax = plt.subplots(2, 4, figsize=(15, 10))
    ax[0, 0].imshow(horizontal_mask)
    ax[0, 0].set_title('Horizontal Mask')
    ax[0, 1].imshow(vertical_mask)
    ax[0, 1].set_title('Vertical Mask')
    ax[0, 2].imshow(object_mask)
    ax[0, 2].set_title('Object Mask')
    ax[0, 3].imshow(combined_mask)
    ax[0, 3].set_title('Combined Mask')
    ax[1, 0].imshow(horizontal_code)
    ax[1, 0].set_title('Horizontal Code')
    ax[1, 1].imshow(vertical_code)
    ax[1, 1].set_title('Vertical Code')
    ax[1, 2].imshow(combined_code)
    ax[1, 2].set_title('Combined Code')
    ax[1, 3].imshow(masked_code)
    ax[1, 3].set_title('Masked Combined Code')

    for axis in ax.flat:
        axis.axis('off')
    plt.tight_layout()
    plt.show()

def visualize_points(pts3, rgb):
    """
    Visualize 3D points in XY, XZ, ZY, and 3D views using matplotlib.

    Parameters
    ----------
    pts3 : numpy.array (shape=(3,N))
        3D points to visualize, where N is the number of points.

    rgb : numpy.array (shape=(N,3))
        RGB colors for each point, where N is the number of points.
        Each row should contain the RGB values for a point.

    """

    fig = plt.figure()
    fig.set_size_inches(8, 8)
    ax = fig.add_subplot(221)
    ax.set_title('XY View')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect('equal')
    ax.scatter(pts3[0,:],pts3[1,:],s=1, c=rgb.T)
    ax = fig.add_subplot(222)
    ax.set_title('XZ View')
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_aspect('equal')
    ax.scatter(pts3[0,:],pts3[2,:],s=1, c=rgb.T)
    ax = fig.add_subplot(223)
    ax.set_title('ZY View')
    ax.set_xlabel('Z')
    ax.set_ylabel('Y')
    ax.set_aspect('equal')
    ax.scatter(pts3[2,:],pts3[1,:],s=1, c=rgb.T)
    ax = fig.add_subplot(224, projection='3d')
    ax.set_title('3D View')
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')
    ax.scatter(pts3[0,:],pts3[2,:],pts3[1,:],s=1, c=rgb.T)
    ax.set_box_aspect([1,1,1])
    xlim = [np.min(pts3[0,:]), np.max(pts3[0,:])]
    ylim = [np.min(pts3[2,:]), np.max(pts3[2,:])]
    zlim = [np.min(pts3[1,:]), np.max(pts3[1,:])]
    max_range = max(
        xlim[1] - xlim[0],
        ylim[1] - ylim[0],
        zlim[1] - zlim[0]
    )
    mid_x = np.mean(xlim)
    mid_y = np.mean(ylim)
    mid_z = np.mean(zlim)
    ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
    ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
    ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
    plt.tight_layout()
    plt.show()

def visualize_mesh(pts3, rgb, faces):
    """
    Visualize a 3D mesh using trimesh.
    
    Parameters
    ----------
    pts3 : numpy.array (shape=(3,N))
        3D points of the mesh vertices.
    
    rgb : numpy.array (shape=(3,N))
        RGB colors for each vertex.
    
    faces : numpy.array (shape=(M,3))
        Indices of the vertices that form each triangular face.

    """

    mesh = trimesh.Trimesh(vertices=pts3.T, faces=faces, vertex_colors=rgb.T)
    mesh.show()