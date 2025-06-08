import matplotlib.pyplot as plt
import trimesh
import numpy as np

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