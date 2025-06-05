import matplotlib.pyplot as plt
import trimesh

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