from scipy.spatial import Delaunay
import numpy as np

def generate_mesh(pts3, pts2L, rgb, box_limits, tri_thresh):
    """
    Generate a mesh from 3D points using Delaunay triangulation and apply pruning based on bounding box limits and triangle edge length.

    Parameters
    ----------
    pts3 : 2D numpy.array (dtype=float)
        Coordinates of N points stored in a array of shape (3,N)

    pts2L : 2D numpy.array (dtype=float)
        Coordinates of N points stored in a array of shape (2,N) seen from the left camera.

    rgb : 2D numpy.array (dtype=float)
        RGB color values for each point, stored in a array of shape (3,N).

    box_limits : array-like
        Limits along the x, y, and z axes of a box containing the object.
        Should be an array of shape (6,) with format [xmin, xmax, ymin, ymax, zmin, zmax].

    tri_thresh : float
        Longest allowed edge length in the mesh. Triangles with edges longer than this value will be removed.

    Returns
    -------
    pts_pruned : 2D numpy.array (dtype=float)
        Pruned 3D points that lie within the specified bounding box.

    indices : 2D numpy.array (dtype=int)
        Indices of the triangles in the mesh after pruning. Each row corresponds to a triangle defined by three vertex indices.

    """
    #
    # bounding box pruning
    #
    mask =  (pts3[0,:] > box_limits[0]) & (pts3[0,:] < box_limits[1]) & \
            (pts3[1,:] > box_limits[2]) & (pts3[1,:] < box_limits[3]) & \
            (pts3[2,:] > box_limits[4]) & (pts3[2,:] < box_limits[5])
    pts3_pruned = pts3[:, mask]
    pts2L_pruned = pts2L[:, mask]
    rgb_pruned = rgb[:, mask]

    #
    # triangulate the 2D points to get the surface mesh
    #

    tri = Delaunay(pts2L_pruned.T)

    #
    # triangle pruning
    #
    def valid_triangle(slice):
        p1 = pts3_pruned[:,slice[0]]
        p2 = pts3_pruned[:,slice[1]]
        p3 = pts3_pruned[:,slice[2]]
        e1 = np.linalg.norm(p1-p2)
        e2 = np.linalg.norm(p2-p3)
        e3 = np.linalg.norm(p3-p1)
        return e1 <= tri_thresh and e2 <= tri_thresh and e3 <= tri_thresh

    tri_mask = np.apply_along_axis(valid_triangle, 1, tri.simplices)
    indices = tri.simplices[tri_mask]

    return pts3_pruned, rgb_pruned, indices