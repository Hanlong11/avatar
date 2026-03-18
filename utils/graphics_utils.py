

import math
import numpy as np
import torch
import trimesh


def rand_point_on_mesh(vert, face, pts_num=7000, init_factor=5):
    mesh = trimesh.Trimesh(vertices=vert, faces=face, process=False)
    try:
        import open3d as o3d

        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(vert)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(face)
        point_cloud = o3d_mesh.sample_points_poisson_disk(number_of_points=pts_num, init_factor=init_factor)
        return np.asarray(point_cloud.points)
    except Exception:
        # Fallback for headless clusters where Open3D fails to load libGL.
        points, _ = trimesh.sample.sample_surface_even(mesh, count=pts_num)
        return points.astype(np.float32)


def polar_decomposition_newton_schulz(A, iteration=4):
    n, m = A.shape[-2:]
    if n != m:
        raise ValueError("Matrix must be square")

    Q = A.clone()
    for _ in range(iteration):
        Q_t = Q.transpose(-2, -1)
        Q = 0.5 * (Q + torch.linalg.inv_ex(Q_t)[0])
    return Q
