import torch
import xatlas
import numpy as np
from nvdiffrec.render import mesh
from nvdiffrec.render import render
from nvdiffrec.render import texture
from nvdiffrec.render import material


###############################################################################
# UV - map geometry & convert to a mesh
###############################################################################


@torch.no_grad()
def xatlas_uvmap(glctx, geometry, mat, FLAGS, eval_mesh):
    # eval_mesh = geometry.getMesh(mat)

    # Create uvs with xatlas
    v_pos = eval_mesh.v_pos.detach().cpu().numpy()
    t_pos_idx = eval_mesh.t_pos_idx.detach().cpu().numpy()
    vmapping, indices, uvs = xatlas.parametrize(v_pos, t_pos_idx)

    # Convert to tensors
    indices_int64 = indices.astype(np.uint64, casting="same_kind").view(np.int64)

    uvs = torch.tensor(uvs, dtype=torch.float32, device="cuda")
    faces = torch.tensor(indices_int64, dtype=torch.int64, device="cuda")

    new_mesh = mesh.Mesh(v_tex=uvs, t_tex_idx=faces, base=eval_mesh)

    mask, kd, ks, normal = render.render_uv(
        glctx, new_mesh, FLAGS.texture_res, eval_mesh.material["kd_ks_normal"]
    )

    if FLAGS.layers > 1:
        kd = torch.cat((kd, torch.rand_like(kd[..., 0:1])), dim=-1)

    kd_min, kd_max = torch.tensor(
        FLAGS.kd_min, dtype=torch.float32, device="cuda"
    ), torch.tensor(FLAGS.kd_max, dtype=torch.float32, device="cuda")
    ks_min, ks_max = torch.tensor(
        FLAGS.ks_min, dtype=torch.float32, device="cuda"
    ), torch.tensor(FLAGS.ks_max, dtype=torch.float32, device="cuda")
    nrm_min, nrm_max = torch.tensor(
        FLAGS.nrm_min, dtype=torch.float32, device="cuda"
    ), torch.tensor(FLAGS.nrm_max, dtype=torch.float32, device="cuda")

    new_mesh.material = material.Material(
        {
            "bsdf": mat["bsdf"],
            "kd": texture.Texture2D(kd, min_max=[kd_min, kd_max]),
            "ks": texture.Texture2D(ks, min_max=[ks_min, ks_max]),
            "normal": texture.Texture2D(normal, min_max=[nrm_min, nrm_max]),
        }
    )

    return new_mesh