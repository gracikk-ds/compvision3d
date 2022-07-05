import torch
import numpy as np
from nvdiffrec.render import texture
from nvdiffrec.render import material
from nvdiffrec.render import mlptexture


###############################################################################
# Utility functions for material
###############################################################################


def initial_guess_material(geometry, mlp, FLAGS, init_mat=None):
    kd_min, kd_max = torch.tensor(
        FLAGS.kd_min, dtype=torch.float32, device="cuda"
    ), torch.tensor(FLAGS.kd_max, dtype=torch.float32, device="cuda")
    ks_min, ks_max = torch.tensor(
        FLAGS.ks_min, dtype=torch.float32, device="cuda"
    ), torch.tensor(FLAGS.ks_max, dtype=torch.float32, device="cuda")
    nrm_min, nrm_max = torch.tensor(
        FLAGS.nrm_min, dtype=torch.float32, device="cuda"
    ), torch.tensor(FLAGS.nrm_max, dtype=torch.float32, device="cuda")
    if mlp:
        mlp_min = torch.cat((kd_min[0:3], ks_min, nrm_min), dim=0)
        mlp_max = torch.cat((kd_max[0:3], ks_max, nrm_max), dim=0)
        mlp_map_opt = mlptexture.MLPTexture3D(
            geometry.getAABB(), channels=9, min_max=[mlp_min, mlp_max]
        )
        mat = material.Material({"kd_ks_normal": mlp_map_opt})
    else:
        # Setup Kd (albedo) and Ks (x, roughness, metalness) textures
        if FLAGS.random_textures or init_mat is None:
            num_channels = 4 if FLAGS.layers > 1 else 3
            kd_init = (
                    torch.rand(size=FLAGS.texture_res + [num_channels], device="cuda")
                    * (kd_max - kd_min)[None, None, 0:num_channels]
                    + kd_min[None, None, 0:num_channels]
            )
            kd_map_opt = texture.create_trainable(
                kd_init, FLAGS.texture_res, not FLAGS.custom_mip, [kd_min, kd_max]
            )

            ksR = np.random.uniform(size=FLAGS.texture_res + [1], low=0.0, high=0.01)
            ksG = np.random.uniform(
                size=FLAGS.texture_res + [1], low=ks_min[1].cpu(), high=ks_max[1].cpu()
            )
            ksB = np.random.uniform(
                size=FLAGS.texture_res + [1], low=ks_min[2].cpu(), high=ks_max[2].cpu()
            )

            ks_map_opt = texture.create_trainable(
                np.concatenate((ksR, ksG, ksB), axis=2),
                FLAGS.texture_res,
                not FLAGS.custom_mip,
                [ks_min, ks_max],
            )
        else:
            kd_map_opt = texture.create_trainable(
                init_mat["kd"],
                FLAGS.texture_res,
                not FLAGS.custom_mip,
                [kd_min, kd_max],
            )
            ks_map_opt = texture.create_trainable(
                init_mat["ks"],
                FLAGS.texture_res,
                not FLAGS.custom_mip,
                [ks_min, ks_max],
            )

        # Setup normal map
        if FLAGS.random_textures or init_mat is None or "normal" not in init_mat:
            normal_map_opt = texture.create_trainable(
                np.array([0, 0, 1]),
                FLAGS.texture_res,
                not FLAGS.custom_mip,
                [nrm_min, nrm_max],
            )
        else:
            normal_map_opt = texture.create_trainable(
                init_mat["normal"],
                FLAGS.texture_res,
                not FLAGS.custom_mip,
                [nrm_min, nrm_max],
            )

        mat = material.Material(
            {"kd": kd_map_opt, "ks": ks_map_opt, "normal": normal_map_opt}
        )

    if init_mat is not None:
        mat["bsdf"] = init_mat["bsdf"]
    else:
        mat["bsdf"] = "pbr"

    return mat