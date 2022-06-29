import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import json
import torch
import argparse
import nvdiffrast.torch as dr
from nvdiffrec.render import obj
from nvdiffrec.render import light
from nvdiffrec.geometry.dlmesh import DLMesh
from nvdiffrec.support.uvmap import xatlas_uvmap
from nvdiffrec.geometry.dmtet import DMTetGeometry
from nvdiffrec.support.training import optimize_mesh
from nvdiffrec.dataset.dataset_nerf import DatasetNERF
from nvdiffrec.support.validation_and_testing import validate
from nvdiffrec.support.material_utility import initial_guess_material

RADIUS = 3.0


def set_flags():
    parser = argparse.ArgumentParser(description="nvdiffrec")
    parser.add_argument("--config", type=str, default=None, help="Config file")
    parser.add_argument("-i", "--iter", type=int, default=5000)
    parser.add_argument("-b", "--batch", type=int, default=1)
    parser.add_argument("-s", "--spp", type=int, default=1)
    parser.add_argument("-l", "--layers", type=int, default=1)
    parser.add_argument("-r", "--train-res", nargs=2, type=int, default=[512, 512])
    parser.add_argument("-dr", "--display-res", type=int, default=None)
    parser.add_argument("-tr", "--texture-res", nargs=2, type=int, default=[1024, 1024])
    parser.add_argument("-di", "--display-interval", type=int, default=0)
    parser.add_argument("-si", "--save-interval", type=int, default=1000)
    parser.add_argument("-lr", "--learning-rate", type=float, default=0.01)
    parser.add_argument("-mr", "--min-roughness", type=float, default=0.08)
    parser.add_argument("-mip", "--custom-mip", action="store_true", default=False)
    parser.add_argument("-rt", "--random-textures", action="store_true", default=False)
    parser.add_argument(
        "-bg",
        "--background",
        default="checker",
        choices=["black", "white", "checker", "reference"],
    )
    parser.add_argument(
        "--loss", default="logl1", choices=["logl1", "logl2", "mse", "smape", "relmse"]
    )
    parser.add_argument("-o", "--out-dir", type=str, default=None)
    parser.add_argument("-rm", "--ref_mesh", type=str)
    parser.add_argument("-bm", "--base-mesh", type=str, default=None)
    parser.add_argument("--validate", type=bool, default=True)

    FLAGS = parser.parse_args()

    FLAGS.mtl_override = None  # Override material of model
    FLAGS.dmtet_grid = 256  # Resolution of initial tet grid.
    FLAGS.mesh_scale = 2.5  # Scale of tet grid box. Adjust to cover the model
    FLAGS.env_scale = 1.0  # Env map intensity multiplier
    FLAGS.envmap = None  # HDR environment probe
    FLAGS.display = (
        None  # Conf validation window/display. E.g. [{"relight" : <path to envlight>}]
    )
    FLAGS.camera_space_light = False  # Fixed light in camera space.
    FLAGS.lock_light = False  # Disable light optimization in the second pass
    FLAGS.lock_pos = False  # Disable vertex position optimization in the second pass
    FLAGS.sdf_regularizer = 0.2  # Weight for sdf regularizer (see paper for details)
    FLAGS.laplace = "relative"  # Mesh Laplacian ["absolute", "relative"]
    FLAGS.laplace_scale = (
        7500  # Weight for sdf regularizer. Default is relative with large weight
    )
    FLAGS.pre_load = True  # Pre-load entire dataset into memory for faster training
    FLAGS.kd_min = [0.0, 0.0, 0.0, 0.0]  # Limits for kd
    FLAGS.kd_max = [1.0, 1.0, 1.0, 1.0]
    FLAGS.ks_min = [0.0, 0.08, 0.0]  # Limits for ks
    FLAGS.ks_max = [1.0, 1.0, 1.0]
    FLAGS.nrm_min = [-1.0, -1.0, 0.0]  # Limits for normal map
    FLAGS.nrm_max = [1.0, 1.0, 1.0]
    FLAGS.cam_near_far = [0.1, 1000.0]
    FLAGS.learn_light = True

    FLAGS.local_rank = 0
    FLAGS.multi_gpu = "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1
    if FLAGS.multi_gpu:
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = "localhost"
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "23456"

        FLAGS.local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(FLAGS.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")

    if FLAGS.config is not None:
        data = json.load(open(FLAGS.config, "r"))
        for key in data:
            FLAGS.__dict__[key] = data[key]

    if FLAGS.display_res is None:
        FLAGS.display_res = FLAGS.train_res

    if FLAGS.local_rank == 0:
        print("Config / Flags:")
        print("---------")
        for key in FLAGS.__dict__.keys():
            print(key, FLAGS.__dict__[key])
        print("---------")

    os.makedirs(FLAGS.out_dir, exist_ok=True)
    return FLAGS


def base_run():
    FLAGS = set_flags()

    glctx = dr.RasterizeGLContext()

    # ===============================================================================
    #  Create data pipeline
    # ===============================================================================
    dataset_train = DatasetNERF(
        os.path.join(FLAGS.ref_mesh, "nerf_transforms.json"),
        FLAGS,
        examples=(FLAGS.iter + 1) * FLAGS.batch,
    )
    dataset_validate = DatasetNERF(
        os.path.join(FLAGS.ref_mesh, "nerf_transforms.json"), FLAGS
    )

    # ===============================================================================
    #  Create env light with trainable parameters
    # ===============================================================================

    if FLAGS.learn_light:
        print("we are learning light")
        lgt = light.create_trainable_env_rnd(512, scale=0.0, bias=0.5)
        print(lgt.type)
    else:
        lgt = light.load_env(FLAGS.envmap, scale=FLAGS.env_scale)

    # Setup geometry for optimization
    geometry = DMTetGeometry(FLAGS.dmtet_grid, FLAGS.mesh_scale, FLAGS)

    # Setup textures, make initial guess from reference if possible
    mat = initial_guess_material(geometry, True, FLAGS)

    # Run optimization
    geometry, mat = optimize_mesh(
        glctx,
        geometry,
        mat,
        lgt,
        dataset_train,
        dataset_validate,
        FLAGS,
        pass_idx=0,
        pass_name="dmtet_pass1",
        optimize_light=FLAGS.learn_light,
    )

    if FLAGS.local_rank == 0 and FLAGS.validate:
        validate(
            glctx,
            geometry,
            mat,
            lgt,
            dataset_validate,
            FLAGS.out_dir,
            FLAGS,
        )

    # save mesh without textures
    eval_mesh = geometry.getMesh(mat)
    os.makedirs(os.path.join(FLAGS.out_dir, "eval_mesh"), exist_ok=True)
    obj.write_obj(os.path.join(FLAGS.out_dir, "eval_mesh/"), eval_mesh)

    return geometry, mat


def refinement_run():
    FLAGS = set_flags()

    glctx = dr.RasterizeGLContext()

    # Trying to create textured mesh from result
    base_mesh = xatlas_uvmap(glctx, geometry, mat, FLAGS)

    # Free temporaries / cached memory
    torch.cuda.empty_cache()
    mat["kd_ks_normal"].cleanup()
    del mat["kd_ks_normal"]

    lgt = lgt.clone()
    geometry = DLMesh(base_mesh, FLAGS)

    if FLAGS.local_rank == 0:
        # Dump mesh for debugging.
        os.makedirs(os.path.join(FLAGS.out_dir, "dmtet_mesh"), exist_ok=True)
        obj.write_obj(os.path.join(FLAGS.out_dir, "dmtet_mesh/"), base_mesh)
        light.save_env_map(os.path.join(FLAGS.out_dir, "dmtet_mesh/probe.hdr"), lgt)

        # ==========================================================================
        #  Pass 2: Train with fixed topology (mesh)
        # ==========================================================================
        geometry, mat = optimize_mesh(
            glctx,
            geometry,
            base_mesh.material,
            lgt,
            dataset_train,
            dataset_validate,
            FLAGS,
            pass_idx=1,
            pass_name="mesh_pass",
            warmup_iter=100,
            optimize_light=FLAGS.learn_light and not FLAGS.lock_light,
            optimize_geometry=not FLAGS.lock_pos,
        )

        # ==========================================================================
        #  Validate
        # ==========================================================================
        if FLAGS.validate and FLAGS.local_rank == 0:
            validate(
                glctx,
                geometry,
                mat,
                lgt,
                dataset_validate,
                os.path.join(FLAGS.out_dir, "validate"),
                FLAGS,
            )

        # ===========================================================================
        #  Dump output
        # ===========================================================================
        if FLAGS.local_rank == 0:
            final_mesh = geometry.getMesh(mat)
            os.makedirs(os.path.join(FLAGS.out_dir, "mesh"), exist_ok=True)
            obj.write_obj(os.path.join(FLAGS.out_dir, "mesh/"), final_mesh)
            light.save_env_map(os.path.join(FLAGS.out_dir, "mesh/probe.hdr"), lgt)

    # ----------------------------------------------------------------------------


if __name__ == "__main__":
    base_run()
