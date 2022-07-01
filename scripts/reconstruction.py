import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import yaml
import torch
import click
import torch
import pickle
from pathlib import Path
import nvdiffrast.torch as dr
from argparse import Namespace
from nvdiffrec.render import obj
from nvdiffrec.render import light
from nvdiffrec.geometry.dlmesh import DLMesh
from nvdiffrec.supports.uvmap import xatlas_uvmap
from nvdiffrec.geometry.dmtet import DMTetGeometry
from nvdiffrec.supports.training import optimize_mesh
from nvdiffrec.dataset.dataset_nerf import DatasetNERF
from nvdiffrec.supports.validation_and_testing import validate
from nvdiffrec.supports.material_utility import initial_guess_material

RADIUS = 3.0

ROOT = Path(__file__).parent.parent


def set_flags(ref_mesh, out_dir):
    FLAGS = Namespace()
    FLAGS.iter = 5000
    FLAGS.batch = 1
    FLAGS.spp = 1
    FLAGS.layers = 1
    FLAGS.train_res = [512, 512]
    FLAGS.display_res = None
    FLAGS.texture_res = [1024, 1024]
    FLAGS.display_interval = 0
    FLAGS.save_interval = 500
    FLAGS.learning_rate = 0.01
    FLAGS.min_roughness = 0.08
    FLAGS.custom_mip = True
    FLAGS.random_textures = True
    FLAGS.background = "white"
    FLAGS.loss = "logl1"
    FLAGS.out_dir = out_dir
    FLAGS.ref_mesh = ref_mesh
    FLAGS.base_mesh = None
    FLAGS.validate = True
    FLAGS.mtl_override = None  # Override material of model
    FLAGS.dmtet_grid = 256  # Resolution of initial tet grid.
    FLAGS.mesh_scale = 2.5  # Scale of tet grid box. Adjust to cover the model
    FLAGS.env_scale = 1.0  # Env map intensity multiplier
    FLAGS.envmap = None  # HDR environment probe
    FLAGS.display = [
        {"latlong": True}, {"bsdf": "kd"}, {"bsdf": "ks"}, {"bsdf": "normal"}
    ]
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
    FLAGS.ks_min = [0.0, 0.2, 0.0]  # Limits for ks
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

    # save best params
    with open(ROOT / Path("params.yaml"), "r") as stream:
        data = yaml.safe_load(stream)
        for key in data:
            FLAGS.__dict__[key] = data[key]

    # if FLAGS.config is not None:
    #     data = json.load(open(FLAGS.config, "r"))
    #     for key in data:
    #         FLAGS.__dict__[key] = data[key]

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


# @click.command()
# @click.option("--ref_mesh", type=str, default="data/processed/configs/", help="Config file")
# @click.option("--out_dir", type=str, default="data/results", help="Config file")
def base_run(ref_mesh, out_dir):

    FLAGS = set_flags(ref_mesh, out_dir)

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

    save mesh without textures
    eval_mesh = geometry.getMesh(mat)
    os.makedirs(os.path.join(FLAGS.out_dir, "eval_mesh"), exist_ok=True)
    obj.write_obj(os.path.join(FLAGS.out_dir, "eval_mesh/"), eval_mesh)
    
    torch.save(mat.state_dict(), os.path.join(FLAGS.out_dir, "mat.pt"))
    
    with open(os.path.join(FLAGS.out_dir, "geometry.pickle"), "wb") as file:
        pickle.dump(geometry, file)
        
    with open(os.path.join(FLAGS.out_dir, "lgt.pickle"), "wb") as file:
        pickle.dump(lgt, file)

    with open(os.path.join(FLAGS.out_dir, "FLAGS.pickle"), "wb") as file:
        pickle.dump(FLAGS, file)


def refinement_run(out_dir):
    
    eval_mesh = obj.load_obj("data/results/eval_mesh/mesh.obj")
    
    # load geometry
    with open(os.path.join(out_dir, "geometry.pickle"), "rb") as file:
        geometry = pickle.load(file)
        
    with open(os.path.join(out_dir, "lgt.pickle"), "rb") as file:
        lgt = pickle.load(file)
        
    with open(os.path.join(out_dir, "FLAGS.pickle"), "rb") as file:
        FLAGS = pickle.load(file)
    
    # load materials
    mat = initial_guess_material(geometry, True, FLAGS)
    mat.load_state_dict(torch.load(os.path.join(out_dir, "mat.pt")))
    
    # load glctx
    glctx = dr.RasterizeGLContext()

    # Trying to create textured mesh from result
    base_mesh = xatlas_uvmap(glctx, geometry, mat, FLAGS, eval_mesh)

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


# if __name__ == "__main__":
#     base_run()
