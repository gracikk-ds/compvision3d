import os
import json
import torch
import numpy as np
from nvdiffrec.render import util
from nvdiffrec.render import light
from losses_and_batch import prepare_batch


###############################################################################
# Validation & testing
###############################################################################


def validate_itr(glctx, target, geometry, opt_material, lgt, FLAGS):
    result_dict = {}
    with torch.no_grad():
        lgt.build_mips()
        if FLAGS.camera_space_light:
            lgt.xfm(target["mv"])

        buffers = geometry.render(glctx, target, lgt, opt_material)

        result_dict["ref"] = util.rgb_to_srgb(target["img"][..., 0:3])[0]
        result_dict["opt"] = util.rgb_to_srgb(buffers["shaded"][..., 0:3])[0]
        result_image = torch.cat([result_dict["opt"], result_dict["ref"]], axis=1)

        if FLAGS.display is not None:
            white_bg = torch.ones_like(target["background"])
            for layer in FLAGS.display:
                if "latlong" in layer and layer["latlong"]:
                    if isinstance(lgt, light.EnvironmentLight):
                        result_dict["light_image"] = util.cubemap_to_latlong(
                            lgt.base, FLAGS.display_res
                        )
                    result_image = torch.cat(
                        [result_image, result_dict["light_image"]], axis=1
                    )
                elif "relight" in layer:
                    if not isinstance(layer["relight"], light.EnvironmentLight):
                        layer["relight"] = light.load_env(layer["relight"])
                    img = geometry.render(glctx, target, layer["relight"], opt_material)
                    result_dict["relight"] = util.rgb_to_srgb(img[..., 0:3])[0]
                    result_image = torch.cat(
                        [result_image, result_dict["relight"]], axis=1
                    )
                elif "bsdf" in layer:
                    buffers = geometry.render(
                        glctx, target, lgt, opt_material, bsdf=layer["bsdf"]
                    )
                    if layer["bsdf"] == "kd":
                        result_dict[layer["bsdf"]] = util.rgb_to_srgb(
                            buffers["shaded"][0, ..., 0:3]
                        )
                    elif layer["bsdf"] == "normal":
                        result_dict[layer["bsdf"]] = (
                                                             buffers["shaded"][0, ...,
                                                             0:3] + 1
                                                     ) * 0.5
                    else:
                        result_dict[layer["bsdf"]] = buffers["shaded"][0, ..., 0:3]
                    result_image = torch.cat(
                        [result_image, result_dict[layer["bsdf"]]], axis=1
                    )

        return result_image, result_dict


def validate(glctx, geometry, opt_material, lgt, dataset_validate, out_dir, FLAGS):
    # ==============================================================================================
    #  Validation loop
    # ==============================================================================================
    mse_values = []
    psnr_values = []

    dataloader_validate = torch.utils.data.DataLoader(
        dataset_validate, batch_size=1, collate_fn=dataset_validate.collate
    )

    os.makedirs(out_dir, exist_ok=True)

    metrics = dict(avg_MSE=None, avg_PSNR=None)

    print("Running validation")
    for it, target in enumerate(dataloader_validate):

        # Mix validation background
        target = prepare_batch(target, FLAGS.background)

        result_image, result_dict = validate_itr(
            glctx, target, geometry, opt_material, lgt, FLAGS
        )

        # Compute metrics
        opt = torch.clamp(result_dict["opt"], 0.0, 1.0)
        ref = torch.clamp(result_dict["ref"], 0.0, 1.0)

        mse = torch.nn.functional.mse_loss(
            opt, ref, size_average=None, reduce=None, reduction="mean"
        ).item()
        mse_values.append(float(mse))
        psnr = util.mse_to_psnr(mse)
        psnr_values.append(float(psnr))

        os.makedirs(out_dir + "/dmtet_validate", exist_ok=True)
        for k in result_dict.keys():
            np_img = result_dict[k].detach().cpu().numpy()
            util.save_image(
                os.path.join(out_dir, "dmtet_validate") +
                "/" +
                ("val_%06d_%s.png" % (it, k)), np_img)

    avg_mse = np.mean(np.array(mse_values))
    avg_psnr = np.mean(np.array(psnr_values))
    metrics["avg_MSE"] = avg_mse
    metrics["avg_PSNR"] = avg_psnr
    with open(os.path.join(out_dir, "metrics.json"), "w") as fp:
        json.dump(metrics, fp)
    print("MSE,      PSNR")
    print("%1.8f, %2.3f" % (avg_mse, avg_psnr))
    return avg_psnr
