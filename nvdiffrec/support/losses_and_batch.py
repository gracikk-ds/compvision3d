import torch
from nvdiffrec.render import util
import nvdiffrec.render.renderutils as ru


###############################################################################
# Loss setup
###############################################################################


@torch.no_grad()
def createLoss(FLAGS):
    if FLAGS.loss == "smape":
        return lambda img, ref: ru.image_loss(img, ref, loss="smape", tonemapper="none")
    elif FLAGS.loss == "mse":
        return lambda img, ref: ru.image_loss(img, ref, loss="mse", tonemapper="none")
    elif FLAGS.loss == "logl1":
        return lambda img, ref: ru.image_loss(
            img, ref, loss="l1", tonemapper="log_srgb"
        )
    elif FLAGS.loss == "logl2":
        return lambda img, ref: ru.image_loss(
            img, ref, loss="mse", tonemapper="log_srgb"
        )
    elif FLAGS.loss == "relmse":
        return lambda img, ref: ru.image_loss(
            img, ref, loss="relmse", tonemapper="none"
        )
    else:
        assert False

###############################################################################
# Mix background into a dataset image
###############################################################################


@torch.no_grad()
def prepare_batch(target, bg_type="black"):
    assert len(target["img"].shape) == 4, "Image shape should be [n, h, w, c]"
    if bg_type == "checker":
        background = torch.tensor(
            util.checkerboard(target["img"].shape[1:3], 8),
            dtype=torch.float32,
            device="cuda",
        )[None, ...]
    elif bg_type == "black":
        background = torch.zeros(
            target["img"].shape[0:3] + (3,), dtype=torch.float32, device="cuda"
        )
    elif bg_type == "white":
        background = torch.ones(
            target["img"].shape[0:3] + (3,), dtype=torch.float32, device="cuda"
        )
    elif bg_type == "reference":
        background = target["img"][..., 0:3]
    elif bg_type == "random":
        background = torch.rand(
            target["img"].shape[0:3] + (3,), dtype=torch.float32, device="cuda"
        )
    else:
        assert False, "Unknown background type %s" % bg_type

    target["mv"] = target["mv"].cuda()
    target["mvp"] = target["mvp"].cuda()
    target["campos"] = target["campos"].cuda()
    target["img"] = target["img"].cuda()
    target["background"] = background

    target["img"] = torch.cat(
        (
            torch.lerp(background, target["img"][..., 0:3], target["img"][..., 3:4]),
            target["img"][..., 3:4],
        ),
        dim=-1,
    )

    return target