import time
import torch
import numpy as np
from nvdiffrec.render import util
from losses_and_batch import createLoss, prepare_batch
from validation_and_testing import validate_itr


###############################################################################
# Main shape fitter function / optimization loop
###############################################################################


class Trainer(torch.nn.Module):
    def __init__(
            self,
            glctx,
            geometry,
            lgt,
            mat,
            optimize_geometry,
            optimize_light,
            image_loss_fn,
            FLAGS,
    ):
        super(Trainer, self).__init__()

        self.glctx = glctx
        self.geometry = geometry
        self.light = lgt
        self.material = mat
        self.optimize_geometry = optimize_geometry
        self.optimize_light = optimize_light
        self.image_loss_fn = image_loss_fn
        self.FLAGS = FLAGS

        if not self.optimize_light:
            with torch.no_grad():
                self.light.build_mips()

        self.params = list(self.material.parameters())
        self.params += list(self.light.parameters()) if optimize_light else []
        self.geo_params = list(self.geometry.parameters()) if optimize_geometry else []

    def forward(self, target, it):
        if self.optimize_light:
            self.light.build_mips()
            if self.FLAGS.camera_space_light:
                self.light.xfm(target["mv"])

        return self.geometry.tick(
            self.glctx, target, self.light, self.material, self.image_loss_fn, it
        )


def optimize_mesh(
        glctx,
        geometry,
        opt_material,
        lgt,
        dataset_train,
        dataset_validate,
        FLAGS,
        warmup_iter=0,
        log_interval=10,
        pass_idx=0,
        pass_name="",
        optimize_light=True,
        optimize_geometry=True,
):
    # ==============================================================================
    #  Setup torch optimizer
    # ==============================================================================

    learning_rate = (
        FLAGS.learning_rate[pass_idx]
        if isinstance(FLAGS.learning_rate, list)
           or isinstance(FLAGS.learning_rate, tuple)
        else FLAGS.learning_rate
    )
    learning_rate_pos = (
        learning_rate[0]
        if isinstance(learning_rate, list) or isinstance(learning_rate, tuple)
        else learning_rate
    )
    learning_rate_mat = (
        learning_rate[1]
        if isinstance(learning_rate, list) or isinstance(learning_rate, tuple)
        else learning_rate
    )

    def lr_schedule(iter, fraction):
        if iter < warmup_iter:
            return iter / warmup_iter
        return max(
            0.0, 10 ** (-(iter - warmup_iter) * 0.0002)
        )  # Exponential falloff from [1.0, 0.1] over 5k epochs.

    # ==============================================================================
    #  Image loss
    # ==============================================================================
    image_loss_fn = createLoss(FLAGS)

    trainer_noddp = Trainer(
        glctx,
        geometry,
        lgt,
        opt_material,
        optimize_geometry,
        optimize_light,
        image_loss_fn,
        FLAGS,
    )

    if FLAGS.multi_gpu:
        # Multi GPU training mode
        import apex
        from apex.parallel import DistributedDataParallel as DDP
        trainer = DDP(trainer_noddp)
        trainer.train()
        if optimize_geometry:
            optimizer_mesh = apex.optimizers.FusedAdam(
                trainer_noddp.geo_params, lr=learning_rate_pos
            )
            scheduler_mesh = torch.optim.lr_scheduler.LambdaLR(
                optimizer_mesh, lr_lambda=lambda x: lr_schedule(x, 0.9)
            )

        optimizer = apex.optimizers.FusedAdam(
            trainer_noddp.params, lr=learning_rate_mat
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda x: lr_schedule(x, 0.9)
        )
    else:
        # Single GPU training mode
        trainer = trainer_noddp
        if optimize_geometry:
            optimizer_mesh = torch.optim.Adam(
                trainer_noddp.geo_params, lr=learning_rate_pos
            )
            scheduler_mesh = torch.optim.lr_scheduler.LambdaLR(
                optimizer_mesh, lr_lambda=lambda x: lr_schedule(x, 0.9)
            )

        optimizer = torch.optim.Adam(trainer_noddp.params, lr=learning_rate_mat)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda x: lr_schedule(x, 0.9)
        )

    # ===============================================================================
    #  Training loop
    # ===============================================================================
    img_cnt = 0
    img_loss_vec = []
    reg_loss_vec = []
    iter_dur_vec = []

    dataloader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=FLAGS.batch,
        collate_fn=dataset_train.collate,
        shuffle=True,
    )
    dataloader_validate = torch.utils.data.DataLoader(
        dataset_validate, batch_size=1, collate_fn=dataset_train.collate
    )

    def cycle(iterable):
        iterator = iter(iterable)
        while True:
            try:
                yield next(iterator)
            except StopIteration:
                iterator = iter(iterable)

    v_it = cycle(dataloader_validate)

    for it, target in enumerate(dataloader_train):

        # Mix randomized background into dataset image
        target = prepare_batch(target, "random")

        # =============================================================================
        #  Display / save outputs. Do it before training so we get initial meshes
        # =============================================================================

        # Show/save image before training step (want to get correct rendering of input)
        if FLAGS.local_rank == 0:
            display_image = FLAGS.display_interval and (
                    it % FLAGS.display_interval == 0
            )
            save_image = FLAGS.save_interval and (it % FLAGS.save_interval == 0)
            if display_image or save_image:
                result_image, result_dict = validate_itr(
                    glctx,
                    prepare_batch(next(v_it), FLAGS.background),
                    geometry,
                    opt_material,
                    lgt,
                    FLAGS,
                )
                np_result_image = result_image.detach().cpu().numpy()
                if display_image:
                    util.display_image(
                        np_result_image, title="%d / %d" % (it, FLAGS.iter)
                    )
                if save_image:
                    util.save_image(
                        FLAGS.out_dir
                        + "/"
                        + ("img_%s_%06d.png" % (pass_name, img_cnt)),
                        np_result_image,
                    )
                    img_cnt = img_cnt + 1

        iter_start_time = time.time()

        # ============================================================================
        #  Zero gradients
        # ============================================================================
        optimizer.zero_grad()
        if optimize_geometry:
            optimizer_mesh.zero_grad()

        # ============================================================================
        #  Training
        # ============================================================================
        img_loss, reg_loss = trainer(target, it)

        # ============================================================================
        #  Final loss
        # ============================================================================
        total_loss = img_loss + reg_loss

        img_loss_vec.append(img_loss.item())
        reg_loss_vec.append(reg_loss.item())

        # ============================================================================
        #  Backpropagate
        # ============================================================================
        total_loss.backward()
        if hasattr(lgt, "base") and lgt.base.grad is not None and optimize_light:
            lgt.base.grad *= 64
        if "kd_ks_normal" in opt_material:
            opt_material["kd_ks_normal"].encoder.params.grad /= 8.0

        optimizer.step()
        scheduler.step()

        if optimize_geometry:
            optimizer_mesh.step()
            scheduler_mesh.step()

        # ===========================================================================
        #  Clamp trainables to reasonable range
        # ===========================================================================
        with torch.no_grad():
            if "kd" in opt_material:
                opt_material["kd"].clamp_()
            if "ks" in opt_material:
                opt_material["ks"].clamp_()
            if "normal" in opt_material:
                opt_material["normal"].clamp_()
                opt_material["normal"].normalize_()
            if lgt is not None:
                lgt.clamp_(min=0.0)

        torch.cuda.current_stream().synchronize()
        iter_dur_vec.append(time.time() - iter_start_time)

        # ============================================================================
        #  Logging
        # ============================================================================
        if it % log_interval == 0 and FLAGS.local_rank == 0:
            img_loss_avg = np.mean(np.asarray(img_loss_vec[-log_interval:]))
            reg_loss_avg = np.mean(np.asarray(reg_loss_vec[-log_interval:]))
            iter_dur_avg = np.mean(np.asarray(iter_dur_vec[-log_interval:]))

            remaining_time = (FLAGS.iter - it) * iter_dur_avg
            print(
                "iter=%5d, img_loss=%.6f, reg_loss=%.6f, lr=%.5f, time=%.1f ms, rem=%s"
                % (
                    it,
                    img_loss_avg,
                    reg_loss_avg,
                    optimizer.param_groups[0]["lr"],
                    iter_dur_avg * 1000,
                    util.time_to_text(remaining_time),
                )
            )

    return geometry, opt_material
