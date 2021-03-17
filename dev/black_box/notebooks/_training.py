import torch
import torch_geometric
import pytorch_lightning as pl
from IPython.core.display import display, Image
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import dill as pickle
import matplotlib.pyplot as plt

from _callbacks import ClearCallback, StoreMetricsCallback
from _defaults import *
import os

from _plots import plot_true_pred, plot_category_validation


def find_test_data(loader, smiles=None):
    if smiles is None:
        smiles = TEST_SMILES
    try:
        loader.test_dataloader()
    except:
        loader.setup()
    sdt = 0
    for subloader in [
        loader.test_dataloader(),
        loader.val_dataloader(),
        loader.train_dataloader(),
    ]:
        for i, d in enumerate(subloader):
            for sd in d.to_data_list():
                if sd.string_data_titles[0][sdt] != "index":
                    sdt = sd.string_data_titles[0].index("smiles")
                if smiles == sd.string_data[0][sdt]:
                    return sd
    raise ValueError()


def default_model_run(
    model_name,
    model,
    loader,
    force_run=False,
    detect_lr=True,
    show_tb=True,
    train=True,
    save=True,
    test=True,
    live_plot=True,
    max_epochs=1000,
    min_epochs=1,
    ignore_load_error=False,
    force_test_data_reload=False,
    early_stopping=True,
    early_stopping_delta=10 ** (-6),
    categories=None,
    plot_ignore_first=True,
    early_stop_patience=DEFAULT_PATIENCE,
):

    data = {"files": {}}
    data["model_name"] = model_name
    data["files"]["model_dir"] = os.path.join("models", data["model_name"])

    data["files"]["logdir"] = os.path.join(data["files"]["model_dir"], "logs")
    data["files"]["tb_logdir"] = os.path.join(data["files"]["logdir"], "tensorboard")

    data["files"]["plot_dir"] = os.path.join(data["files"]["model_dir"], "plots")

    os.makedirs(data["files"]["plot_dir"], exist_ok=True)
    data["files"]["true_pred_plt"] = os.path.join(data["files"]["plot_dir"], "tvp.png")
    data["files"]["lr_optim_plot"] = os.path.join(data["files"]["plot_dir"], "lrp.png")
    data["files"]["metrics_plot"] = os.path.join(
        data["files"]["plot_dir"], "metrics.png"
    )
    data["files"]["cat_plot"] = os.path.join(
        data["files"]["plot_dir"], "cat_validation.png"
    )
    data["files"]["model_plot"] = os.path.join(
        data["files"]["plot_dir"], "model_plot.png"
    )
    data["files"]["model_plot_img"] = os.path.join(
        data["files"]["plot_dir"], "model_plot_img.png"
    )
    data["files"]["img_graph_plot"] = os.path.join(
        data["files"]["plot_dir"], "img_graph_plot"
    )
    os.makedirs(data["files"]["img_graph_plot"], exist_ok=True)

    data["files"]["model_checkpoint"] = os.path.join(
        data["files"]["model_dir"], "model.ckpt"
    )
    data["files"]["test_data_file"] = os.path.join(
        data["files"]["model_dir"], "test_data.pickle"
    )
    data["files"]["test_batch_file"] = os.path.join(
        data["files"]["model_dir"], "test_batch.pickle"
    )

    if not force_run:
        try:
            model = model.__class__.load_from_checkpoint(
                data["files"]["model_checkpoint"],
                map_location=lambda storage, location: storage,
            )
        except:
            if ignore_load_error:
                pass
            else:
                force_run = True
        try:
            test_data = pickle.load(open(data["files"]["test_data_file"], "rb"))
            test_batch = pickle.load(open(data["files"]["test_batch_file"], "rb"))
        except:
            force_run = True

    if force_test_data_reload or force_run:
        test_data = find_test_data(loader)
        test_batch = iter(torch_geometric.data.DataLoader([test_data])).next()
        pickle.dump(test_data, open(data["files"]["test_data_file"], "wb"))
        pickle.dump(test_batch, open(data["files"]["test_batch_file"], "wb"))

    if force_run:

        try:
            loader.test_dataloader()
        except:
            loader.setup()

        if detect_lr:
            lr_trainer = pl.Trainer()
            lr_finder = lr_trainer.tuner.lr_find(
                model, train_dataloader=loader.train_dataloader(), max_lr=10 ** 2
            )
            fig = lr_finder.plot(suggest=True)

            plt.savefig(data["files"]["lr_optim_plot"], dpi=DEFAULT_DPI)
            plt.close()

            model.lr = lr_finder.suggestion()
            print("set lr to", model.lr)

        if train or test:
            clear_cb = ClearCallback()
            tb_logger = TensorBoardLogger(data["files"]["tb_logdir"])
            checkpoint_callback = ModelCheckpoint(monitor="val_loss", verbose=True)
            metrics_cb = StoreMetricsCallback(
                live_plot=live_plot,
                final_save=data["files"]["metrics_plot"],
                plot_ignore_first=plot_ignore_first,
            )
            early_stop_cb = EarlyStopping(
                monitor="val_loss",
                patience=early_stop_patience,
                min_delta=early_stopping_delta,
            )

            cb = []

            if early_stopping:
                cb = [early_stop_cb]

            cb.extend(
                [
                    checkpoint_callback,
                    clear_cb,
                    metrics_cb,
                ]
            )

            trainer = pl.Trainer(
                max_epochs=max_epochs,
                gpus=torch.cuda.device_count(),
                callbacks=cb,
                logger=tb_logger,
                terminate_on_nan=True,
                min_epochs=min_epochs,
            )

        if train:
            trainer.fit(model, loader)
            model = model.__class__.load_from_checkpoint(
                checkpoint_callback.best_model_path
            )

        if test:
            trainer.test(model=model, ckpt_path=None)

        if save:
            trainer.save_checkpoint(data["files"]["model_checkpoint"])

    model.to("cpu")
    if REDRAW or force_run:
        plot_true_pred(model, loader, target_file=data["files"]["true_pred_plt"])
        if categories:
            plot_category_validation(
                model, loader, categories, target_file=data["files"]["cat_plot"]
            )

    if hasattr(model, "to_graphviz_from_batch"):
        g = model.to_graphviz_from_batch(test_batch, reduced=True)
        g.format = "png"
        # g.engine="fdp"
        g.render(
            filename=os.path.basename(data["files"]["model_plot"]).replace(
                "." + g.format, ""
            ),
            directory=os.path.dirname(data["files"]["model_plot"]),
        )

    if hasattr(model, "to_graphviz_images_from_batch"):
        g = model.to_graphviz_images_from_batch(
            test_batch,
            path=os.path.abspath(
                data["files"]["img_graph_plot"],
            ),
        )
        g.format = "png"
        # g.engine="fdp"
        g.render(
            filename=os.path.basename(data["files"]["model_plot_img"]).replace(
                "." + g.format, ""
            ),
            directory=os.path.dirname(data["files"]["model_plot"]),
        )

    if os.path.exists(data["files"]["lr_optim_plot"]):
        display(Image(data["files"]["lr_optim_plot"], width=DEFAULT_IMG_PLOT_WIDTH))
    if os.path.exists(data["files"]["metrics_plot"]):
        display(Image(data["files"]["metrics_plot"], width=DEFAULT_IMG_PLOT_WIDTH))
    if os.path.exists(data["files"]["true_pred_plt"]):
        display(Image(data["files"]["true_pred_plt"], width=DEFAULT_IMG_PLOT_WIDTH))
    if os.path.exists(data["files"]["cat_plot"]):
        display(Image(data["files"]["cat_plot"], width=DEFAULT_IMG_PLOT_WIDTH))

    if os.path.exists(data["files"]["model_plot"]):
        display(Image(data["files"]["model_plot"], width=DEFAULT_IMG_PLOT_WIDTH))
    if os.path.exists(data["files"]["model_plot_img"]):
        display(Image(data["files"]["model_plot_img"], width=DEFAULT_IMG_PLOT_WIDTH))

    data["test_data"] = test_data
    data["test_batch"] = test_batch

    try:
        data["trainer"] = trainer
    except:
        pass
    return model, data
