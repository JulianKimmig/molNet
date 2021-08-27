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
import numpy as np
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



def create_model_struct(model_name):
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
    #data["files"]["test_data_file"] = os.path.join(
    #    data["files"]["model_dir"], "test_data.pickle"
    #)
    data["files"]["test_batch_file"] = os.path.join(
        data["files"]["model_dir"], "test_batch.pickle"
    )
    
    return data
    
    
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
    #force_test_data_reload=False,
    early_stopping=True,
    early_stopping_delta=10 ** (-6),
    categories=None,
    plot_ignore_first=True,
    early_stop_patience=DEFAULT_PATIENCE,
):

    
    data = create_model_struct(model_name)

    if not force_run:
        try:
            #model = model.__class__.load_from_checkpoint(
            #    data["files"]["model_checkpoint"],
            #    map_location=lambda storage, location: storage,
            #)
            checkpoint = torch.load(data["files"]["model_checkpoint"]) # ie, model_best.pth.tar
            model.load_state_dict(checkpoint['state_dict'])
        except Exception as e:
            print(e)
            if ignore_load_error:
                pass
            else:
                raise e
                force_run = True
        #try:
        #    test_data = pickle.load(open(data["files"]["test_data_file"], "rb"))
        #    test_batch = pickle.load(open(data["files"]["test_batch_file"], "rb"))
        #except:
        #    force_run = True

    #if force_test_data_reload or force_run:
        #test_data = find_test_data(loader)
        #test_batch = iter(torch_geometric.data.DataLoader([test_data])).next()
        #pickle.dump(test_data, open(data["files"]["test_data_file"], "wb"))
        #pickle.dump(test_batch, open(data["files"]["test_batch_file"], "wb"))

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
            #model = model.__class__.load_from_checkpoint(
            #    checkpoint_callback.best_model_path,
            #)
            checkpoint = torch.load(checkpoint_callback.best_model_path) # ie, model_best.pth.tar
            model.load_state_dict(checkpoint['state_dict'])

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

    #data["test_data"] = test_data
    #data["test_batch"] = test_batch

    try:
        data["trainer"] = trainer
    except:
        pass
    return model, data


from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.loggers.base import rank_zero_experiment
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

class DictLogger(LightningLoggerBase):
    
    def __init__(self,n,monitor=None,mode=min):
        super().__init__(n)
        self.metrics=[]
        self._monitor=monitor
        self._best=None
        self._mode=mode
        
    @property
    def name(self):
        return 'MyLogger'

    @property
    @rank_zero_experiment
    def experiment(self):
        # Return the experiment object associated with this logger.
        pass

    @property
    def version(self):
        # Return the experiment version, int or str.
        return '0.1'

    @rank_zero_only
    def log_hyperparams(self, params):
        # params is an argparse.Namespace
        # your code to record hyperparameters goes here
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step):
        # metrics is a dictionary of metric names and values
        # your code to record metrics goes here
        #if step%500==0:
        if self._monitor is not None:
            if self._monitor in metrics:
                if self._best is None:
                    self._best = metrics[self._monitor]
                else:
                    self._best = self._mode(self._best,metrics[self._monitor])
        print(step,metrics," "*5,end="\r")
        self.metrics.append(metrics)

    @rank_zero_only
    def save(self):
        # Optional. Any code necessary to save logger data goes here
        # If you implement this, remember to call `super().save()`
        # at the start of the method (important for aggregation of metrics)
        super().save()

    @rank_zero_only
    def finalize(self, status):
        # Optional. Any code that needs to be run after training
        # finishes goes here
        pass
    
    @property
    def best_model_score(self):
        return self._best

    
class trialwrapperf():
    def __init__(self,n):
        self._f = n
    
    def to_trial(self,trial):
        return getattr(trial,self._f)(*self._args,**self._kwargs)
    
    def __call__(self,*args,**kwargs):
        self._args=args
        self._kwargs=kwargs
        return self
    
class trialwrapper():   
    def __getattr__(self,n):
        try:
            return super().__getattr__(n)
        except AttributeError:
            return trialwrapperf(n)
    
def load_study(model_name):
    import optuna
    data = create_model_struct(model_name)
    study = optuna.create_study(study_name="{}_study".format(model_name),
                            direction="minimize",
                            storage="sqlite:///"+os.path.join(data["files"]["model_dir"],"optuna.db"),
                           load_if_exists=True
                           )
    return study



def suggest_multiple(self,values,names,minimum=0,maximum=None):
    if maximum is None:
        maximum=len(values)
    
    
    mnames=np.array(names)
    np.random.shuffle(mnames)
    
    selects=[] 
    for i,n in enumerate(mnames):
        add=self.suggest_categorical(n,[False,True])
        if i<=minimum:
            add=True
        elif i>=maximum:
            add=False
        if add:
            selects.append(values[names.index(n)])
            self.set_user_attr(n,add)
            self.params[n] = add
    
    return selects
    
import optuna
optuna.Trial.suggest_multiple=suggest_multiple



from pytorch_lightning.callbacks.base import Callback
class StopOnValCallback(Callback):
    def __init__(self,monitor,target,mode=min):
        self._monitor=monitor
        self._target=target
        self._mode=mode
        
    def on_train_epoch_end(self, trainer, pl_module,*args,**kwargs):
        logs = trainer.callback_metrics
        current = logs.get(self._monitor)
        if current is None:
            return
        should_stop = current ==  self._mode(self._target,current)
        #should_stop = trainer.training_type_plugin.reduce_boolean_decision(should_stop)
        trainer.should_stop = trainer.should_stop or should_stop
        if should_stop:
            trainer.should_stop = True
            
def optimize_model(model_class,model_name,data_loader,fixed_params={},suggestion_params={},trainer_params={},n_trials=1000,verbose=False,callbacks=[],reps_per_trial=3):
    from pytorch_lightning.utilities import rank_zero_only
    from pytorch_lightning.loggers import LightningLoggerBase
    from pytorch_lightning.loggers.base import rank_zero_experiment
    from pytorch_lightning.callbacks.early_stopping import EarlyStopping
    
       
    
    
    data = create_model_struct(model_name)
    
    study=load_study(model_name)
    
    def objective(trial):

            print({**fixed_params,
                                **{n:trialwrapper.to_trial(trial) for n,trialwrapper in suggestion_params.items()}})
          
            
            
            if verbose:
                display(trial.params)
            #display(_model)
            metrics=[]
            for i in range(reps_per_trial):
                
                _tainer_params={k:v for k,v in trainer_params.items()}
            
                if "max_epochs" not in _tainer_params:
                    _tainer_params["max_epochs"]=200

                if "progress_bar_refresh_rate" not in _tainer_params:
                    _tainer_params["progress_bar_refresh_rate"]=0

                if "gpus" not in _tainer_params or _tainer_params["gpus"]=="auto":
                    _tainer_params["gpus"]=torch.cuda.device_count()

                if "callbacks" in _tainer_params:
                    _tainer_params["callbacks"] = _tainer_params["callbacks"]()
                else:
                    _tainer_params["callbacks"]=[]
                    
                if len(metrics)>0:
                    _tainer_params["callbacks"].append(
                        StopOnValCallback(
                            target = max(metrics),
                            monitor = "val_loss",
                            mode = min,
                        )
                    )
                
                logger=DictLogger(trial.number,
                                  monitor="val_loss",
                                  mode=min
                                 )
                   
                
                _model = model_class(**fixed_params,
                                    **{n:trialwrapper.to_trial(trial) for n,trialwrapper in suggestion_params.items()},
                                    )#
                trainer = pl.Trainer(
                    **_tainer_params,
                    checkpoint_callback=False,
                    logger=logger,
                    )

                trainer.fit(_model,data_loader)
                
                metrics.append(logger.best_model_score)
            
                try:
                    if study.best_trial.value < logger.best_model_score:
                        break
                except ValueError:
                    pass
            
            if len(metrics)>1:
                return max(metrics)
            else:
                return metrics[0]
#for i in range(20):
#optuna.study.delete_study(study_name="tune_study",
#                            storage="sqlite:///"+os.path.join(data_MPModel1["files"]["model_dir"],"optuna.db"),
#                           )

    study.optimize(objective, n_trials=n_trials)
    return study
    