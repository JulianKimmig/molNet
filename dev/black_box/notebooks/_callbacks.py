import pytorch_lightning as pl
from IPython.display import clear_output
import matplotlib.pyplot as plt
from _defaults import *


class ClearCallback(pl.Callback):
    def on_validation_epoch_end(self, *args, **kwargs):
        self.clear()

    def clear(self):
        clear_output(wait=True)


class StoreMetricsCallback(pl.Callback):
    def __init__(
        self, live_plot=True, final_save=None, plot_ignore_first=True, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.data = {}
        self.live_plot = live_plot
        self.final_save = final_save
        self.plot_ignore_first = plot_ignore_first

    def plot_data(self, save=None):
        plt.figure()
        for label, data in self.data.items():
            if self.plot_ignore_first and len(data[0]) > 1:
                plt.plot(data[0][1:], data[1][1:], label=label)
            else:
                plt.plot(data[0], data[1], label=label)
        plt.legend()
        if save:
            plt.savefig(save, dpi=DEFAULT_DPI)
        else:
            plt.show()
        plt.close()

    def on_validation_epoch_end(self, trainer, pl_module, *args, **kwargs):
        ep = trainer.current_epoch
        for k, v in trainer.callback_metrics.items():
            if k not in self.data:
                self.data[k] = ([], [])
            self.data[k][0].append(ep)
            self.data[k][1].append(v.detach().cpu().numpy())
        if self.live_plot:
            self.plot_data()

        # if 'val_loss' in self.data:
        #    display(','.join([str(i) for i in self.data["val_loss"][1]]))

    # on_validation_epoch_end = on_epoch_end
    # on_test_epoch_end = on_epoch_end
    # on_train_epoch_end = on_epoch_end

    def on_train_end(self, trainer, pl_module):
        self.plot_data(self.final_save)
