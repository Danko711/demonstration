import torch
from collections import defaultdict

from catalyst.utils import get_activation_fn
from catalyst.dl.core import Callback, CallbackOrder, RunnerState

from utils.metrics import spearman_correlation


def spearman_correlation_callback(
    targets: torch.Tensor,
    outputs: torch.Tensor,
    activation: str = "Sigmoid"
):
    """
    Computes the spearman correlation metric

    Args:
        targets (list): A list of elements that are to be predicted
        outputs (list):  A list of predicted elements
        activation (str): An torch.nn activation applied to the outputs.
            Must be one of ["none", "Sigmoid", "Softmax2d"]

    Returns:
        double:  Spearman correlation score
    """
    activation_fn = get_activation_fn(activation)
    y_pred = activation_fn(outputs).cpu().detach().numpy()
    y_true = targets.cpu().detach().numpy()
    spearm_corr = spearman_correlation(y_true, y_pred)

    return spearm_corr


# Example for catalyst callback
class SpearmanCorrelationMetricCallback(Callback):
    """
    Spearman corellation score metric callback.
    """

    def __init__(
        self,
        input_key: str = "targets",
        output_key: str = "logits",
        prefix: str = "spearman",
        activation: str = "Sigmoid",
        **metric_params
    ):
        """
        Args:
            input_key (str): input key to use for metric calculation
                specifies our ``y_true``.
            output_key (str): output key to use for metric calculation;
                specifies our ``y_pred``
            activation (str): An torch.nn activation applied to the outputs.
                Must be one of ['none', 'Sigmoid', 'Softmax2d']
        """
        super().__init__(CallbackOrder.Metric)
        self.input_key = input_key
        self.output_key = output_key
        self.metric_fn = spearman_correlation_callback
        self.prefix = prefix
        self.activation = activation
        self.metric_params = metric_params

        self.predictions = defaultdict(lambda: [])
        self.metric_value = None

    def on_loader_start(self, state: RunnerState):
        self.predictions = defaultdict(lambda: [])

    def on_batch_end(self, state: RunnerState):
        targets = state.input[self.input_key]
        outputs = state.output[self.output_key]
        self.predictions[self.input_key].append(targets.detach().cpu())
        self.predictions[self.output_key].append(outputs.detach().cpu())
        # For calculation metric for each batch
        # metric = self.metric_fn(outputs, targets, **self.metric_params)
        # state.metrics.add_batch_value(name=self.prefix, value=metric)

    def on_loader_end(self, state: RunnerState):
        self.predictions = {
            key: torch.cat(value, dim=0)
            for key, value in self.predictions.items()
        }
        targets = self.predictions[self.input_key]
        outputs = self.predictions[self.output_key]
        value = self.metric_fn(
            targets, outputs
        )
        state.metrics.epoch_values[state.loader_name][self.prefix] = value
        self.metric_value = value
