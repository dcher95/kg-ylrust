import torch
from datetime import datetime
import os

def set_seeds(seed: int=42):
    """Sets random sets for torch operations.

    Args:
        seed (int, optional): Random seed to set. Defaults to 42.
    """
    # Set the seed for general torch operations
    torch.manual_seed(seed)
    # Set the seed for CUDA torch operations (ones that happen on the GPU)
    torch.cuda.manual_seed(seed)

# from torch.utils.tensorboard import SummaryWriter
# def create_writer(experiment_name: str,
#                   model_name: str,
#                   extra: str=None) -> SummaryWriter:
#     """Creates a SummaryWriter instance saving to a specific log_dir.

#     log_dir is a combination of runs/timestamp/experiment_name/model_name/extra.

#     Where timestamp is the current date in YYYY-MM-DD format.

#     Args:
#         experiment_name (str): Name of experiment.
#         model_name (str): Name of model.
#         extra (str, optional): Anything extra to add to the directory. Defaults to None.

#     Returns:
#         SummaryWriter: Instance of a writer saving to log_dir.

#     Example usage:
#         # Create a writer saving to "runs/2022-06-04/data_10_percent/effnetb2/5_epochs/"
#         writer = create_writer(experiment_name="data_10_percent",
#                                model_name="effnetb2",
#                                extra="5_epochs")
#         # The above is the same as:
#         writer = SummaryWriter(log_dir="runs/2022-06-04/data_10_percent/effnetb2/5_epochs/")
#     """

#     # Get timestamp of current date (all experiments on certain day live in same folder)
#     timestamp = datetime.now().strftime("%Y-%m-%d") # returns current date in YYYY-MM-DD format

#     if extra:
#         # Create log directory path
#         log_dir = os.path.join("runs", timestamp, experiment_name, model_name, extra)
#     else:
#         log_dir = os.path.join("runs", timestamp, experiment_name, model_name)

#     print(f"[INFO] Created SummaryWriter, saving to: {log_dir}...")
#     return SummaryWriter(log_dir=log_dir)

# def test_tensorboard():
#     writer = create_writer("test_experiment", "test_model")
#     writer.add_scalar('test_metric', 1, 0)
#     writer.close()

# if __name__ == "__main__":
#     test_tensorboard()
