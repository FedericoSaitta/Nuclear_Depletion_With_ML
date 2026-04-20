from omegaconf import OmegaConf
import torch

# Personal Imports
import ML.datamodule.dnn_datamodule as dnn_datamodule
import ML.datamodule.neural_ode_datamodule as node_datamodule
from ML.models.dnn_model import DNN_Model
from ML.models.neural_ode import NODE_Model
import ML.models.modes as modes

if __name__ == "__main__":
  # Allow use of tensor cores if present
  torch.set_float32_matmul_precision('high')
 
  cfg = OmegaConf.load("main_config.yaml")
  lightning_mode = cfg.runtime.mode

  if cfg.runtime.model == 'DNN': 
    model = DNN_Model
    datamodule = dnn_datamodule.DNN_Datamodule(cfg)
  elif cfg.runtime.model == 'NODE': 
    model = NODE_Model
    datamodule = node_datamodule.NODE_Datamodule(cfg)

  if (lightning_mode == 'train'): modes.train_and_test(datamodule, model, cfg)
  elif (lightning_mode == 'train_from_ckp'): modes.train_from_checkpoint_and_test(datamodule, model, cfg)
  elif (lightning_mode == 'inference'): modes.inference(datamodule, model, cfg)
