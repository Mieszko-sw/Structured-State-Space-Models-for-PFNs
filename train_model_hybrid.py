#------------------------------------------------------------------------------------------------
#                                        IMPORTS
#------------------------------------------------------------------------------------------------
import os
print(f"currently in {os.getcwd()}")
from datetime import datetime

import torch
# Limit PyTorch CUDA use
#torch.cuda.set_per_process_memory_fraction(0.6)
import json

from pathlib import Path

import numpy as np
import wandb
from tabpfn.scripts.model_builder import save_model
from tabpfn.scripts.model_builder_custom import get_model
from tabpfn.scripts.model_configs import *
from tabpfn.scripts.epoch_callback import epoch_callback
from tabpfn.priors.utils import uniform_int_sampler_f

from evaluation_helper import EvalHelper

#------------------------------------------------------------------------------------------------
#                                       END IMPORTS
#------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------
#                                        PARAMETER
#------------------------------------------------------------------------------------------------

# Other Parameters
base_path = '.'
max_features = 100
large_datasets = True

# Others
json_file_path = "tabpfn_original_config.json"

#------------------------------------------------------------------------------------------------
#                                      END PARAMETER
#------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------
#                                         CONFIG
#------------------------------------------------------------------------------------------------

with open(json_file_path, "r") as f: config = json.load(f)
    
# Fill in stuff that could not be loaded properly into the config.json
uniform_int_sampler_f = (lambda a, b : lambda : round(np.random.uniform(a, b)))
choice_values = [
    torch.nn.modules.activation.Tanh, 
    torch.nn.modules.linear.Identity,
    torch.nn.modules.activation.ReLU
    ]

config["differentiable_hyperparameters"]["prior_mlp_activations"]["choice_values"] = choice_values
config["num_classes"] = uniform_int_sampler_f(2, config['max_num_classes']) # Wrong Function
config["num_features_used"] = uniform_int_sampler_f(1, max_features)

#------------------------------------------------------------------------------------------------
#                                        END CONFIG
#------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------
#                                          CUSTOM
#------------------------------------------------------------------------------------------------

model_type = "hybrid"
checkpoint_model_name = "hybrid_hydra_transformer_x4"

config['batch_size'] = 64 
config['emsize'] = 512 
config["epochs"] = 200
config["bptt"] = 1024
config["max_eval_pos"] = 1000        

config["num_steps"] = 1024

config["hybrid_layer_types"] = ["hydra", "transformer"] * 4
config["nlayers"] = len(config["hybrid_layer_types"])
config["enable_autocast"] = True
config["enable_transformer_full_attn"] = False
config["bootstrap_samples"] = 4096          # Default would be 0. 
config["permutation_repeat"] = 0

device = "cuda:2"
ENABLE_DATA_PARALLEL = False

#os.environ["SLURM_PROCID"]="1"

#------------------------------------------------------------------------------------------------
#                                        END CUSTOM
#------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------
#                                           WANDB
#------------------------------------------------------------------------------------------------

wandb_project = "mamba_project"
wandb_job_type = f"create_{model_type}_model"
wandb_run_name = f"{model_type} {config['nlayers']}l {config['emsize']}e {config['batch_size']}b"

wandb_config= config

wandb_run = wandb.init(project=wandb_project,job_type=wandb_job_type,config=wandb_config, name=wandb_run_name, group="DDP")
print(f"W&B run URL: {wandb_run.url}")

#------------------------------------------------------------------------------------------------
#                                         END WANDB
#------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------
#                                           MODEL
#------------------------------------------------------------------------------------------------

# Evaluation during training:
eval_class = EvalHelper()

# Get the model 
#model = get_model(config, device, should_train=True, verbose=0) # , state_dict=model[2].state_dict()
hybrid_model = get_model(config, 
                              device, 
                              should_train=True, 
                              verbose=1,
                              epoch_callback=epoch_callback,
                              use_autocast=config["enable_autocast"], 
                              evaluation_class=eval_class,
                              permutation_repeat=config["permutation_repeat"],
                              bootstrap_samples = config["bootstrap_samples"],
                              enable_data_parallel=ENABLE_DATA_PARALLEL,
                              model_type=model_type
                              ) # , state_dict=model[2].state_dict()

(hp_embedding, data, _), targets, single_eval_pos = next(iter(hybrid_model[3]))

# Save Hybrid Hydra/Transformer Model
save_model(hybrid_model[2], 
           base_path, 
           f"tabpfn/models_diff/{checkpoint_model_name}_{config['nlayers']}l.cpkt",
           config
           )

#------------------------------------------------------------------------------------------------
#                                         END MODEL
#------------------------------------------------------------------------------------------------


wandb_run.finish()

print("works")




