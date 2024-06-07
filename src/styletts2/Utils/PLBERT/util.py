import os
import yaml
import torch
from transformers import AlbertConfig, AlbertModel
from collections import OrderedDict

class CustomAlbert(AlbertModel):
    def forward(self, *args, **kwargs):
        # Call the original forward method
        outputs = super().forward(*args, **kwargs)

        # Only return the last_hidden_state
        return outputs.last_hidden_state


def load_plbert(log_dir, config_path=None, checkpoint_path=None):
    """
    Load the pre-trained ALBERT model from the specified directory.

    :param log_dir: Directory containing the model checkpoints.
    :param config_path: Optional path to the configuration file. Defaults to "config.yml" in log_dir.
    :param checkpoint_path: Optional path to the specific checkpoint file. If not provided, the latest checkpoint is used.
    :return: Loaded ALBERT model.
    """
    if not config_path:
        config_path = os.path.join(log_dir, "config.yml")
    plbert_config = yaml.safe_load(open(config_path))
    
    albert_base_configuration = AlbertConfig(**plbert_config['model_params'])
    bert = CustomAlbert(albert_base_configuration)

    if not checkpoint_path:
        files = os.listdir(log_dir)
        ckpts = [f for f in files if f.startswith("step_") and os.path.isfile(os.path.join(log_dir, f))]
        iters = [int(f.split('_')[-1].split('.')[0]) for f in ckpts]
        iters = sorted(iters)[-1]
        checkpoint_path = os.path.join(log_dir, "step_" + str(iters) + ".t7")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['net']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        if name.startswith('encoder.'):
            name = name[8:] # remove `encoder.`
        new_state_dict[name] = v
    try:
        del new_state_dict["embeddings.position_ids"]
    except KeyError:
        pass
    bert.load_state_dict(new_state_dict, strict=False)
    
    return bert
