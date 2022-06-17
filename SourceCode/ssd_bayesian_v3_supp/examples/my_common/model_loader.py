import torch
from examples.my_common.example_logger import logger

def load_resuming_model_state_dict_and_checkpoint_from_path(resuming_checkpoint_path):
    
    logger.info('Resuming from checkpoint {}...'.format(resuming_checkpoint_path))
    resuming_checkpoint = torch.load(resuming_checkpoint_path, map_location='cpu')
    # use checkpoint itself in case only the state dict was saved,
    # i.e. the checkpoint was created with `torch.save(module.state_dict())`
    resuming_model_state_dict = resuming_checkpoint.get('state_dict', resuming_checkpoint)
    return resuming_model_state_dict, resuming_checkpoint
