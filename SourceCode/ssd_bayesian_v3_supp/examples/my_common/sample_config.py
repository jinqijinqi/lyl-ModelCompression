#from addict import Dict

import argparse
import os

try:
    import examples.my_common.jstyleson as json
except ImportError:
    import json

def from_json(path):
    with open(path,encoding='UTF-8') as f:
        loaded_json = json.load(f)
    return loaded_json

def update_from_args(config, args, argparser=None):
    if argparser is not None:
        known_args = argparser.parse_known_args()[0]
        default_args = {k for k, v in vars(args).items() if vars(known_args)[k] == v}
    else:
        default_args = {k for k, v in vars(args).items() if v is None}
    for key, value in vars(args).items():
        if key not in default_args or key not in config:
            config[key] = value
    return config

def create_sample_config(args, parser):
    nncf_config = from_json(args.config)
    sample_config = from_json(args.config)

    sample_config = argparse.Namespace(**update_from_args(sample_config, args, parser))
    # sample_config.add_argument("--nncf_config", defalut=None, type=dict)
    sample_config.nncf_config = nncf_config

    return sample_config


