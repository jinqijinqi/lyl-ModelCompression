import numpy as np
import collections

def prunning_weights(model_pruned, model, masks, output_mask):
    post_weight_mus = []

    for i, (layer, mask) in enumerate(zip(model.kl_list, masks)):
        # compute posteriors
        post_weight_mu, post_weight_var = layer.compute_posterior_params()
        post_weight_mu = post_weight_mu.cpu().data
        post_weight_mus.append(post_weight_mu)

    weights_mus = collections.OrderedDict()
    pruned_weights = collections.OrderedDict()
    mask_idx = 0
    for m,v in model.state_dict().items():
        if ('log' not in m) and ('z' not in m):
            m = m.replace('_mu','')
            weights_mus[m] = v

    for m,v in model_pruned.state_dict().items():
        raw_v = weights_mus[m]
        mask = masks[mask_idx]
        single_mask = output_mask[mask_idx]
        if len(v.size()) > 2:  # 卷积层的权值剪枝
            pruned_weight = post_weight_mus[mask_idx][mask.astype('bool').squeeze()].reshape(v.size())
        elif len(v.size()) == 2: # 全连接层的权值剪枝
            pruned_weight = raw_v.reshape(-1)[mask.reshape(-1).astype('bool')].reshape(v.size())
        elif len(v.size()) == 1: # BN层数的权值以及bias以及卷积层的bias的剪枝
            pruned_weight = raw_v[single_mask.astype('bool').squeeze()].reshape(v.size())
        else:
            assert 'num_batches_tracked' in m
            pruned_weight = raw_v
            mask_idx += 1
        pruned_weights[m] = pruned_weight
    return pruned_weights

def lenet_prunning_weights(model_pruned, model, masks, output_mask):
    post_weight_mus = []

    for i, (layer, mask) in enumerate(zip(model.kl_list, masks)):
        # compute posteriors
        post_weight_mu, post_weight_var = layer.compute_posterior_params()
        post_weight_mu = post_weight_mu.cpu().data
        post_weight_mus.append(post_weight_mu)

    weights_mus = collections.OrderedDict()
    pruned_weights = collections.OrderedDict()
    mask_idx = 0
    for m,v in model.state_dict().items():
        if ('log' not in m) and ('z' not in m):
            m = m.replace('_mu','')
            weights_mus[m] = v

    for m,v in model_pruned.state_dict().items():
        single_mask = output_mask[mask_idx]
        if "conv2" in m:
            # m = m.replace("conv2.conv","conv2")
            single_mask = np.ones_like(single_mask)
        raw_v = weights_mus[m]
        mask = masks[mask_idx]
        if len(v.size()) > 2:  # 卷积层的权值剪枝
            pruned_weight = post_weight_mus[mask_idx][mask.astype('bool').squeeze()].reshape(v.size())
        elif len(v.size()) == 2: # 全连接层的权值剪枝
            pruned_weight = raw_v.reshape(-1)[mask.reshape(-1).astype('bool')].reshape(v.size())
        else: # BN层数的权值以及bias以及卷积层的bias的剪枝
            pruned_weight = raw_v[single_mask.astype('bool').squeeze()].reshape(v.size())
            mask_idx += 1
        pruned_weights[m] = pruned_weight
    return pruned_weights