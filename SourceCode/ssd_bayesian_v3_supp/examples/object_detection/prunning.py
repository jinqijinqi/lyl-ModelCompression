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
        if mask_idx < len(model.kl_list): # backbone（VGG）与额外特征层均采用了贝叶斯卷积
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
        else:  # 输出头结构使用的是原始的卷积层(nn.Conv2d),所以可直接将权值赋值过来
            pruned_weights[m] = raw_v
    return pruned_weights

# def prunning_weights(model_pruned, model, masks, output_mask): # l9v2
#     post_weight_mus = []
#
#     for i, (layer, mask) in enumerate(zip(model.kl_list, masks)):
#         # compute posteriors
#         post_weight_mu, post_weight_var = layer.compute_posterior_params()
#         post_weight_mu = post_weight_mu.cpu().data
#         post_weight_mus.append(post_weight_mu)
#
#     weights_mus = collections.OrderedDict()
#     pruned_weights = collections.OrderedDict()
#     mask_idx = 0
#
#     for m,v in model.state_dict().items():
#         if ('log' not in m) and ('z' not in m):
#             m = m.replace('_mu','')
#             weights_mus[m] = v
#
#     for m,v in model_pruned.state_dict().items():
#         raw_v = weights_mus[m]
#         if mask_idx < len(model.kl_list):
#             if mask_idx<4:
#                 mask = masks[mask_idx]
#                 single_mask = output_mask[mask_idx]
#                 if len(v.size()) > 2:  # 卷积层的权值剪枝
#                     pruned_weight = post_weight_mus[mask_idx][mask.astype('bool').squeeze()].reshape(v.size())
#                 elif len(v.size()) == 2: # 全连接层的权值剪枝
#                     pruned_weight = raw_v.reshape(-1)[mask.reshape(-1).astype('bool')].reshape(v.size())
#                 elif len(v.size()) == 1: # BN层数的权值以及bias以及卷积层的bias的剪枝
#                     pruned_weight = raw_v[single_mask.astype('bool').squeeze()].reshape(v.size())
#                 else:
#                     assert 'num_batches_tracked' in m
#                     pruned_weight = raw_v
#                     mask_idx += 1
#             elif mask_idx==4:
#                 if len(v.size()) > 2:
#                     t_mask = np.expand_dims(output_mask[3], axis=0) * np.expand_dims(np.ones(raw_v.size(0)), axis=1)
#                     pruned_weight = post_weight_mus[4][torch.from_numpy(t_mask.astype('bool'))].reshape(v.size())
#                 else:
#                     pruned_weight = raw_v
#                     if 'num_batches_tracked' in m:
#                         mask_idx += 1
#             else:
#                 if len(v.size()) > 2:
#                     pruned_weight = post_weight_mus[mask_idx]
#                 else:
#                     pruned_weight = raw_v
#                     if 'num_batches_tracked' in m:
#                         mask_idx += 1
#             pruned_weights[m] = pruned_weight
#         else :
#             pruned_weights[m] = raw_v
#
#     tt = model_pruned.state_dict()
#     return pruned_weights