import torch
import numpy as np


def _sample_topp(probs):

    # =====  Code from  fairseq/search.py _sample_topp ======

    # sort the last dimension (vocab dimension) in descending order
    sorted_probs, sorted_indices = probs.sort(descending=True)

    # compute a mask to indicate the words to be included in the top-P set.
    cumsum_probs = sorted_probs.cumsum(dim=2)
    mask = cumsum_probs.lt(sampling_topp)

    # note that mask was computed by 'lt'. One more word needs to be included
    # so that the cumulative probability mass can exceed p.
    cumsum_mask = mask.cumsum(dim=2)
    last_included = cumsum_mask[:, :, :1]
    mask = mask.scatter_(2, last_included, 1)

    # truncate unnecessary dims.
    max_dim = last_included.max()
    truncated_mask = mask[:, :, :max_dim + 1]
    truncated_probs = sorted_probs[:, :, :max_dim + 1]
    truncated_indices = sorted_indices[:, :, :max_dim + 1]

    # trim the words that are not in top-P by setting their probabilities
    # to 0, so that they would not be sampled later.
    trim_mask = 1 - truncated_mask
    trimed_probs = truncated_probs.masked_fill_(trim_mask, 0)
    return trimed_probs, truncated_indices

    # ========================================================


if __name__ == '__main__':
    np.random.seed(1234)
    torch.manual_seed(1234)

    sampling_topp = 0.9
    probs = torch.softmax(torch.randn(1, 1, 10), dim=-1)
    # probs = tensor([0.0545, 0.0779, 0.0189, 0.0647, 0.0282, 0.0862, 0.0656, 0.1041, 0.0399, 0.4600])
    print('probs =', probs[0][0])

    trimed_probs, truncated_indices = _sample_topp(probs)

    cum_probs = trimed_probs.cumsum(dim=-1)[0][0]
    # cumsum = tensor([0.4600, 0.5641])
    print('cumsum =', cum_probs)
    # Will throw AssertionError
    assert float(cum_probs[-1]) >= sampling_topp
