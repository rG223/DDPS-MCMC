import utils
import numpy as np


def test_volume(data_loader, method, GP, scores, hv, rotation):
    score_values = np.array([])
    s = []
    for batch in data_loader:
        batch = utils.dict_to_cuda(batch)
        _, output, test_rays = method.eval_step(batch=batch, GP=GP, rotation=rotation)
        for l in output:
            batch.update(l)
            s.append([s(**batch) for s in scores])
        if score_values.size == 0:
            score_values = np.array(s)
        else:
            score_values += np.array(s)
    volume = hv.compute(score_values) if score_values.shape[1] < 5 else -1
    return volume