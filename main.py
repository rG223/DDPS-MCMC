import torch
import random
import numpy as np
import copy
import argparse
import os
import pathlib
import time
import json
from torch.utils import data
import settings as s
import utils
from objectives import from_name
from hv import HyperVolume
from utils import circle_points, dot_product_angle
import scipy.optimize as optimize
from methods import HypernetMethod, ParetoMTLMethod, SingleTaskMethod, COSMOSMethod, MGDAMethod, UniformScalingMethod, \
    BALPREMethod, _solve, BalHypernetMethod
from scores import from_objectives
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.optimize import fsolve
from fastsort import fast_non_dominated_sort
import math
import scipy.interpolate as spi
import pandas as pd
import scipy

# seed now to be save and overwrite later
np.random.seed(1)
random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
torch.cuda.manual_seed_all(1)


def solve(ipo1, pref, n_obj, bias):
    xb, yb = bias[0], bias[1]

    def func1D(paramlist):
        x1 = paramlist[0]
        return [(x1 - xb) - (pref[0] / pref[1]) * (spi.splev(x1, ipo1) - yb)]
    if n_obj == 2:
        solved = fsolve(func1D, 0)
    # elif n_obj ==3:
    #     solved = fsolve(func2D, np.array([0] * n_obj))
    else:
        raise ValueError('n_obj > 2')
    return solved


# fitting function exp


def target_func(x, a0, a1, a2):
    return a0 * np.exp(-x / a1) + a2


def traget_inver(x, a0, a1, a2):
    return -(a1 * np.log((x - a2) / a0))


def method_from_name(method, **kwargs):
    if method == 'ParetoMTL':
        return ParetoMTLMethod(**kwargs)
    elif method == 'balpre':
        return BALPREMethod(**kwargs)
    elif 'balhyper' in method:
        return BalHypernetMethod(**kwargs)
    elif 'cosmos' in method:
        return COSMOSMethod(**kwargs)
    elif method == 'SingleTask':
        return SingleTaskMethod(**kwargs)
    elif 'hyper' in method:
        return HypernetMethod(**kwargs)
    elif method == 'mgda':
        return MGDAMethod(**kwargs)
    elif method == 'uniform':
        return UniformScalingMethod(**kwargs)
    else:
        raise ValueError("Unkown method {}".format(method))


epoch_max = -1
volume_max = -1
elapsed_time = 0
test_volume = -1
used_value = None
best_point = None

def evaluate(j, e, method, scores, data_loader, logdir, reference_point, split, result_dict, settings, GP=None):
    assert split in ['train', 'val', 'test']
    global volume_max
    global epoch_max
    global test_volume
    global best_point
    global used_value
    score_values = np.array([])
    score_values_t = np.array([])
    test_rays = circle_points(15, dim=method.K)
    for batch in data_loader:
        batch = utils.dict_to_cuda(batch)
        if GP is not None:
            batch_t = copy.deepcopy(batch)
        # more than one solution for some solvers
        s = []
        st = []
        if GP is not None and 'bal' in settings['method']:
            output, output_t, test_rays, trans_rays = method.eval_step(batch=batch, GP=GP)
        elif settings['method'] == 'SingleTask':
            output = method.eval_step(batch=batch)
        else:
            output, test_rays = method.eval_step(batch=batch)
        if GP is None:
            for l in output:
                batch.update(l)
                s.append([s(**batch) for s in scores])
        else:
            for l, l_t in zip(output, output_t):
                batch.update(l)
                s.append([s(**batch) for s in scores])
                batch_t.update(l_t)
                st.append([s(**batch_t) for s in scores])
        if score_values.size == 0 and GP is None:
            score_values = np.array(s)
        elif score_values.size == 0 and GP is not None:
            score_values = np.array(s)
            score_values_t = np.array(st)
        elif score_values.size != 0 and GP is None:
            score_values += np.array(s)
        else:
            score_values += np.array(s)
            score_values_t += np.array(st)
    bias_point = np.zeros((1, method.K))
    cos = [1, 1]
    score_values /= len(data_loader)
    score_values_t /= len(data_loader)
    hv = HyperVolume(reference_point)
    # load gauss model
    # fit score
    effect = True
    lengt = 0
    alpha = [1, 1]
    if split == 'val' and 'bal' in settings['method']:
        if e < settings['eval_every'] and GP is None:
            used_value = score_values
        elif e >= settings['eval_every'] and GP is None:
            used_value = np.concatenate([used_value, score_values], axis=0)
        else:
            used_value = np.concatenate([used_value, score_values_t, score_values], axis=0)

        rank, F = fast_non_dominated_sort(used_value)
        guass_data = []
        f_number = 0
        front = F[0]
        if len(front) / len(test_rays) < 1 / 3:
            effect = False
            # GP = None
        while len(front) < 3:
            f_number += 1
            front += F[f_number]
        FS = np.array([used_value[f, :] for f in front])
        Mean_score = np.mean(FS, axis=0)
        FS = FS[FS[:, 0].argsort()]
        ipo1 = spi.splrep(np.array(FS[:, 0]), np.array(FS[:, 1]), k=1)

        # calculate score
        mean_x = pd.Series(sorted(FS[:, 0])).diff(1).dropna().mean()
        S_0 = np.array([used_value[f, :] for f in F[0]])
        print('f_number:', f_number)
        print('rank:', rank)
        print('S0:', S_0)
        print('MS:', Mean_score)
        lengt = np.sqrt(
            (0.2 * (np.max(FS[:, 1]) - np.min(FS[:, 1])))**2+ (0.2 *
                (np.max(FS[:, 0]) - np.min(FS[:, 0])))**2
        )

        next_v = np.array([np.min(FS[:, 0]), np.min(FS[:, 1])])
        w = 1 - (e/15) * 0.6
        if e > settings['eval_every']-1 and 'best_point' in result_dict['start_0'][list(result_dict['start_0'].keys())[-1]].keys() and settings['PSO']:
            best_v = result_dict['start_0'][list(result_dict['start_0'].keys())[-1]]['best_point']
            first = random.random()
            if e > 2*(settings['eval_every'] - 1)+1:
                print('111')
                second = random.random()
                last_v = result_dict['start_0'][list(result_dict['start_0'].keys())[-2]]['point']
                last_len = result_dict['start_0'][list(result_dict['start_0'].keys())[-2]]['length']
                next_direction = second * np.array(last_v) + first*np.array(best_v) + w * next_v
                next_direction /= next_direction.sum()
                next_len = last_len * 0.5 + lengt * 0.5
                Tan = next_direction[1] / next_direction[0]
                bias_point[0][0] = np.min(FS[:, 0]) - (np.sqrt(1/(1+Tan**2)))*next_len
                bias_point[0][1] = np.min(FS[:, 1]) - (Tan/math.sqrt(Tan**2+1))*next_len
            else:
                print('222')
                best_len = result_dict['start_0'][list(result_dict['start_0'].keys())[-1]]['length']
                next_len = best_len * 0.5 + lengt * 0.5
                next_direction = w * next_v + first*np.array(best_v)
                next_direction /= next_direction.sum()
                Tan = next_direction[1] / next_direction[0]
                bias_point[0][0] = np.min(FS[:, 0]) - (np.sqrt(1/(1+Tan**2)))*next_len
                bias_point[0][1] = np.min(FS[:, 1]) - (Tan/math.sqrt(Tan**2+1))*next_len
            print('direction:', next_direction)
            print('next_len:', next_len)
        else:
            print('333')
            bias_point[0][0] = np.min(FS[:, 0]) - 0.2 * (np.max(FS[:, 1]) - np.min(FS[:, 1]))
            bias_point[0][1] = np.min(FS[:, 1]) - 0.2 * (np.max(FS[:, 0]) - np.min(FS[:, 0]))

        print(bias_point)
        bias_point[0][0] = max(bias_point[0][0], 0)
        bias_point[0][1] = max(bias_point[0][1], 0)
        # print(bias_point)
    if split == 'val' and 'bal' in settings['method']:
        for r in range(len(test_rays)):
            S_x = solve(ipo1, test_rays[r], method.K, bias_point.flatten())
            S_y = spi.splev(S_x, ipo1)
            S = [S_x[0], S_y[0]]
            # if 'bal' in settings['method']:
            S = [s / sum(S) for s in S]
            guass_data.append(S)

    if 'bal' in settings['method'] and split == 'val':
        gpr = []
        guass_data = np.array(guass_data)
        if effect:
            for n in range(method.K):
                gpr1 = GaussianProcessRegressor(random_state=0).fit(test_rays[:, n].reshape(-1, 1), guass_data[:, n])
                gpr.append(gpr1)
                print('Score:', gpr1.score(test_rays[:, n].reshape(-1, 1), guass_data[:, n]))
        else:
            gpr = GP
        print('test_rays:', test_rays)
        print('transform_rays:', guass_data)
    # Computing hyper-volume for many objectives is expensive

    if GP is None:
        chose_value = score_values
        rays = test_rays
    else:
        rays = trans_rays
        chose_value = score_values_t

    KL = scipy.stats.entropy(np.sum(score_values * rays, axis=1) / np.sum(score_values * rays, axis=1).sum(),
                             (np.ones((1, len(test_rays))) / method.K).flatten())
    volume = hv.compute(chose_value) if chose_value.shape[1] < 5 else -1
    if len(scores) == 2:
        pareto_front = utils.ParetoFront([s.__class__.__name__ for s in scores], logdir, "{}_{:03d}".format(split, e))
        pareto_front.append(chose_value)
        pareto_front.plot()
    print('Unif.: ', 1 - KL)
    last_score = (1 - KL) * volume
    result = {
        "scores": chose_value.tolist(),
        "hv": volume,
        'Unif': 1 - KL,
        'last_score': last_score,
    }

    if split == 'val':
        if volume > volume_max:
            volume_max = volume
            epoch_max = e
            best_point = bias_point[0]
        result.update({
            "max_epoch_so_far": epoch_max,
            "max_volume_so_far": volume_max,
            "training_time_so_far": elapsed_time,
            "best_point": best_point.tolist(),
            "point": bias_point[0].tolist(),
            'length': lengt,
        })
    elif split == 'test':
        result.update({
            "training_time_so_far": elapsed_time,
        })
        if volume > test_volume:
            test_volume = volume
            print('the Max Testing Set Volume:{}'.format(volume))
        else:
            print('this epoch is not best volume, best volume is{}'.format(test_volume))

    result.update(method.log())

    if f"epoch_{e}" in result_dict[f"start_{j}"]:
        result_dict[f"start_{j}"][f"epoch_{e}"].update(result)
    else:
        result_dict[f"start_{j}"][f"epoch_{e}"] = result

    with open(pathlib.Path(logdir) / f"{split}_results.json", "w") as file:
        json.dump(result_dict, file)

    if 'bal' in settings['method'] and split == 'val':
        return gpr, result_dict, effect, len(front) / len(test_rays), cos, alpha, Mean_score, FS
    else:
        return None, result_dict, effect, None, None, alpha, None, None


def main(settings):
    print("start processig with settings", settings)
    utils.set_seed(settings['seed'])
    GP = None
    global elapsed_time
    effect = False
    rate = None
    FS = None
    cos = [1, 1]
    alpha = [1, 1]
    ms = [1, 1]
    # create the experiment folders
    logdir = os.path.join(settings['logdir'], settings['method'], settings['dataset'], utils.get_runname(settings))
    pathlib.Path(logdir).mkdir(parents=True, exist_ok=True)

    # prepare
    train_set = utils.dataset_from_name(split='train', **settings)
    val_set = utils.dataset_from_name(split='val', **settings)
    test_set = utils.dataset_from_name(split='test', **settings)

    train_loader = data.DataLoader(train_set, settings['batch_size'], shuffle=False,
                                   num_workers=settings['num_workers'])
    val_loader = data.DataLoader(val_set, settings['batch_size'], shuffle=False, num_workers=settings['num_workers'])
    test_loader = data.DataLoader(test_set, settings['batch_size'], settings['num_workers'])

    objectives = from_name(settings.pop('objectives'), train_set.task_names())
    scores = from_objectives(objectives)

    rm1 = utils.RunningMean(400)
    rm2 = utils.RunningMean(400)
    method = method_from_name(objectives=objectives, **settings)

    train_results = dict(settings=settings, num_parameters=utils.num_parameters(method.model_params()))
    val_results = dict(settings=settings, num_parameters=utils.num_parameters(method.model_params()))
    test_results = dict(settings=settings, num_parameters=utils.num_parameters(method.model_params()))

    with open(pathlib.Path(logdir) / "settings.json", "w") as file:
        json.dump(train_results, file)

    # main
    for j in range(settings['num_starts']):
        train_results[f"start_{j}"] = {}
        val_results[f"start_{j}"] = {}
        test_results[f"start_{j}"] = {}

        optimizer = torch.optim.Adam(method.model_params(), settings['lr'])
        if settings['use_scheduler']:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, settings['scheduler_milestones'],
                                                             gamma=settings['scheduler_gamma'])

        for e in range(settings['epochs']):
            print(f"Epoch {e}")
            tick = time.time()
            method.new_epoch(e)
            for b, batch in enumerate(train_loader):
                batch = utils.dict_to_cuda(batch)
                optimizer.zero_grad()
                if 'bal' in settings['method']:
                    stats = method.step(batch=batch, GP=GP, alpha=alpha, effect=effect, cos=cos, rate=rate, ms=ms, FS=FS)
                else:
                    stats = method.step(batch)
                optimizer.step()

                loss, sim = stats if isinstance(stats, tuple) else (stats, 0)
                if b % 50 == 0:
                    print(
                        "Epoch {:03d}, batch {:03d}, train_loss {:.4f}, sim {:.4f}, rm train_loss {:.3f}, rm sim {:.3f}".format(
                            e, b, loss, sim, rm1(loss), rm2(sim)))

            tock = time.time()
            elapsed_time += (tock - tick)

            if settings['use_scheduler']:
                val_results[f"start_{j}"][f"epoch_{e}"] = {'lr': scheduler.get_last_lr()[0]}
                scheduler.step()

            # run eval on train set (mainly for debugging)
            if settings['train_eval_every'] > 0 and (e + 1) % settings['train_eval_every'] == 0:
                train_results = evaluate(j, e, method, scores, train_loader, logdir,
                                         reference_point=settings['reference_point'],
                                         split='train',
                                         result_dict=train_results,
                                         settings=settings)

            if settings['eval_every'] > 0 and (e + 1) % settings['eval_every'] == 0:
                # Validation results
                GP, val_results, effect, rate, cos, alpha, ms, FS = evaluate(j, e, method, scores, val_loader, logdir,
                                                                         reference_point=settings['reference_point'],
                                                                         split='val',
                                                                         result_dict=val_results,
                                                                         settings=settings,
                                                                         GP=GP)
                # Test results
                # if effect:
                if e < settings['eval_every']:
                    _, test_results, _, _, _, _, _,_ = evaluate(j, e, method, scores, test_loader, logdir,
                                                              reference_point=settings['reference_point'],
                                                              split='test',
                                                              result_dict=test_results,
                                                              settings=settings,
                                                              GP=None)
                else:
                    _, test_results, _, _, _, _, _, _ = evaluate(j, e, method, scores, test_loader, logdir,
                                                              reference_point=settings['reference_point'],
                                                              split='test',
                                                              result_dict=test_results,
                                                              settings=settings,
                                                              GP=GP)

            # Checkpoints
            if settings['checkpoint_every'] > 0 and (e + 1) % settings['checkpoint_every'] == 0:
                pathlib.Path(os.path.join(logdir, 'checkpoints')).mkdir(parents=True, exist_ok=True)
                torch.save(method.model.state_dict(),
                           os.path.join(logdir, 'checkpoints', 'c_{}-{:03d}.pth'.format(j, e)))

        print("epoch_max={}, val_volume_max={},test_volume_max".format(epoch_max, volume_max, test_volume))
        pathlib.Path(os.path.join(logdir, 'checkpoints')).mkdir(parents=True, exist_ok=True)
        torch.save(method.model.state_dict(), os.path.join(logdir, 'checkpoints', 'c_{}-{:03d}.pth'.format(j, 999999)))

    return volume_max


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', default='mm', help="The dataset to run on.")
    parser.add_argument('--method', '-m', default='cosmos', help="The method to generate the Pareto front.")
    parser.add_argument('--seed', '-s', default=1, type=int, help="Seed")
    parser.add_argument('--task_id', '-t', default=None, type=int,
                        help='Task id to run single task in parallel. If not set then sequentially.')
    args = parser.parse_args()

    settings = s.generic

    if args.method == 'single_task':
        settings.update(s.SingleTaskSolver)
        if args.task_id is not None:
            settings['num_starts'] = 1
            settings['task_id'] = args.task_id
    elif args.method == 'balpre':
        settings.update(s.balpre)
    elif args.method == 'cosmos':
        settings.update(s.cosmos)
    elif args.method == 'hyper_ln':
        settings.update(s.hyperSolver_ln)
    elif args.method == 'balhyper_ln':
        settings.update(s.balhyperSolver_ln)
    elif args.method == 'hyper_epo':
        settings.update(s.hyperSolver_epo)
    elif args.method == 'balhyper_epo':
        settings.update(s.balhyperSolver_epo)
    elif args.method == 'pmtl':
        settings.update(s.paretoMTL)
    elif args.method == 'mgda':
        settings.update(s.mgda)
    elif args.method == 'uniform':
        settings.update(s.uniform_scaling)

    if args.dataset == 'mm':
        settings.update(s.multi_mnist)
    elif args.dataset == 'adult':
        settings.update(s.adult)
    elif args.dataset == 'mfm':
        settings.update(s.multi_fashion_mnist)
    elif args.dataset == 'fm':
        settings.update(s.multi_fashion)
    elif args.dataset == 'credit':
        settings.update(s.credit)
    elif args.dataset == 'compass':
        settings.update(s.compass)
    elif args.dataset == 'celeba':
        settings.update(s.celeba)

    settings['seed'] = args.seed

    return settings


if __name__ == "__main__":
    settings = parse_args()
    main(settings)
