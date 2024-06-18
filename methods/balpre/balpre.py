import torch
import torch.nn as nn
import numpy as np
import random
from utils import num_parameters, model_from_dataset, circle_points
from ..base import BaseMethod
from scipy.optimize import fsolve
import scipy.interpolate as spi
import math

def solve(ipo1, pref, n_obj, bias):
    xb, yb = bias[0], bias[1]

    def func1D(paramlist):
        x1 = paramlist[0]
        return [(x1 - xb) - (pref[0] / pref[1]) * (spi.splev(x1, ipo1) - yb)]

    # def func2D(paramlist):
    #     x1, x2, x3 = paramlist[0], paramlist[1], paramlist[2]
    #     return [coef[0]+coef[1]*x1 + coef[2]*x2+coef[3]*x1*x2+coef[4]*x1**2+coef[5]*x2**2-x3,
    #             x1 - (pref[0]/pref[2])*x2,
    #             x2 - (pref[1]/pref[2])*x3]
    if n_obj == 2:
        solved = fsolve(func1D, 0.5)
    # elif n_obj ==3:
    #     solved = fsolve(func2D, np.array([0] * n_obj))
    else:
        raise ValueError('n_obj > 2')
    return solved


class Upsampler(nn.Module):

    def __init__(self, K, child_model, input_dim):
        """
        In case of tabular data: append the sampled rays to the data instances (no upsampling)
        In case of image data: use a transposed CNN for the sampled rays.
        """
        super().__init__()

        if len(input_dim) == 1:
            # tabular data
            self.tabular = True
        elif len(input_dim) == 3:
            # image data
            self.tabular = False
            self.transposed_cnn = nn.Sequential(
                nn.ConvTranspose2d(K, K, kernel_size=4, stride=1, padding=0, bias=False),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(K, K, kernel_size=6, stride=2, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Upsample(input_dim[-2:])
            )
        else:
            raise ValueError(f"Unknown dataset structure, expected 1 or 3 dimensions, got {dim}")

        self.child_model = child_model

    def forward(self, batch):
        x = batch['data']

        b = x.shape[0]
        a = batch['alpha'].repeat(b, 1)

        if not self.tabular:
            # use transposed convolution
            a = a.reshape(b, len(batch['alpha']), 1, 1)
            a = self.transposed_cnn(a)

        x = torch.cat((x, a), dim=1)
        return self.child_model(dict(data=x))

    def private_params(self):
        if hasattr(self.child_model, 'private_params'):
            return self.child_model.private_params()
        else:
            return []


def _solve(weight, n_obj):
    if n_obj == 2:
        def __solve(paramlist):
            x1, x2 = paramlist[0], paramlist[1]
            return [x1 + x2 - 1,
                    x1 * weight[1] - x2 * weight[0]]

        s = fsolve(__solve, weight)
    elif n_obj == 3:
        def ___solve(paramlist):
            x1, x2, x3 = paramlist[0], paramlist[1], paramlist[2]
            return [x1 + x2 + x3 - 1,
                    x1 * weight[2] - x3 * weight[0],
                    x2 * weight[2] - x3 * weight[1]]

        s = fsolve(___solve, np.zeros(1, n_obj).flatten())
    else:
        raise ValueError('n_obj > 2 is not support at present')

    return s


class BALPREMethod(BaseMethod):

    def __init__(self, objectives, alpha, lamda, dim, n_test_rays, **kwargs):
        """
        Args:
            objectives: A list of objectives
            alpha: Dirichlet sampling parameter (list or float)
            lamda: Cosine similarity penalty
            dim: Dimensions of the data
            n_test_rays: The number of test rays used for evaluation.
        """
        self.objectives = objectives
        self.K = len(objectives)
        self.alpha = alpha
        self.n_test_rays = n_test_rays
        self.lamda = lamda

        dim = list(dim)
        dim[0] = dim[0] + self.K

        model = model_from_dataset(method='cosmos', dim=dim, **kwargs)
        self.model = Upsampler(self.K, model, dim).cuda()

        self.n_params = num_parameters(self.model)
        print("Number of parameters: {}".format(self.n_params))

    def step(self, batch, GP=None, cricle=None, alpha=None, effect=False, rate=None, cos=[1, 1], ms=[1, 1], FS=None):
        # step 1: sample alphas
        if isinstance(self.alpha, list):
            prefer_v = np.random.dirichlet(self.alpha, 1).astype(np.float32).flatten()

        elif self.alpha > 0:
            prefer_v = np.random.dirichlet([self.alpha for _ in range(self.K)], 1).astype(np.float32).flatten()

        else:
            raise ValueError(f"Unknown value for alpha: {self.alpha}, expecting list or float.")
        # Balancing preference vector
        prob = random.random()
        prefer_v[0] = math.floor(10 * prefer_v[0]) / 10
        prefer_v[1] = math.ceil(10 * prefer_v[1]) / 10
        if GP is not None and prob > 0.2:
            prefer_v = np.random.dirichlet([ms[1], ms[0]], 1).astype(np.float32).flatten()
            prefer_v[0] = math.floor(10 * prefer_v[0]) / 10
            prefer_v[1] = math.ceil(10 * prefer_v[1]) / 10
            # print('Before transform:', prefer_v)
            bias_optimal_point = []
            for obj in range(self.K):
                bias_optimal_point.append(GP[obj].predict(prefer_v[obj].reshape(-1, 1))[0])
            prefer_v = np.array(bias_optimal_point)
            prefer_v /= prefer_v.sum()
            # print('After transforms', prefer_v)
        batch['alpha'] = torch.from_numpy(prefer_v.astype(np.float32)).cuda()

        # step 2: calculate loss
        self.model.zero_grad()
        logits = self.model(batch)
        batch.update(logits)
        loss_total = None
        task_losses = []

        for a, objective in zip(batch['alpha'], self.objectives):
            task_loss = objective(**batch)
            loss_total = a * task_loss if not loss_total else loss_total + a * task_loss
            task_losses.append(task_loss)
        cossim = torch.nn.functional.cosine_similarity(torch.stack(task_losses), batch['alpha'], dim=0)
        loss_total -= self.lamda * cossim

        loss_total.backward()
        return loss_total.item(), cossim.item()

    def eval_step(self, batch, test_rays=None, GP=None):
        self.model.eval()
        logits = []
        logits_t = []
        return_rays = []
        trans_rays = []
        with torch.no_grad():
            if test_rays is None:
                test_rays = circle_points(self.n_test_rays, dim=self.K)

            for ray in test_rays:
                ray /= sum(ray)
                return_rays.append(ray)
                if GP is not None:
                    transform_ray = []
                    for obj in range(self.K):
                        transform_ray.append(GP[obj].predict(ray[obj].reshape(1, -1))[0])
                    ray2 = np.array(transform_ray)
                    ray2 /= sum(ray2)
                    trans_rays.append(ray2)
                    ray2 = torch.from_numpy(ray2.astype(np.float32)).cuda()
                    batch['alpha'] = ray2
                    logit_t = self.model(batch)
                    logits_t.append(logit_t)

                ray = torch.from_numpy(ray.astype(np.float32)).cuda()
                batch['alpha'] = ray
                logit = self.model(batch)
                logits.append(logit)
        if GP is not None:
            return logits, logits_t, np.array(return_rays), trans_rays
        else:
            return logits, np.array(return_rays)

