import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import math
from utils import num_parameters, circle_points
from ..base import BaseMethod

from .models import PHNHyper, PHNTarget
from .solvers import LinearScalarizationSolver, EPOSolver


class LeNetPHNHyper(PHNHyper):
    pass


class LeNetPHNTargetWrapper(PHNTarget):

    def forward(self, x, weights=None):
        logits = super().forward(x, weights)
        return dict(logits_2=logits[0], logits_4=logits[1])


# fully connected hyper version
# this is unfortunately not published, therefore I implemented it myself
class FCPHNHyper(nn.Module):
    
    def __init__(self, dim, ray_hidden_dim=100, n_tasks=2):
        super().__init__()

        self.feature_dim = dim[0]

        self.ray_mlp = nn.Sequential(
            nn.Linear(n_tasks, ray_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, ray_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, ray_hidden_dim),
        )

        self.fc_0_weights = nn.Linear(ray_hidden_dim, 60*self.feature_dim)
        self.fc_0_bias = nn.Linear(ray_hidden_dim, 60)
        self.fc_1_weights = nn.Linear(ray_hidden_dim, 25*60)
        self.fc_1_bias = nn.Linear(ray_hidden_dim, 25)
        self.fc_2_weights = nn.Linear(ray_hidden_dim, 1*25)
        self.fc_2_bias = nn.Linear(ray_hidden_dim, 1)

    def forward(self, ray):
        x = self.ray_mlp(ray)
        out_dict = {
            'fc0.weights': self.fc_0_weights(x).reshape(60, self.feature_dim),
            'fc0.bias': self.fc_0_bias(x),
            'fc1.weights': self.fc_1_weights(x).reshape(25, 60),
            'fc1.bias': self.fc_1_bias(x),
            'fc2.weights': self.fc_2_weights(x).reshape(1, 25),
            'fc2.bias': self.fc_2_bias(x),
        }
        return out_dict


class FCPHNTarget(nn.Module):

    def forward(self, x, weights):
        x = F.linear(
            x,
            weight=weights['fc0.weights'],
            bias=weights['fc0.bias']
        )
        x = F.relu(x)
        x = F.linear(
            x,
            weight=weights['fc1.weights'],
            bias=weights['fc1.bias']
        )
        x = F.relu(x)
        x = F.linear(
            x,
            weight=weights['fc2.weights'],
            bias=weights['fc2.bias']
        )
        return {'logits': x}


class BalHypernetMethod(BaseMethod):

    def __init__(self, objectives, dim, n_test_rays, alpha, internal_solver, **kwargs):
        self.objectives = objectives
        self.n_test_rays = n_test_rays
        self.alpha = alpha
        self.K = len(objectives)

        if len(dim) == 1:
            # tabular
            hnet = FCPHNHyper(dim, ray_hidden_dim=100)
            net = FCPHNTarget()
        elif len(dim) ==3:
            # image
            hnet: nn.Module = LeNetPHNHyper([7, 3], ray_hidden_dim=100)
            net: nn.Module = LeNetPHNTargetWrapper([7, 3])
        else:
            raise ValueError(f"Unkown dim {dim}, expected len 1 or len 3")

        print("Number of parameters: {}".format(num_parameters(hnet)))

        self.model = hnet.cuda()
        self.net = net.cuda()


        if internal_solver == 'linear':
            self.solver = LinearScalarizationSolver(n_tasks=len(objectives))
        elif internal_solver == 'epo':
            self.solver = EPOSolver(n_tasks=len(objectives), n_params=num_parameters(hnet))


    def step(self, batch, GP=None, cricle=None, alpha=None, effect=False, rate=None, cos=[1, 1], ms=[1, 1], FS=None):
        if self.alpha > 0:
            ray = np.random.dirichlet([self.alpha for _ in range(len(self.objectives))], 1).astype(np.float32).flatten()

        else:
            alpha = torch.empty(1, ).uniform_(0., 1.)
            ray = np.array([alpha.item(), 1 - alpha.item()])
        prob = random.random()
        ray[0] = math.floor(10 * ray[0]) / 10
        ray[1] = math.ceil(10 * ray[1]) / 10
        if GP is not None and prob > 0.5:
            ray = np.random.dirichlet([ms[1], ms[0]], 1).astype(np.float32).flatten()
            ray[0] = math.floor(10 * ray[0]) / 10
            ray[1] = math.ceil(10 * ray[1]) / 10
            # print('Before transform:', prefer_v)
            bias_optimal_point = []
            for obj in range(self.K):
                bias_optimal_point.append(GP[obj].predict(ray[obj].reshape(-1, 1))[0])
            ray = np.array(bias_optimal_point)
            ray /= ray.sum()
            # print('After transforms', prefer_v)
        ray = torch.from_numpy(ray.astype(np.float32)).cuda()
        img = batch['data']
        # print(self.model)
        # print(self.net)

        weights = self.model(ray)

        batch.update(self.net(img, weights))
        losses = torch.stack([o(**batch) for o in self.objectives])
        cossim = torch.nn.functional.cosine_similarity(losses, ray, dim=0)
        ray = ray.squeeze(0)
        loss = self.solver(losses, ray, list(self.model.parameters()))

        loss -= 3 * cossim
        loss.backward()

        return loss.item()
    
    def eval_step(self, batch, GP=None):
        self.model.eval()
        return_rays = []
        test_rays = circle_points(self.n_test_rays, dim=self.K)
        logits_t = []
        logits = []
        trans_rays = []
        for ray in test_rays:
            ray /= ray.sum()
            return_rays.append(ray)
            if GP is not None:
                transform_ray = []
                for obj in range(self.K):
                    transform_ray.append(GP[obj].predict(ray[obj].reshape(1, -1))[0])
                ray2 = np.array(transform_ray)
                ray2 /= sum(ray2)
                trans_rays.append(ray2)
                ray2 = torch.from_numpy(ray2.astype(np.float32)).cuda()
                weights = self.model(ray2)
                logit_t = self.net(batch['data'], weights)
                logits_t.append(logit_t)

            ray = torch.from_numpy(ray.astype(np.float32)).cuda()
            weights = self.model(ray)
            logits.append(self.net(batch['data'], weights))
        if GP is not None:
            return logits, logits_t, np.array(return_rays), trans_rays
        else:
            return logits, np.array(return_rays)




