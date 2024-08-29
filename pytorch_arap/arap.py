import numpy as np
import torch
import torch.nn as nn


def cotmatrix_entries(V, F):

    l2 = squared_edge_lengths(V, F)
    l = torch.sqrt(l2)

    # Double area
    dbla = double_area(l)

    # Compute cotangent of angles
    C0 = (l2[:, 1] + l2[:, 2] - l2[:, 0]) / dbla / 4
    C1 = (l2[:, 2] + l2[:, 0] - l2[:, 1]) / dbla / 4
    C2 = (l2[:, 0] + l2[:, 1] - l2[:, 2]) / dbla / 4

    C = torch.stack([C0, C1, C2]).t()

    return C


def squared_edge_lengths(V, F):
    L0 = V[F[:, 1]] - V[F[:, 2]]
    L1 = V[F[:, 2]] - V[F[:, 0]]
    L2 = V[F[:, 0]] - V[F[:, 1]]

    L0_len = torch.sum(L0 ** 2, dim=1)
    L1_len = torch.sum(L1 ** 2, dim=1)
    L2_len = torch.sum(L2 ** 2, dim=1)

    return torch.stack([L0_len, L1_len, L2_len]).t()


def double_area(l):
    arg = (l[:, 0] + (l[:, 1] + l[:, 2])) * \
          (l[:, 2] - (l[:, 0] - l[:, 1])) * \
          (l[:, 2] + (l[:, 0] - l[:, 1])) * \
          (l[:, 0] + (l[:, 1] - l[:, 2]))
    dbla = torch.sqrt(arg) / 2

    # Replace NaNs with zeros
    dbla[dbla != dbla] = 0.

    return dbla


class ARAPLoss(nn.Module):
    """
    As-Rigid-As-Possible loss function. (Sorkine and Alexa, 2007)
    Only SPOKES_AND_RIMS ARAP loss for 3D triangle mesh is supported.
    """

    def __init__(self, V: np.ndarray, F: np.ndarray, device='cpu'):
        """
        :param V: Vertex (libigl-style)
        :param F: Face (libigl-style)
        """

        super(ARAPLoss, self).__init__()
        self.V_numpy = V
        self.F_numpy = F

        self.V_torch = torch.tensor(V, dtype=torch.float32).to(device)
        self.F_torch = torch.tensor(F, dtype=torch.int64).to(device)

        self.L = cotmatrix_entries(self.V_torch, self.F_torch).to(device)

        # Element list & Edge weight list
        elem_list = [[] for _ in range(self.V_torch.shape[0])]
        weights_list = [[] for _ in range(self.V_torch.shape[0])]

        for T, weights in zip(self.F_torch, self.L):
            for i in range(3):
                e_s = T[i]
                e_t = T[(i + 1) % 3]
                weight = weights[i]

                for j in range(3):
                    elem_list[T[j]].append([e_s, e_t])
                    weights_list[T[j]].append(weight)

        max_elem_count = max([len(e) for e in elem_list])

        self.elem_list_tensor = -torch.ones((self.V_torch.shape[0], max_elem_count, 2), dtype=torch.long).to(device)
        self.weights_list_tensor = torch.zeros((self.V_torch.shape[0], max_elem_count), dtype=torch.float32).to(device)

        for i, (e, w) in enumerate(zip(elem_list, weights_list)):
            self.elem_list_tensor[i, :len(e)] = torch.tensor(e).to(device)
            self.weights_list_tensor[i, :len(w)] = torch.tensor(w).to(device)

        self.elem_list_tensor = self.elem_list_tensor.to(device)
        self.weights_list_tensor = self.weights_list_tensor.to(device)

        self.elem_edge_vecs_rest = self.get_elem_edge_vecs(self.V_torch).to(device)
        self.elem_weights = self.weights_list_tensor.unsqueeze(-1).to(device)

    def get_elem_edge_vecs(self, V):
        e_start = V[self.elem_list_tensor[:, :, 0]]
        e_end = V[self.elem_list_tensor[:, :, 1]]

        return e_end - e_start

    def find_rotation(self, V_deformed):
        elem_edge_vecs_deformed = self.get_elem_edge_vecs(V_deformed)

        elem_edge_vecs_rest = self.elem_edge_vecs_rest
        elem_weights = self.elem_weights

        elem_edge_vecs_rest_weighted = elem_edge_vecs_rest * elem_weights

        elem_S = torch.bmm(elem_edge_vecs_rest_weighted.transpose(-1, -2), elem_edge_vecs_deformed)

        elem_U, _, elem_Vt = torch.linalg.svd(elem_S, full_matrices=False)

        elem_R = torch.bmm(elem_Vt.transpose(-1, -2), elem_U.transpose(-1, -2))

        return elem_R

    def forward(self, V_deformed):
        """
        :param V_deformed: Deformed vertex (libigl-style)
        :return: Loss
        """

        elem_R = self.find_rotation(V_deformed)

        elem_edge_vecs_deformed = self.get_elem_edge_vecs(V_deformed)

        elem_edge_vecs_rest = self.elem_edge_vecs_rest
        elem_weights = self.elem_weights

        elem_edge_vecs_rest_rotated = torch.matmul(elem_R.unsqueeze(1), elem_edge_vecs_rest.unsqueeze(-1)).squeeze(-1)

        elem_diff = elem_edge_vecs_deformed - elem_edge_vecs_rest_rotated

        loss = torch.sum(elem_diff ** 2 * elem_weights)

        return loss
