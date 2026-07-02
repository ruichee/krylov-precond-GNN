import numpy as np
from krylov.pcg import preconditioned_conjugate_gradient

import torch
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree\


class MessagePassingLayers(MessagePassing):
    pass


