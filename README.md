# Simple PyTorch As-Rigid-As-Possible Energy

This is a simple PyTorch implementation of the As-Rigid-As-Possible energy (Sorkine and Alexa, 2007).

## Installation

```bash
pip install git+https://github.com/ataga101/simple_pytorch_arap.git
```

## Usage

```python
from pytorch_arap import ARAPLoss

# V: vertices of the mesh (N, 3) (libigl format)
# F: faces of the mesh (M, 3) (libigl format)
# V_deformed: deformed vertices of the mesh (N, 3) (libigl format)
loss_fn = ARAPLoss(V, F)
loss = loss_fn(V_deformed)
```
