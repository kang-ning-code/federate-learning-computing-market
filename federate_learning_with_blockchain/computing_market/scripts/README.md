scripts to start cluster simulator
```python
# activate env
conda activate deeplearning_common
# use console
brownie console

# import ClusterSimualtor
from scripts.simulator import ClusterSimulator
cs = ClusterSimulator()
cm = ComputingMarket[0]
# access specific client in cluster
c0 = cs.clients[0]
c0i = c0.invoker
c0t = c0.trainer
c0ip = c0.ipfs
c1 = cs.clients[1]
c1i = c1.invoker
c1t = c1.trainer
c1ip = c1.ipfs
# start 
```