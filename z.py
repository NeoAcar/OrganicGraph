from models import GATModel
from dataset import MeltingPointDataset

mdl = GATModel(node_in_dim=5, edge_in_dim=5)
print(mdl)
