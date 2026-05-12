import torch
import numpy as np
from torch_geometric.data import Batch

from gcn_utils import (
    create_pytorch_geometric_graph_data_list_from_smiles_and_labels
)

from mpnn_utils import mol_to_graph


def predict_boiling_point(model, smiles_list, device='cpu', mode='gcn'):
    model.eval()

    if mode == 'gcn':

        data_list = (
            create_pytorch_geometric_graph_data_list_from_smiles_and_labels(
                smiles_list,
                [0.0] * len(smiles_list)
            )
        )

    elif mode == 'mpnn':

        data_list = [
            mol_to_graph(smi, 0.0)
            for smi in smiles_list
        ]

        # Remove invalid molecules
        data_list = [d for d in data_list if d is not None]

    else:
        raise ValueError("Unsupported mode")

    batch = Batch.from_data_list(data_list).to(device)

    with torch.no_grad():
        predictions = model(batch)

    predictions = (
        predictions
        .cpu()
        .numpy()
        .flatten()
    )

    return np.round(predictions, 3).tolist()
