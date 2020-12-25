import os.path as osp
from datetime import datetime

import argparse
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Flickr
from torch_geometric.data import GraphSAINTRandomWalkSampler
from torch_geometric.nn import GraphConv
from torch_geometric.utils import degree

class Net(torch.nn.Module):
    def __init__(self, hidden_channels, in_channels, out_channels):
        super(Net, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = GraphConv(in_channels, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.conv3 = GraphConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(3 * hidden_channels, out_channels)

    def set_aggr(self, aggr):
        self.conv1.aggr = aggr
        self.conv2.aggr = aggr
        self.conv3.aggr = aggr

    def forward(self, x0, edge_index, edge_weight=None):
        x1 = F.relu(self.conv1(x0, edge_index, edge_weight))
        x1 = F.dropout(x1, p=0.2, training=self.training)
        x2 = F.relu(self.conv2(x1, edge_index, edge_weight))
        x2 = F.dropout(x2, p=0.2, training=self.training)
        x3 = F.relu(self.conv3(x2, edge_index, edge_weight))
        x3 = F.dropout(x3, p=0.2, training=self.training)
        x = torch.cat([x1, x2, x3], dim=-1)
        x = self.lin(x)
        return x.log_softmax(dim=-1)


def train(model, device, optimizer, loader, use_normalization):
    model.train()
    # model.set_aggr('add' if args.use_normalization else 'mean')
    model.set_aggr('mean')

    total_loss = total_examples = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()

        if use_normalization:
            edge_weight = data.edge_norm * data.edge_weight
            out = model(data.x, data.edge_index, edge_weight)
            loss = F.nll_loss(out, data.y, reduction='none')
            loss = (loss * data.node_norm)[data.train_mask].sum()
        else:
            out = model(data.x, data.edge_index)
            loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])

        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_nodes
        total_examples += data.num_nodes
    return total_loss / total_examples


@torch.no_grad()
def test(model, data, device):
    model.eval()
    model.set_aggr('mean')

    out = model(data.x.to(device), data.edge_index.to(device))
    pred = out.argmax(dim=-1)
    correct = pred.eq(data.y.to(device))

    accs = []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        accs.append(correct[mask].sum().item() / mask.sum().item())
    return accs
    
def main():
    parser = argparse.ArgumentParser(description='Flicker (Graph-SAINT) - Single GPU')
    parser.add_argument('--use_normalization', action='store_true')
    parser.add_argument('--batch_size', type=int, default=6000)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--dataset_path', type=str, default="/fs/class-projects/fall2020/cmsc498p/c498p000/data/Flickr")
    parser.add_argument('--log_path', type=str, default="graph_saint_single.txt")
    parser.add_argument('--addr', type=str, default='localhost')
    parser.add_argument('--port', type=str, default='12358')
    args = parser.parse_args()
    print(args)

    path = osp.join(args.dataset_path)
    dataset = Flickr(path)
    print("Done loading.")

    data = dataset[0]
    row, col = data.edge_index
    data.edge_weight = 1. / degree(col, data.num_nodes)[col]  # Norm by in-degree.

    loader = GraphSAINTRandomWalkSampler(data, batch_size=args.batch_size, walk_length=2,
                                        num_steps=5, sample_coverage=100,
                                        save_dir=dataset.processed_dir,
                                        num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(hidden_channels=256, in_channels=dataset.num_node_features, out_channels=dataset.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    f = open(args.log_path, 'w')

    for epoch in range(1, args.epochs + 1):
        start_epoch = datetime.now()
        loss = train(model, device, optimizer, loader, args.use_normalization)
        end_epoch = datetime.now()

        train_acc, val_acc, test_acc = test(model, data, device)
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}', ",\tTime: ", str(end_epoch - start_epoch), f', Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')  
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}', ",\tTime: ", str(end_epoch - start_epoch), f', Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}', file=f)  
        f.flush()
        
    f.close()

if __name__ == '__main__':
    main()