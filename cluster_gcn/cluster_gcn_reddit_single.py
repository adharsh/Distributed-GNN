import argparse

from datetime import datetime
import os.path as osp

import torch
import torch.nn.functional as F
from torch.nn import ModuleList
from tqdm import tqdm
from torch_geometric.datasets import Reddit
from torch_geometric.data import ClusterData, ClusterLoader, NeighborSampler
from torch_geometric.nn import SAGEConv

class Net(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Net, self).__init__()
        self.convs = ModuleList(
            [SAGEConv(in_channels, 128),
             SAGEConv(128, out_channels)])

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return F.log_softmax(x, dim=-1)

    def inference(self, x_all, subgraph_loader, device):
        pbar = tqdm(total=x_all.size(0) * len(self.convs))
        pbar.set_description('Evaluating')

        for i, conv in enumerate(self.convs):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = conv((x, x_target), edge_index)
                if i != len(self.convs) - 1:
                    x = F.relu(x)
                xs.append(x.cpu())

                pbar.update(batch_size)

            x_all = torch.cat(xs, dim=0)

        pbar.close()

        return x_all

def train(model, train_loader, device, optimizer):
    model.train()

    total_loss = total_nodes = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index)
        loss = F.nll_loss(out[batch.train_mask], batch.y[batch.train_mask])
        loss.backward()
        optimizer.step()

        nodes = batch.train_mask.sum().item()
        total_loss += loss.item() * nodes
        total_nodes += nodes

    return total_loss / total_nodes


@torch.no_grad()
def test(model, data, subgraph_loader, device):  # Inference should be performed on the full graph.
    model.eval()
    
    out = model.inference(data.x, subgraph_loader, device)
    y_pred = out.argmax(dim=-1)

    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        correct = y_pred[mask].eq(data.y[mask]).sum().item()
        accs.append(correct / mask.sum().item())
    return accs

def main():
    parser = argparse.ArgumentParser(description='Reddit (Cluster-GCN) - Single GPU')
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--eval_steps', type=int, default=10)
    parser.add_argument('--eval_enable', type=bool, default=True)    
    parser.add_argument('--dataset_path', type=str, default="/fs/class-projects/fall2020/cmsc498p/c498p000/data/Reddit")
    parser.add_argument('--log_path', type=str, default="cluster_gcn_reddit_single.txt")
    args = parser.parse_args()
    print(args)

    path = osp.join(args.dataset_path)
    dataset = Reddit(path)
    print("Done loading.")

    data = dataset[0]

    cluster_data = ClusterData(data, num_parts=1500, recursive=False,
                            save_dir=dataset.processed_dir)
    train_loader = ClusterLoader(cluster_data, batch_size=args.batch_size, shuffle=True,
                                num_workers=12)

    subgraph_loader = NeighborSampler(data.edge_index, sizes=[-1], batch_size=1024//8,
                                    shuffle=False, num_workers=12)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(dataset.num_features, dataset.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    f = open(args.log_path, 'w')
    for epoch in range(1, args.epochs + 1):
        start_epoch = datetime.now()
        loss = train(model, train_loader, device, optimizer)
        end_epoch = datetime.now()

        if epoch%args.eval_steps == 0 and args.eval_enable:
            train_acc, val_acc, test_acc = test(model, data, subgraph_loader, device)
            print(f'Epoch: {epoch:02d}, Loss: {loss:.4f},', "\tTime: ", str(end_epoch - start_epoch), f', Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')  
            print(f'Epoch: {epoch:02d}, Loss: {loss:.4f},', "\tTime: ", str(end_epoch - start_epoch), f', Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}', file=f)  
            f.flush()
        else:
            print(f'Epoch: {epoch:02d}, Loss: {loss:.4f},', "\tTime: ", str(end_epoch - start_epoch))  
            print(f'Epoch: {epoch:02d}, Loss: {loss:.4f},', "\tTime: ", str(end_epoch - start_epoch), file=f)  
            f.flush()

    f.close()


if __name__ == "__main__":
    torch.manual_seed(0)
    main()