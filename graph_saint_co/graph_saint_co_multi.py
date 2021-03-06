import os
import os.path as osp

import argparse

import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from torch_geometric.data import GraphSAINTRandomWalkSampler, NeighborSampler
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import subgraph

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from datetime import datetime
from tqdm import tqdm

class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index, edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index, edge_weight)
        return torch.log_softmax(x, dim=-1)

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

def to_inductive(data):
    mask = data.train_mask
    data.x = data.x[mask]
    data.y = data.y[mask]
    data.train_mask = data.train_mask[mask]
    data.test_mask = None
    data.edge_index, _ = subgraph(mask, data.edge_index, None,
                                  relabel_nodes=True, num_nodes=data.num_nodes)
    data.num_nodes = mask.sum().item()
    return data


@torch.no_grad()
def test(model, data, evaluator, subgraph_loader, rank):
    model.eval()

    out = model.inference(data.x, subgraph_loader, rank)

    y_true = data.y
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': y_true[data.train_mask],
        'y_pred': y_pred[data.train_mask]
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': y_true[data.valid_mask],
        'y_pred': y_pred[data.valid_mask]
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y_true[data.test_mask],
        'y_pred': y_pred[data.test_mask]
    })['acc']

    return train_acc, valid_acc, test_acc


def run(rank, args, world_size):
    os.environ['MASTER_ADDR'] = args.addr
    os.environ['MASTER_PORT'] = args.port
    dist.init_process_group('nccl', rank=rank, world_size=world_size)

    path = osp.join(args.dataset_path)
    dataset = PygNodePropPredDataset('ogbn-products', path)
    split_idx = dataset.get_idx_split()
    data = dataset[0]

    for key, idx in split_idx.items():
        mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        mask[idx] = True
        data[f'{key}_mask'] = mask

    sampler_data = data
    if args.inductive:
        sampler_data = to_inductive(data)

    train_sampler = DistributedSampler(sampler_data, num_replicas=world_size,
                                   rank=rank)

    loader = GraphSAINTRandomWalkSampler(sampler_data,
                                         batch_size=args.batch_size/world_size,
                                         walk_length=args.walk_length,
                                         num_steps=args.num_steps,
                                         sample_coverage=0,
                                         save_dir=dataset.processed_dir,
                                         sampler=train_sampler)

    subgraph_loader = NeighborSampler(data.edge_index, sizes=[-1],
                                      batch_size=1024, shuffle=False,
                                      num_workers=12)

    model = SAGE(data.x.size(-1), args.hidden_channels, dataset.num_classes,
                 args.num_layers, args.dropout).to(rank)
    
    model = DistributedDataParallel(model, device_ids=[rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr*world_size)
    evaluator = Evaluator(name='ogbn-products')

    if rank == 0:
        f = open(args.log_path, 'w')

    for epoch in range(1, args.epochs + 1):
        if rank == 0:
            start_epoch = datetime.now()

        model.train()
        total_loss = 0
        for data in loader:
            data = data.to(rank)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            y = data.y.squeeze(1)
            loss = F.nll_loss(out[data.train_mask], y[data.train_mask])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        loss = total_loss / len(loader)
            
        dist.barrier()

        if rank == 0:
            end_epoch = datetime.now()
            if epoch % args.eval_steps == 0 and args.eval_enable:
                result = test(model, data, evaluator, subgraph_loader, rank)

                train_acc, val_acc, test_acc = result
                print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, ', "\tTime: ", str(end_epoch - start_epoch), f', Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')  
                print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, ', "\tTime: ", str(end_epoch - start_epoch), f', Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}', file=f)  
                f.flush()
            else:
                print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, ', "\tTime: ", str(end_epoch - start_epoch))  
                print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, ', "\tTime: ", str(end_epoch - start_epoch), file=f)  
                f.flush()

        dist.barrier()
    
    if rank == 0:
        f.close()
    dist.destroy_process_group()


if __name__ == '__main__':
    torch.manual_seed(0)
    
    parser = argparse.ArgumentParser(description='OGBN-Products (GraphSAINT) - Multi GPU')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--inductive', action='store_true')
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=20000)
    parser.add_argument('--walk_length', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--num_steps', type=int, default=30)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--eval_steps', type=int, default=10)
    parser.add_argument('--eval_enable', type=bool, default=False)
    parser.add_argument('--dataset_path', type=str, default="/fs/class-projects/fall2020/cmsc498p/c498p000/data/Products")
    parser.add_argument('--log_path', type=str, default="graph_saint_co_multi.txt")
    parser.add_argument('--addr', type=str, default='localhost')
    parser.add_argument('--port', type=str, default='12357')
    
    args = parser.parse_args()
    print(args)
    
    world_size = torch.cuda.device_count()
    print('Number of GPU\'s in use: ', world_size)
    mp.spawn(run, args=(args, world_size, ), nprocs=world_size, join=True)