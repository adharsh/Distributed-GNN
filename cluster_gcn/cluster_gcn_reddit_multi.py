import argparse
import os
import os.path as osp

import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d as BatchNorm
from torch_geometric.data import ClusterData, ClusterLoader, NeighborSampler
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from torch.nn import ModuleList
from torch_geometric.nn import SAGEConv


from torch_geometric.datasets import Reddit
from datetime import datetime

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

    def get_convs(self):
        return self.convs

def run(rank, world_size, num_workers, args):
    # Boiler plate code to set up DistributedDataParallel
    os.environ['MASTER_ADDR'] = args.addr
    os.environ['MASTER_PORT'] = args.port
    dist.init_process_group('nccl', rank=rank, world_size=world_size)

    path = osp.join(args.dataset_path)
    dataset = Reddit(path)

    data = dataset[0]

    # Creating clusters, then passing in the distributed train_sampler into the ClusterLoader
    cluster_data = ClusterData(data, num_parts=1500, recursive=False,
                            save_dir=dataset.processed_dir)

    train_sampler = DistributedSampler(cluster_data, num_replicas=world_size,
                                        rank=rank)

    train_loader = ClusterLoader(cluster_data, batch_size=args.batch_size, num_workers=num_workers, sampler=train_sampler)

    model = Net(dataset.num_features, dataset.num_classes).to(rank)
    model_convs = model.get_convs()
    model = DistributedDataParallel(model, device_ids=[rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Thread 0 will do evaluation
    if rank == 0:
        f = open(args.log_path, 'w')
        subgraph_loader = NeighborSampler(data.edge_index, sizes=[-1], batch_size=16, #for inference/testing
                                    shuffle=False, num_workers=num_workers)
    
    
    for epoch in range(1, args.epochs + 1):
        
        # Thread 0 will time 1 epoch (training + loss calculation)
        if rank == 0:
            start_epoch = datetime.now()

        model.train()

        total_loss = total_nodes = 0
        for batch in train_loader:
            batch = batch.to(rank)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)
            loss = F.nll_loss(out[batch.train_mask], batch.y[batch.train_mask])
            loss.backward()
            optimizer.step()

            nodes = batch.train_mask.sum().item()
            total_loss += loss.item() * nodes
            total_nodes += nodes
        loss = total_loss / total_nodes

        dist.barrier()
        # Block threads to make sure syncs are performed

        if rank == 0:  # Perform evaluation on the GPU
            end_epoch = datetime.now()
            if epoch % args.eval_steps == 0 and args.eval_enable:

                ## Test code block -->
                model.eval()

                ## Inference code block -->
                x_all = data.x
                pbar = tqdm(total=x_all.size(0) * len(model_convs))
                pbar.set_description('Evaluating')

                for i, conv in enumerate(model_convs):
                    xs = []
                    for batch_size, n_id, adj in subgraph_loader:
                        edge_index, _, size = adj.to(rank)
                        x = x_all[n_id].to(rank)
                        x_target = x[:size[1]]
                        x = conv((x, x_target), edge_index)
                        if i != len(model_convs) - 1:
                            x = F.relu(x)
                        xs.append(x.cpu())

                        pbar.update(batch_size)

                    x_all = torch.cat(xs, dim=0)

                pbar.close()
                ## Inference code block <--

                out = x_all
                y_pred = out.argmax(dim=-1)

                accs = []
                for mask in [data.train_mask, data.val_mask, data.test_mask]:
                    correct = y_pred[mask].eq(data.y[mask]).sum().item()
                    accs.append(correct / mask.sum().item())
                ### Test code block  <--

                train_acc, val_acc, test_acc = accs
                print(f'Epoch: {epoch:02d}, Loss: {loss:.4f},', "\tTime: ", str(end_epoch - start_epoch), f', Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')  
                print(f'Epoch: {epoch:02d}, Loss: {loss:.4f},', "\tTime: ", str(end_epoch - start_epoch), f', Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}', file=f)  
                f.flush()

            else:
                print(f'Epoch: {epoch:02d}, Loss: {loss:.4f},', "\tTime: ", str(datetime.now() - start_epoch))    
                print(f'Epoch: {epoch:02d}, Loss: {loss:.4f},', "\tTime: ", str(datetime.now() - start_epoch), file=f)    
                f.flush()
        
        dist.barrier()
        
    if rank == 0:  
        f.close()
    dist.destroy_process_group()


if __name__ == '__main__':
    torch.manual_seed(0)
    parser = argparse.ArgumentParser(description='Reddit (Cluster-GCN) - Multi GPU')
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--eval_steps', type=int, default=10)
    parser.add_argument('--eval_enable', type=bool, default=False)    
    parser.add_argument('--dataset_path', type=str, default="/fs/class-projects/fall2020/cmsc498p/c498p000/data/Reddit")
    parser.add_argument('--log_path', type=str, default="cluster_gcn_reddit_multi.txt")
    parser.add_argument('--addr', type=str, default='localhost')
    parser.add_argument('--port', type=str, default='12356')
    args = parser.parse_args()
    print(args)

    world_size = torch.cuda.device_count()
    print('Number of GPU\'s in use: ', world_size)

    num_workers = 0
    mp.spawn(run, args=(world_size, num_workers, args, ), nprocs=world_size, join=True) 