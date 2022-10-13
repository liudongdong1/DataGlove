import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,GATConv
from torch_geometric.nn import global_mean_pool
import numpy as np
import os
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
import torch_geometric.utils as ut

np.random.seed(42)

# Model 
class GCN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(num_node_features, hidden_channels)        
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        
        self.conv4 = GCNConv(num_node_features, hidden_channels)
        self.conv5 = GCNConv(hidden_channels, hidden_channels)
        
        self.conv6 = GCNConv(num_node_features, hidden_channels)

        self.lin = Linear(hidden_channels*3, num_classes)

    def forward(self, x, edge_index, batch):
    
        
        #print(type(x))
        #print(edge_index.shape)
        #print('x',x.shape)
        # 1.1 Obtain node embeddings, first exactor
        x1 = self.conv1(x, edge_index)        
        x1 = x1.relu()
        x1 = self.conv2(x1, edge_index)
        x1 = x1.relu()
        x1 = self.conv3(x1, edge_index)
        #print('After First feature exactor:',x1.shape)
        # 1.2 Readout layer
        x1 = global_mean_pool(x1, batch)  # [batch_size, hidden_channels]
       # print('After readout:',x1.shape)
        
        
        # 2.1 Obtain node embeddings, second exactor
        x2 = self.conv4(x, edge_index)
        x2 = x2.relu()
        x2 = self.conv5(x2, edge_index)
        #print('After Second feature exactor:',x2.shape)
        # 2.2 Readout layer
        x2 = global_mean_pool(x2, batch)  # [batch_size, hidden_channels]
        #print('After readout:',x2.shape)
        
        
        # 3.1 Obtain node embeddings, second exactor
        x3 = self.conv6(x, edge_index)
       # print('After Third feature exactor:',x3.shape)       
        # 3.2 Readout layer
        x3 = global_mean_pool(x3, batch)  # [batch_size, hidden_channels]
        #print('After readout:',x3.shape)
        
        # 4. attention part
        x_cat = torch.cat((x1,x2,x3),1)
        #print('After cat:',x_cat.shape,edge_index.shape)
  
        # 5. Apply a final classifier
        x_final = F.dropout(x_cat, p=0.5, training=self.training)
        #print('Before classifier:',x_final.shape)
        x_final = self.lin(x_final)
        #print('After classifier:',x_final.shape)

        return x_final


#model = GCN(num_node_features=6, hidden_channels=64, num_classes=10)
#print(model)


#————————————————————————————————————————————————————————————————————————————————————————
lableWord = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M',
             13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y',
             25: 'Z'}

num_edge_features = 1
label_dic = dict(zip(lableWord.values(), lableWord.keys()))


def read_flex_data(raw_dir):
    data = []
    for tempfile in os.listdir(raw_dir):
        filename = os.path.join(raw_dir, tempfile)
        tempdata = np.loadtxt(filename, delimiter=',')
        label = label_dic[tempfile.split(".")[0]]
        print("filename: {} label {} ".format(tempfile, label))
        label = np.full((len(tempdata), 1), label, dtype=np.int)
        tempdata = np.column_stack((tempdata, label))
        if len(data) == 0:
            data = tempdata
        else:
            data = np.row_stack((data, tempdata))
    
    return data


def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma


#生成邻接矩阵，先增加了均值，然后做了归一化操作，这里应该要改成除以向量的L1范数
def graphfeatureHandle(record):
    # todo
    meanvalue = np.mean(record, axis=0)
    data = np.append(record, meanvalue)
    data = standardization(data)
    adj = []
    temp = []
    for i in range(0, len(data)):
        for j in range(0, len(data)):
            if data[i] - data[j] > 0:
                temp.append(data[i] - data[j])  # 只有正直，负值用0表示
            else:
                temp.append(0)

    data = np.array(temp)
    
    data = np.reshape(data, (6, 6))  # 直接data.reshape() 不起作用
    return data


def handleToData(data_list):
    listData = []
    #print('asdadasdaas')
    for i in range(len(data_list)):
        adj = graphfeatureHandle(data_list[i][:-1])
        #print('adj',adj.shape)
        #print("adj",adj,"\n np.nonzero(adj)",np.nonzero(adj))
        source_nodes, target_nodes = np.nonzero(adj)
        source_nodes = source_nodes.reshape((1, -1))
        target_nodes = target_nodes.reshape((1, -1))
        #print(target_nodes.shape,type(target_nodes),"value:",target_nodes,"shape:",target_nodes.shape)
        #print(source_nodes.shape,type(source_nodes),"value:",source_nodes)
        
        
        edge_index = torch.tensor(np.concatenate((source_nodes, target_nodes), axis=0),
                                  dtype=torch.long)  # edge_index should be long type
        #print('before',edge_index.shape)
          #torch.Size([2, 14])
        a = torch.Tensor([[i]*6 for i in range(6)]).reshape((1,-1))
        b = a.T.reshape((1,-1))
        
        edge_index = (torch.cat((a,b),0)).type(torch.long)
        #print('after',edge_index.shape)
        #print(edge_index.shape)
        # edge_weight = adj
        edge_weight = []
        for i in range(len(source_nodes[0])):
            edge_weight.append(adj[source_nodes[0][i]][target_nodes[0][i]])
        edge_weight = np.asarray(edge_weight)
        edge_weight = np.asarray(adj)
        edge_weight = torch.tensor(edge_weight.reshape((-1, num_edge_features)),
                                   dtype=torch.float)  # edge_index should be float type
        
        #print('edge_index',edge_index)
        #print("edge_index_shape:",edge_index.shape)
        #print("edge_weight_shape:",edge_weight.shape)
        temp = data_list[i][:-1]
        meanvalue = np.mean(temp, axis=0)
        x = torch.Tensor(np.reshape(np.append(temp, meanvalue),(6,1)))
        #x = torch.tensor([[i] for i in range(26)], dtype=torch.int)
        # y should be long type, graph label should not be a 0-dimesion tensor
        # use [graph_label[i]] ranther than graph_label[i]
        y = torch.tensor([data_list[i][-1]], dtype=torch.long)

        data = Data(x=x, edge_index=edge_index, y=y, edge_attr=edge_weight)
        #print(data)
        listData.append(data)
        # break
    #print('listData',listData.shape)
    return listData

#data = read_flex_data('/home/iot/jupyter/root_dir/liudongdong/data/flexData/char1/')
#print('datalist',data)
#print('flex',data.shape)
#data2=handleToData(data)

#print('graphfeature',data2.shape)

#——————————————————————————————————————————————————————————————————————————————————————————————————
class GraphFlexSensorInMem(InMemoryDataset):
    """
    Graph classification
    """

    def __init__(self, root, transform=None, pre_transform=None):
        super(GraphFlexSensorInMem, self).__init__(root, transform, pre_transform)
        #         classtype='d'
        #         if classtype[0]=='d':
        #             self.label_dic=dict(zip(lableDigit.values(), lableDigit.keys()))
        #         else:
        #             self.label_dic=dict(zip(lableWord.values(), lableWord.keys()))
        print(self.processed_paths[0])
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['data_graph.pt']

    def download(self):
        pass

    def process(self):
        data = read_flex_data('/home/iot/jupyter/root_dir/liudongdong/data/flexData/chars/char1')
        print('Flex data shape',data.shape)
        #print('Flex data shape',data)
        np.random.shuffle(data)
        data_chip = data[0:1000]
        #print('Flex data shape',data)
        #print('asdadasdaas')
        data_list = handleToData(data_chip)
        #print('Graph data shape',data.shape)
        data, slices = self.collate(data_list)  # Here used to be [data] for one graph
        #print(data)
        #print(slices)
        print("save file")
        torch.save((data, slices), self.processed_paths[0])


#test_dataset = GraphFlexSensorInMem(root='./dataset/char/train')
#print(test_dataset[1])

#——————————————————————————————————————————————————————————————————————————————————————————————————————
from torch_geometric.data import DataLoader
#GraphFlexSensorInMem(root='./dataset/char/train').process()
#GraphFlexSensorInMem(root='./dataset/char/test').process()
train_dataset = GraphFlexSensorInMem(root='/home/iot/jupyter/root_dir/liudongdong/src/ai/dataset/char/train')
test_dataset = GraphFlexSensorInMem(root='/home/iot/jupyter/root_dir/liudongdong/src/ai/dataset/char/test')

torch.manual_seed(12345)
train_dataset=train_dataset.shuffle()
test_dataset = test_dataset.shuffle()
#print(train_dataset[0])  #Data(edge_attr=[15, 1], edge_index=[2, 15], x=[26], y=[1])
#print(test_dataset[1])
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
validation_loader = DataLoader(train_dataset,batch_size=64,shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)
# for step, data in enumerate(test_loader):
#     print(f'Step {step + 1}:')
#     print('=======')
#     print(f'Number of graphs in the current batch: {data.num_graphs}')
#     print(data) #Batch(batch=[1169], edge_attr=[2592, 4], edge_index=[2, 2592], x=[1169, 7], y=[64])
#     print()
#     if step>2:
#         break
#——————————————————————————————————————————————————————————————————————————————————————————————————————


#GCN(
#  (conv1): GCNConv(7, 64)
#  (conv2): GCNConv(64, 64)
#  (conv3): GCNConv(64, 64)
#  (lin): Linear(in_features=64, out_features=2, bias=True)
#)

# num_node_features,hidden_channels,num_classes
model = GCN(num_node_features=1,hidden_channels=64,num_classes=26)
print('Model created:',model)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

def train(train_loader):
    model.train()
    for data in train_loader:  # Iterate in batches over the training dataset.
         out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
         loss = criterion(out, data.y)  # Compute the loss.
         loss.backward()  # Derive gradients.
         optimizer.step()  # Update parameters based on gradients.
         optimizer.zero_grad()  # Clear gradients.



def test(loader):
     print('Testing')
     model.eval()
     correct = 0
     for data in loader:  # Iterate in batches over the training/test dataset.
         out = model(data.x, data.edge_index, data.batch)
         #print(out.shape)
        
         pred = out.argmax(dim=1)  # Use the class with highest probability.
         #print('pred',pred)
         #print('y',data.y)
         correct += int((pred == data.y).sum())  # Check against ground-truth labels.
     #print(correct / len(loader.dataset))
     return correct / len(loader.dataset)  # Derive ratio of correct predictions.


for epoch in range(1, 15):
    train(train_loader)
    #train(test_loader)
    train_acc = test(train_loader)
    #train_acc = 0
    validate = test(validation_loader)
    test_acc = test(test_loader)
    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Validate Acc: {validate:.4f}, Test Acc: {test_acc:.4f}')


#————————————————————————————————————————————————————————————————————————————————————————————————
# 本文件旨在为所有数据集提供一个统一的数据处理手段，待处理的文件需要有以下四列构成：FromID，ToID，weight，snapshotID。输出为一个tg中由data
# 组成的dataset。
import os
import torch
import torch_geometric as tg

from torch_geometric.data import Data
from torch.utils.data import Dataset


# window_size记录每一个data由多少张snapshot构成
def load_edges(file_path, edge_window_size=1, to_undirected=True, reform_values=False, do_coalesce=True):
    # 从文件中读取数据，主要构建出edge_index, edge_attr, graph_idx, num_nodes四个值（整个数据集的，并对其做一点变换（变为无向图，重组织边权重之类））
    with open(file_path, 'r') as f:
        data = f.read().split('\n')[1:-1]
        data = [[x for x in line.split(',')] for line in data]

        edge_index = [[int(line[0]), int(line[1])] for line in data]
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        edge_index = edge_index - edge_index.min()

        # 对edge_index重排序，令id连续
        nodes, edge_index = edge_index.unique(return_inverse=True)

        edge_index = edge_index.t().contiguous()
        num_nodes = nodes.shape[0]

        edge_attr = [float(line[2]) for line in data]
        edge_attr = torch.tensor(edge_attr, dtype=torch.long)

        # 重置边权重
        pos_indices = edge_attr > 0
        neg_indices = edge_attr <= 0
        edge_attr[pos_indices] = 1
        edge_attr[neg_indices] = -1

        graph_idx = [int(line[3]) for line in data]
        graph_idx = torch.tensor(graph_idx, dtype=torch.long)

    max_time = graph_idx.max().item()
    min_time = graph_idx.min().item()

    if to_undirected:
        edge_index = torch.cat((edge_index, edge_index[[1, 0], :]), dim=-1)
        edge_attr = torch.cat((edge_attr, edge_attr), dim=-1)
        graph_idx = torch.cat((graph_idx, graph_idx), dim=-1)

    # 这里的处理，针对edge_weight的值有正有负的情形，将每一个snapshot上的多次出现的同一边的权重加和作为其边标签，将出现次数（不论正负）作为边权重
    if reform_values:

        neg_mask = edge_attr == -1
        neg_sp_indices = torch.cat((edge_index[:, neg_mask], graph_idx[neg_mask].view(1, -1)), dim=0)
        neg_sp_values = edge_attr[neg_mask]
        neg_sp_edges = torch.sparse.Tensor(neg_sp_indices,
                                           neg_sp_values,
                                           torch.Size([num_nodes,
                                                       num_nodes,
                                                       max_time+1])).coalesce()

        pos_mask = edge_attr == 1
        pos_sp_indices = torch.cat((edge_index[:, pos_mask], graph_idx[pos_mask].view(1, -1)), dim=0)
        pos_sp_values = edge_attr[pos_mask]
        pos_sp_edges = torch.sparse.Tensor(pos_sp_indices,
                                           pos_sp_values,
                                           torch.Size([num_nodes,
                                                       num_nodes,
                                                       max_time+1])).coalesce()

        pos_sp_edges *= 1000
        sp_edges = (pos_sp_edges - neg_sp_edges).coalesce()
        vals = sp_edges._values()
        neg_vals = vals % 1000
        pos_vals = vals // 1000

        edge_attr = pos_vals + neg_vals
        edge_indices = sp_edges._indices()
        edge_index = edge_indices[:2]
        graph_idx = edge_indices[2]

    # 开始构造dataset
    data_list = []
    for i in range(graph_idx.max().item() + 1):
        mask = (graph_idx > (i - edge_window_size)) & (graph_idx <= i)
        data = Data()
        data.edge_index = edge_index[:, mask]
        data.edge_attr = edge_attr[mask]
        data.num_nodes = num_nodes
        data_list.append(data)

    if do_coalesce:
        for data in data_list:
            data.coalesce()

    return data_list, num_nodes, max_time, min_time


class DynamicDataset(Dataset):
    def __init__(self, args):
        super(DynamicDataset, self).__init__()

        file_path = os.path.join(args.folder, args.edges_file)

        self.data_list, self.num_nodes, self.max_time, self.min_time = load_edges(file_path)
        # # max_time是所有的graph idx中的最大值,而不是总数(因为idx从0开始计数)
        # self.max_time = len(self.data_list) - 1

    def __getitem__(self, idx):
        return self.data_list[idx]

    def __len__(self):
        return len(self.data_list)