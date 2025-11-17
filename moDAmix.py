import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import pandas as pd 
import os
import numpy as np
import sys

class SourceDataset(Dataset):
    def __init__(self, x_data, x_gene, y_data):
        self.x_data = x_data
        self.x_gene = x_gene
        self.y_data = y_data
        
    def __getitem__(self, index): 
        return self.x_data[index], self.x_gene[index], self.y_data[index]
        
    def __len__(self): 
        return self.x_data.shape[0]


class UnlabelDataset(Dataset):
    def __init__(self, x_data, x_gene):
        self.x_data = x_data
        self.x_gene = x_gene
        
    def __getitem__(self, index): 
        return self.x_data[index], self.x_gene[index]
        
    def __len__(self): 
        return self.x_data.shape[0]


class DomainDataset(Dataset) :
    def __init__(self, x_data, x_gene, y_data, z_data):
        self.x_data = x_data
        self.x_gene = x_gene
        self.y_data = y_data
        self.z_data = z_data

    def __getitem__(self, index): 
        return self.x_data[index], self.x_gene[index], self.y_data[index], self.z_data[index]
        
    def __len__(self): 
        return self.x_data.shape[0]

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

if device == "cuda" :
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

print(f"Using {device} device")

result_dir = "./results" 
os.makedirs(result_dir, exist_ok = True)

data_dir = sys.argv[1]

sourceDataDir = data_dir
targetDataDir = data_dir

x_filename = os.path.join(sourceDataDir, sys.argv[2])
y_filename = os.path.join(sourceDataDir, sys.argv[4])
target_filename = os.path.join(targetDataDir, sys.argv[5])

x_gene_filename = os.path.join(sourceDataDir, sys.argv[3])
target_gene_filename = os.path.join(targetDataDir, sys.argv[6])

raw_x = pd.read_csv(x_filename, index_col = 0)
raw_y = pd.read_csv(y_filename, index_col = 0)

raw_target_x = pd.read_csv(target_filename, index_col = 0)

raw_x_gene = pd.read_csv(x_gene_filename, index_col = 0)
raw_target_x_gene = pd.read_csv(target_gene_filename, index_col = 0)

sample_id_list = raw_x.index.tolist()
sample_id_list.extend(raw_target_x.index.tolist())

sample_id_list_gene = raw_x_gene.index.tolist()
sample_id_list_gene.extend(raw_target_x_gene.index.tolist())

raw_target_domain_y = raw_target_x['domain_idx'].tolist()
raw_target_domain_y_gene = raw_target_x_gene['domain_idx'].tolist()

raw_y_colname = raw_y.columns.tolist()[0]
y_train = raw_y[raw_y_colname].tolist()
num_subtype = len(set(y_train))
y_train = np.array(y_train)

del raw_target_x['domain_idx']
del raw_target_x['Batch']

del raw_target_x_gene['domain_idx']
del raw_target_x_gene['Batch']

raw_target_x = raw_target_x.values
x_train = raw_x.values

raw_target_x_gene = raw_target_x_gene.values
x_train_gene = raw_x_gene.values

domain_x = np.append(x_train, raw_target_x, axis = 0)
domain_x_gene = np.append(x_train_gene, raw_target_x_gene, axis = 0)

raw_source_domain_y = np.zeros(len(y_train), dtype = int) # TCGA label : 0
domain_y = np.append(raw_source_domain_y, raw_target_domain_y)

raw_source_domain_y_gene = np.zeros(len(raw_x_gene), dtype = int) # TCGA label : 0
domain_y_gene = np.append(raw_source_domain_y_gene, raw_target_domain_y_gene)

num_domain = len(set(domain_y))

x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)

domain_x = torch.from_numpy(domain_x)
domain_y = torch.from_numpy(domain_y)

x_train_gene = torch.from_numpy(x_train_gene)
domain_x_gene = torch.from_numpy(domain_x_gene)
domain_y_gene = torch.from_numpy(domain_y_gene)


target_x = torch.from_numpy(raw_target_x)
target_x_gene = torch.from_numpy(raw_target_x_gene)

target_init_y = torch.randint(low=0, high=num_subtype, size = (len(target_x),))

#domain_z : domain_subtype
domain_z = torch.cat((y_train, target_init_y), 0)

num_feature = len(x_train[0])
num_feature_gene = len(x_train_gene[0])
num_train = len(x_train)
num_test = len(raw_target_x)

train_dataset = SourceDataset(x_train, x_train_gene, y_train)
domain_dataset = DomainDataset(domain_x, domain_x_gene, domain_y, domain_z)
target_dataset = SourceDataset(target_x, target_x_gene, target_init_y)

batch_size = 128
target_batch_size = 128
test_target_batch_size = 64

train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
domain_dataloader = DataLoader(domain_dataset, batch_size = batch_size, shuffle = True)
target_dataloader = DataLoader(target_dataset, batch_size = target_batch_size, shuffle = False)

n_fe_embed1 = 1024
n_fe_embed2 = 512

n_mo_fe_embed1 = 512
n_mo_fe_embed2 = 256

n_c_h1 = 128
n_c_h2 = 64
n_d_h1 = 256
n_d_h2 = 64

class SingleOmicsFeatureExtractor(nn.Module) :
    def __init__(self, n_input) :
        super().__init__()
        self.feature_layer = nn.Sequential(
            nn.Linear(n_input, n_fe_embed1),
            nn.LeakyReLU(),
            nn.Linear(n_fe_embed1, n_fe_embed2),
            nn.LeakyReLU()
            )
    def forward(self, x) :
        embedding = self.feature_layer(x)
        return embedding


class MultiOmicsFeatureExtractor(nn.Module) :
    def __init__(self) :
        super().__init__()
        self.feature_layer = nn.Sequential(
            nn.Linear(n_fe_embed2*2, n_mo_fe_embed1),
            nn.LeakyReLU(),
            nn.Linear(n_mo_fe_embed1, n_mo_fe_embed2),
            nn.LeakyReLU()
            )
    def forward(self, x) :
        embedding = self.feature_layer(x)
        return embedding


class DomainDiscriminator(nn.Module) :
    def __init__(self, n_fe_h2) :
        super().__init__()
        self.disc_layer = nn.Sequential(
            nn.Linear(n_fe_h2, n_d_h1),
            nn.LeakyReLU(),
            nn.Linear(n_d_h1, n_d_h2),
            nn.LeakyReLU(),
            nn.Linear(n_d_h2, num_domain)
            )
    def forward(self, x) :
        domain_logits = self.disc_layer(x)
        return domain_logits


class SubtypeClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        #self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(n_mo_fe_embed2, n_c_h1),
            nn.LeakyReLU(),
            nn.Linear(n_c_h1, n_c_h2),
            nn.LeakyReLU(),
            nn.Linear(n_c_h2, num_subtype)
        )
    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


fe_model_methyl = SingleOmicsFeatureExtractor(num_feature).to(device)
fe_model_gene = SingleOmicsFeatureExtractor(num_feature_gene).to(device)
fe_model_multiomics = MultiOmicsFeatureExtractor().to(device)

domain_disc_methyl_model = DomainDiscriminator(n_fe_embed2).to(device)
domain_disc_gene_model = DomainDiscriminator(n_fe_embed2).to(device)
domain_disc_multiomics_model = DomainDiscriminator(n_mo_fe_embed2).to(device)

subtype_pred_model = SubtypeClassifier().to(device)

c_loss = nn.CrossEntropyLoss() # Already have softmax
domain_loss = nn.CrossEntropyLoss() # Already have softmax

fe_methyl_optimizer = torch.optim.Adam(fe_model_methyl.parameters(), lr=1e-4)
fe_gene_optimizer = torch.optim.Adam(fe_model_gene.parameters(), lr=1e-4)
fe_multiomics_optimizer = torch.optim.Adam(fe_model_multiomics.parameters(), lr=1e-4)

c_optimizer = torch.optim.Adam(subtype_pred_model.parameters(), lr=1e-5)

d_methyl_optimizer = torch.optim.Adam(domain_disc_methyl_model.parameters(), lr=1e-6)
d_gene_optimizer = torch.optim.Adam(domain_disc_gene_model.parameters(), lr=1e-6)
d_multiomics_optimizer = torch.optim.Adam(domain_disc_multiomics_model.parameters(), lr=1e-6)


def pretrain_classifier(epoch, dataloader, fe_model_methyl, fe_model_gene, fe_model_multiomics, c_model, c_loss, fe_methyl_optimizer, fe_gene_optimizer, fe_multiomics_optimizer, c_optimizer):
    size = len(dataloader.dataset)
    correct = 0
    for batch, (X, X_gene, y) in enumerate(dataloader):
        X, X_gene, y = X.to(device), X_gene.to(device), y.to(device)
        X = X.float()
        X_gene = X_gene.float()
        embed_methyl = fe_model_methyl(X)
        embed_gene = fe_model_gene(X_gene)
        embed_concated = torch.cat((embed_methyl, embed_gene), 1)
        embed_multiomics = fe_model_multiomics(embed_concated)
        pred = c_model(embed_multiomics)
        loss = c_loss(pred, y)
        fe_methyl_optimizer.zero_grad()
        fe_gene_optimizer.zero_grad()
        fe_multiomics_optimizer.zero_grad()
        c_optimizer.zero_grad()
        loss.backward()
        fe_methyl_optimizer.step()
        fe_gene_optimizer.step()
        fe_multiomics_optimizer.step()
        c_optimizer.step()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    loss = loss.item()
    correct /= size
    if epoch % 10 == 0 :
        print(f"[PT Epoch {epoch+1}] \tTraining loss: {loss:>5f}, Training Accuracy: {(100*correct):>0.2f}%")


def adversarial_train_disc_single_omics(epoch, dataloader, fe_model_methyl, fe_model_gene, d_model_methyl, d_model_gene, domain_loss, fe_methyl_optimizer, fe_gene_optimizer, d_methyl_optimizer, d_gene_optimizer) :
    size = len(dataloader.dataset)
    correct_methyl = 0
    correct_gene = 0
    for batch, (X, X_gene, y, z_subtype) in enumerate(dataloader):
        X, X_gene, y = X.to(device), X_gene.to(device), y.to(device)
        X = X.float()
        X_gene = X_gene.float()
        embed_methyl = fe_model_methyl(X)
        pred = d_model_methyl(embed_methyl)
        d_loss = domain_loss(pred, y)
        # Backpropagation for methyl
        fe_methyl_optimizer.zero_grad()
        d_methyl_optimizer.zero_grad()
        d_loss.backward()
        d_methyl_optimizer.step()
        correct_methyl += (pred.argmax(1) == y).type(torch.float).sum().item()
        #
        embed_gene = fe_model_gene(X_gene)
        pred_gene = d_model_gene(embed_gene)
        d_gene_loss = domain_loss(pred_gene, y)
        fe_gene_optimizer.zero_grad()
        d_gene_optimizer.zero_grad()
        d_gene_loss.backward()
        d_gene_optimizer.step()
        correct_gene += (pred_gene.argmax(1) == y).type(torch.float).sum().item()
    d_loss = d_loss.item()
    d_gene_loss = d_gene_loss.item()
    correct_methyl /= size 
    correct_gene /= size
    if t % 10 == 0 :
        print(f"[AT-S Epoch {epoch+1}] Disc me loss: {d_loss:>5f} (Acc: {(100*correct_methyl):>0.2f}%), Disc gene loss: {d_gene_loss:>5f} (Acc: {(100*correct_gene):>0.2f}%)", end = ", ")


def adversarial_train_disc_multiomics(epoch, dataloader, fe_model_methyl, fe_model_gene, fe_model_multiomics, d_model_multiomics, domain_loss, fe_multiomics_optimizer, d_multiomics_optimizer) :
    size = len(dataloader.dataset)
    correct = 0
    for batch, (X, X_gene, y, z_subtype) in enumerate(dataloader):
        X, X_gene, y = X.to(device), X_gene.to(device), y.to(device)
        X = X.float()
        X_gene = X_gene.float()
        #
        embed_methyl = fe_model_methyl(X)
        embed_gene = fe_model_gene(X_gene)
        embed_concated = torch.cat((embed_methyl, embed_gene), 1)
        embed_multiomics = fe_model_multiomics(embed_concated)
        #
        pred = d_model_multiomics(embed_multiomics)
        d_loss = domain_loss(pred, y)
        # Backpropagation
        fe_multiomics_optimizer.zero_grad()
        d_multiomics_optimizer.zero_grad()
        d_loss.backward()
        d_multiomics_optimizer.step()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    d_loss = d_loss.item()
    correct /= size 
    if t % 10 == 0 :
        print(f"[AT-M Epoch {epoch+1}] Disc loss: {d_loss:>5f}, Training Accuracy: {(100*correct):>0.2f}%", end = ", ")


def adversarial_train_fe_single_omics(epoch, dataloader, fe_model_methyl, fe_model_gene, d_model_methyl, d_model_gene, domain_loss, fe_methyl_optimizer, fe_gene_optimizer, d_methyl_optimizer, d_gene_optimizer) :
    size = len(dataloader.dataset)
    for batch, (X, X_gene, y, z_subtype) in enumerate(dataloader):
        X, X_gene, y = X.to(device), X_gene.to(device), y.to(device)
        X = X.float()
        X_gene = X_gene.float()
        embed_methyl = fe_model_methyl(X)
        pred = d_model_methyl(embed_methyl)
        fake_y = torch.randint(low=0, high=num_domain, size = (len(y),))
        fake_y = fake_y.to(device)
        g_loss = domain_loss(pred, fake_y)
        # Backpropagation
        fe_methyl_optimizer.zero_grad()
        d_methyl_optimizer.zero_grad()
        g_loss.backward()
        fe_methyl_optimizer.step()
        # Gene
        embed_gene = fe_model_gene(X_gene)
        pred_gene = d_model_gene(embed_gene)
        fake_y_gene = torch.randint(low=0, high=num_domain, size = (len(y),))
        fake_y_gene = fake_y_gene.to(device)
        g_gene_loss = domain_loss(pred_gene, fake_y)
        fe_gene_optimizer.zero_grad()
        d_gene_optimizer.zero_grad()
        g_gene_loss.backward()
        fe_gene_optimizer.step()
    g_loss = g_loss.item()
    g_gene_loss = g_gene_loss.item()
    if epoch % 10 == 0:
        print(f"Gen methyl loss: {g_loss:>5f}, Gene gene loss: {g_gene_loss:>5f}")


def adversarial_train_fe_multiomics(epoch, dataloader, fe_model_methyl, fe_model_gene, fe_model_multiomics, d_model_multiomics, domain_loss, fe_multiomics_optimizer, d_multiomics_optimizer) :
    size = len(dataloader.dataset)
    for batch, (X, X_gene, y, z_subtype) in enumerate(dataloader):
        X, X_gene, y = X.to(device), X_gene.to(device), y.to(device)
        X = X.float()
        X_gene = X_gene.float()
        #
        embed_methyl = fe_model_methyl(X)
        embed_gene = fe_model_gene(X_gene)
        embed_concated = torch.cat((embed_methyl, embed_gene), 1)
        embed_multiomics = fe_model_multiomics(embed_concated)
        #
        pred = d_model_multiomics(embed_multiomics)
        fake_y = torch.randint(low=0, high=num_domain, size = (len(y),))
        fake_y = fake_y.to(device)
        g_loss = domain_loss(pred, fake_y)
        # Backpropagation
        fe_multiomics_optimizer.zero_grad()
        d_multiomics_optimizer.zero_grad()
        g_loss.backward()
        fe_multiomics_optimizer.step()
    g_loss = g_loss.item()
    if epoch % 10 == 0:
        print(f"Gen multi loss: {g_loss:>5f}")




def class_alignment_train(epoch, domain_dataloader, fe_model_methyl, fe_model_gene, fe_model_multiomics, fe_methyl_optimizer, fe_gene_optimizer, fe_multiomics_optimizer, c_optimizer) :
    for batch, (X, X_gene, y_domain, z_subtype) in enumerate(domain_dataloader):
        X, X_gene, y_domain, z_subtype = X.to(device), X_gene.to(device), y_domain.to(device), z_subtype.to(device)
        X = X.float()
        X_gene = X_gene.float()
        batch_subtype_list = z_subtype.unique()
        #
        embed_methyl = fe_model_methyl(X)
        embed_gene = fe_model_gene(X_gene)
        embed_concated = torch.cat((embed_methyl, embed_gene), 1)
        X_embed = fe_model_multiomics(embed_concated)
        #
        align_loss = torch.zeros((1) ,dtype = torch.float64)
        align_loss = align_loss.to(device)
        #
        for subtype in batch_subtype_list :
            sample_idx_list = (z_subtype == subtype).nonzero(as_tuple = True)[0]
            if len(sample_idx_list) < 1 :
                continue
            #else :
            tmp_x = X_embed[sample_idx_list]
            tmp_y = y_domain[sample_idx_list]
            tmp_z = z_subtype[sample_idx_list]
            batch_domain_list = tmp_y.unique()
            domain_centroid_stack = []
            for domain in batch_domain_list :
                domain_idx_list = (tmp_y == domain).nonzero(as_tuple = True)[0]
                if len(domain_idx_list) != 1 :
                    tmp_x_domain = tmp_x[domain_idx_list]
                    tmp_centroid = torch.div(torch.sum(tmp_x_domain, dim = 0), len(domain_idx_list))
                    domain_centroid_stack.append(tmp_centroid)
            if len(domain_centroid_stack) == 0 :
                continue
            else :
                domain_centroid_stack = torch.stack(domain_centroid_stack)
            subtype_centroid = torch.mean(domain_centroid_stack, dim = 0)
            # Duplicate the subtype centroid to get dist with each domain_centroid
            subtype_centroid_stack = []
            for i in range(len(domain_centroid_stack)) :
                subtype_centroid_stack.append(subtype_centroid)
            subtype_centroid_stack = torch.stack(subtype_centroid_stack)
            pdist_stack = nn.L1Loss()(subtype_centroid_stack, domain_centroid_stack)
            align_loss +=  torch.mean(pdist_stack, dim = 0)
        if align_loss == 0.0 :
            continue
        align_loss = align_loss / len(batch_subtype_list)
        fe_methyl_optimizer.zero_grad()
        fe_gene_optimizer.zero_grad()
        fe_multiomics_optimizer.zero_grad()
        c_optimizer.zero_grad()
        align_loss.backward()
        fe_methyl_optimizer.step()
        fe_gene_optimizer.step()
        fe_multiomics_optimizer.step()
        c_optimizer.step()
    align_loss = align_loss.item()
    if epoch % 10 == 0 :
        print(f"[CA Epoch {epoch+1}] align loss: {align_loss:>5f}\n")


def ssl_train_classifier(epoch, source_dataloader, target_dataloader, fe_model_methyl, fe_model_gene, fe_model_multiomics, c_model, c_loss, fe_methyl_optimizer, fe_gene_optimizer, fe_multiomics_optimizer, c_optimizer) :
    source_size = len(source_dataloader.dataset)
    target_size = len(target_dataloader.dataset)
    #
    # 1. Obtain the pseudo-label for target dataset
    #
    target_pseudo_label = torch.empty((0), dtype = torch.int64)
    target_pseudo_label = target_pseudo_label.to(device)
    #
    for batch, (target_X, target_X_gene, target_y) in enumerate(target_dataloader):
        target_X, target_X_gene, target_y = target_X.to(device), target_X_gene.to(device), target_y.to(device)
        target_X = target_X.float()
        target_X_gene = target_X_gene.float()
        #
        embed_methyl = fe_model_methyl(target_X)
        embed_gene = fe_model_gene(target_X_gene)
        embed_concated = torch.cat((embed_methyl, embed_gene), 1)
        extracted_feature = fe_model_multiomics(embed_concated)
        #
        #extracted_feature = fe_model(target_X)
        batch_target_pred = c_model(extracted_feature)
        batch_pseudo_label = batch_target_pred.argmax(1)
        target_pseudo_label = torch.cat((target_pseudo_label, batch_pseudo_label), 0)
        if batch == 0 :
            target_loss = c_loss(batch_target_pred, target_y)
        else :
            target_loss = target_loss + c_loss(batch_target_pred, target_y)
    target_loss = target_loss / (batch + 1)
    #
    # Define alpha value
    alpha_f = 0.01
    t1 = 100
    t2 = 200
    if epoch < t1 :
        alpha = 0
    elif epoch < t2 :
        alpha = (epoch - t1) / (t2 - t1) * alpha_f
    else :
        alpha = alpha_f
    #
    # 2. Calculate the loss for the source dataset
    #
    correct = 0
    for batch, (source_X, source_X_gene, source_y) in enumerate(source_dataloader):
        source_X, source_X_gene, source_y = source_X.to(device), source_X_gene.to(device), source_y.to(device)
        source_X = source_X.float()
        source_X_gene = source_X_gene.float()
        #
        source_embed_methyl = fe_model_methyl(source_X)
        source_embed_gene = fe_model_gene(source_X_gene)
        source_embed_concated = torch.cat((source_embed_methyl, source_embed_gene), 1)
        source_extracted_feature = fe_model_multiomics(source_embed_concated)
        #
        #source_extracted_feature = fe_model(source_X)
        source_pred = c_model(source_extracted_feature)
        source_loss = c_loss(source_pred, source_y)
        ssl_loss = source_loss + alpha * target_loss
        # Backpropogation
        target_loss.detach_()
        fe_methyl_optimizer.zero_grad()
        fe_gene_optimizer.zero_grad()
        fe_multiomics_optimizer.zero_grad()
        c_optimizer.zero_grad()
        ssl_loss.backward() #retain_graph=True
        fe_methyl_optimizer.step()
        fe_gene_optimizer.step()
        fe_multiomics_optimizer.step()
        c_optimizer.step()
        correct += (source_pred.argmax(1) == source_y).type(torch.float).sum().item()
    ssl_loss = ssl_loss.item()
    source_loss = source_loss.item()
    target_loss = target_loss.item()
    correct /= source_size
    if epoch % 10 == 0 :
        print(f"[SSL Epoch {epoch+1}] alpha : {alpha:>3f}, SSL loss: {ssl_loss:>5f}, source loss: {source_loss:>5f}, target loss: {target_loss:>4f}, source ACC: {(100*correct):>0.2f}%\n")
    return target_pseudo_label



def get_embed(dataloader, fe_model_methyl, fe_model_gene, fe_model_multiomics, c_model) :
    fe_model_methyl.eval()
    fe_model_gene.eval()
    fe_model_multiomics.eval()
    c_model.eval()
    X_embed_list = []
    y_list = []
    with torch.no_grad() :
        for batch, (X, X_gene, y) in enumerate(dataloader):
            X, X_gene, y = X.to(device), X_gene.to(device), y.to(device)
            X = X.float()
            #
            embed_methyl = fe_model_methyl(X)
            embed_gene = fe_model_gene(X_gene)
            embed_concated = torch.cat((embed_methyl, embed_gene), 1)
            X_embed = fe_model_multiomics(embed_concated)
            #
            X_embed_list.append(X_embed)
            y_list.append(y)
    X_embed_list = torch.cat(X_embed_list, 0)
    y_list = torch.cat(y_list, 0)
    return X_embed_list, y_list



def get_embed_domain(domain_dataloader, fe_model_methyl, fe_model_gene, fe_model_multiomics, c_model) :
    fe_model_methyl.eval()
    fe_model_gene.eval()
    fe_model_multiomics.eval()
    c_model.eval()
    X_embed_list = []
    X_embed_methyl_list = []
    X_embed_gene_list = []
    domain_list = []
    pred_subtype_list = []
    label_list = [] # Can be used only for source dataset
    with torch.no_grad() :
        for batch, (X, X_gene, y, z) in enumerate(domain_dataloader):
            X, X_gene, y, z = X.to(device), X_gene.to(device), y.to(device), z.to(device)
            X = X.float()
            X_gene = X_gene.float()
            #
            embed_methyl = fe_model_methyl(X)
            embed_gene = fe_model_gene(X_gene)
            embed_concated = torch.cat((embed_methyl, embed_gene), 1)
            X_embed = fe_model_multiomics(embed_concated)
            #
            #X_embed = fe_model(X)
            pred = c_model(X_embed)
            pred_subtype_list.append(pred.argmax(1))
            X_embed_list.append(X_embed)
            domain_list.append(y)
            label_list.append(z)
            X_embed_methyl_list.append(embed_methyl)
            X_embed_gene_list.append(embed_gene)
    X_embed_list = torch.cat(X_embed_list, 0)
    X_embed_methyl_list = torch.cat(X_embed_methyl_list, 0)
    X_embed_gene_list = torch.cat(X_embed_gene_list, 0)
    pred_subtype_list = torch.cat(pred_subtype_list, 0)
    domain_list = torch.cat(domain_list, 0)
    label_list = torch.cat(label_list, 0)
    return X_embed_list, domain_list, pred_subtype_list, label_list, X_embed_methyl_list, X_embed_gene_list


pt_epochs = 20#500
ad_train_epochs = 20#500
ssl_train_epochs = 20#500
ft_epochs = 20#800


# 1. Pre-training
for t in range(pt_epochs):
    pretrain_classifier(t, train_dataloader, fe_model_methyl, fe_model_gene, fe_model_multiomics, subtype_pred_model, c_loss, fe_methyl_optimizer, fe_gene_optimizer, fe_multiomics_optimizer, c_optimizer)
    

# 2-1. Adversarial training (Single-omics)
for t in range(ad_train_epochs):
    adversarial_train_disc_single_omics(t, domain_dataloader, fe_model_methyl, fe_model_gene, domain_disc_methyl_model, domain_disc_gene_model, domain_loss, fe_methyl_optimizer, fe_gene_optimizer, d_methyl_optimizer, d_gene_optimizer)
    adversarial_train_fe_single_omics(t, domain_dataloader, fe_model_methyl, fe_model_gene, domain_disc_methyl_model, domain_disc_gene_model, domain_loss, fe_methyl_optimizer, fe_gene_optimizer, d_methyl_optimizer, d_gene_optimizer)


# 2-2. Adversarial training (Multiomics)
for t in range(ad_train_epochs):
    adversarial_train_disc_multiomics(t, domain_dataloader, fe_model_methyl, fe_model_gene, fe_model_multiomics, domain_disc_multiomics_model, domain_loss, fe_multiomics_optimizer, d_multiomics_optimizer)
    adversarial_train_fe_multiomics(t, domain_dataloader, fe_model_methyl, fe_model_gene, fe_model_multiomics, domain_disc_multiomics_model, domain_loss, fe_multiomics_optimizer, d_multiomics_optimizer)


# 3. SSL training
for t in range(ssl_train_epochs) :
    target_pseudo_label = ssl_train_classifier(t, train_dataloader, target_dataloader, fe_model_methyl, fe_model_gene, fe_model_multiomics, subtype_pred_model, c_loss, fe_methyl_optimizer, fe_gene_optimizer, fe_multiomics_optimizer, c_optimizer)
    target_dataset = SourceDataset(target_x, target_x_gene, target_pseudo_label)
    target_dataloader = DataLoader(target_dataset, batch_size = target_batch_size)


# 4. Fine-tuning 
for t in range(ft_epochs) :
    # SSL
    target_pseudo_label = ssl_train_classifier(t, train_dataloader, target_dataloader, fe_model_methyl, fe_model_gene, fe_model_multiomics, subtype_pred_model, c_loss, fe_methyl_optimizer, fe_gene_optimizer, fe_multiomics_optimizer, c_optimizer)
    target_dataset = SourceDataset(target_x, target_x_gene, target_pseudo_label)
    target_dataloader = DataLoader(target_dataset, batch_size = target_batch_size)

    # CA
    target_pseudo_label = target_pseudo_label.to("cpu")
    domain_z = torch.cat((y_train, target_pseudo_label), 0)
    domain_dataset = DomainDataset(domain_x, domain_x_gene, domain_y, domain_z)
    domain_dataloader = DataLoader(domain_dataset, batch_size = batch_size, shuffle = True)
    class_alignment_train(t, domain_dataloader, fe_model_methyl, fe_model_gene, fe_model_multiomics, fe_methyl_optimizer, fe_gene_optimizer, fe_multiomics_optimizer, c_optimizer)



domain_dataset = DomainDataset(domain_x, domain_x_gene, domain_y, domain_z)
domain_dataloader = DataLoader(domain_dataset, batch_size = batch_size, shuffle = False)

data_X_embed, domain_label, pred_subtype, label_subtype, methyl_embed, gene_embed = get_embed_domain(domain_dataloader, fe_model_methyl, fe_model_gene, fe_model_multiomics, subtype_pred_model)
data_X_embed = data_X_embed.detach().cpu().numpy()
domain_label = domain_label.detach().cpu().numpy()
pred_subtype = pred_subtype.detach().cpu().numpy()
label_subtype = label_subtype.detach().cpu().numpy()
#methyl_embed = methyl_embed.detach().cpu().numpy()
#gene_embed = gene_embed.detach().cpu().numpy()

data_X_embed = pd.DataFrame(data_X_embed)
data_X_embed['Batch'] = domain_label
data_X_embed['Pred_subtype'] = pred_subtype
data_X_embed['Label_subtype'] = label_subtype

data_X_embed.index = sample_id_list
domain_info = pd.read_csv(os.path.join(sourceDataDir, "batch_category_info.csv"), index_col = 1)
subtype_info = pd.read_csv(os.path.join(sourceDataDir, "subtype_category_info.csv"), index_col = 1)

domain_info = domain_info.to_dict()
domain_info['batch'][0] = 'Source'
subtype_info = subtype_info.to_dict()

data_X_embed['Pred_subtype'] = data_X_embed['Pred_subtype'].replace(subtype_info['subtype'])
data_X_embed['Label_subtype'] = data_X_embed['Label_subtype'].replace(subtype_info['subtype'])
data_X_embed['Batch'] = data_X_embed['Batch'].replace(domain_info['batch'])

data_X_embed.to_csv(os.path.join(result_dir, "batch_corrected_features.csv"), mode = "w", index = True)

target_pred = data_X_embed[['Batch','Pred_subtype']]
target_pred = target_pred[target_pred['Batch'] != 'Source']
target_pred.to_csv(os.path.join(result_dir, "results_target_prediction.csv"), mode = "w", index = True)

