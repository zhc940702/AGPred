import os
import numpy as np
import pandas as pd
from rdkit import Chem
from tqdm import tqdm
from torch.nn import BCEWithLogitsLoss
import torch
from torch_geometric.data import Dataset
from deepchem.feat import MolGraphConvFeaturizer
from torch_geometric.loader import DataLoader
from torch.optim import SGD
from model import GNN, GAT, CAM
from config import TrainingConfig
from torch.optim.lr_scheduler import ExponentialLR
from rdkit import Chem
from rdkit.Chem import AllChem
from torch.optim import Adam
import pickle
import random
import shutil
from cal_property import calculate_property


class MoleculeDataset(Dataset):
    def __init__(self, root: str, filename: str, test: bool = False):
        self.root = root
        self.filename = filename
        self.test = test
        self.raw_data = pd.read_csv(self.raw_paths[0]).reset_index()
        # with open('ben_targets2index.pickle', 'rb') as file:
        #     token2index = pickle.load(file)
        # self.token2index = token2index
        # self.vocab_size = len(token2index)
        with open('./normalize.pkl', 'rb') as w:
            norm = pickle.load(w)
        self.property_mean, self.property_std = norm

        super().__init__(root, transform=None, pre_transform=None)

    @property
    def raw_file_names(self) -> str:
        return self.filename

    @property
    def processed_file_names(self):
        return [self._get_processed_filename(idx) for idx in list(self.raw_data.index)]

    def download(self) -> None:
        pass

    def process(self) -> None:
        featurizer = MolGraphConvFeaturizer(use_edges=True)
        for idx, row in tqdm(self.raw_data.iterrows(), total=self.raw_data.shape[0]):
            # Featurize molecule
            mol = Chem.MolFromSmiles(row["smiles"])
            f = featurizer._featurize(mol)
            data = f.to_pyg_graph()
            # print(data.x.shape)
            data.y = self._get_label(row["label"])
            # data.y = self._get_label(row["HIV_active"])
            data.smiles = row["smiles"]
            # token = eval(row["token"])
            data.ids = 1
            # data.feat_17 = eval(row["feat_17"])
            # data.feat_kg = eval(row["feat_kg"])
            data.morgan = list(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=512))
            data.prop = (calculate_property(row["smiles"]) - self.property_mean) / self.property_std

            processed_fname = self._get_processed_filename(idx)
            torch.save(data, os.path.join(self.processed_dir, processed_fname))

    def _get_label(self, label):
        label = np.asarray([label])
        return torch.tensor(label, dtype=torch.int64)

    def _get_processed_filename(self, idx: int) -> str:
        return f"data_test_{idx}.pt" if self.test else f"data_{idx}.pt"

    def len(self) -> int:
        return self.raw_data.shape[0]

    def get(self, idx: int):
        processed_fname = self._get_processed_filename(idx)
        data = torch.load(os.path.join(self.processed_dir, processed_fname))
        return data


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 函数：删除已处理文件夹
def remove_processed_folder(root_folder):
    processed_folder = os.path.join(root_folder, "processed")
    if os.path.exists(processed_folder):
        shutil.rmtree(processed_folder)


if __name__ == '__main__':
    from sklearn.metrics import auc, accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, \
        precision_recall_curve

    # 固定随机种子
    seed_everything(42)
    # target编码字典
    # with open('ben_targets2index.pickle', 'rb') as file:
    #     token2index = pickle.load(file)
    # vocab_size = len(token2index)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config = TrainingConfig()

    fold_results = []
    for fold in range(10):
        print('fold_num: ', fold)
        # 前一折预处理数据的文件删除掉
        remove_processed_folder("./dataset")
        # train_file = f"./fold_data/benchmark_1_train_fold_{fold}.csv"
        # test_file = f"./fold_data/benchmark_1_test_fold_{fold}.csv"
        train_file = f"benchmark_train_fold_{fold}.csv"
        test_file = f"benchmark_test_fold_{fold}.csv"
        # train_file = f"./fold_data/DB_P4_train_fold_{fold}.csv"
        # test_file = f"./fold_data/DB_P4_test_fold_{fold}.csv"
        train_dataset = MoleculeDataset(root="dataset", filename=train_file)
        train_loader = DataLoader(
            train_dataset, batch_size=16, shuffle=True
        )
        test_dataset = MoleculeDataset(root="dataset", filename=test_file, test=True)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        sample_data = train_dataset[0]
        (feature_dim, edge_dim) = (sample_data.x.shape[1], sample_data.edge_attr.shape[1])

        # embedding_matrix = np.load('embedding_matrix.npy')
        # embedding_tensor = torch.tensor(embedding_matrix, dtype=torch.float32)
        # # model = GNN(feature_size=feature_dim, edge_size=edge_dim, config=config)
        model = CAM(feature_size=feature_dim, edge_size=edge_dim, config=config)
        #
        # model.bert.embedding.tok_embed.weight.data.copy_(embedding_tensor)
        # model = GAT(in_dim = 30, hidden_dim = 256, num_heads=3, num_classes=2)
        model = model.to(device)

        # optimizer = SGD(
        #     params=model.parameters(),
        #     lr=config.learning_rate,
        #     momentum=config.sgd_momentum,
        #     weight_decay=config.weight_decay,
        # )

        optimizer = Adam(
            params=model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # optimizer = Adam([
        #         {'params': model.prop_bert.parameters(), 'lr': 0.00001},
        #         {'params': model.gnn.parameters(), 'lr': 0.00005},
        #         {'params': [param for name, param in model.named_parameters()
        #                     if 'prop_bert' not in name and 'gnn' not in name and 'cross_attention1' not in name], 'lr': 0.00005},
        #         # {'params': model.cross_attention1.parameters(), 'lr': 0.000005}
        #     ])

        pos_weight = torch.tensor([config.pos_weight], dtype=torch.float32).to(device)
        loss_fn = BCEWithLogitsLoss(pos_weight=pos_weight)
        # scheduler = ExponentialLR(optimizer=optimizer, gamma=config.scheduler_gamma)

        EPOCH = 200
        best_auc = 0
        best_aupr = 0
        best_acc = 0
        best_recall = 0
        best_precision = 0
        best_f1 = 0
        for epoch_num in range(EPOCH):
            final_loss = 0
            all_preds = []
            all_labels = []
            probs = []
            train_all_preds = []
            train_all_labels = []
            train_probs = []
            test_smiles = []
            step = 0
            running_loss = 0.0
            print('epoch_num:', epoch_num)
            for num, data in enumerate(train_loader):
                model.train()
                optimizer.zero_grad()
                data = data.to(device)
                smiles = data.smiles
                prop = data.prop
                # print(prop.shape)
                prop = prop.view(-1, 53)
                # other = torch.tensor(data.other_feature).to(device)\
                morgan = torch.FloatTensor(data.morgan).to(device)
                # ids是编码后的靶标序列
                ids = data.ids
                pred, _ = model(data, morgan, ids, prop, None)
                loss = loss_fn(torch.squeeze(pred), data.y.float())
                loss.backward()
                optimizer.step()
                # scheduler.step()
                step += 1
                running_loss += loss.item()

            print('loss of this epoch: ', running_loss)
            model.eval()
            for num, data1 in enumerate(test_loader):
                data1 = data1.to(device)
                # other1 = torch.tensor(data1.other_feature).to(device)
                # pred, _ = model(data1.x.float(), data1.edge_index, data1.edge_attr.float(), data1.batch)
                morgan1 = torch.FloatTensor(data1.morgan).to(device)
                ids1 = data1.ids
                # feat_171 = torch.FloatTensor(data1.feat_17).to(device)
                # feat_kg1 = torch.FloatTensor(data1.feat_kg).to(device)
                # pred, _ = model(data1, morgan1)
                # pred, _ = model(data1, morgan1, ids1)
                prop1 = data1.prop
                # print(prop.shape)
                prop1 = prop1.view(-1, 53)
                pred, _ = model(data1, morgan1, ids1, prop1, None)
                label_raw = torch.sigmoid(pred).cpu().detach().numpy()
                all_preds.append(np.rint(label_raw))
                probs.append(label_raw)
                all_labels.append(data1.y.cpu().detach().numpy())
                test_smiles += data1.smiles

            all_preds = np.concatenate(all_preds).ravel()
            all_labels = np.concatenate(all_labels).ravel()
            probs = np.concatenate(probs).ravel()
            # print(test_smiles)
            # print(len(test_smiles))
            # print(len(probs))
            test_smiles_dict = {test_smile: all_preds for test_smile, all_preds in zip(test_smiles, all_preds)}
            # print(test_smiles_dict)
            # For classification task
            iprecision, irecall, ithresholds = precision_recall_curve(all_labels, probs, pos_label=1,
                                                                      sample_weight=None)
            # 这里recall要写在前面。
            AUPR = auc(irecall, iprecision)
            accuracy = accuracy_score(all_labels, all_preds)
            recall = recall_score(all_labels, all_preds)
            precision = precision_score(all_labels, all_preds)
            f1 = f1_score(all_labels, all_preds)
            AUC = roc_auc_score(all_labels, probs)
            if AUC > best_auc:
                best_auc = AUC
                best_acc = accuracy
                best_recall = recall
                best_precision = precision
                best_f1 = f1
                best_aupr = AUPR
                tingzhi = 0
            else:
                tingzhi = tingzhi + 1
            # if tingzhi >= 30:
            #     break
            print("AUC:", AUC)
            print(f"AUPR: {AUPR}")
            print(f"Accuracy: {accuracy}")
            print(f"Recall: {recall}")
            print(f"Precision: {precision}")
            print(f"F1 Score: {f1}")
        print("best_AUC:", best_auc)
        print("best_AUPR:", best_aupr)
        print("best_acc:", best_acc)
        print("best_precision:", best_precision)
        print("best_recall:", best_recall)
        print("best_f1:", best_f1)
        fold_results.append((best_auc, best_aupr, best_acc, best_precision, best_recall, best_f1))

total_auc = sum([result[0] for result in fold_results]) / len(fold_results)
total_aupr = sum([result[1] for result in fold_results]) / len(fold_results)
total_acc = sum([result[2] for result in fold_results]) / len(fold_results)
total_precision = sum([result[3] for result in fold_results]) / len(fold_results)
total_recall = sum([result[4] for result in fold_results]) / len(fold_results)
total_f1 = sum([result[5] for result in fold_results]) / len(fold_results)
print("Overall 10-Fold AUC:", total_auc)
print("Overall 10-Fold AUPR:", total_aupr)
print("Overall 10-Fold Accuracy:", total_acc)
print("Overall 10-Fold Precision:", total_precision)
print("Overall 10-Fold Recall:", total_recall)
print("Overall 10-Fold f1:", total_f1)
