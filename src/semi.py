"""Semi-supervised family: MAIN_unified_SMOTE_semi and unified_mixup_semi.

Both notebooks share their data pipeline, model zoo and a two-phase
training driver. They differ only in the phase-2 step
``train_on_cls_embeddings`` (SMOTE oversampling vs. mixup), which is the
actual subject of each notebook. That method is left abstract here and
supplied by each notebook via a small ``SemiMain`` subclass.
"""
import math
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from transformers import BertConfig, BertModel
from transformers import RobertaConfig, RobertaModel

from .common import Classifier


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
class TextDataset(Dataset):
    def __init__(self, data, flag='train'):
        self.data = data
        self.flag = flag

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        ind = torch.tensor(row['id'], dtype=torch.int)
        text = torch.tensor(row['text'][:2048], dtype=torch.long)
        if self.flag != 'test':
            label = torch.tensor(row['label'], dtype=torch.long)
            return text, label, ind
        else:
            return text, ind


class DataFactory(object):
    def __init__(self, path, max_len=512, flag='train'):
        self.path = path
        self.flag = flag
        self.max_len = max_len
        self.df = pd.read_json(self.path, lines=True)

    def collate_fn(self, batch):
        texts, labels, ids = zip(*batch)
        padded_texts = pad_sequence(texts, batch_first=True, padding_value=0)
        mask = (padded_texts != 0).long()
        labels = torch.stack(labels)
        ids = torch.stack(ids)
        return padded_texts, mask, labels, ids

    def collate_fn_test(self, batch):
        texts, ids = zip(*batch)
        padded_texts = pad_sequence(texts, batch_first=True, padding_value=0)
        mask = (padded_texts != 0).long()
        ids = torch.stack(ids)
        return padded_texts, mask, ids

    def get_dataloader(self, batch_size=32):
        if self.flag != 'test':
            train_data, val_data = train_test_split(self.df, test_size=0.3, random_state=42)
            train_dataset = TextDataset(train_data, flag='train')
            val_dataset = TextDataset(val_data, flag='val')
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=self.collate_fn)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=self.collate_fn)
            return train_loader, val_loader
        elif self.flag == 'semi':
            semi_dataset = TextDataset(self.df, flag='semi')
            semi_loader = DataLoader(semi_dataset, batch_size=batch_size, shuffle=True, collate_fn=self.collate_fn)
            return semi_loader
        else:
            test_dataset = TextDataset(self.df, flag='test')
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=self.collate_fn_test)
            return test_loader


# --------------------------------------------------------------------------- #
# Losses
# --------------------------------------------------------------------------- #
def mmd_loss(a, b):
    if a.size(0) < 2 or b.size(0) < 2:
        return torch.tensor(0., device=a.device)
    mu_a, mu_b = a.mean(0), b.mean(0)
    return torch.sum((mu_a - mu_b) ** 2)


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
class BERTClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, hidden_dim, num_layers, num_classes=2):
        super().__init__()
        # self.embedding = nn.Embedding(vocab_size, embed_dim)
        # self.positional_encoding = nn.Parameter(torch.zeros(1, 2048, embed_dim))
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, embed_dim)
        )

        self.res = nn.Linear(embed_dim, 128)

        self.classifier = nn.Linear(embed_dim, num_classes)

        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim)
        )
        config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=embed_dim,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=hidden_dim,
            max_position_embeddings=4096,
            output_hidden_states=True
        )
        self.bert = BertModel(config)

    def forward(self, x, x_mask):
        if x.size(1) == 0:
            # Return dummy logits and embeddings with appropriate batch size
            batch_size = x.size(0)
            dummy_logits = torch.zeros(batch_size, self.fc[-1].out_features, device=x.device)
            dummy_z = torch.zeros(batch_size, self.proj[-1].out_features, device=x.device)
            return dummy_logits, dummy_z

        # x_embed = self.embedding(x) + self.positional_encoding[:, :x.size(1), :]
        out = self.bert(input_ids=x,
                        attention_mask=x_mask)
        cls_vec = out.hidden_states[-1][:, 0, :]
        logits = self.classifier(self.fc(cls_vec) + cls_vec)
        # z = self.proj(cls_vec)
        # z = F.normalize(z, dim=1)
        return logits, cls_vec

    def extract_cls_from_tokens(self, dataloader, device):
        self.eval()
        features, labels, ids = [], [], []
        with torch.no_grad():
            for x, x_mask, y, ids_batch in dataloader:
                # x_mask = (x != 0).long()
                x, x_mask = x.to(device), x_mask.to(device)
                _, cls_vec = self(x, x_mask)
                features.append(cls_vec.cpu())
                labels.append(y.cpu())
                ids.extend([i.item() if isinstance(i, torch.Tensor) else int(i) for i in ids_batch])
        return torch.cat(features), torch.cat(labels), ids


class RoBERTaClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, hidden_dim, num_layers, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, 2048, embed_dim))
        self.fc = nn.Linear(embed_dim, num_classes)
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(embed_dim),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, hidden_dim)
        )
        config = RobertaConfig(
            vocab_size=vocab_size,
            hidden_size=embed_dim,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=hidden_dim,
            max_position_embeddings=4096,
        )
        self.roberta = RobertaModel(config)

    def forward(self, x, x_mask):
        if x.size(1) == 0:
            # Return dummy logits and embeddings with appropriate batch size
            batch_size = x.size(0)
            dummy_logits = torch.zeros(batch_size, self.fc.out_features, device=x.device)
            dummy_z = torch.zeros(batch_size, self.proj[-1].out_features, device=x.device)
            return dummy_logits, dummy_z

        # x_embed = self.embedding(x) + self.positional_encoding[:, :x.size(1), :]
        out = self.roberta(input_ids=x,
                        attention_mask=x_mask)
        sequence_output = out.last_hidden_state
        cls_vec = sequence_output[:, 0, :]
        logits = self.fc(cls_vec)
        z = self.proj(cls_vec)
        z = F.normalize(z, dim=1)
        return logits, z


class LSTM_BERT_Classifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, lstm_hidden_dim, num_heads, hidden_dim, num_layers, num_classes=2):
        super().__init__()

        # Step 1: Embedding + LSTM
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, lstm_hidden_dim, batch_first=True, bidirectional=True)

        # Projection to match BERT/Transformer input size
        self.project = nn.Linear(2 * lstm_hidden_dim, embed_dim)

        # Step 2: BERT-style Transformer encoder
        config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=embed_dim,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=hidden_dim,
            max_position_embeddings=4096,
            output_hidden_states=True
        )
        self.bert = BertModel(config)

        # Step 3: Classifier head
        self.classifier = nn.Linear(embed_dim, num_classes)

        # Optional projection head for MMD / SupCon
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, x, x_mask):
        B, L = x.shape

        # Step 1: Embedding + LSTM
        x_embed = self.embedding(x)                        # (B, L, E)
        x_lstm, _ = self.lstm(x_embed)                    # (B, L, 2H)
        z_lstm = x_lstm.mean(dim=1)                       # (B, 2H) for MMD (optional)

        # Step 2: Project and feed to BERT
        x_proj = self.project(x_lstm)                     # (B, L, E)
        out = self.bert(inputs_embeds=x_proj, attention_mask=x_mask)
        cls_vec = out.last_hidden_state[:, 0, :]          # (B, E)

        # Step 3: Classification
        logits = self.classifier(cls_vec)

        # Optional projection for SupCon or MMD
        z = self.proj(cls_vec)
        z = F.normalize(z, dim=1)

        return logits, z, z_lstm


# --------------------------------------------------------------------------- #
# Two-phase training driver
# --------------------------------------------------------------------------- #
class SemiMain(object):
    """Shared driver for the semi-supervised notebooks.

    Every method here was byte-identical between the two notebooks; only the
    phase-2 ``train_on_cls_embeddings`` step differed, so it is abstract and
    each notebook supplies its own subclass implementation.
    """

    def __init__(self, configs):
        random.seed(configs.seed)
        np.random.seed(configs.seed)
        torch.manual_seed(configs.seed)
        self.configs = configs
        self.name = configs.name
        self.embed_dim = configs.embed_dim
        self.num_heads = configs.num_heads
        self.hidden_dim = configs.hidden_dim
        self.lstm_hidden_dim = configs.lstm_hidden_dim
        self.num_layers = configs.num_layers
        self.num_classes = configs.num_classes
        self.criterion = nn.CrossEntropyLoss()
        self.data1 = DataFactory(configs.path1, flag='train')
        self.data2 = DataFactory(configs.path2, flag='val')
        self.data_test = DataFactory(configs.test_path, flag='test')
        self.trainloader_1, self.valloader_1 = self.data1.get_dataloader(batch_size=configs.batch_size1)
        self.trainloader_2, self.valloader_2 = self.data2.get_dataloader(batch_size=configs.batch_size2)
        self.testloader = self.data_test.get_dataloader()
        self.second_trainloader = None
        self.second_valloader = None

        self.vocab_size = 17212
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.model = Classifier(self.vocab_size, self.embed_dim, self.num_heads, self.hidden_dim, self.num_layers, self.num_classes).to(self.device)
        self.model = BERTClassifier(self.vocab_size, self.embed_dim, self.num_heads, self.hidden_dim, self.num_layers, self.num_classes).to(self.device)
        # self.model = RoBERTaClassifier(self.vocab_size, self.embed_dim, self.num_heads, self.hidden_dim, self.num_layers, self.num_classes).to(self.device)
        # self.model = LSTM_BERT_Classifier(self.vocab_size, self.embed_dim, self.lstm_hidden_dim, self.num_heads, self.hidden_dim, self.num_layers, self.num_classes).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.num_epochs = configs.num_epochs
        self.metric1 = accuracy_score
        self.metric2 = f1_score
        self.tau = configs.tau

    def __save__(self):
        path = os.path.join(f'../checkpoints/{self.name}')
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.model.state_dict(), f'{path}/{self.name}.pt')

    def __load__(self):
        path = os.path.join(f'../checkpoints/{self.name}')
        if not os.path.exists(path):
            os.makedirs(path)
        self.model.load_state_dict(torch.load(f'{path}/{self.name}.pt', map_location=self.device))

    def supcon_loss(self, z: torch.Tensor, y: torch.Tensor, T: float = 0.07) -> torch.Tensor:
        """
        Computes the Supervised Contrastive Loss (SupCon) for a batch.

        Args:
            z: Tensor of shape (N, D), L2-normalized projection vectors.
            y: LongTensor of shape (N,), integer class labels.
            T: Float, temperature parameter.

        Returns:
            A scalar Tensor containing the mean SupCon loss over the batch.
        """
        N = z.size(0)

        # 1) Pairwise cosine similarities scaled by temperature -> (N, N)
        sim = torch.matmul(z, z.T) / T

        # 2) Numerical stability: subtract max per row
        sim_max, _ = sim.max(dim=1, keepdim=True)
        sim = sim - sim_max.detach()

        # 3) Exponentiate and zero out self-similarities on the diagonal
        exp_sim = torch.exp(sim)
        eye_mask = torch.eye(N, dtype=torch.bool, device=z.device)
        exp_sim = exp_sim.masked_fill(eye_mask, 0.0)

        # 4) Build mask for positives: same label across batch
        pos_mask = y.unsqueeze(0) == y.unsqueeze(1)  # shape (N, N)

        # 5) Sum of positive similarities and sum of all similarities
        pos_sum = (exp_sim * pos_mask.float()).sum(dim=1)
        all_sum = exp_sim.sum(dim=1)

        # 6) Avoid log(0) by clamping to a small epsilon
        eps = 1e-6
        pos_sum = pos_sum.clamp_min(eps)
        all_sum = all_sum.clamp_min(eps)

        # 7) Compute per-sample loss and then average
        loss_i = -torch.log(pos_sum / all_sum)
        return loss_i.mean()

    def train(self):
        loader_1 = self.trainloader_1
        loader_2 = self.trainloader_2
        min_loss = math.inf
        patience = 3
        for epoch in range(self.num_epochs):
            self.model.train()
            epoch_loss = []
            epoch_acc = []
            epoch_f1 = []
            for (x1, x1_mask, y1, ind1), (x2, x2_mask, y2, ind2) in tqdm(zip(loader_1, loader_2)):
                x1, x1_mask, y1, ind1 = x1.to(self.device), x1_mask.to(self.device), y1.to(self.device), ind1.to(self.device)
                x2, x2_mask, y2, ind2 = x2.to(self.device), x2_mask.to(self.device), y2.to(self.device), ind2.to(self.device)
                domain = torch.cat([torch.zeros_like(y1), torch.ones_like(y2)], dim=0)

                L = max(x1.size(1), x2.size(1))
                # pad the second dim (seq_len) to L
                x1 = F.pad(x1, (0, L - x1.size(1)))       # (left, right) for dim=1
                x2 = F.pad(x2, (0, L - x2.size(1)))
                m1 = F.pad(x1_mask, (0, L - x1_mask.size(1)))
                m2 = F.pad(x2_mask, (0, L - x2_mask.size(1)))

                x = torch.cat([x1, x2], dim=0)
                mask = torch.cat([m1, m2], dim=0)
                y = torch.cat([y1, y2], dim=0)

                self.optimizer.zero_grad()
                outputs, feats = self.model(x, mask)
                pred = torch.argmax(outputs, dim=1)
                loss_ce = self.criterion(outputs, y)
                loss_supcon = self.supcon_loss(outputs, y)

                z_d1_ai = feats[(domain == 0) & (y == 1)]
                z_d2_ai = feats[(domain == 1) & (y == 1)]
                z_d1_h = feats[(domain == 0) & (y == 0)]
                z_d2_h = feats[(domain == 1) & (y == 0)]
                loss_mmd_ai = mmd_loss(z_d1_ai, z_d2_ai)
                loss_mmd_human = mmd_loss(z_d1_h, z_d2_h)
                loss_mmd = loss_mmd_ai + loss_mmd_human

                if epoch <= 6:
                    loss = loss_ce
                else:
                    loss = loss_ce + self.configs.lambda_mmd*loss_mmd
                # loss = loss_ce + self.tau*loss_supcon
                loss.backward()
                self.optimizer.step()
                epoch_loss.append(loss.item())
                epoch_acc.append(self.metric1(pred.detach().cpu(), y.detach().cpu()))
                epoch_f1.append(self.metric2(pred.detach().cpu(), y.detach().cpu(), average='macro'))

            epoch_loss = np.mean(epoch_loss)
            epoch_acc = np.mean(epoch_acc)
            epoch_f1 = np.mean(epoch_f1)
            print(f"Epoch {epoch + 1:>3}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}, F1: {epoch_f1:.4f}")

            vali_loss = self.validation()
            self.model.train()
            if vali_loss < min_loss:
                min_loss = vali_loss
                self.__save__()
            else:
                patience -= 1

            if not patience:
                break

        self.__load__()

    def validation(self):
        loader_1 = self.valloader_1
        loader_2 = self.valloader_2
        self.model.eval()
        with torch.no_grad():
            vali_loss = []
            vali_acc = []
            vali_f1 = []
            for (x1, x1_mask, y1, ind1), (x2, x2_mask, y2, ind2) in tqdm(zip(loader_1, loader_2)):
                x1, x1_mask, y1, ind1 = x1.to(self.device), x1_mask.to(self.device), y1.to(self.device), ind1.to(self.device)
                x2, x2_mask, y2, ind2 = x2.to(self.device), x2_mask.to(self.device), y2.to(self.device), ind2.to(self.device)
                domain = torch.cat([torch.zeros_like(y1), torch.ones_like(y2)], dim=0)

                L = max(x1.size(1), x2.size(1))
                # pad the second dim (seq_len) to L
                x1 = F.pad(x1, (0, L - x1.size(1)))       # (left, right) for dim=1
                x2 = F.pad(x2, (0, L - x2.size(1)))
                m1 = F.pad(x1_mask, (0, L - x1_mask.size(1)))
                m2 = F.pad(x2_mask, (0, L - x2_mask.size(1)))

                x = torch.cat([x1, x2], dim=0)
                mask = torch.cat([m1, m2], dim=0)
                y = torch.cat([y1, y2], dim=0)
                outputs, feats = self.model(x, mask)
                pred = torch.argmax(outputs, dim=1)
                loss = self.criterion(outputs, y)
                vali_loss.append(loss.item())
                vali_acc.append(self.metric1(pred.detach().cpu(), y.detach().cpu()))
                vali_f1.append(self.metric2(pred.detach().cpu(), y.detach().cpu(), average='macro'))

            vali_loss = np.mean(vali_loss)
            vali_acc = np.mean(vali_acc)
            vali_f1 = np.mean(vali_f1)
            print(f"Validation Loss: {vali_loss:.4f}, Accuracy: {vali_acc:.4f}, F1: {vali_f1:.4f}")
            return vali_loss

    def collect_pseudo_labels(self, threshold=0.95):
        import json
        loader_1 = self.valloader_1
        loader_2 = self.valloader_2
        self.model.eval()
        pseudo_samples_1, pseudo_samples_2 = [], []
        low_conf_samples_1, low_conf_samples_2 = [], []

        with torch.no_grad():
            for (x1, x1_mask, y1, ind1), (x2, x2_mask, y2, ind2) in tqdm(zip(loader_1, loader_2)):
                x1, x1_mask, ind1 = x1.to(self.device), x1_mask.to(self.device), ind1.to(self.device)
                x2, x2_mask, ind2 = x2.to(self.device), x2_mask.to(self.device), ind2.to(self.device)

                L = max(x1.size(1), x2.size(1))
                x1 = F.pad(x1, (0, L - x1.size(1)))
                x2 = F.pad(x2, (0, L - x2.size(1)))
                m1 = F.pad(x1_mask, (0, L - x1_mask.size(1)))
                m2 = F.pad(x2_mask, (0, L - x2_mask.size(1)))

                x = torch.cat([x1, x2], dim=0)
                mask = torch.cat([m1, m2], dim=0)
                ind = torch.cat([ind1, ind2], dim=0)

                outputs, _ = self.model(x, mask)
                prob = torch.softmax(outputs, dim=1)
                max_prob, pseudo_label = prob.max(dim=1)

                y = torch.cat([y1, y2], dim=0).to(self.device)  # true labels for all samples

                for i in range(len(x)):
                    pred = pseudo_label[i]
                    item = (x[i].cpu(), pred.cpu(), ind[i].cpu())
                    if i < len(x1):  # domain 1
                        if max_prob[i] > threshold and pred == y[i]:
                            pseudo_samples_1.append(item)
                        else:
                            low_conf_samples_1.append(item)
                    else:  # domain 2
                        if max_prob[i] > threshold and pred == y[i]:
                            pseudo_samples_2.append(item)
                        else:
                            low_conf_samples_2.append(item)

        print(f"Collected {len(pseudo_samples_1)} + {len(pseudo_samples_2)} pseudo-labeled samples, {len(low_conf_samples_1)} + {len(low_conf_samples_2)} remaining in val.")

        return pseudo_samples_1, pseudo_samples_2, low_conf_samples_1, low_conf_samples_2

    def test(self):
        test_loader = self.testloader
        self.model.eval()
        self.__load__()
        all_ids, all_preds = [], []
        with torch.no_grad():
            for x, x_mask, ind in tqdm(test_loader):
                x, x_mask = x.to(self.device), x_mask.to(self.device)
                outputs, _ = self.model(x, x_mask)
                pred = torch.argmax(outputs, dim=1).cpu()
                all_ids.append(ind)
                all_preds.append(pred)
        all_ids = torch.cat(all_ids).numpy()
        all_preds = torch.cat(all_preds).numpy()
        df = pd.DataFrame({
            "id":   all_ids,
            "class": all_preds
        })
        df.to_csv(self.configs.save_path, index=False)
        print(f"Saved -> {self.configs.save_path}")

    def train_on_cls_embeddings(self, *args, **kwargs):
        """Phase-2 head fine-tuning. Notebook-specific (SMOTE vs mixup);
        each notebook provides this via a SemiMain subclass."""
        raise NotImplementedError(
            "train_on_cls_embeddings must be implemented by a SemiMain subclass "
            "(the SMOTE and mixup notebooks differ only in this phase-2 step)."
        )

    def second_validation(self):
        """
        Run full model on token-level validation loader.
        Expects each batch to be (x, x_mask, y, ids).
        Returns val-loss for early stopping.
        """
        self.model.eval()
        device      = self.device
        val_loader  = self.second_valloader
        criterion   = nn.CrossEntropyLoss()

        total_loss, total_acc, total_f1 = [], [], []

        with torch.no_grad():
            for x, x_mask, y, _ in val_loader:
                x, x_mask, y = x.to(device), x_mask.to(device), y.to(device)

                # === Full forward pass (embedding + BERT + head) ===
                logits, _ = self.model(x, x_mask)

                loss = criterion(logits, y)
                pred = torch.argmax(logits, dim=1)

                total_loss.append(loss.item())
                total_acc.append(self.metric1(pred.cpu(), y.cpu()))
                total_f1.append(self.metric2(pred.cpu(), y.cpu(), average='macro'))

        avg_loss = np.mean(total_loss)
        avg_acc  = np.mean(total_acc)
        avg_f1   = np.mean(total_f1)

        print(f"[Second Val] Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}, F1: {avg_f1:.4f}")
        return avg_loss
