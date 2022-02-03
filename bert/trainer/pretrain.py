import int as int
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
import random
from tqdm import tqdm
from bert.dataset import iterator

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

class Pretrain :
    def __init__(self,
                 batch_size:int = 30,
                 seq_len:int = 256,
                 device:[str, None] = None,
                 epochs:int = 20,
                 pad_idx:int = 0):

        self._load_data = False
        self._load_model = False

        self._batch_size = batch_size
        self._seq_len = seq_len
        if device is None :
            self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else :
            self._device = device
        self._epochs = epochs

        self._criterion1 = nn.NLLLoss(ignore_index=pad_idx).to(device)
        self._criterion2 = nn.NLLLoss().to(device)

    def load_data(self, train_fname, valid_fname, tokenizer_fname):
        self._train_dataset = iterator.BertIterator(filename=train_fname,
                                              tokenizer_model_path=tokenizer_fname,
                                              seq_len=self._seq_len)
        self._train_dataloader = DataLoader(self._train_dataset, batch_size=self._batch_size, shuffle=True)

        self._valid_dataset = iterator.BertIterator(filename=valid_fname,
                                              tokenizer_model_path=tokenizer_fname,
                                              seq_len=self._seq_len)
        self._valid_dataloader = DataLoader(self._valid_dataset, batch_size=self._batch_size, shuffle=True)
        self._load_data = True

    def load_model(self, model):
        assert self._load_data, "load_data() should be first"

        self._model = model.to(self._device)
        self._optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
        self._scheduler = optim.lr_scheduler.OneCycleLR(self._optimizer, max_lr=1e-1, pct_start=0.01,
                                                  steps_per_epoch=len(self._train_dataloader), epochs=self._epochs,
                                                  total_steps=self._epochs * len(self._train_dataloader),
                                                  anneal_strategy='linear')
        self._load_model = True

    def _train(self, model, dataloader, optimizer, scheduler):

        model.train()

        epoch_total_loss = 0
        epoch_mlm_loss = 0
        epoch_nsp_loss = 0

        epoch_mlm_acc = 0
        epoch_nsp_acc = 0
        cnt = 0

        for data in tqdm(dataloader, desc='train'):
            data = {k: v.to(self._device) for k, v in data.items()}

            optimizer.zero_grad()
            mlm_pred, nsp_pred = model(data['text'], data['seg'])

            # calculate loss from masked language modeling
            mlm_loss = self._criterion1(mlm_pred.transpose(1, 2), data['mlm'])

            # calculate loss from next sentence prediction
            nsp_loss = self._criterion2(nsp_pred, data['nsp'].long().to(self._device))

            # merge two loss equally
            loss = mlm_loss + nsp_loss

            loss.backward()
            optimizer.step()

            # calculate acc for mlm
            acc = (mlm_pred.view(-1, mlm_pred.shape[-1]).argmax(1) == data['mlm'].view(-1)).sum().item() / \
                  data['mlm'].view(-1).shape[0]
            epoch_mlm_acc += acc

            # calculate acc from nsp
            acc = (nsp_pred.argmax(1) == data['nsp']).sum() / data['nsp'].shape[0]
            epoch_nsp_acc += acc

            epoch_total_loss += loss.item()
            epoch_mlm_loss += mlm_loss.item()
            epoch_nsp_loss += nsp_loss.item()
            cnt += 1
            scheduler.step()

        print(
            f'\tTrain Total Loss: {epoch_total_loss / cnt:.3f} | Train MLM Loss: {epoch_mlm_loss / cnt:.3f} | Train NSP Loss: {epoch_nsp_loss / cnt:.3f}\
            | MLM ACC : {epoch_mlm_acc / cnt: .3f} | NSP ACC : {epoch_nsp_acc / cnt: .3f} | Learning Rate : {scheduler.get_last_lr()[0]:.3f}')

    def _evaluate(self, model, dataloader):

        model.eval()

        epoch_total_loss = 0
        epoch_mlm_loss = 0
        epoch_nsp_loss = 0

        epoch_mlm_acc = 0
        epoch_nsp_acc = 0
        cnt = 0

        with torch.no_grad():
            for data in tqdm(dataloader, desc='valid'):
                data = {k: v.to(self._device) for k, v in data.items()}

                mlm_pred, nsp_pred = model(data['text'], data['seg'])

                mlm_loss = self._criterion1(mlm_pred.transpose(1, 2), data['mlm'])

                nsp_loss = self._criterion2(nsp_pred, data['nsp'].long())

                loss = mlm_loss + nsp_loss

                acc = (mlm_pred.view(-1, mlm_pred.shape[-1]).argmax(1) == data['mlm'].view(-1)).sum().item() / \
                      data['mlm'].view(-1).shape[0]
                epoch_mlm_acc += acc

                acc = (nsp_pred.argmax(1) == data['nsp']).sum() / data['nsp'].shape[0]
                epoch_nsp_acc += acc

                epoch_total_loss += loss.item()
                epoch_mlm_loss += mlm_loss.item()
                epoch_nsp_loss += nsp_loss.item()
                cnt += 1

            print(
                f'\Valid Total Loss: {epoch_total_loss / cnt:.3f} | Valid MLM Loss: {epoch_mlm_loss / cnt:.3f} | Valid NSP Loss: {epoch_nsp_loss / cnt:.3f}\
                | MLM ACC : {epoch_mlm_acc / cnt: .3f} | NSP ACC : {epoch_nsp_acc / cnt: .3f}')

    def run(self):
        assert self._load_data and self._load_model, "Do load_data() and load_model() first"

        for _ in range(self._epochs):
            self._train(self._model, self._train_dataloader, self._optimizer, self._scheduler)
            self._evaluate(self._model, self._valid_dataloader)
