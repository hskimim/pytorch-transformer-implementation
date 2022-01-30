from . import preprocess as p

import torch
from torch.utils.data import Dataset

import numpy as np
import random
import pandas as pd # to load pickle

class BertIterator(Dataset) : 
    def __init__(self,
                filename = '../data/prep_train.txt',
                tokenizer_model_path = '../data/m.model',
                 seq_len=256,
                in_memory=True) : 
        
        if in_memory is False : 
            NotImplementedError("Only in-memory version is supported")
            
        self.docs = self._load_txt_in_memory(filename)
        self.prep = p.TxtProcessor(tokenizer_model_path)
        self.seq_len = seq_len
        
    def __len__(self) : 
        return self.length
    
    def __getitem__(self, item) : 
        txt1, txt2 = self._sample_txt_from_line(item)
        
        wi1, mask_l1 = self._generate_mask(txt1)
        wi2, mask_l2 = self._generate_mask(txt2)        
        wi2, mask_lv2, nsp_l = self._generate_nsp(wi2, mask_l2)
        
        wi, seg = self._concat_sequences(wi1, wi2)
        mask_l = self._concat_sequences(mask_l1, mask_l2, is_mlm=True)
        
        return {
            'text' : wi,
            'mlm' : mask_l,
            'nsp' : nsp_l,
            'seg' : seg,
        }

    def _load_txt_in_memory(self, fname) : 
        docs = open(fname).read().splitlines()
        self.length = len(docs) # take end-line
        return docs
    
    def _sample_txt_from_line(self, idx, get_pair=True) :
        txt1, txt2 = self.docs[idx].split("\t")
        if get_pair :
            return txt1, txt2
        else : 
            return txt2        
    
    def _generate_mask(self, txt) : 
        wi = np.array(self.prep.preprocess(txt))
        
        # random-sampling mask targeted index
        index_arr = np.arange(len(wi))
        np.random.shuffle(index_arr)
        index_arr = index_arr[:int(index_arr.shape[0]*0.15)]

        mask_label = np.zeros(len(wi))
        mask_label[index_arr] = wi[index_arr]
        
        # seperate mask targeted index into 3 conditions
        mask_idx_arr = index_arr[:int(len(index_arr)*0.8)]
        replace_idx_arr = index_arr[int(len(index_arr)*0.8):int(len(index_arr)*0.9)]

        # apply masking
        wi[mask_idx_arr] = self.prep.mask_id        
        random_alloc_wi = np.random.randint(5, self.prep.vocab_size, size=replace_idx_arr.shape[0])        
        wi[replace_idx_arr] = random_alloc_wi
        
        return wi, mask_label

    def _generate_nsp(self, wi, label) : 
        p = random.random()
        if p > 0.5 : # NotNext
            rand_sample_idx = np.random.randint(low=5, high=self.length, size=1).item()
            txt = self._sample_txt_from_line(idx=rand_sample_idx, get_pair=False)
            wi, label = self._generate_mask(txt)
            return wi, label, self.prep.nsp_label['NotNext']
        return wi, label, self.prep.nsp_label['IsNext']

    def _concat_sequences(self, wi1, wi2, is_mlm=False) : 
        pad_length = self.seq_len - 3 - wi1.shape[0] - wi2.shape[0]
        if not is_mlm : # for txt token padding
            
            cated = [self.prep.cls_id] + wi1.tolist() + [self.prep.sep_id] + wi2.tolist() + [self.prep.sep_id]
            cated = cated[:self.seq_len] # list type
            padded = torch.tensor(cated + [self.prep.pad_id] * pad_length).long().contiguous()
            
            seg = [1] * (len(wi1) + 2) + [2] * (len(wi2)+1)
            seg = seg[:self.seq_len] # list type
            seg = torch.tensor(seg + [self.prep.pad_id] * pad_length).long().contiguous()
            
            return padded, seg
        else : # padding for mlm label array
            cated = [0] + wi1.tolist() + [0] + wi2.tolist() + [0]
            cated = cated[:self.seq_len] # list type
            padded = torch.tensor(cated + [0] * pad_length).long().contiguous()            
        return padded
