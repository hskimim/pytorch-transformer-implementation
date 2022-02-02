from . import preprocess as p
from torch.utils.data import Dataset
import torch
import numpy as np

class RobertaIterator(Dataset) :
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
        txt1 = self._sample_txt_from_line(item, get_pair=False)
        wi, mask_l = self._generate_mask(txt1)
        wi, mask_l = self._padding(wi, mask_l)
        return {
            'text' : wi,
            'mlm' : mask_l,
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
        """Dynamic-masking"""
        wi = np.array(self.prep.preprocess(txt))
        
        # random-sampling mask targeted index
        index_arr = np.arange(len(wi))
        np.random.shuffle(index_arr)
        index_arr = index_arr[:int(index_arr.shape[0]*0.15)]

        mask_label = np.zeros(len(wi))
        mask_label[index_arr] = wi[index_arr]
        
        # seperate mask targeted index into 3 conditions (except changed case)
        mask_idx_arr = index_arr[:int(len(index_arr)*0.8)]
        replace_idx_arr = index_arr[int(len(index_arr)*0.8):int(len(index_arr)*0.9)]

        # apply masking
        wi[mask_idx_arr] = self.prep.mask_id
        random_alloc_wi = np.random.randint(low=0, high=self.prep.vocab_size, size=replace_idx_arr.shape[0])
        wi[replace_idx_arr] = random_alloc_wi
        
        return wi, mask_label

    def _padding(self, tokens, mlm):
        pad_length = self.seq_len - 2 - tokens.shape[0] # 3 : [CLS], [SEP]
        cated = [self.prep.cls_id] + tokens.tolist() + [self.prep.sep_id]
        cated = cated[:self.seq_len]  # list type
        padded_token = torch.tensor(cated + [self.prep.pad_id] * pad_length).long().contiguous()

        cated = [0] + mlm.tolist() + [0]
        cated = cated[:self.seq_len]  # list type
        padded_mlm = torch.tensor(cated + [0] * pad_length).long().contiguous()

        return padded_token, padded_mlm

    @property
    def vocab(self):
        return self.prep