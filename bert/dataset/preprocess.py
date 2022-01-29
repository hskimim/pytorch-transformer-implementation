import sentencepiece as spm
import pandas as pd # to load pickle
import numpy as np

class TxtProcessor() : 
    def __init__(self, tokenizer_model_path = '../data/m.model') : 
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load('../data/m.model')
        self._nsp_label = {'IsNext':1, 'NotNext':0}
        
    def _wi_map(self, word) : 
        return self._lemma_dict.get(word, self._lemma_dict['<unk>'])
    
    def sent_tokenize(self, txt) : 
        tokens = self.tokenizer(txt)
        return tokens
    
    def word_indexing(self, word) : 
        return self._wi_map(word)
    
    def preprocess(self, txt) : 
        tokens = self.sent_tokenize(txt)
        wi_arr = np.vectorize(self._wi_map)(tokens)
        return wi_arr
    
    @property
    def lemma_dict(self) : 
        return self._lemma_dict
    
    @property
    def mask_id(self) : 
        return self._lemma_dict['<mask>']
    
    @property
    def unk_id(self) : 
        return self._lemma_dict['<unk>']

    @property
    def pad_id(self) : 
        return self._lemma_dict['<pad>']

    @property
    def sep_id(self) : 
        return self._lemma_dict['<sep>']
    
    @property
    def cls_id(self) : 
        return self._lemma_dict['<cls>']
    
    @property
    def nsp_label(self) : 
        return self._nsp_label