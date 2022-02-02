import sentencepiece as spm


class TxtProcessor() : 
    def __init__(self, tokenizer_model_path = '../data/m.model') : 
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(tokenizer_model_path)
        self._nsp_label = {'IsNext':1, 'NotNext':0}
        self.vocab_size = self.sp.vocab_size()
        
    def preprocess(self, txt) : 
        wi_arr = self.sp.EncodeAsIds(txt)[1:] 
        # there is always whitespace in 1st index of wi arr
        return wi_arr

    @property
    def mask_id(self) :
        return self.sp.PieceToId('[MASK]')

    @property
    def unk_id(self) :
        return self.sp.unk_id()

    @property
    def pad_id(self) :
        return self.sp.pad_id()

    @property
    def sep_id(self) :
        return self.sp.PieceToId('[SEP]')

    @property
    def cls_id(self) :
        return self.sp.PieceToId('[CLS]')

    @property
    def nsp_label(self) :
        return self._nsp_label