from bert.trainer.pretrain import Pretrain as BertTrain
from albert.dataset import iterator

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

class Pretrain(BertTrain) :

    def load_data(self, train_fname, valid_fname, tokenizer_fname):
        self._train_dataset = iterator.AlbertIterator(filename=train_fname,
                                              tokenizer_model_path=tokenizer_fname,
                                              seq_len=self._seq_len)
        self._train_dataloader = DataLoader(self._train_dataset, batch_size=self._batch_size, shuffle=True)

        self._valid_dataset = iterator.AlbertIterator(filename=valid_fname,
                                              tokenizer_model_path=tokenizer_fname,
                                              seq_len=self._seq_len)
        self._valid_dataloader = DataLoader(self._valid_dataset, batch_size=self._batch_size, shuffle=True)
        self._load_data = True
