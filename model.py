import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils.rnn as rnn_utils
from torchcrf import CRF


class LstmCrf(nn.Module):

    def __init__(self, w2v, num_tags, hidden_dim):
        super().__init__()
        
        self.num_tags = num_tags

        self.word_embeds = nn.Embedding.from_pretrained(w2v.vectors)
        self.lstm = nn.LSTM(w2v.dim, hidden_dim // 2, num_layers=1, bidirectional=True)
        self.hidden2tag = nn.Linear(hidden_dim, self.num_tags)
        self.crf = CRF(num_tags)

    def forward(self, input_ids, seq_lengths, label_ids):  # L x B
        embeds = self.word_embeds(input_ids)
        packed_embeds = rnn_utils.pack_padded_sequence(embeds, seq_lengths)
        packed_lstm, _ = self.lstm(packed_embeds)
        lstm, _ = rnn_utils.pad_packed_sequence(packed_lstm)  # L x B x H

        tag_logits = self.hidden2tag(lstm)  # L x B x S

        mask = self.get_mask(seq_lengths).to(tag_logits.device)
        loss = -self.crf(tag_logits, label_ids, mask=mask)

        # decode
        pred_ids = self.crf.decode(tag_logits, mask=mask)

        '''
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(tag_logits.view(-1, self.num_tags), label_ids.view(-1))
        '''

        return loss, pred_ids
    
    @staticmethod
    def get_mask(seq_lengths):
        max_len = seq_lengths[0]
        all_mask = []
        for length in seq_lengths:
            mask = [1] * length + [0] * (max_len - length)
            all_mask.append(mask)
        return torch.tensor(all_mask, dtype=torch.uint8).permute(1, 0)  # L x B
