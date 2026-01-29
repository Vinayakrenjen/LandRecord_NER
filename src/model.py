import torch
import torch.nn as nn
from torchcrf import CRF

class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim, dropout=0.5, pretrained_embeddings=None):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.dropout = nn.Dropout(dropout)

        # 1. Embedding Layer
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Load pre-trained weights if provided
        if pretrained_embeddings is not None:
            self.word_embeds.weight.data.copy_(pretrained_embeddings)
            # Keep fine-tuning enabled
            self.word_embeds.weight.requires_grad = True

        # 2. LSTM Layer
        # bidirectional=True means it reads Left->Right AND Right->Left
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True, batch_first=True, dropout=dropout if dropout > 0 else 0)

        # 3. Linear Layer (maps LSTM output to Tag Space)
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # 4. CRF Layer (The Logic Layer)
        self.crf = CRF(self.tagset_size, batch_first=True)

    def forward(self, sentence, tags, mask):
        # 1. Get Embeddings
        embeds = self.word_embeds(sentence) # (Batch, Seq, EmbDim)
        embeds = self.dropout(embeds) # Apply Dropout

        # 2. Run LSTM
        lstm_out, _ = self.lstm(embeds) # (Batch, Seq, HiddenDim)
        lstm_out = self.dropout(lstm_out) # Apply Dropout

        # 3. Project to Tag Space
        emissions = self.hidden2tag(lstm_out) # (Batch, Seq, TagSetSize)

        # 4. Calculate Loss (Negative Log Likelihood)
        # multiply by -1 because CRF returns log-likelihood, we want to minimize NLL
        loss = -self.crf(emissions, tags, mask=mask.bool(), reduction='mean')
        return loss

    def predict(self, sentence, mask):
        # 1. Get Embeddings
        embeds = self.word_embeds(sentence)
        embeds = self.dropout(embeds)

        # 2. Run LSTM
        lstm_out, _ = self.lstm(embeds)
        lstm_out = self.dropout(lstm_out)

        # 3. Project to Tag Space
        emissions = self.hidden2tag(lstm_out)

        # 4. Decode Best Path
        best_paths = self.crf.decode(emissions, mask=mask.bool())
        return best_paths
