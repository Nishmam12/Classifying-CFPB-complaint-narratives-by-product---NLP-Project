"""
Recurrent architectures for the CFPB task.

The one substantive change from the original notebook class is that the final
hidden state is now read through pack_padded_sequence. Previously the RNN was
walked across all 256 positions including trailing <PAD>, so for a median
119-word complaint the classifier read a state produced by ~137 padding steps.
That penalises every model and is the most likely cause of the SimpleRNN
collapse (val accuracy 0.0997, below the 1/9 random baseline).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, TensorDataset

from common import log, metrics

CELLS = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}


class RecurrentClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, cell_type="lstm",
                 num_layers=1, bidirectional=False, dropout=0.3, pretrained_weights=None):
        super().__init__()
        self.cell_type = cell_type.lower()
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        if pretrained_weights is not None:
            self.embedding = nn.Embedding.from_pretrained(
                pretrained_weights, freeze=False, padding_idx=0
            )
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        if self.cell_type not in CELLS:
            raise ValueError(f"Unsupported cell_type: {cell_type}")
        self.rnn = CELLS[self.cell_type](
            embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True,
            bidirectional=bidirectional, dropout=dropout if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * self.num_directions, num_classes)

    def forward(self, x, lengths):
        embedded = self.embedding(x)
        packed = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        if self.cell_type == "lstm":
            _, (hidden, _) = self.rnn(packed)
        else:
            _, hidden = self.rnn(packed)

        if self.bidirectional:
            last = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            last = hidden[-1]
        return self.fc(self.dropout(last))


def make_loader(ids, lens, y, batch_size, shuffle):
    return DataLoader(
        TensorDataset(
            torch.from_numpy(ids.astype(np.int64)),
            torch.from_numpy(lens.astype(np.int64)),
            torch.from_numpy(np.asarray(y).astype(np.int64)),
        ),
        batch_size=batch_size,
        shuffle=shuffle,
    )


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds = []
    for xb, lb, _ in loader:
        logits = model(xb.to(device, non_blocking=True), lb)
        preds.append(torch.argmax(logits, dim=1).cpu().numpy())
    return np.concatenate(preds)


def train_model(model, train_loader, eval_loader, criterion, optimizer, epochs, device,
                y_eval, label="", restore_best=False):
    """Train, returning the metrics of the best epoch and that epoch's predictions.

    The original loop returned best-epoch metrics but left the caller holding
    the final-epoch weights. With restore_best=True the best-epoch weights are
    reloaded into the model before returning, so that callers which go on to
    score a *different* split (test) use the same model the metrics describe.
    """
    import copy

    model = model.to(device)
    best = {"macro_f1": -1.0}
    best_preds = None
    best_state = None
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for xb, lb, yb in train_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            optimizer.zero_grad()
            loss = criterion(model(xb, lb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total += loss.item()
        preds = predict(model, eval_loader, device)
        m = metrics(y_eval, preds)
        log(f"    {label} epoch {epoch}/{epochs} loss={total/len(train_loader):.4f} "
            f"macro_f1={m['macro_f1']:.4f} acc={m['accuracy']:.4f}")
        if m["macro_f1"] > best["macro_f1"]:
            best, best_preds = m, preds
            if restore_best:
                best_state = copy.deepcopy(model.state_dict())
    if restore_best and best_state is not None:
        model.load_state_dict(best_state)
    return best, best_preds


def build(cfg, emb, num_classes, device):
    model = RecurrentClassifier(
        vocab_size=emb.shape[0],
        embedding_dim=emb.shape[1],
        hidden_dim=cfg["hidden_dim"],
        num_classes=num_classes,
        cell_type=cfg["cell_type"],
        bidirectional=cfg["bidirectional"],
        dropout=cfg["dropout"],
        pretrained_weights=torch.from_numpy(emb),
    )
    return model.to(device), optim.Adam(model.parameters(), lr=cfg["lr"])
