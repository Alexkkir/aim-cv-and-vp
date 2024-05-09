from collections import defaultdict
from tqdm.notebook import tqdm
from typing import Tuple

import torch
from torch import optim



def train_model(
    model: object,
    train_loader: object,
    test_loader: object,
    epochs: int,
    lr: float,
    use_tqdm: bool = False,
    use_cuda: bool = False,
    loss_key: str = "total_loss",
) -> Tuple[dict, dict]:
    
    fmt_head = "{:<20s} {:<20s}"
    fmt_val =  "{:<20.4f} {:<20.4f}"

    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_losses = defaultdict(list)
    test_losses = defaultdict(list)
    forrange = tqdm(range(epochs)) if use_tqdm else range(epochs)
    if use_cuda:
        model = model.cuda()
        
    print(fmt_head.format('Train Loss', 'Val Loss'))

    for epoch in tqdm(forrange):
        model.train()
        train_loss = model.train_epoch(train_loader, optimizer, use_cuda, loss_key)
        test_loss = model.eval_model(test_loader, use_cuda)

        for k in train_loss.keys():
            train_losses[k].extend(train_loss[k])
            test_losses[k].append(test_loss[k])

        print(fmt_val.format(train_loss[loss_key][-1], test_loss[loss_key]))
    return dict(train_losses), dict(test_losses)
