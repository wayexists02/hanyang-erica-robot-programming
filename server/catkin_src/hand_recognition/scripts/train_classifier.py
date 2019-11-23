import torch
from torch import nn, optim
from models.classifierVGG import ClassifierVGG
# from models.classifier import Classifier
from models.dataloader import DataLoader
from env import *


CLF_CKPT_PATH = NOTHING_CLF_CKPT_PATH
# CLF_CKPT_PATH = SIGN_CLF_CKPT_PATH

def main():
    train_loader = DataLoader(train=True, noise=True, flip=True, batch_size=BATCH_SIZE)
    valid_loader = DataLoader(train=False, noise=False, flip=False, batch_size=BATCH_SIZE)

    model = ClassifierVGG(NOTHING_CAT).cuda()
    model.load(CLF_CKPT_PATH)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=ETA, weight_decay=1e-1)

    top_valid_acc = 0.0

    for e in range(EPOCHS):
        train_loss = 0.0
        train_acc = 0.0
        valid_loss = 0.0
        valid_acc = 0.0

        for x, y in train_loader.next_batch():
            x = torch.FloatTensor(x).cuda()
            y = torch.LongTensor(y).cuda()

            logps = model(x)
            loss = criterion(logps, y)
            
            with torch.no_grad():
                ps = torch.exp(logps)
                ps_k, top_k = ps.topk(1, dim=1)
                equal = top_k == y.view(*top_k.size())
                
                train_acc += torch.mean(equal.type(torch.FloatTensor))
                train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            model.eval()

            for x, y in valid_loader.next_batch():
                x = torch.FloatTensor(x).cuda()
                y = torch.LongTensor(y).cuda()

                logps = model(x)
                loss = criterion(logps, y)

                ps = torch.exp(logps)
                ps_k, top_k = ps.topk(1, dim=1)
                equal = top_k == y.view(*top_k.size())
                
                valid_acc += torch.mean(equal.type(torch.FloatTensor))
                valid_loss += loss.item()

            train_loss /= len(train_loader)
            train_acc /= len(train_loader)
            valid_loss /= len(valid_loader)
            valid_acc /= len(valid_loader)

            print(f"Epoch {e+1}/{EPOCHS}")
            print(f"Train loss: {train_loss:.8f}")
            print(f"Train acc: {train_acc:.4f}")
            print(f"Valid loss: {valid_loss:.8f}")
            print(f"Valid acc: {valid_acc:.4f}")

            if top_valid_acc <= valid_acc:
                top_valid_acc = valid_acc
                model.save(CLF_CKPT_PATH, top_valid_acc)

            model.train()


if __name__ == "__main__":
    main()
