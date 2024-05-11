import random
from itertools import zip_longest

import wandb
from torchsummary import summary

from crnn import CRNN
from generator.dataset_generator import BarcodeDataset, TestBarcodeDataset
import torch
import numpy as np

run = wandb.init(project="barcode_detection_recognition")
config = wandb.config
wandb.config.vocab = '0123456789'
wandb.config.seed = 44
wandb.config.bs = 64
wandb.config.epoch_size = 1024
wandb.config.lr = 5e-4
wandb.config.epochs = 48

random.seed(wandb.config.seed)
np.random.seed(wandb.config.seed)
torch.manual_seed(wandb.config.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(wandb.config.seed)

criterion = torch.nn.CTCLoss(zero_infinity=True)

crnn = CRNN(
    cnn_backbone_name='resnet18d',
    cnn_backbone_pretrained=True,
    cnn_output_size=4608,
    rnn_features_num=128,
    rnn_dropout=0.1,
    rnn_bidirectional=True,
    rnn_num_layers=2,
    num_classes=11
)
crnn.train()
summary(model=crnn, depth=4)
wandb.watch(crnn, log_freq=10, log="parameters", log_graph=True)

barcode_loader = torch.utils.data.DataLoader(
    BarcodeDataset(epoch_size=wandb.config.epoch_size, vocab=wandb.config.vocab),
    batch_size=wandb.config.bs
)

test_barcode_loader = torch.utils.data.DataLoader(
    TestBarcodeDataset(vocab=wandb.config.vocab,
                       directory='/Users/rimma_vakhreeva/PycharmProjects/barcode_detection_recognition/images/cropped_test_images'),
)

optimizer = torch.optim.Adam(crnn.parameters(), lr=wandb.config.lr)
lambda1 = lambda epoch: 1 - epoch / 700
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def calculate_accuracy(gt_texts, pred_texts):
    accuracy = []
    for gt_text, pred_text in zip(gt_texts, pred_texts):
        correct, text_length = 0, 0
        for gt_ch, pred_ch in zip_longest(gt_text, pred_text):
            text_length += 1
            if gt_ch == pred_ch:
                correct += 1
        accuracy.append(correct / text_length)
    return np.mean(accuracy)


if __name__ == '__main__':
    crnn.to(device)
    crnn.train()

    for epoch in range(wandb.config.epochs):
        for iter_n, (input_images, targets, target_lengths, gt_texts) in enumerate(barcode_loader):
            input_images = input_images.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)

            output = crnn(input_images)
            pred_texts = crnn.decode_output(output, wandb.config.vocab)
            train_acc = calculate_accuracy(gt_texts, pred_texts)

            input_lengths = [output.size(0) for _ in input_images]
            input_lengths = torch.LongTensor(input_lengths)
            target_lengths = torch.flatten(target_lengths)

            loss = criterion(output, targets, input_lengths, target_lengths)
            if loss.isinf() or loss.isnan():
                print(f"loss is inf or nan, skipping the batch")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(crnn.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            crnn.eval()
            with torch.no_grad():
                test_losses = []
                for input_images, targets, target_lengths, gt_texts in test_barcode_loader:
                    input_images = input_images.to(device)
                    targets = targets.to(device)
                    target_lengths = target_lengths.to(device)

                    output = crnn(input_images)
                    pred_texts = crnn.decode_output(output, wandb.config.vocab)
                    test_acc = calculate_accuracy(gt_texts, pred_texts)

                    input_lengths = [output.size(0) for _ in input_images]
                    input_lengths = torch.LongTensor(input_lengths)
                    target_lengths = torch.flatten(target_lengths)

                    loss = criterion(output, targets, input_lengths, target_lengths)
                    if not (loss.isinf() or loss.isnan()):
                        test_losses.append(loss)
                test_loss = sum(test_losses) / len(test_losses)
            crnn.train()

            run.log({
                "epoch": epoch,
                "iter_n": iter_n,
                "train_loss": loss.item(),
                "lr": optimizer.param_groups[0]['lr'],
                "train_acc": train_acc,
                "test_loss": test_loss,
                "test_acc": test_acc
            })
            print(f"epoch {epoch}\t"
                  f"loss {loss.item():.5f}\t"
                  f"lr: {optimizer.param_groups[0]['lr']:.5f}\t"
                  f"train_acc: {train_acc:.5f}\t"
                  f"test_loss: {test_loss:.5f}\t"
                  f"test_acc: {test_acc:.5f}")

            cp = crnn.state_dict()
            torch.save(cp, "../../crnn_last.pt")
