"""Модуль содержит в себе реализацию CRNN модели."""
from typing import List

import timm
import torch
from torch.nn import TransformerDecoderLayer, TransformerDecoder


class CRNN(torch.nn.Module):
    """Реализует CRNN модель для OCR задачи.
    
    CNN-backbone берется из timm, в RNN части стоит GRU.
    """

    def __init__(
            self, 
            cnn_backbone_name: str,
            cnn_backbone_pretrained: bool,
            cnn_output_size: int,
            rnn_features_num: int,
            rnn_dropout: float,
            rnn_bidirectional: bool,
            rnn_num_layers: int,
            num_classes: int,
            
    ) -> None:
        """
        Parameters: (лучше вынести в config, в этом примере перечислены параметры для простоты)
        
            cnn_backbone_name: имя backbone-а 
            cnn_backbone_pretrained: True, чтобы брать предобученный бэкбон, иначе False
            cnn_output_size: размер выходного тензора из CNN (можно сделать прогон в init-е и посчитать)
            rnn_features_num: размер выхода каждого GRU слоя
            rnn_dropout: если не ноль, добавляет dropout после каждого GRU слоя с указанным значением
            rnn_bidirectional: True, чтобы использовать двунаправленную GRU, иначе False
            rnn_num_layers: количество слоев GRU,
            num_classes: Количество классов - длина алфавита + 1.
        """
        super().__init__()

        self.backbone = timm.create_model(
            cnn_backbone_name, pretrained=cnn_backbone_pretrained,
        )
        self.backbone.global_pool = torch.nn.Identity()
        self.backbone.fc = torch.nn.Identity()

        self.gate = torch.nn.Linear(cnn_output_size, rnn_features_num)

        self.rnn = torch.nn.GRU(
            rnn_features_num,
            rnn_features_num,
            dropout=rnn_dropout,
            bidirectional=rnn_bidirectional,
            num_layers=rnn_num_layers,
        )

        classifier_in_features = rnn_features_num
        if rnn_bidirectional:
            classifier_in_features = 2 * rnn_features_num

        self.fc = torch.nn.Linear(classifier_in_features, num_classes)
        self.softmax = torch.nn.LogSoftmax(dim=2)
        self.init_weights()

    def init_weights(self):
        torch.nn.init.kaiming_normal_(self.gate.weight, mode='fan_out', nonlinearity='relu')
        if self.gate.bias is not None:
            self.gate.bias.data.fill_(0.01)

        for name, param in self.rnn.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

        torch.nn.init.xavier_uniform_(self.fc.weight)
        if self.fc.bias is not None:
            self.fc.bias.data.fill_(0.01)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:    # noqa: WPS210
        """
        Прямой проход по графу вычислений.

        Parameters:
            tensor: Изображение-тензор.

        Returns:
            Сырые логиты для каждой позиции в карте признаков.
        """
        cnn_features = self.backbone(tensor)
        batch_size, channels, height, width = cnn_features.shape
        cnn_features = cnn_features.view(
            batch_size, height * channels, width,
        ).permute(2, 0, 1)
        cnn_features = torch.nn.functional.relu(self.gate(cnn_features))
        rnn_output, _ = self.rnn(cnn_features)
        logits = self.fc(rnn_output)
        output = self.softmax(logits)
        return output

    # def decode_output(self, pred: torch.Tensor, vocab: str) -> List[str]:
    #     texts = []
    #     index2char = {idx + 1: char for idx, char in enumerate(vocab)}
    #     index2char[0] = "eos"
    #     for idx in range(pred.shape[1]):
    #         classes_b = pred[:, idx, :].argmax(dim=1).cpu().numpy().tolist()
    #         finished_chars = []
    #         last_char = None
    #         meet_other_than_zero = False
    #         for c_idx, ch in enumerate(map(lambda x: index2char[x], classes_b)):
    #             if ch == "eos" and not meet_other_than_zero:
    #                 continue
    #
    #             meet_other_than_zero = True
    #             if ch == "eos":
    #                 break
    #
    #             if last_char == ch:
    #                 continue
    #
    #             last_char = ch
    #             finished_chars.append(ch)
    #
    #         texts.append("".join(finished_chars))
    #
    #     return texts

    def decode_output(self, pred: torch.Tensor, vocab: str) -> List[str]:
        texts = []
        index2char = {idx + 1: char for idx, char in enumerate(vocab)}
        index2char[0] = ""
        for idx in range(pred.shape[1]):
            classes_b = pred[:, idx, :].argmax(dim=1).cpu().numpy().tolist()
            chars = list(map(lambda x: index2char[x], classes_b))[:13]
            texts.append("".join(chars))
        return texts


class TransformerOcr(torch.nn.Module):
    def __init__(
            self,
            cnn_backbone_name: str,
            cnn_backbone_pretrained: bool,
            cnn_output_size: int,
            transformer_features_num: int,
            transformer_dropout: float,
            transformer_nhead: int,
            transformer_num_layers: int,
            num_classes: int,

    ) -> None:
        super().__init__()
        self.num_classes = num_classes

        # Initialize the CNN backbone
        self.backbone = timm.create_model(
            cnn_backbone_name, pretrained=cnn_backbone_pretrained,
        )
        self.backbone.global_pool = torch.nn.Identity()
        self.backbone.fc = torch.nn.Identity()

        # Project CNN output to transformer feature size
        self.gate = torch.nn.Linear(cnn_output_size, transformer_features_num)

        # Transformer Decoder Setup
        decoder_layer = TransformerDecoderLayer(
            d_model=transformer_features_num,
            nhead=transformer_nhead,
            dropout=transformer_dropout,
        )
        self.transformer_decoder = TransformerDecoder(
            decoder_layer,
            num_layers=transformer_num_layers
        )

        # Classifier
        self.fc = torch.nn.Linear(transformer_features_num, num_classes)
        self.softmax = torch.nn.LogSoftmax(dim=2)
        self.init_weights()

    def init_weights(self):
        torch.nn.init.kaiming_normal_(self.gate.weight, mode='fan_out', nonlinearity='relu')
        if self.gate.bias is not None:
            self.gate.bias.data.fill_(0.01)

        torch.nn.init.xavier_uniform_(self.fc.weight)
        if self.fc.bias is not None:
            self.fc.bias.data.fill_(0.01)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        # Compute CNN features
        cnn_features = self.backbone(tensor)
        batch_size, channels, height, width = cnn_features.shape
        cnn_features = cnn_features.view(
            batch_size, height * channels, width,
        ).permute(2, 0, 1)
        cnn_features = torch.nn.functional.relu(self.gate(cnn_features))

        # Prepare a dummy memory for the transformer if needed
        memory = torch.zeros_like(cnn_features)

        # Decode features through the transformer
        transformer_output = self.transformer_decoder(cnn_features, memory)

        # Apply classifier
        logits = self.fc(transformer_output)
        output = self.softmax(logits)
        return output


    def decode_output(self, pred: torch.Tensor, vocab: str) -> List[str]:
        texts = []
        index2char = {idx + 1: char for idx, char in enumerate(vocab)}
        index2char[0] = ""
        index2char[len(vocab) + 1] = "<eos>"
        for idx in range(pred.shape[1]):
            classes_b = pred[:, idx, :].argmax(dim=1).cpu().numpy().tolist()
            chars = list(map(lambda x: index2char[x], classes_b))
            text = "".join(chars).split("<eos>")[0]
            texts.append(text)
        return texts


