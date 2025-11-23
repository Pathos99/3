import os
import math
import argparse
import torch
from torch import optim
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_
from torch.nn import functional as F
from model import Encoder, Decoder, Seq2Seq
from utils import load_dataset


def parse_arguments():
    p = argparse.ArgumentParser(description='Hyperparams')
    p.add_argument('-epochs', type=int, default=100,
                   help='number of epochs for train')
    p.add_argument('-batch_size', type=int, default=32,
                   help='number of epochs for train')
    p.add_argument('-lr', type=float, default=0.0001,
                   help='initial learning rate')
    p.add_argument('-grad_clip', type=float, default=10.0,
                   help='in case of gradient explosion')
    return p.parse_args()

def test(model, test_iter, vocab_size, DE, EN):
    with torch.no_grad():
        model.eval
        pad = EN.vocab.stoi['<pad>']
        trgs = []
        preds = []
        for b, batch in enumerate(test_iter):
            src, len_src = batch.src
            trg, len_trg = batch.trg
            # print(src.size(), trg.size())
            src = src.data.cuda()
            trg = trg.data.cuda()
            output = model(src, trg, teacher_forcing_ratio=0.0)
            print(trg, output)
            trgs.append(trg)
            preds.append(output)
        return trgs, preds



def main():
    args = parse_arguments()
    hidden_size = 512
    embed_size = 256
    assert torch.cuda.is_available()

    print("[!] preparing dataset...")
    train_iter, val_iter, test_iter, DE, EN = load_dataset(args.batch_size)
    de_size, en_size = len(DE.vocab), len(EN.vocab)
    print("[TRAIN]:%d (dataset:%d)\t[TEST]:%d (dataset:%d)"
          % (len(train_iter), len(train_iter.dataset),
             len(test_iter), len(test_iter.dataset)))
    print("[DE_vocab]:%d [en_vocab]:%d" % (de_size, en_size))

    print("[!] loading models...")
    encoder = Encoder(de_size, embed_size, hidden_size,
                      n_layers=2, dropout=0.5)
    decoder = Decoder(embed_size, hidden_size, en_size,
                      n_layers=1, dropout=0.5)
    seq2seq = Seq2Seq(encoder, decoder).cuda()
    seq2seq.load_state_dict(torch.load('.save/seq2seq_10.pt'))
    print(seq2seq)
    trgs, preds = test(seq2seq, test_iter, en_size, DE, EN)
    print(len(trgs), len(preds))
    trgs = torch.cat(trgs, -1)
    preds = torch.cat(preds, -1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:
        print("[STOP]", e)
