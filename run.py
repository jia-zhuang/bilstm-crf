import numpy as np
import argparse
from tqdm import tqdm, trange
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchtext.vocab import GloVe

from seqeval.metrics import precision_score, recall_score, f1_score, classification_report

from utils import NerDataset, get_labels, collate_fn
from model import LstmCrf

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train(args, model, dataset):
    tb_writer = SummaryWriter()

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Training
    global_step = 0
    tr_loss, logging_loss = 0., 0.
    for _ in trange(args.epochs, desc="Epoch"):
        for batch in tqdm(dataloader):
            # forward
            inputs = {k: v.to(args.device) for k, v in batch.items()}

            outputs = model(**inputs)
            loss = outputs[0]
            tr_loss += loss.item()

            if global_step % args.logging_steps == 0:
                tb_writer.add_scalar('loss', (tr_loss - logging_loss) / args.logging_steps, global_step)
                logging_loss = tr_loss

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1

    tb_writer.close()

'''
def align_predictions(predictions, label_ids, label_list, ignore_index=-100):  # B x L
    preds = np.argmax(predictions, axis=2)

    batch_size, seq_len = preds.shape

    out_label_list = [[] for _ in range(batch_size)]
    preds_list = [[] for _ in range(batch_size)]

    for i in range(batch_size):
        for j in range(seq_len):
            if label_ids[i, j] != ignore_index:
                out_label_list[i].append(label_list[label_ids[i, j]])
                preds_list[i].append(label_list[preds[i, j]])
    
    return preds_list, out_label_list
'''
def align_predictions(predictions, label_ids, label_list):
    ''' for CRF
        predictions: List[List[int]], B x L
        label_ids: B x L
    '''
    batch_size = len(predictions)
    out_label_list = [[] for _ in range(batch_size)]
    preds_list = [[] for _ in range(batch_size)]

    for i in range(batch_size):
        for j in range(len(predictions[i])):
            out_label_list[i].append(label_list[label_ids[i, j]])
            preds_list[i].append(label_list[predictions[i][j]])    
    
    return preds_list, out_label_list
    

def eval(args, model, dataset, label_list):
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    all_true_labels, all_pred_labels = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            inputs = {k: v.to(args.device) for k, v in batch.items()}
            outputs = model(**inputs)
            
            #predictions = outputs[1].permute(1, 0, 2).detach().cpu().numpy()
            predictions = outputs[1]
            label_ids = batch['label_ids'].permute(1, 0).detach().cpu().numpy()

            preds_list, out_label_list = align_predictions(predictions, label_ids, label_list)

            all_true_labels += out_label_list
            all_pred_labels += preds_list

    report = classification_report(all_true_labels, all_pred_labels)
    logger.info(report)
    

    return {
        'precision': precision_score(all_true_labels, all_pred_labels),
        'recall': recall_score(all_true_labels, all_pred_labels),
        'f1': f1_score(all_true_labels, all_pred_labels),
    } 


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default=None, type=str, required=True)
    parser.add_argument('--w2v_path', default=None, type=str, required=True)
    parser.add_argument('--labels', default=None, type=str, required=True)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--epochs', default=3, type=int)
    parser.add_argument('--logging_steps', default=20, type=int)
    parser.add_argument('--learning_rate', default=5e-3, type=float)
    args = parser.parse_args()

    args.device = torch.device('cuda')

    labels = get_labels(args.labels)
    glove = GloVe(cache=args.w2v_path)

    # model
    model = LstmCrf(w2v=glove, num_tags=len(labels), hidden_dim=512)
    model.to(args.device)
    
    # dataset
    train_dataset = NerDataset(args.data_dir, labels, glove, mode='train')
    eval_dataset = NerDataset(args.data_dir, labels, glove, mode='dev')

    # train
    train(args, model, train_dataset)

    # eval
    result = eval(args, model, eval_dataset, labels)

    print(result)


if __name__ == '__main__':
    main()