import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import argparse
import pandas as pd
import numpy as np
from datetime import datetime
import json
import gc
import random
import re
from tqdm import tqdm

from sklearn.model_selection import train_test_split

from transformers import BartForConditionalGeneration, BartConfig
from transformers import PreTrainedTokenizerFast
from pytorch_pretrained_bert.optimization import BertAdam

from text.dataset import SummaryDataset

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from gcp.bucket import Bucket_processor
from ABS_PATH import ABS_PATH

KEY_PATH = ABS_PATH + "/../gcp_auth_key/capstone-352301-ef799c59a451.json"


def accuracy_function(real, pred):
    accuracies = torch.eq(real, torch.argmax(pred, dim=2))
    mask = torch.logical_not(torch.eq(real, -100))
    accuracies = torch.logical_and(mask, accuracies)
    accuracies = accuracies.clone().detach()
    mask = mask.clone().detach()

    return torch.sum(accuracies)/torch.sum(mask)

def loss_function(real, pred, criterion = nn.CrossEntropyLoss()):
    mask = torch.logical_not(torch.eq(real, -100))
    loss_ = criterion(pred.permute(0,2,1), real)
    mask = mask.clone().detach()
    loss_ = mask * loss_

    return torch.sum(loss_)/torch.sum(mask)


def train_step(batch_item, epoch, batch, training, model, optimizer):#, device):
    input_ids = batch_item['input_ids']#.to(device)
    attention_mask = batch_item['attention_mask']#.to(device)
    decoder_input_ids = batch_item['decoder_input_ids']#.to(device)
    decoder_attention_mask = batch_item['decoder_attention_mask']#.to(device)
    labels = batch_item['labels']#.to(device)
    # print(input_Ids)
    # print(attention_mask)
    # print(decoder_input_ids)
    # print(labels)
    if training is True:
        model.train()
        model.model.encoder.config.gradient_checkpointing = True
        model.model.decoder.config.gradient_checkpointing = True
        optimizer.zero_grad()
        '''
        with torch.cuda.amp.autocast():
            output = model(input_ids=input_ids,
                           attention_mask=attention_mask,
                           decoder_input_ids=decoder_input_ids,
                           decoder_attention_mask=decoder_attention_mask,
                           labels=labels, return_dict=True)

            loss = output.loss
            # loss2 = loss_function(labels, output.logits)
        '''
        #print('check7')
        output = model(input_ids=input_ids,
                       attention_mask=attention_mask,
                       decoder_input_ids=decoder_input_ids,
                       decoder_attention_mask=decoder_attention_mask,
                       labels=labels, return_dict=True)
        #print('check8')
        #print(output)
        loss = output.loss
        acc = accuracy_function(labels, output.logits)

        loss.backward()

        optimizer.step()


        lr = optimizer.param_groups[0]["lr"]
        return loss, acc, round(lr, 10)
    else:
        model.eval()
        with torch.no_grad():
            output = model(input_ids=input_ids,
                           attention_mask=attention_mask,
                           decoder_input_ids=decoder_input_ids,
                           decoder_attention_mask=decoder_attention_mask,
                           labels=labels, return_dict=True)
            loss = output.loss
            # loss = loss_function(labels, output.logits)

        acc = accuracy_function(labels, output.logits)

        return loss, acc

def main(args,model_name_list):
    bucket_processor = Bucket_processor(args.auth_key_path, args.gcp_project_id, args.gcs_bucket_name)
    # gcs에 data가 있다고 가정함
    bucket_processor.download_from_bucket(args.bucket_data_path, args.local_save_path)


    dt_now = datetime.now()

    # gpu count
    #n_gpu = torch.cuda.device_count()

    # random seed 설정
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    #if n_gpu > 0:
    #    torch.cuda.manual_seed_all(args.seed)
    # GPU 사용
    #device = torch.device("cpu")  # cuda:0
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if (device.type == 'cuda') and (torch.cuda.device_count() > 1):
        print('Multi GPU activate')
        net = nn.DataParallel(netG, device_ids=list(range(NGPU)))
    '''
    tokenizer= None
    model = None
    for model_name in model_name_list:
        tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)
        model = BartForConditionalGeneration.from_pretrained(model_name)#.to(device)
        config = BartConfig.from_pretrained(model_name)

    data_df, train_path = pd.DataFrame(), args.train_data_name
    print(args.data_path + train_path)
    data_df = pd.read_csv(args.data_path + train_path)
    #data_df = processor.get_train_examples(args.data_path + train_path)

    # split train, dev from train_data
    train_data, eval_data = train_test_split(data_df, test_size=0.1, shuffle=True, random_state=args.seed)

    train_dataset = SummaryDataset(train_data, args.max_len, tokenizer)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = args.batch_size, num_workers=1, shuffle=True)

    eval_dataset = SummaryDataset(eval_data, args.max_len, tokenizer)
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size= args.batch_size, num_workers=1, shuffle=True)

    del data_df

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    # optimizer 설정
    num_train_optimization_steps = int(len(train_data) / args.batch_size) * args.num_epochs

    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=args.learning_rate,
                         warmup=args.warmup_ratio,
                         t_total=num_train_optimization_steps)

    # optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    loss_plot, val_loss_plot = [], []
    acc_plot, val_acc_plot = [], []

    for epoch in range(args.num_epochs):
        total_loss, total_val_loss = 0, 0
        total_acc, total_val_acc = 0, 0
        tqdm_dataset = tqdm(enumerate(train_dataloader))
        pytorch_total_params = sum(p.numel() for p in model.parameters())
        print(pytorch_total_params)
        training = True
        for batch, batch_item in tqdm_dataset:
            # for item, value in batch_item.items():
            # print(item)
            # print(value.shape)
            batch_loss, batch_acc, lr = train_step(batch_item, epoch, batch, training,model,optimizer)#,device)
            total_loss += batch_loss
            total_acc += batch_acc

            tqdm_dataset.set_postfix({
                'Epoch': epoch + 1,
                'LR': lr,
                'Loss': '{:06f}'.format(batch_loss.item()),
                'Total Loss': '{:06f}'.format(total_loss / (batch + 1)),
                'Total ACC': '{:06f}'.format(total_acc / (batch + 1))
            })
        loss_plot.append(total_loss / (batch + 1))
        acc_plot.append(total_acc / (batch + 1))

        #torch.save(model.state_dict(), 'epoch {} weight.pt'.format(epoch + 1))

        tqdm_dataset = tqdm(enumerate(eval_dataloader))
        training = False
        for batch, batch_item in tqdm_dataset:
            batch_loss, batch_acc = train_step(batch_item, epoch, batch, training, model,optimizer)#,device)
            # batch_item, epoch, batch, training,model,optimizer,device
            total_val_loss += batch_loss
            total_val_acc += batch_acc

            tqdm_dataset.set_postfix({
                'Epoch': epoch + 1,
                'Val Loss': '{:06f}'.format(batch_loss.item()),
                'Total Val Loss': '{:06f}'.format(total_val_loss / (batch + 1)),
                'Total Val ACC': '{:06f}'.format(total_val_acc / (batch + 1))
            })
        val_loss_plot.append(total_val_loss / (batch + 1))
        val_acc_plot.append(total_val_acc / (batch + 1))

    print("\nbye")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--num_epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_len", type=int, default=1024)
    parser.add_argument("--hidden_size", type=int, default=768)

    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--dropout_ratio", type=float, default=0.1)

    parser.add_argument("--max_grad_norm", type=int, default=1)
    parser.add_argument("--log_interval", type=int, default=200)
    parser.add_argument("--early_stopping_patience", type=int, default=5)

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--data_path", type=str, default=ABS_PATH + '/../data')
    parser.add_argument("--output_path", type=str, default=ABS_PATH + '/../output')
    parser.add_argument("--result_path", type=str, default=ABS_PATH + '/../result')
    parser.add_argument("--personal_model_path", type=str, default= ABS_PATH + '/../personal_models')

    parser.add_argument("--train_data_name", type=str, default='/train/preprocessed/beauty_health.csv')
    #parser.add_argument("--train_data_name", type=str, default="/save/save.csv")

    parser.add_argument("--train_data_num", type=int, default= 1000)
    parser.add_argument("--dev_data_num", type=int ,default=300)

    parser.add_argument("--is_personal_model", type=bool, default=False)
    parser.add_argument("--is_ensemble_test", type=bool, default=True)

    parser.add_argument("--auth_key_path", type=str, default=KEY_PATH)
    parser.add_argument("--gcp_project_id", type=str, default="capstone-352301")
    parser.add_argument("--gcs_bucket_name", type=str, default="capstone_mlops_data")

    parser.add_argument("--bucket_data_path", type=str, default="capstone_data/text/train/preprocessed/beauty_health.csv")
    parser.add_argument("--local_save_path", type=str, default=ABS_PATH + "/../data/train/preprocessed/beauty_health.csv")

    # parser.add_argument("--model_name", type=str, default='kykim/electra-kor-base')
    # parser.add_argument("--model_name", type=str, default='monologg/koelectra-base-v3-discriminator')
    # parser.add_argument("--model_name", type=str, default='klue/roberta-large')
    # parser.add_argument("--model_name", type=str, default='tunib/electra-ko-base')

    # parser.add_argument("--model_name", type=str, default='beomi/KcELECTRA-base')

    model_name_list = [
                       "ainize/kobart-news"
                       ]

    args = parser.parse_args()
    main(args, model_name_list)