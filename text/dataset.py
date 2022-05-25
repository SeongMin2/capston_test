import torch
from torch.utils.data import Dataset

class SummaryDataset(Dataset):
    def __init__(self, dataframe, max_seq_len, tokenizer) -> None:
        self.dataframe = dataframe
        self.bos_token = '<s>'
        self.eos_token = '</s>'
        self.max_seq_len = max_seq_len  # special token 포함해서 고려한 길이가 max_seq_len임

        self.tokenizer = tokenizer

    def __len__(self):
        return self.dataframe.shape[0]

    def make_input_id_mask(self, tokens, index):
        input_id = self.tokenizer.convert_tokens_to_ids(tokens)  # <s> 와 </s> token 포함된 상태에서 진행하는것임
        attention_mask = [1] * len(input_id)
        if len(input_id) < self.max_seq_len:
            while len(input_id) < self.max_seq_len:
                input_id += [self.tokenizer.pad_token_id]
                attention_mask += [0]
        else:
            input_id = input_id[:self.max_seq_len - 1] + [
                self.tokenizer.eos_token_id]
            attention_mask = attention_mask[:self.max_seq_len]  # 1028개 까지만 수용
        return input_id, attention_mask

    def __getitem__(self, index):
        target_row = self.dataframe.iloc[index]
        context, summary = target_row['context'], target_row['summary']
        context_tokens = [self.bos_token] + \
                         self.tokenizer.tokenize(context) + [self.eos_token]
        summary_tokens = [self.bos_token] + \
                         self.tokenizer.tokenize(summary) + [self.eos_token]  # 애초에 summary token은 max_seq_len 보다 훨씬 짧음
        encoder_input_id, encoder_attention_mask = self.make_input_id_mask(
            context_tokens, index)
        decoder_input_id, decoder_attention_mask = self.make_input_id_mask(
            summary_tokens, index)  # decoder에 넣기 위해 딱 맞는 형태로 갖춤
        # labels = self.tokenizer.convert_tokens_to_ids(summary_tokens[1:self.max_seq_len])
        labels = self.tokenizer.convert_tokens_to_ids(
            summary_tokens[1:(
                        self.max_seq_len + 1)])  # 아니 애초에 summary token의 길이보다 더 긴 indexing을 하는데 말이되나, labels을 만드는 의도가 머지 -> decoder에서 예측된 놈들이 나오는 것이니까 맞네 이게
        if len(labels) < self.max_seq_len:
            while len(labels) < self.max_seq_len:
                # for cross entropy loss masking
                labels += [-100]

        return {'input_ids': torch.tensor(encoder_input_id, dtype=torch.long),
                'attention_mask': torch.tensor(encoder_attention_mask, dtype=torch.long),
                'decoder_input_ids': torch.tensor(decoder_input_id[:128], dtype=torch.long),
                'decoder_attention_mask': torch.tensor(decoder_attention_mask[:128], dtype=torch.long),
                'labels': torch.tensor(labels[:128], dtype=torch.long)}
