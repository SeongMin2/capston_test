import pandas as pd
import numpy as np
import re
import json
import argparse
import os
from ABS_PATH import ABS_PATH

ABS_PATH = ABS_PATH

class Ft_Processor():
    def get_train_examples(self, data_dir):
        # pass
        return self.preprocessing(data_dir)

    def get_val_examples(self, data_dir):
        # pass
        return self.preprocessing(data_dir)

    def get_labels(self,):
        pass

    def read_json(self, data_dir):
        with open(data_dir, encoding="UTF8") as f:
            JSON_DATA = json.loads(f.read())
        return JSON_DATA

    def replace2abbrev(self, m):
        m = str(m.group())
        m = re.findall('\((.*?)\)', m)[0]
        return m

    def preprocessing(self, data_dir):
        json_data = self.read_json(data_dir)
        data_df = pd.DataFrame(columns=['uid', 'type', 'topic','partici_num','utter_num', 'context', 'summary'])
        uid = 0

        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                   u"\U00002500-\U00002BEF"  # chinese char
                                   u"\U00010000-\U0010ffff"
                                   u"\u2640-\u2642"
                                   u"\u2600-\u2B55"
                                   u"\u200d"
                                   u"\u23cf"
                                   u"\u23e9"
                                   u"\u231a"
                                   u"\ufe0f"  # dingbats
                                   u"\u3030"
                                   "]+", flags=re.UNICODE)

        for data in json_data['data'][0:args.train_data_num]:
            data_df.loc[uid, 'uid'] = uid
            data_df.loc[uid, 'type'] = data['header']['dialogueInfo']['type']
            data_df.loc[uid, 'topic'] = data['header']['dialogueInfo']['topic']
            data_df.loc[uid, 'partici_num'] = data['header']['dialogueInfo']['numberOfParticipants']
            data_df.loc[uid, 'utter_num'] = data['header']['dialogueInfo']['numberOfUtterances']
            context = ''
            for utter in data['body']['dialogue']:
                context = context + utter['utterance'] + ' '
                # if utter['utteranceID'] == 'U'+str(data['header']['dialogueInfo']['numberOfUtterances']):
                #    context = context + utter['utterance']

                # else:
                #    context = context + utter['utterance'] + '/ '

            context = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ·!』\\‘’|\(\)\[\]\<\>`\'…》]', '', context)  # 특수문자 제거
            context = re.sub('(([ㄱ-힣])\\2{1,})', '', context)
            context = emoji_pattern.sub(r'', context)
            data_df.loc[uid, 'context'] = context

            data_df.loc[uid, 'summary'] = re.sub('[ㄱ-힣]+\([^)]*\)', self.replace2abbrev, data['body']['summary'])
            uid += 1

        return data_df

def main(args):
    processor = Ft_Processor()
    df = processor.get_train_examples(args.train_path)
    df.to_csv(args.save_path)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_path", type=str, default=ABS_PATH+"/data/train/beauty_health.json")
    parser.add_argument("--valid_path", type=str, default=ABS_PATH+"/data/valid/beauty_health.json")
    parser.add_argument("--train_data_num", type=int, default=8)
    parser.add_argument("--save_path", type=str, default=ABS_PATH+"/data/save/save.csv")


    args = parser.parse_args()
    main(args)