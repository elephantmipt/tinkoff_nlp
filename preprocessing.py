import re
from tqdm import tqdm
from typing import List
import pandas as pd
from sklearn.model_selection import train_test_split
import youtokentome as yttm
import random
from itertools import chain
from tqdm import tqdm
from allennlp.data.vocabulary import Vocabulary
tqdm.pandas()


class TrainTestSplit:
    def __init__(self, inp_path, out_path, train_path, test_path, voc_path, ms_path,
                 test_size=0.1, bpe_path='', bpe=False):
        df = pd.read_csv(inp_path)

        #df['msg'] = df['msg'].apply(lambda x: x)

        df['msg_parsed'] = df.msg.apply(self._preproc)
        df['msg_splitted_len'] = df.msg.apply(lambda x: len(self._preproc(x).split()))
        df = df[df['msg_splitted_len'] > 1]
        print('making mistakes...')
        df['msg_0.01'] = df['msg_parsed'].progress_apply(lambda x : mistakes_maker(x, 0.01))
        df['msg_0.02'] = df['msg_parsed'].progress_apply(lambda x: mistakes_maker(x, 0.02))
        df['msg_0.03'] = df['msg_parsed'].progress_apply(lambda x: mistakes_maker(x, 0.03))
        df['msg_0.05'] = df['msg_parsed'].progress_apply(lambda x: mistakes_maker(x, 0.05))
        df['msg_0.1'] = df['msg_parsed'].progress_apply(lambda x: mistakes_maker(x, 0.1))
        msg_all = list(chain(df['msg_0.01'].values, df['msg_0.02'].values,
                  df['msg_0.03'].values, df['msg_0.05'].values,
                  df['msg_0.1'].values))
        print('creating tmp files...')
        with open(voc_path, 'w') as out:
            for msg in tqdm(msg_all):
                out.write(msg+'\n')
        with open(ms_path+'0.01.csv', 'w') as out:
            for msg in df['msg_0.01'].values:
                out.write(msg+'\n')
        with open(ms_path+'0.02.csv', 'w') as out:
            for msg in df['msg_0.02'].values:
                out.write(msg+'\n')
        with open(ms_path+'0.03.csv', 'w') as out:
            for msg in df['msg_0.03'].values:
                out.write(msg+'\n')
        with open(ms_path+'0.05.csv', 'w') as out:
            for msg in df['msg_0.05'].values:
                out.write(msg+'\n')
        with open(ms_path+'0.1.csv', 'w') as out:
            for msg in df['msg_0.1'].values:
                out.write(msg+'\n')

        with open(out_path, 'w') as out:
            for msg in df.msg_parsed.values:
                out.write(msg+'\n')
        print('done')
        if bpe:
            yttm.BPE.train(model=bpe_path, vocab_size=5000, data=voc_path, coverage=0.999, n_threads=-1)

        X_train, X_test = train_test_split(df.msg_parsed.values, test_size=test_size, random_state=9)
        with open(train_path, 'w') as inp:
            for msg in X_train:
                inp.write(msg+'\n')
        with open(test_path, 'w') as inp:
            for msg in X_test:
                msg = str(msg.encode('utf-8'))
                inp.write(msg+'\n')

    def _preproc(self, msg: str) -> List[str]:
        x = msg
        number_re = r'[+]*[(]{0,1}[0-9]{1,4}[)]{0,1}[-\s\./0-9]*'
        x = re.sub('[\d]{4,99}', ' [phone_number] ', x)
        x = re.sub(number_re, ' [number] ', x)
        x = x.strip().lower()
        x = re.sub('[\s.]л[\s.]', ' лет ', x)
        x = re.sub('[\s.]г[\s.]', ' года ', x)
        x = re.sub('[\s.]м[\s.]', ' мужчина ', x)
        x = re.sub('^м[\s.]', ' мужчина ', x)
        x = re.sub('[\s.]ж[\s.]', ' женщина ', x)
        x = re.sub('[\s.]женщ[\s.]', ' женщина ', x)
        x = re.sub('^ж[\s.]', ' женщина ', x)
        x = re.sub('[\s.]д[\s.]', ' девушка ', x)
        x = re.sub('[\s.]дев[\s.]', ' девушка ', x)
        x = re.sub('[\s.]поз[\s.]', ' познакомится ', x)
        x = re.sub('^поз[\s.]', ' познакомится ', x)
        x = re.sub('[\s.]позн[\s.]', ' познакомится ', x)
        x = re.sub('^позн[\s.]', ' познакомится ', x)
        x = re.sub('^познк[\s.]', ' познакомится ', x)
        x = re.sub('^д[\s.]', 'девушка ', x)
        x = re.sub('[\s.]п[\s.]', ' парень ', x)
        x = re.sub('^п[\s.0-9]', ' парень ', x)
        x = re.sub('[\s]пар[\s.]', ' парень ', x)
        x = re.sub('^пар[\s.]', ' парень ', x)
        x = re.sub('[\s]жен[\s.]', ' женщина ', x)
        x = re.sub('норм[\s.]', 'нормальный ', x)
        x = re.sub('симп[\s.]', ' симпотичная ', x)
        x = re.sub('сим[\s.]', ' симпотичным ', x)
        x = re.sub('сер[\s.]', 'серьезных ', x)
        x = re.sub('отн[\s.]', 'отношений ', x)
        x = re.sub('[\s.]с\\о[\s.]', ' серьезных отношений ', x)
        x = re.sub('[.?!,]', ' ', x)
        x = x.strip().lower()
        return x


def mistakes_maker(msg, mistakes_rate):
    msg_ = list(msg)
    for i in range(len(msg)):
        rv = random.randrange(1, 1001)
        if rv <= mistakes_rate:
            msg_[i] = random.choice(list('йцукенгшщзхъфывапролджэячсмитьбю'))
    return str(msg_)
