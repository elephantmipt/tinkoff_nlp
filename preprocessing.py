import re
from tqdm import tqdm
from typing import List
import pandas as pd
from sklearn.model_selection import train_test_split
import youtokentome as yttm


class TrainTestSplit():
    def __init__(self, inp_path, out_path, train_path, test_path, test_size=0.1, bpe_path=''):
        df = pd.read_csv(inp_path)
        df['msg_parsed'] = df.msg.apply(self._preproc)
        df['msg_splitted_len'] = df.msg.apply(lambda x: len(self._preproc(x).split()))
        df = df[df['msg_splitted_len'] > 1]
        with open(out_path, 'w') as out:
            for msg in df.msg_parsed.values:
                out.write(msg+'\n')
        if len(bpe_path) > 0:
            yttm.BPE.train(model=bpe_path, vocab_size=5000, data=out_path, coverage=0.999, n_threads=-1)
        X_train, X_test = train_test_split(df.msg_parsed.values, test_size=test_size)
        with open(train_path, 'w') as inp:
            for msg in X_train:
                inp.write(msg+'\n')
        with open(test_path, 'w') as inp:
            for msg in X_test:
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
