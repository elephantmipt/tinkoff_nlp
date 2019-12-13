import re
from tqdm import tqdm
from typing import List

class Preprocessing():
    def __init__(self, inp_path, out_path):
        with open(inp_path, 'r') as inp:
            with open(out_path, 'w') as out:
                pbar = tqdm(inp)
                pbar.set_description('Preprocessing')
                for line in pbar:
                    if len(self._preproc(line).split()) != 0:
                        out.write(self._preproc(line) + '\n')

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