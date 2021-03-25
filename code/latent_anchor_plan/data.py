import os
import torch

from collections import Counter

from utils import load_pickle, make_vocab
import constants #import emotion_states
import copy

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.counter = Counter()
        self.total = 0
        if True:
            pad = '<pad>'
            self.idx2word.append(pad)
            self.word2idx[pad] = len(self.idx2word) - 1
            emotion_states = constants.get_emotion_states('basic')
            for emotion in emotion_states: #_basic:
                self.idx2word.append(emotion)
                self.word2idx[emotion] = len(self.idx2word) - 1

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        token_id = self.word2idx[word]
        self.counter[token_id] += 1
        self.total += 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


def find_emotion_label(text, emotion_vocab_list):
    words = text.strip().split()
    for j,k in enumerate(emotion_vocab_list):
        for kk in constants.emotion_states_keyword_mapper[k]:
            if kk.lower() in words:
                return j,k
    return -1,'<pad>'


class Corpus(object):
    def __init__(self, applyDict=False, **kwargs):
        #TODO: document this
        """
        :param applyDict:
        :param kwargs: 'train_path' 'dev_path' 'test_path', 'dict_path', 'applyDict'
        """
        self.debug = False
        # if 'debug' in kwargs.keys():
        #     self.debug = kwargs['debug']
        self.keywords_dist = {}
        emotion_special_processing = kwargs['emotion_special_processing']
        emotion_type = kwargs['emotion_type']
        use_emotion_supervision = kwargs['use_emotion_supervision']
        if applyDict:
            print("[CORPUS]: Loading dictionary from provided path : ", kwargs['dict_path'])
            self.dictionary = load_pickle(kwargs['dict_path'])  # a previously saved pickle of a Dictionary
            print('[dictionary]: len(dictionary.word2idx) = ', len(self.dictionary.word2idx))
            if 'train_path' in kwargs.keys():
                self.train = self.tokenize(kwargs['train_path'],applyDict=applyDict,
                                           emotion_special_processing=emotion_special_processing,
                                           emotion_type=emotion_type,
                                           use_emotion_supervision=use_emotion_supervision)
            if 'dev_path' in kwargs.keys():
                self.valid = self.tokenize(kwargs['dev_path'],applyDict=applyDict,
                                           emotion_special_processing=emotion_special_processing,
                                           emotion_type=emotion_type,
                                           use_emotion_supervision=use_emotion_supervision)
            if 'test_path' in kwargs.keys():
                self.test = self.tokenize(kwargs['test_path'],applyDict=applyDict,
                                           emotion_special_processing=emotion_special_processing,
                                           emotion_type=emotion_type,
                                           use_emotion_supervision=use_emotion_supervision)
        else:
            self.dictionary = Dictionary()
            if 'train_path' in kwargs.keys():
                self.train = self.tokenize(kwargs['train_path'],
                                           emotion_special_processing=emotion_special_processing,
                                           emotion_type=emotion_type,
                                           use_emotion_supervision=use_emotion_supervision)
            if 'dev_path' in kwargs.keys():
                self.valid = self.tokenize(kwargs['dev_path'],
                                           emotion_special_processing=emotion_special_processing,
                                           emotion_type=emotion_type,
                                           use_emotion_supervision=use_emotion_supervision)
            if 'test_path' in kwargs.keys():
                self.test = self.tokenize(kwargs['test_path'],
                                           emotion_special_processing=emotion_special_processing,
                                           emotion_type=emotion_type,
                                          use_emotion_supervision=use_emotion_supervision)
            # save file when done
            make_vocab(self.dictionary, kwargs['output'])


    def tokenize(self, path, applyDict=False,
                 gen_mode=False,
                 emotion_special_processing=False,
                 emotion_type=None,
                 use_emotion_supervision=False):
        debug = self.debug
        """Tokenizes a text file."""
        assert os.path.exists(path), path
        # Add words to the dictionary
        self.dictionary.add_word('<pad>')
        tokens = 0
        if not applyDict: # index also
            with open(path, 'r') as f:
                for line in f:
                    words = line.split() + ['<eos>']
                    tokens += len(words)
                    for word in words:
                        self.dictionary.add_word(word)
        # Tokenize file content
        if not tokens: # else apply
            with open(path, 'r') as f:
                for line in f:
                    words = line.split() + ['<eos>']
                    tokens += len(words)
        ret = []

        if gen_mode:
            ids = torch.LongTensor(tokens)  # init the LongTensor to size of tokens, pretty large and 1D
            token = 0
            with open(path, 'r') as f:
                for line in f:
                    words = line.split() + ['<eos>']
                    for word in words:
                        ids[token] = self.dictionary.word2idx.get(word, 0)
                        token += 1
            ret = ids
            return ret

        else:
            emotion_states = constants.get_emotion_states(emotion_type)
            not_found_cnt = 0
            with open(path, 'r') as f:
                lines_processed = 0
                skip_cnt = 0
                for line in f:
                    lines_processed+=1
                    sample = line #' '.join( line.split() + ['<eos>'] )
                    tmp = sample.split('<EOT>')
                    title = tmp[0].strip() + ' <EOT>'
                    cur = {'title': title, 'eos':'<eos>', 'eol':'<EOL>'}
                    lines = []
                    # print(" --->>> tmp = ", tmp)
                    if tmp[1].strip().count('<EOL>')>0:
                        tmp = tmp[1].strip().split('<EOL>')
                        keywords = tmp[0].strip() + ' <EOL>' #.split()
                        cur['keywords'] = keywords
                    if tmp[1].strip().count('</s>')>0:
                        lines = tmp[1].strip().split('</s>')[1:]
                        if len(lines)!=5:
                            skip_cnt+=1
                            continue
                        if 'keywords' in cur and len(cur['keywords'].strip().split())!=6:
                            # print("kws = ", keywords)
                            skip_cnt+=1
                            continue
                        for kw in cur['keywords'].strip().split()[:-1]:
                            self.keywords_dist[kw] = self.keywords_dist.get(kw,0) + 1
                    for j,line in enumerate(lines):
                        cur['line'+str(j)] = '</s> '+line.strip()
                        if use_emotion_supervision:
                            cur['label'+str(j)] = find_emotion_label(line.strip(), emotion_states)
                    for k in cur:
                        if k.count('label')>0:
                            continue
                        words = cur[k].split()
                        ids = torch.LongTensor(len(words))  # init the LongTensor to size of tokens, pretty large and 1D
                        token = 0
                        for word in words:
                            ids[token] = self.dictionary.word2idx.get(word, 0)# TODO ****** - should have unk id here
                            if word not in self.dictionary.word2idx:
                                not_found_cnt += 1
                            token += 1
                        cur[k] = ids
                    ret.append(cur)
                print("*** skip_cnt = ", skip_cnt)
                print("*** not_found_cnt = ", not_found_cnt)
            return ret





