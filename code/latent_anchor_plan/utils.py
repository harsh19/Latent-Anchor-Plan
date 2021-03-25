import torch
import _pickle as pickle
import numpy as np
import constants

ignore_type_tokens_lists =  {
    'default':['<pad>'],
    'non_content_set2': ['<pad>', '.', '</s>', ',', '!','<EOT>'] + constants.stopwords
}


def get_emotion_vocab_list(dictionary, emotion_type, use_cuda):
    emotion_states = constants.get_emotion_states(emotion_type)
    num_states = len(emotion_states)
    emotion_vocab_list = torch.LongTensor(1, num_states).zero_()
    for j, emotion in enumerate(emotion_states):
        emotion_vocab_list[0, j] = dictionary.word2idx[emotion]
    if use_cuda:
        emotion_vocab_list = emotion_vocab_list.cuda()
    print("[get_emotion_vocab_list] emotion_states = ", emotion_states)
    print("[get_emotion_vocab_list] emotion_vocab_list = ", emotion_vocab_list)
    return emotion_vocab_list


def repackage_hidden(h):
    """Wraps hidden states in new Tensors,
    to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def pad(lst, pad_id, pad_prepend):
    # print("pad: lst = ", lst)
    maxlen = max([v.size()[0] for v in lst])
    ret = torch.LongTensor(len(lst), maxlen)
    for i,v in enumerate(lst):
        szi = len(v)
        if pad_prepend:
            ret[i, maxlen - szi:] = v
            ret[i, :maxlen - szi] = pad_id
        else:
            ret[i,:szi] = v
            ret[i,szi:] = pad_id
    # print("[pad]: returning ", ret)
    return ret

def pad_array(bs, pad_id):
    ret = torch.LongTensor(bs, 1)
    ret[:,:] = pad_id
    return ret


global_ignore_tokens = None

def batchify(data, bsz, args, dictionary):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = len(data) // bsz
    data = [ row for row in data]
    data = [data[i*bsz:(i+1)*bsz] for i in range(nbatch)]
    print("=========>>>> data : num batches ", len(data))
    return data


def get_batch(source, i,
              args,
              seq_len=None,
              evaluation=False,
              mode='prior',
              dictionary=None,
              pad_prepend=False,
              provided_kw=None,
              sanity_check_title_joined=False,
              sanity_check_title=False,
              kw_need_targets=True,
              print_more=False,
              emotion_vocab_tensor=None,
              num_sentences_cond=None):

    tmp = source[i] # tmp is one babtch. with a list of elements
    # each element is a list of title,plot,sentences
    # print("tmp = ", tmp)
    padid = dictionary.word2idx['<pad>']
    global global_ignore_tokens

    if mode == 'title': #used
        # eos = tmp[0][-1][-1:]
        if sanity_check_title:
            if sanity_check_title_joined:
                data = pad_array(len(tmp), padid)
            else:
                tmp = [ row['title'][:-1] for row in tmp] # exclude EOT symbol
                data = pad(tmp,padid,pad_prepend)
        else:
            if sanity_check_title_joined:
                data = pad_array(len(tmp), padid)
            else:
                tmp = [ row['title'] for row in tmp]
                data = pad(tmp,padid,pad_prepend)
        if print_more: #True or np.random.rand()<0.005:
            for i in range(len(data)):
                datai = data[i]
                tokens = [dictionary.idx2word[idx] for idx in datai.data.cpu().numpy()]
                print("[TITLE] datai = ", tokens)
        data = data.t()  # seqlen,bsz
        if args.cuda:
            data = data.cuda()
        return data

    elif mode == 'title_ae': # used
        # eos = tmp[0][-1][-1:]
        if args.emotion_special_processing:
            tmp = [ row['title'] for j in range(1) for row in tmp]
        else:
            tmp = [row['title'] for j in range(5) for row in tmp]
        data = pad(tmp,padid,pad_prepend)
        if print_more: #True or np.random.rand()<0.005:
            for i in range(len(data)):
                datai = data[i]
                tokens = [dictionary.idx2word[idx] for idx in datai.data.cpu().numpy()]
                print("[TITLE] datai = ", tokens)
        data = data.t()  # seqlen,bsz
        if args.cuda:
            data = data.cuda()
        return data


    elif mode=='plot':
        # eos = tmp[0][-1][-1:]
        tmp = [ torch.cat([row['title'][-1:],row['keywords'][:-1]]) for row in tmp ] # prepend EOT; remove EOL from end
        data = pad(tmp,padid,pad_prepend)
        if print_more: #True or np.random.rand()<0.005:
            for i in range(len(data)):
                datai = data[i]
                tokens = [dictionary.idx2word[idx] for idx in datai.data.cpu().numpy()]
                print("[PLOT] datai = ", tokens)
            print()
        data = data.t()  # seqlen, bsz #*bsz:(i+1)*bsz]
        if args.cuda:
            data = data.cuda()
        inp = data[:-1, :]
        target = data[1:, :].contiguous() #.view(-1)
        # sz = target.size()[0]
        # mask = np.ones(sz)
        # mask = get_mask(target, dictionary)
        return inp, target

    elif mode=='posterior_single_ae': # used
        # eos = tmp[0][-1][-1:]
        ret = []
        emotion_targets = []
        bsz = len(tmp)
        if global_ignore_tokens is not None:
            ignore_tokens = global_ignore_tokens
        else:
            ignore_tokens = [dictionary.word2idx[token] for token in ignore_type_tokens_lists[args.infer_nw_ignore_token_type]
                     if token in dictionary.word2idx]
            global_ignore_tokens = ignore_tokens
        for row in tmp:
            cnt = 5
            if args.emotion_special_processing:
                cnt = 1
            for j in range(cnt):
                # linej = row['line'+str(j)]
                linej = row['line'+str(j)][1:] #****
                if args.new_posterior_batch_defn:
                    linej = torch.cat([row['title'], row['line' + str(j)]])
                    for ignore_tokensj in ignore_tokens:
                        linej = linej[linej!=ignore_tokensj]
                elif args.new_posterior_batch_defn_notitle:
                    assert not args.new_posterior_batch_defn
                    for ignore_tokensj in ignore_tokens:
                        linej = linej[linej != ignore_tokensj]
                ret.append(linej)
                if args.use_emotion_supervision:
                    raise NotImplementedError
        data = pad(ret,padid,pad_prepend)
        if print_more:
            for i in range(len(data)):
                datai = data[i]
                tokens = [dictionary.idx2word[idx] for idx in datai.data.cpu().numpy()]
                print("[POSTERIOR] datai = ", tokens)
        # data = data.view(bsz,5,-1)
        if args.use_emotion_supervision:
            raise NotImplementedError
        else:
            if args.cuda:
                data = data.cuda()
        return data # B,5,len

    elif mode=='posterior':
        # eos = tmp[0][-1][-1:]
        ret = []
        bsz = len(tmp)
        if global_ignore_tokens is not None:
            ignore_tokens = global_ignore_tokens
        else:
            ignore_tokens = [dictionary.word2idx[token] for token in ignore_type_tokens_lists[args.infer_nw_ignore_token_type]
                     if token in dictionary.word2idx]
            global_ignore_tokens = ignore_tokens
            print("global_ignore_tokens = ", global_ignore_tokens)
            print("ignore_type_tokens_lists[args.infer_nw_ignore_token_type] = ", ignore_type_tokens_lists[args.infer_nw_ignore_token_type])

        for row in tmp:
            for j in range(5):
                linej = row['line'+str(j)]
                if args.new_posterior_batch_defn:
                    linej = torch.cat([row['title'], row['line' + str(j)]])
                    for ignore_tokensj in ignore_tokens:
                        linej = linej[linej!=ignore_tokensj]
                elif args.new_posterior_batch_defn_notitle:
                    assert not args.new_posterior_batch_defn
                    for ignore_tokensj in ignore_tokens:
                        linej = linej[linej != ignore_tokensj]
                ret.append(linej)
        data = pad(ret,padid,pad_prepend)
        if print_more:
            for i in range(len(data)):
                datai = data[i]
                tokens = [dictionary.idx2word[idx] for idx in datai.data.cpu().numpy()]
                print("[POSTERIOR] datai = ", tokens)
        data = data.view(bsz,5,-1)
        if args.cuda:
            data = data.cuda()
        return data # B,5,len
        # todo - return in reverse order ?

    elif mode in ['conditional_single_ae']:
        ret = []
        for row in tmp:
            cnt = 5
            if args.emotion_special_processing:
                cnt = 1
            for j in range(cnt):
                ret.append(torch.cat([row['eol'],row['line'+str(j)],row['eos']]) )
        tmp = ret
        data = pad(tmp,padid,pad_prepend)
        if print_more: #True or np.random.rand() < 0.005:
            for i in range(len(data)):
                datai = data[i]
                tokens = [dictionary.idx2word[idx] for idx in datai.data.cpu().numpy()]
                print("[CONDITIONAL] datai = ", tokens)
        data = data.t() # seqlen, bsz #*bsz:(i+1)*bsz]
        if args.cuda:
            data = data.cuda()
        inp = data[:-1,:]
        target = data[1:,:].contiguous() #.view(-1)
        return inp, target

    elif mode in ['conditional','conditional_title','conditional_gtkw','conditional_lines']:
        if mode == 'conditional':
            if args.new_decoder:
                provided_kw = provided_kw.cpu()
                # tmp = [torch.cat( [row['eol'], sample[0:1], row['line0'], sample[1:2], row['line1'], sample[2:3],
                #                    row['line2'],sample[3:4], row['line3'], sample[4:5], row['line4'], row['eos']])
                #                        for row,sample in zip(tmp,provided_kw)]
                tmp = [torch.cat([row['eol'], row['eos'], sample[0:1], row['line0'], row['eos'], sample[1:2], row['line1'], row['eos'], sample[2:3],row['line2'], row['eos'], sample[3:4], row['line3'], row['eos'], sample[4:5], row['line4'], row['eos']])
                       for row, sample in zip(tmp, provided_kw)]
            else:
                tmp = [torch.cat( [row['eol'],row['line0'],row['line1'],row['line2'],row['line3'],row['line4'],row['eos']] )
                    for row in tmp]
            if sanity_check_title_joined:
                assert False
        elif mode=='conditional_title':
            if sanity_check_title_joined:
                tmp = [torch.cat([row['title'],row['line0'], row['line1'], row['line2'], row['line3'], row['line4'],
                                  row['eos']]) for row in tmp]
            elif sanity_check_title:
                tmp = [torch.cat([row['title'][-1:], row['line0'], row['line1'], row['line2'], row['line3'], row['line4'],
                                  row['eos']]) for row in tmp]
            else:
                tmp = [torch.cat([row['line0'], row['line1'], row['line2'], row['line3'], row['line4'], row['eos']])
                       for row in tmp]
        elif mode=='conditional_gtkw':
            if sanity_check_title_joined:
                assert False
            else:
                tmp = [torch.cat(
                    [row['eol'], row['line0'], row['line1'], row['line2'], row['line3'], row['line4'], row['eos']])
                       for row in tmp]
        elif mode == 'conditional_lines':
            if args.new_decoder:
                assert False
            ret = []
            for row in tmp:
                if not sanity_check_title:
                    t = [row['eol']]
                else:
                    t = [row['title'][-1:]]  #eot
                for j in range(num_sentences_cond):
                    t.append(row['line'+str(j)])
                # t.append(row['eos'])
                t.append(row['line0'][0:1]) # TODO
                ret.append(torch.cat(t))
            tmp = ret
            if sanity_check_title_joined:
                assert False
        data = pad(tmp,padid,pad_prepend)
        if print_more: #True or np.random.rand() < 0.005:
            for i in range(len(data)):
                datai = data[i]
                tokens = [dictionary.idx2word[idx] for idx in datai.data.cpu().numpy()]
                print("[CONDITIONAL] datai = ", tokens)
        data = data.t() # seqlen, bsz #*bsz:(i+1)*bsz]
        if args.cuda:
            data = data.cuda()
        inp = data[:-1,:]
        target = data[1:,:].contiguous() #.view(-1)
        # target: seqlen,bs
        return inp, target

    elif mode == 'kw_classification':
        raise NotImplementedError

    elif mode == 'emotion_classification':
        raise NotImplementedError

state = 'title'
def get_mask(target, dictionary, typ='plot'):
    global state
    assert typ in ['plot','noplot']
    sz = target.size()[0]
    mask = np.ones(sz)
    tokens = [ dictionary.idx2word[idx] for idx in target.data.cpu().numpy() ]
    # print("tokens = ", tokens)
    for j,token in enumerate(tokens):
        if state in ['title','plot']:
            mask[j] = 0
        elif state == 'story':
            pass
        if token == '<EOT>':
            assert state == 'title'
            if typ=='plot':
                state = 'plot'
            else:
                state = 'story'
        elif token == '<EOL>':
            assert state == 'plot'
            assert typ == 'plot'
            state = 'story'
        elif token == '<eos>':
            state = 'title'
        if token=='<pad>':
            mask[j] = 0
            #0/0
    # print("tokens[:5] = ", tokens[:5])
    # print("mask[:5]= ", mask[:5])
    #0/0
    return mask

# pad_id = corpus.dictionary.word2idx['<pad>']

def get_batch_story_only(source, i, args, seq_len=None, evaluation=False, dictionary=None, data_typ='plot'):
    print("="*33)
    # seq_len = min(seq_len if seq_len else args.bptt, len(source) - 1 - i)
    # data = source[i:i+seq_len]
    data = source[i].t()
    inp = data[:-1, :]
    target = data[1:, :].contiguous().view(-1)
    # target = source[i+1:i+1+seq_len].view(-1)
    mask = get_mask( target, dictionary, typ=data_typ )
    # print("inp = ", inp, inp.size())
    # print("target = ", target, target.size())
    # print("mask = ", mask)
    return inp, target, mask


def load_pickle(path):
    with open(path, 'rb') as fin:
        obj = pickle.load(fin)
    return obj


def make_vocab(corpus_dictionary, vocab_path):
    """take data, create pickle of vocabulary"""
    with open(vocab_path, 'wb') as fout:
        pickle.dump(corpus_dictionary, fout)
    print('[UTILS] Saved dictionary to', vocab_path)
    print('[word2idx]: len(self.word2idx) = ', len(corpus_dictionary.word2idx))
    print('')


def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape).cuda()
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return torch.nn.Functional.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, temperature, latent_dim, categorical_dim):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    y_hard = (y_hard - y).detach() + y
    return y_hard.view(-1,latent_dim*categorical_dim)



