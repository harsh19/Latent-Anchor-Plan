
import argparse
import sys
import numpy, math
import torch
import torch.nn as nn
from numbers import Number
from utils import batchify, get_batch, repackage_hidden, get_batch_story_only
import torch.nn.functional as F


# https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits
#



def generate_plot_samples(model, data_source, args, corpus):

    eos_id = corpus.dictionary.word2idx['<eos>']
    eot_id = corpus.dictionary.word2idx['<EOT>']
    ntokens = len(corpus.dictionary)
    input = torch.rand(1, 1).mul(ntokens).long()  # , volatile=True)
    if args.cuda:
        input = input.cuda()

    model.eval()

    with open(args.outf, 'w') as outf: #, open('gold_4sent.txt', 'w') as gf:
        data = data_source #corpus.tokenize(args.conditional_data, applyDict=True).tolist()
        # this is a list of ids corresponding to words from the word2idx dict
        nsent = 0

        while nsent < args.sents:

            #################

            mode = 'title'  # 'prior'
            # pad_prepend = False
            pad_prepend = True
            title_data = get_batch(data_source, nsent, args, seq_len=None, mode=mode,
                                   dictionary=corpus.dictionary, pad_prepend=pad_prepend,
                                   sanity_check_title_joined=args.sanity_check_title_joined)
            _, bs = title_data.size()
            assert bs==1
            hidden_prior = model.init_hidden(bs)  # args.batch_size)
            output_prior, hidden_prior = model(title_data, hidden_prior, return_h=False)
            hidden = hidden_prior  # [-1]
            output = output_prior.view(-1,bs,output_prior.size()[1])[-1, :, :]

            if args.print_cond_data:
                assert title_data.size()[1] == 1
                for word_idx in title_data.data:
                    word = corpus.dictionary.idx2word[word_idx.item()]
                    outf.write(word + ' ')

            #################

            exist_word = set()
            for i in range(5):
                output = model.prior_model(output)
                word_weights = output.squeeze().data.div(args.temperature).exp().cpu()
                if args.top_p>0.0:
                    word_weights = output.squeeze().div(args.temperature)
                    word_weights = top_k_top_p_filtering(word_weights, top_k=0, top_p=args.top_p, filter_value=-float('Inf'))
                    word_weights = word_weights.exp().cpu()
                samples = torch.multinomial(word_weights, 5)
                if args.dedup:
                    for word_idx in samples:
                        word_idx = word_idx.item()
                        if word_idx not in exist_word:
                            break
                    exist_word.add(word_idx)
                else:
                    word_idx = samples[0]
                input.data.fill_(word_idx)
                output, hidden = model(input, hidden)
                word = corpus.dictionary.idx2word[word_idx]
                outf.write(word + ' ')
            outf.write('\n')
            print('| Generated {} sentences'.format(nsent+1), file=sys.stderr)
            nsent += 1
        outf.flush()




def generate_story_samples(model, data_source, args, corpus,
                           use_provided_plots=False,
                           completion_mode='latent_var_model',
                           use_posterior_samples=False, samples_per_tile=5):

    assert completion_mode in ['latent_var_model','title_only']
    print("completion_mode = ",completion_mode)

    eos_id = corpus.dictionary.word2idx['<eos>']
    eot_id = corpus.dictionary.word2idx['<EOT>']
    eol_id = corpus.dictionary.word2idx['<EOL>']
    beginning_of_line_idx = corpus.dictionary.word2idx['</s>']
    ntokens = len(corpus.dictionary)
    input = torch.rand(1, 1).mul(ntokens).long()  # , volatile=True)
    if args.cuda:
        input = input.cuda()

    model.eval()

    with open(args.outf, 'w') as outf: #, open('gold_4sent.txt', 'w') as gf:
        nsent = 0

        while nsent < args.sents:

            for k in range(samples_per_tile):

                #################

                mode = 'title'  # 'prior'
                if use_provided_plots:
                    mode='title_plot_new'
                # pad_prepend = False
                pad_prepend = True
                title_data = get_batch(data_source, nsent, args, seq_len=None, mode=mode,
                                       dictionary=corpus.dictionary, pad_prepend=pad_prepend,
                                       sanity_check_title_joined=args.sanity_check_title_joined,
                                       print_more=True)
                _, bs = title_data.size()
                assert bs==1
                hidden_prior = model.init_hidden(bs)  # args.batch_size)
                output_prior, hidden_prior = model(title_data, hidden_prior, return_h=False)
                hidden = hidden_prior  # [-1]
                output = output_prior.view(-1,bs,output_prior.size()[1])[-1, :, :]

                if args.print_cond_data:
                    assert title_data.size()[1] == 1
                    for word_idx in title_data.data:
                        word = corpus.dictionary.idx2word[word_idx.item()]
                        outf.write(word + ' ')

                if completion_mode in ['latent_var_model']:

                    ################# generate plot

                    if use_provided_plots:
                        pass
                    elif use_posterior_samples:
                        sentences_data = get_batch(data_source, nsent, args, seq_len=None, mode='posterior',
                                                   dictionary=corpus.dictionary)
                        sample, _ = model.get_sample_from_posterior(
                            sentences_data=sentences_data)  # todo - add temperature here ?
                        output, hidden = model(sample.t(), hidden, return_h=False)
                        for s in sample.view(-1).data.cpu().numpy():
                            word = corpus.dictionary.idx2word[s]
                            outf.write(word + ' ')
                    else:
                        exist_word = set()
                        prev_words = []
                        for i in range(5):
                            output = model.prior_model(output)
                            word_weights = output.squeeze().data.div(args.temperature).exp().cpu()
                            if args.top_p > 0.0:
                                word_weights = output.squeeze().div(args.temperature) # todo -> not adding temp to this ?
                                word_weights = top_k_top_p_filtering(word_weights, top_k=0, top_p=args.top_p,
                                                                     filter_value=-float('Inf'))
                                word_weights = word_weights.exp().cpu()
                            samples = torch.multinomial(word_weights, 5)
                            if args.dedup:
                                for word_idx in samples:
                                    word_idx = word_idx.item()
                                    if word_idx not in exist_word:
                                        break
                                exist_word.add(word_idx)
                            elif args.previous_word_prior_dedup:
                                for word_idx in samples:
                                    word_idx = word_idx.item()
                                    if len(prev_words) == 0 or prev_words[-1] != word_idx:
                                        break
                                prev_words.append(word_idx)
                            else:
                                word_idx = samples[0]
                            # print("word_idx = ", word_idx)
                            input.data.fill_(word_idx)
                            output, hidden = model(input, hidden)
                            word = corpus.dictionary.idx2word[word_idx]
                            outf.write(word + ' ')

                    # firtst add eol
                    input.data.fill_(eol_id)
                    output, hidden = model(input, hidden)
                    word = corpus.dictionary.idx2word[eol_id]
                    outf.write(word + ' ')


                ################# generate story

                cur_line_num = 0
                exist_word = set()
                for j in range(args.words):

                    output = model.decoder(output)
                    word_weights = output.squeeze().data.div(args.temperature).exp().cpu()
                    if args.top_p > 0.0:
                        word_weights = output.squeeze().div(args.temperature)
                        word_weights = top_k_top_p_filtering(word_weights, top_k=0, top_p=args.top_p,
                                                             filter_value=-float('Inf'))
                        word_weights = word_weights.exp().cpu()
                    samples = torch.multinomial(word_weights, 5)
                    if args.sentence_dedup: # **sentence_dedup
                        print("[sentence_dedup]: ")
                        for word_idx in samples:
                            word_idx = word_idx.item()
                            if word_idx not in exist_word:
                                break
                            word_idx = samples[0]
                        exist_word.add(word_idx)
                    else:
                        word_idx = samples[0]

                    if word_idx == beginning_of_line_idx or (word_idx == eos_id and cur_line_num==5):
                        exist_word = set()
                        cur_line_num += 1
                        print(" found beginning_of_line_idx; cur_line_num = ", cur_line_num)
                        if cur_line_num >= 6:
                            outf.write('\n')
                            break

                    input.data.fill_(word_idx)
                    output, hidden = model(input, hidden)
                    word = corpus.dictionary.idx2word[word_idx]
                    outf.write(word + ' ')

                outf.write('\n')

                print('| Generated {} sentences'.format(nsent+1), file=sys.stderr)

            nsent += 1

        # gf.flush()
        outf.flush()





def encoder_analysis(model, data_source, args, corpus):

    model.eval()
    assert args.inference_use_argmax
    import json

    with open(args.outf, 'w') as outf:
        nsent = 0

        while nsent < args.sents:

            #################

            sentences_data = get_batch(data_source, nsent, args, seq_len=None,
                                       mode='posterior',
                                       dictionary=corpus.dictionary)
            #  B,5,len
            if args.use_emotion_supervision:
                sentences_data, emotion_targets = sentences_data

            sentences_data_txt = []
            for i in range(5):
                sentences_datai = data_source[nsent][0]['line'+str(i)]
                t = []
                for st in sentences_datai.view(-1).data:
                    word = corpus.dictionary.idx2word[st.item()]
                    t.append(word)
                sentences_data_txt.append(' '.join(t))
                print("sentences_data_txti = ", [corpus.dictionary.idx2word[st.item()] for st in sentences_datai])

            kws = []
            if args.rake_as_inference:
                assert False

            else:
                sample, vals = model.get_sample_from_posterior(sentences_data=sentences_data,
                                                            print_more=True,
                                                            completion_mode=True,
                                                            temperature=args.inference_temperature,
                                                            use_argmax=args.inference_use_argmax,)
                for st in sample.view(-1).data:
                    word = corpus.dictionary.idx2word[st.item()]
                    kws.append(word)
                print("vals = ", vals)
                print("kws = ", kws)
                print("sentences_data[0] = ", sentences_data[0])

            cur = {'sentences':sentences_data_txt,
                   'keywords':kws
                   }
            print("---->>> ", json.dumps(cur))

            outf.write(json.dumps(cur))
            outf.write('\n')

            nsent += 1

        outf.flush()


