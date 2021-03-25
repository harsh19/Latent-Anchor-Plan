###############################################################################
# Language Modeling on Penn Tree Bank
#
# This file generates new sentences sampled from the language model
#
###############################################################################

import argparse
import sys
import numpy, math
import torch
import torch.nn as nn
from numbers import Number
from utils import batchify, get_batch, repackage_hidden, get_batch_story_only
import torch.nn.functional as F
import generate_samples

import data

def str2bool(val):
    if val.lower() in ['false']:
        return False
    elif val.lower() in ['true']:
        return True

parser = argparse.ArgumentParser(description='PyTorch Language Model')

# Model parameters.
#parser.add_argument('--train-data', type=str, default='data/penn/train.txt',
#                    help='location of the training data corpus. Used for the rescore_story function')
parser.add_argument('--vocab', type=str, default='../models/vocab.pickle',
                    help='path to a pickle of the vocab used in training the model')
parser.add_argument('--keywords', type=str, default='',
                    help='location of the file for validation keywords')
parser.add_argument('--conditional-data', type=str, default='',
                    help='location of the file that contains the content that the generation conditions on')
parser.add_argument('--happy-endings', type=str, default='',
                    help='location of the file for all happy endings')
parser.add_argument('--sad-endings', type=str, default='',
                    help='location of the file for all sad endings')
parser.add_argument('--story-body', type=str, default='',
                    help='location of the file for story body')
parser.add_argument('--true-endings', type=str, default='',
                    help='location of the file for true endings')
parser.add_argument('--fake-endings', type=str, default='',
                    help='location of the file for fake endings')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (LSTM, QRNN)')
parser.add_argument('--checkpoint', type=str, default='./model.pt',
                    help='model checkpoint to use')
parser.add_argument('--task', type=str, default='generate',
                    choices=['test_eval', 'prior_samples', 'rake_overlap', 'story_samples','inference_samples', 'cond_on_plot', 'marginalize_posterior_eval', 'evaluate_iwnll', 'story_completion', 'controllability_evals','encoder_analysis', 'story_samples_using_posterior', 'story_samples_single'],
                    help='specify the generation task')
parser.add_argument('--outf', type=str, default='generated.txt',
                    help='output file for generated text')
parser.add_argument('--sents', type=int, default='40',
                    help='number of sentences to generate')
parser.add_argument('--words', type=int, default='1000',
                    help='number of words to generate')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--dedup', action='store_true',
                    help='de-duplication')
parser.add_argument('--previous_word_prior_dedup', type=str2bool, default = 'false',
                    help='de-duplication')
parser.add_argument('--sentence_dedup', type=str2bool, default = 'false',
                    help='de-duplication')
parser.add_argument('--print-cond-data', action='store_true',
                    help='whether to print the prompt on which conditionally generated text is conditioned')
parser.add_argument('--bptt', type=int, default=5000,
                    help='sequence length')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature - higher will increase diversity')
parser.add_argument('--log-interval', type=int, default=100,
                    help='reporting interval')
parser.add_argument('--test-data', type=str, default='data/penn/test.txt',
                    help='location of the test data corpus') # new
parser.add_argument('--sanity_check_title', action='store_true', default=False,
                    help='')
parser.add_argument('--sanity_check_title_joined', action='store_true', default=False,
                    help='')
parser.add_argument('--latent_plot_typ', default='kw')
parser.add_argument('--iwnll_num_samples', default=5, type=int)
parser.add_argument('--pretrain_ae_mode', default='false', type=str2bool)
parser.add_argument('--sanity_check_unconditional', action='store_true', default=False,
                    help='')
parser.add_argument('--rake_model', type=str2bool, default='false', help='Use baseline model')
parser.add_argument('--inference_nw_kl_num_samples', type=int, default=1)
parser.add_argument('--debug_mode', type=str2bool, default='false') #action='store_true')
parser.add_argument('--use_argmax', type=str2bool, default='false') #action='store_true')
parser.add_argument('--emotion_special_processing', type=str2bool, default='false')
parser.add_argument('--emotion_type', type=str, default='basic')
parser.add_argument('--new_posterior_batch_defn', type=str2bool, default='true')
parser.add_argument('--infer_nw_ignore_token_type', type=str, default='non_content_set2')
parser.add_argument('--new_decoder', type=str2bool, default='false')
parser.add_argument('--num_sentences_cond', type=int, default=4)
parser.add_argument('--use_emotion_supervision', type=str2bool, default='false')
parser.add_argument('--completion_mode', default='latent_var_model', choices=['latent_var_model','uncon','title_only'])
parser.add_argument('--rake_as_inference', default='false', type=str2bool)
parser.add_argument('--inference_use_argmax', default='false', type=str2bool)
parser.add_argument('--inference_temperature', default=1.0, type=float)
parser.add_argument('--inference_nw_uniform_distribution', default='false', type=str2bool)
parser.add_argument('--new_posterior_batch_defn_notitle', type=str2bool, default='false')
parser.add_argument('--top_p', type=float, default=0.0,
                    help='topp sampling')

args = parser.parse_args()

print("args = ", args)

def log_sum_exp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation
    value.exp().sum(dim, keepdim).log()
    """
    # TODO: torch.max(value, dim=None) threw an error at time of writing
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0),
            dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        if isinstance(sum_exp, Number):
            return m + math.log(sum_exp)
        else:
            return m + torch.log(sum_exp)

def evaluate(data, hidden, args):
    bdata = batchify(torch.LongTensor(data), test_batch_size, args, dictionary=corpus.dictionary)
    source, targets = get_batch(bdata, 0, args, evaluation=True)
    loutput, lhidden = model(source, hidden)
    output_flat = loutput.view(-1, ntokens)
    total_loss = criterion(output_flat, targets).data
    return total_loss[0], lhidden

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

if args.temperature < 1e-3:
    parser.error("--temperature has to be greater or equal 1e-3")

with open(args.checkpoint, 'rb') as f:
    model, criterion, optimizer = torch.load(f, map_location=lambda storage, loc: storage)
print("criterion = ", criterion)

model.eval()
if not hasattr(model, 'tie_weights'):
    model.tie_weights = True
if args.model == 'QRNN':
    model.reset()

if args.cuda:
    model.cuda()
else:
    model.cpu()


corpus = data.Corpus(applyDict=True, dict_path=args.vocab,
                     emotion_special_processing=args.emotion_special_processing,
                     emotion_type=args.emotion_type,
                     use_emotion_supervision=False)
ntokens = len(corpus.dictionary)
padid = corpus.dictionary.word2idx['<pad>']
hidden = model.init_hidden(1)
input = torch.rand(1, 1).mul(ntokens).long()  #, volatile=True)
print('ntokens', ntokens, file=sys.stderr)
if args.cuda:
    input.data = input.data.cuda()

######### GLOBALS #########
eos_id = corpus.dictionary.word2idx['<eos>']
eot_id = corpus.dictionary.word2idx['<EOT>']
delimiter = '#'  # this delimits multi-word phrases. Only used to prevent the delimiter from being deduped in cond_generate when flag present
delimiter_idx = corpus.dictionary.word2idx[delimiter]
pad_id = corpus.dictionary.word2idx['<pad>']

if args.task != 'cond_generate_plot':
    beginning_of_line_idx = corpus.dictionary.word2idx['</s>']
    print('eos id:', eos_id, file=sys.stderr)
else:
    pass


def evaluate_datasplit(data_source, batch_size=10, use_argmax=False):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    print("use_argmax = ", use_argmax)
    if args.model == 'QRNN': model.reset()
    total_loss = 0
    total_elbo = 0
    total_loss_kl = 0
    total_loss_conditional = 0
    total_loss_conditional_per_token = 0
    total_valid_tokens_cnt = 0
    total_prior_loss_per_token = 0
    ntokens = len(corpus.dictionary)

    for i in range(0, len(data_source)):  # - 1, args.bptt):

        ################# Title
        # title is always observed
        if i == 0:
            print_more = True
        else:
            print_more = False

        mode = 'title'  # 'prior'
        pad_prepend = True
        title_data = get_batch(data_source, i, args, seq_len=None, mode=mode,
                               dictionary=corpus.dictionary, pad_prepend=pad_prepend,
                               sanity_check_title_joined=args.sanity_check_title_joined or args.sanity_check_unconditional,
                               sanity_check_title=args.sanity_check_title or args.rake_model,
                               print_more=print_more)
        _, bs = title_data.size()
        hidden_prior = model.init_hidden(bs)  # args.batch_size)
        output_prior, hidden_prior = model(title_data, hidden_prior, return_h=False)
        prior_context = output_prior, hidden_prior  # [-1]

        ################# Plan
        if not (args.rake_model or args.pretrain_ae_mode): # LAP model (except for pretraining inference n/w)
            sentences_data = get_batch(data_source, i, args, seq_len=None, mode='posterior',
                                       dictionary=corpus.dictionary, print_more=print_more)
            sample, _ = model.get_sample_from_posterior(sentences_data=sentences_data,
                                                        use_argmax=use_argmax)
            # sample: bs,5
            # KL(q(z|x,t)||p(z|t))
            kl_loss = model.compute_kl_loss(sentences_data=sentences_data,
                                            prior_context=prior_context,
                                            num_samples=args.inference_nw_kl_num_samples)  # bs
            # Pass Sample
            output, hidden = model(sample.t(), hidden_prior, return_h=False)

        elif args.pretrain_ae_mode:  # AE pretraining mode -- skip computing KL
            sentences_data = get_batch(data_source, i, args, seq_len=None, mode='posterior',
                                       dictionary=corpus.dictionary, print_more=print_more)
            sample, _ = model.get_sample_from_posterior(sentences_data=sentences_data,
                                                        use_argmax=use_argmax)  # todo - add temperature here ?
            output, hidden = model(sample.t(), hidden_prior, return_h=False)
            kl_loss = torch.zeros(title_data.size()[1])
            if args.cuda:
                kl_loss = kl_loss.cuda()

        elif args.rake_model: # Rake tagged Supervised plan
            mode = 'plot'
            data, targets_plot = get_batch(data_source,
                                           i,
                                           args,
                                           seq_len=None,
                                           mode=mode,
                                           dictionary=corpus.dictionary,
                                           sanity_check_title_joined=args.sanity_check_title_joined,
                                           sanity_check_title=args.sanity_check_title,
                                           print_more=print_more)
            output_plot, hidden = model(data, hidden_prior, return_h=False)
            seqlen, bs = data.size()
            prior_loss = []
            prior_loss_per_token = 0.0
            output_plot = output_plot.view(seqlen, bs, -1)
            for bsi in range(bs):
                outputi = output_plot[:, bsi, :]
                targetsi = targets_plot[:, bsi]
                idx = (targetsi != pad_id)
                valid_tokens_cnt = torch.sum(idx).data.item()
                assert valid_tokens_cnt == 5
                outputi = model.prior_model(outputi)
                output_flat = outputi.view(-1, ntokens)
                loss_prior_b = criterion(output_flat, targetsi).sum()
                prior_loss.append(loss_prior_b)  # NLL values
                prior_loss_per_token += loss_prior_b
            prior_loss = torch.stack(prior_loss)  # bs
            prior_loss_per_token /= (bs * 5.0)
            if args.rake_as_inference:
                kl_loss = prior_loss
                # Rake tags can be viewed as an inference network
            else:
                kl_loss = torch.zeros(title_data.size()[1])
                if args.cuda:
                    kl_loss = kl_loss.cuda()
        else:
            kl_loss = torch.zeros(bs)
            if args.cuda:
                kl_loss = kl_loss.cuda()
            hidden = hidden_prior

        ################# p(y|t,z)
        mode = 'conditional'
        data, targets = get_batch(data_source,
                                  i,
                                  args,
                                  seq_len=None,
                                  mode=mode,
                                  dictionary=corpus.dictionary,
                                  sanity_check_title=args.sanity_check_title,
                                  sanity_check_title_joined=args.sanity_check_title_joined,
                                  print_more=print_more)
        seqlen, bs = targets.size()
        output, hidden = model(data, hidden)
        output = output.view(seqlen, bs, -1)
        cond_loss_vals = []
        cond_loss_vals_per_token = []
        for bsi in range(bs):
            outputi = output[:, bsi, :]
            targetsi = targets[:, bsi]
            idx = (targetsi != pad_id)
            valid_tokens_cnt = torch.sum(idx).data.item()
            total_valid_tokens_cnt += valid_tokens_cnt
            outputi = model.decoder(outputi)
            output_flat = outputi.view(-1, ntokens)
            loss_b = criterion(output_flat, targetsi).sum()  # todo -- is sum needed here ?
            cond_loss_vals.append(loss_b)  # NLL values
            cond_loss_vals_per_token.append(loss_b / valid_tokens_cnt)
        cond_loss_vals = torch.stack(cond_loss_vals)  # bs
        cond_loss_vals_per_token = torch.stack(cond_loss_vals_per_token)  # bs
        negative_elbo = cond_loss_vals + kl_loss  # bs and bs -> bs
        # elbo variable holds value of negative ELBO. which is -log p(y|x) + KL(q||p)
        cur_loss = negative_elbo  # bs
        if args.rake_model:
            if args.rake_as_inference:
                # have already compute kl_loss term ussing rake as inference network
                pass
            else:
                # for supervised plan, compute -log p (z|t) - log p(y|z,t)
                # first term is computed earlier and stored in prior_loss
                # second term is stored in negative_elbo (kl_loss variable was set to 0 )
                cur_loss =  prior_loss + negative_elbo # B
                total_valid_tokens_cnt += (bs * 5)
                # include plan token count to total count

        total_loss_conditional += cond_loss_vals.mean().data
        total_loss_conditional_per_token += cond_loss_vals_per_token.mean().data
        total_loss_kl += kl_loss.mean().data
        total_loss += cur_loss.sum().data  # notice 'sum' here
        total_elbo += negative_elbo.mean().data
        if args.rake_model:
            total_prior_loss_per_token += prior_loss_per_token.data
        if args.debug_mode:
            break

    print("EVAL: total_loss_cond = ", total_loss_conditional,
          " ||| total_loss_kl = ", total_loss_kl,
          " || total_elbo = ", total_elbo)
    print("EVAL : loss_cond = ", total_loss_conditional / len(data_source),
          "loss_kl = ", total_loss_kl / len(data_source),
          "loss (total_loss_sum/ total_valid_tokens_cnt) = ", total_loss.item() / total_valid_tokens_cnt,
          "loss_elbo = ", total_elbo.item() / len(data_source),
          "prior_loss_per_token = ", total_prior_loss_per_token / len(data_source),
          "loss_cond_per_token = ", total_loss_conditional_per_token / len(data_source))
    print("EVAL: total_valid_tokens_cnt = ", total_valid_tokens_cnt)
    return total_loss.item() * 1.0 / total_valid_tokens_cnt



def evaluate_iwnll(data_source, num_samples=5):
    '''
    - num_samples
    - eval and no_grad modes
    - sample z from posterior. run prior as well as conditional
    - compute log p(z,x) - log q(z)
    - do this for each sample.   
    - do logsumexp minus log(num_samples)
    '''
    # Turn on evaluation mode which disables dropout.
    model.eval()
    iw_nll = 0
    total_valid_tokens_cnt = 0
    ntokens = len(corpus.dictionary)
    num_samples_tensor = torch.tensor(1.0*num_samples,dtype=torch.float32)
    if args.cuda:
        num_samples_tensor = num_samples_tensor.cuda()

    for i in range(0, len(data_source)):

        mode = 'title'
        pad_prepend = True
        title_data = get_batch(data_source, i, args, seq_len=None, mode=mode,
                               dictionary=corpus.dictionary, pad_prepend=pad_prepend,
                               sanity_check_title_joined=args.sanity_check_title_joined or args.sanity_check_unconditional,
                               sanity_check_title=args.sanity_check_title or args.rake_model)
        _, bs = title_data.size()
        hidden_prior = model.init_hidden(bs)  # args.batch_size)
        output_prior, hidden_prior = model(title_data, hidden_prior, return_h=False)

        sentences_data = get_batch(data_source, i, args, seq_len=None, mode='posterior',
                                   dictionary=corpus.dictionary)
        probs = model.get_posterior_dist(sentences_data=sentences_data)
        distrib = torch.distributions.Categorical(probs=probs) # q(z)

        mode = 'conditional'
        data, targets = get_batch(data_source,
                                  i,
                                  args,
                                  seq_len=None,
                                  mode=mode,
                                  dictionary=corpus.dictionary,
                                  sanity_check_title=args.sanity_check_title,
                                  sanity_check_title_joined=args.sanity_check_title_joined)
        seqlen, bs = targets.size()

        cur_loss_acrossz = []
        for j in range(num_samples):

            with torch.no_grad():

                ## Sample z; Compute q(z)
                position_indices = distrib.sample()  # Bnew
                plot_log_probs = distrib.log_prob(position_indices)  # Bnew
                logq_z = -plot_log_probs.view(bs, 5).sum(dim=1)  # bs
                # log q(z) = \sum_j=1^5 log(q_zj)
                vocab_indices = sentences_data.view(bs * 5, -1).gather(dim=1,
                                                       index=position_indices.unsqueeze(1)).squeeze(1)  # Bnew
                sample = vocab_indices.view(bs, 5)  # bs,5

                ## Compute p(z)
                output, hidden = model(sample.t(), hidden_prior, return_h=False)
                prior_loss = []
                output_plot = output.view(5, bs, -1)
                for bsi in range(bs):
                    outputi = output_plot[:, bsi, :] # 5,ntokens
                    targetsi = sample[bsi,:] # 5
                    outputi = model.prior_model(outputi)
                    output_flat = outputi.view(-1, ntokens)
                    loss_prior_b = criterion(output_flat, targetsi).sum()
                    prior_loss.append(loss_prior_b)  # NLL values
                prior_loss = torch.stack(prior_loss)  # bs
                negative_logp_z = prior_loss

                ## Compute p(x|z)
                output, hidden = model(data, hidden)
                output = output.view(seqlen, bs, -1)
                cond_loss_vals = []
                for bsi in range(bs):
                    outputi = output[:, bsi, :]
                    targetsi = targets[:, bsi]
                    idx = (targetsi != pad_id)
                    valid_tokens_cnt = torch.sum(idx).data.item()
                    if j == 0:  # count only once
                        total_valid_tokens_cnt += valid_tokens_cnt
                    outputi = model.decoder(outputi)
                    output_flat = outputi.view(-1, ntokens)
                    loss_b = criterion(output_flat, targetsi).sum()
                    cond_loss_vals.append(loss_b)  # NLL values
                negative_logp_x_givenz = torch.stack(cond_loss_vals)  # bs

                ## Compute p(x,z)
                negative_logp_x_z = negative_logp_x_givenz + negative_logp_z  # bs and bs -> bs

                ## Compute log p(x,z) - log q(z)
                cur_value = -negative_logp_x_z - logq_z # bs
                cur_loss_acrossz.append(cur_value)

        cur_loss_acrossz = torch.stack(cur_loss_acrossz, dim=0).t()  # bs,num_samples
        # We have [log(v1),log(v2),..] and we want log(v1+v2+...) -> use logsumexp for this
        # log[ (1/K) \sum_k p(x,z)/q(z) ]
        cur_loss_acrossz = torch.logsumexp(cur_loss_acrossz, dim=1) - torch.log(num_samples_tensor)  # bs
        iw_nll -= cur_loss_acrossz.sum().data  # notice 'sum' here

        if i % 100 == 0:
            print("eval: total_loss = ", iw_nll,
                  "eval: total_valid_tokens_cnt = ", total_valid_tokens_cnt,
                  "eval: total_loss/total_valid_tokens_cnt = ", iw_nll / total_valid_tokens_cnt
                  )

        if args.debug_mode:
            break

    print("EVAL: total_loss = ", iw_nll)
    print("EVAL: total_valid_tokens_cnt = ", total_valid_tokens_cnt)
    return iw_nll.item() * 1.0 / total_valid_tokens_cnt


with torch.no_grad():

    if args.task == 'test_eval':
        # Run on test data.
        test_batch_size = 10
        tokenized_test =  corpus.tokenize(args.test_data, applyDict=True)
        test_data = batchify(tokenized_test, test_batch_size, args, dictionary=corpus.dictionary)
        test_loss = evaluate_datasplit(test_data, test_batch_size,
                                       use_argmax=args.use_argmax)
        print('=' * 89)
        print('| test_eval | test loss {:5.2f} | test ppl {:8.2f} | test bpc {:8.3f}'.format(
            test_loss, math.exp(test_loss), test_loss / math.log(2)))
        print('=' * 89)

    elif args.task == 'evaluate_iwnll':
        # Run on test data.
        test_batch_size = 10
        tokenized_test =  corpus.tokenize(args.test_data, applyDict=True)
        test_data = batchify(tokenized_test, test_batch_size, args, dictionary=corpus.dictionary)
        test_loss = evaluate_iwnll(test_data, num_samples=args.iwnll_num_samples) #, test_batch_size)
        print('=' * 89)
        print('| test_eval | test loss {:5.2f} | test ppl {:8.2f} | test bpc {:8.3f}'.format(
            test_loss, math.exp(test_loss), test_loss / math.log(2)))
        print('=' * 89)

    elif args.task == 'prior_samples':
        test_batch_size = 1
        tokenized_test = corpus.tokenize(args.test_data, applyDict=True)
        test_data = batchify(tokenized_test, test_batch_size, args, dictionary=corpus.dictionary)
        generate_samples.generate_plot_samples(model, test_data, args, corpus)
        print('=' * 89)

    elif args.task == 'story_samples':
        test_batch_size = 1
        tokenized_test = corpus.tokenize(args.test_data, applyDict=True)
        test_data = batchify(tokenized_test, test_batch_size, args, dictionary=corpus.dictionary)
        generate_samples.generate_story_samples(model, test_data, args, corpus,
                                                completion_mode=args.completion_mode)
        print('=' * 89)


    elif args.task == 'story_samples_single':
        test_batch_size = 1
        tokenized_test = corpus.tokenize(args.test_data, applyDict=True)
        test_data = batchify(tokenized_test, test_batch_size, args, dictionary=corpus.dictionary)
        generate_samples.generate_story_samples(model, test_data, args, corpus,
                                                completion_mode=args.completion_mode,
                                                samples_per_tile=1)
        print('=' * 89)

    elif args.task == 'story_samples_using_posterior':
        test_batch_size = 1
        tokenized_test = corpus.tokenize(args.test_data, applyDict=True)
        test_data = batchify(tokenized_test, test_batch_size, args, dictionary=corpus.dictionary)
        generate_samples.generate_story_samples(model, test_data, args, corpus, use_posterior_samples=True)
        print('=' * 89)

    elif args.task == 'encoder_analysis':
        test_batch_size = 1
        tokenized_test = corpus.tokenize(args.test_data, applyDict=True)
        test_data = batchify(tokenized_test, test_batch_size, args, dictionary=corpus.dictionary)
        generate_samples.encoder_analysis(model, test_data, args, corpus)
        print('=' * 89)
