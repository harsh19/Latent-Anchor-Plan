import argparse
import time
import math
import numpy as np
import sys
import torch
import torch.nn as nn

import data
import model_inference as model


from utils import batchify, get_batch, repackage_hidden, load_pickle

def str2bool(val):
    if val.lower() in ['false']:
        return False
    elif val.lower() in ['true']:
        return True

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--train-data', type=str, default='rocstory_plan_write/ROCStories_all_merge_tokenize.titlesepkeysepstory.train',
                    help='location of the training data corpus')
parser.add_argument('--valid-data', type=str, default='rocstory_plan_write/ROCStories_all_merge_tokenize.titlesepkeysepstory.dev',
                    help='location of the valid data corpus')
parser.add_argument('--test-data', type=str, default='rocstory_plan_write/ROCStories_all_merge_tokenize.titlesepkeysepstory.test',
                    help='location of the test data corpus')
parser.add_argument('--vocab-file', type=str, default='',
                    help='filename to save the vocabulary pickle to')
parser.add_argument('--applyDict', type=str2bool, default='false',
                    help='if loading dictionary from elsewhere')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (LSTM, QRNN, GRU)')
parser.add_argument('--emsize', type=int, default=400,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=1150,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=3,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=30,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=50,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=80, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=70,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.4,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--dropouth', type=float, default=0.3,
                    help='dropout for rnn layers (0 = no dropout)')
parser.add_argument('--dropouti', type=float, default=0.65,
                    help='dropout for input embedding layers (0 = no dropout)')
parser.add_argument('--dropoute', type=float, default=0.1,
                    help='dropout to remove words from embedding layer (0 = no dropout)')
parser.add_argument('--wdrop', type=float, default=0.5,
                    help='amount of weight dropout to apply to the RNN hidden to hidden matrix')
parser.add_argument('--tied', action='store_false',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--nonmono', type=int, default=5,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
randomhash = ''.join(str(time.time()).split('.'))
parser.add_argument('--save', type=str,  default=randomhash+'.pt',
                    help='path to save the final model')
parser.add_argument('--alpha', type=float, default=2,
                    help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
parser.add_argument('--beta', type=float, default=1,
                    help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
parser.add_argument('--beta_prior', type=float, default=0.0,
                    help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
parser.add_argument('--gamma', type=float, default=0.0,
                    help='inference nw and other l2 on model weights')
parser.add_argument('--wdecay', type=float, default=1.2e-6,
                    help='weight decay applied to all weights')
parser.add_argument('--resume', type=str,  default='',
                    help='path of model to resume')
parser.add_argument('--optimizer', type=str,  default='sgd',
                    help='optimizer to use (sgd, adam)')
parser.add_argument('--when', nargs="+", type=int, default=[-1],
                    help='When (which epochs) to divide the learning rate by 10 - accepts multiple')
parser.add_argument('--task_type', default='prior',
                    help='prior/conditional')
parser.add_argument('--moving_avg_ratio', default=0.1,
                    help='')
parser.add_argument('--posterior_loss_wt', default=1.0, type=float,
                    help='')
parser.add_argument('--use_fixed_uniform_prior', action='store_true', default=False,
                    help='')
parser.add_argument('--sanity_check_title', action='store_true', default=False,
                    help='')
parser.add_argument('--sanity_check_title_joined', action='store_true', default=False,
                    help='')
parser.add_argument('--sanity_check_unconditional', action='store_true', default=False,
                    help='')
parser.add_argument('--rake_model', type=str2bool, default='false', help='Use baseline model')
parser.add_argument('--infer_nw_skip_first_token', type=str2bool, default='false',
                    help='')
parser.add_argument('--infer_nw_arch_type', default='bilstm',
                    help='')
parser.add_argument('--inference_pretrained_model_path', default=None,
                    help='')
parser.add_argument('--inference_pretrained_model_path_extractinference', default='false',help='',type=str2bool)
parser.add_argument('--infer_nw_ignore_token_type', type=str, default='non_content_set2')
parser.add_argument('--infer_nw_share_encoder', type=str2bool, default='false')
parser.add_argument('--inference_nw_frozen', type=str2bool, default='false') #action='store_true')
parser.add_argument('--inference_nw_kl_num_samples', type=int, default=1)
parser.add_argument('--debug_mode', type=str2bool, default='false') #action='store_true')
parser.add_argument('--train_inference_nw_only', type=str2bool, default='false')
parser.add_argument('--run_name', type=str, default='default')
parser.add_argument('--anneal_function', type=str, default='constant')
parser.add_argument('--anneal_k', type=float, default=0.0025)
parser.add_argument('--anneal_x0', type=int, default=2500)
parser.add_argument('--temp_function', type=str, default='constant')
parser.add_argument('--temp_k', type=float, default=0.0025)
parser.add_argument('--temp_x0', type=int, default=25000)
parser.add_argument('--temp_inf', type=float, default=0.7)
parser.add_argument('--latent_plot_typ', default='kw')
parser.add_argument('--load_pretrained_model_path', default=None, type=str)
parser.add_argument('--pretrain_ae_mode', default='false', type=str2bool)
parser.add_argument('--inference_nw_uniform_distribution', default='false', type=str2bool)
parser.add_argument('--inference_nw_first_word_distribution', default='false', type=str2bool)
parser.add_argument('--train_prior_only', type=str2bool, default='false')  # this freezes the backbone
parser.add_argument('--print_cond_data', type=str2bool, default='true')
parser.add_argument('--temperature', type=float, default=0.7)
parser.add_argument('--new_posterior_batch_defn', type=str2bool, default='true')
parser.add_argument('--emotion_type', type=str, default='basic')
parser.add_argument('--emotion_special_processing', type=str2bool, default='false')
parser.add_argument('--position_entropy', type=str2bool, default='false')
parser.add_argument('--entropy', type=str2bool, default='false')
parser.add_argument('--entropy_max', type=float, default=10.0)
parser.add_argument('--entropy_min', type=float, default=0.1)
parser.add_argument('--use_emotion_supervision', type=str2bool, default='false')
parser.add_argument('--new_decoder', type=str2bool, default='false')
parser.add_argument('--strict_load', type=str2bool, default='true')
parser.add_argument('--gamma_scores_l2', type=float, default=0.0,help='')
parser.add_argument('--prior_regularize', type=str2bool, default='false', help='')
parser.add_argument('--use_moving_avg_baseline_priorreg', type=str2bool, default='false', help='')
parser.add_argument('--prior_reg_loss_fact', type=float, default=1.0, help='')
parser.add_argument('--new_posterior_batch_defn_notitle', type=str2bool, default='false')
parser.add_argument('--regularize_posterior', type=str2bool, default='false')
parser.add_argument('--top_p', type=float, default=0.0, help='topp sampling')
parser.add_argument('--dedup', action='store_true', help='de-duplication')
parser.add_argument('--previous_word_prior_dedup', type=str2bool, default = 'false', help='de-duplication')
parser.add_argument('--sentence_dedup', type=str2bool, default = 'false', help='de-duplication')

# parser.add_argument('--debug', action='store_true')
args = parser.parse_args()
#args.tied = True
print("#"*21)
print("## args = ", args)
print("#"*21)
print()

if args.rake_model:
    assert args.latent_plot_typ == 'kw', "For RAKE model, latent_plot_typ must be kw"

# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################

def model_save(fn):
    with open(fn, 'wb') as f:
        torch.save([model, criterion, optimizer], f)

def model_load(fn):
    global model, criterion, optimizer
    with open(fn, 'rb') as f:
        model, criterion, optimizer = torch.load(f)

print('Producing dataset...')
dict_path = None
if args.applyDict:
    dict_path = args.vocab_file
corpus = data.Corpus(applyDict=args.applyDict, train_path=args.train_data, dev_path=args.valid_data,
                         test_path=args.test_data, output=args.vocab_file, dict_path=dict_path,
                         emotion_special_processing=args.emotion_special_processing,
                         emotion_type=args.emotion_type,
                         use_emotion_supervision=args.use_emotion_supervision)

eval_batch_size = 10
test_batch_size = 1
train_data = batchify(corpus.train, args.batch_size, args, dictionary=corpus.dictionary)
val_data = batchify(corpus.valid, eval_batch_size, args, dictionary=corpus.dictionary)
test_data = batchify(corpus.test, test_batch_size, args, dictionary=corpus.dictionary)
pad_id = corpus.dictionary.word2idx['<pad>']
sent_start_id = corpus.dictionary.word2idx['</s>']

moving_avg_baseline = None
moving_avg_baseline_priorreg = None
task_type = args.task_type

###############################################################################
# Build the model
###############################################################################

padid = corpus.dictionary.word2idx['<pad>']
criterion = nn.CrossEntropyLoss(size_average=False, ignore_index=padid)
criterion_prior = nn.CrossEntropyLoss(ignore_index=padid)

ntokens = len(corpus.dictionary)

model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.dropouth,
                       args.dropouti,
                       args.dropoute, args.wdrop, args.tied,
                       use_fixed_uniform_prior=args.use_fixed_uniform_prior,
                       cuda=args.cuda,
                       dictionary=corpus.dictionary,
                       latent_plot_typ=args.latent_plot_typ,
                       infer_nw_arch_type=args.infer_nw_arch_type,
                       infer_nw_ignore_token_type=args.infer_nw_ignore_token_type,
                       inference_pretrained_model_path=args.inference_pretrained_model_path,
                       inference_pretrained_model_path_extractinference=args.inference_pretrained_model_path_extractinference,
                       infer_nw_share_encoder=args.infer_nw_share_encoder,
                       inference_nw_frozen=args.inference_nw_frozen)
print("model = ", model)

if args.load_pretrained_model_path is not None:
    print("***** Loading model from ", args.load_pretrained_model_path)
    with open(args.load_pretrained_model_path, 'rb') as f:
        mdl, _, _ = torch.load(f)
        state_dict = mdl.state_dict()
        model.load_state_dict(state_dict,strict=args.strict_load)
        print(" *** Loading keys: ", state_dict.keys())
        # if mdl.infer_nw_share_encoder:
        print("**** Inference n/w encoder will be initialized with saved model irrespective of infer_nw_share_encoder")

if args.cuda:
    model = model.cuda()
###
params = list(model.parameters()) #+ list(criterion.parameters())
total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in params if x.size())
print('Args:', args)
print('Model total parameters:', total_params)

if args.train_prior_only:
    assert False

if args.train_inference_nw_only:
    assert args.load_pretrained_model_path is not None
    print("")
    print("* Training INFERENCE NW only")
    print("")
    params = list(model.inference_nw.parameters())
    if args.infer_nw_share_encoder:  # ** new
        assert False
    else:
        print("***** Note that  train_inference_nw_only is true ; and encoder is SEPARATE. deepcopying encoder")
        import copy
        model.inference_nw.encoder = copy.deepcopy(model.inference_nw.encoder)
        # if it was shared earlier, then now it will get detached
    total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in params if x.size())
    print('Model trainable total parameters (considering the train_inference_nw_only):', total_params)


##################################


# def generate_plot_samples(model, data_source, args, corpus):
#     generate_samples.generate_plot_samples(model, data_source, args, corpus)
#
# def generate_story_samples(model, data_source, args, corpus, use_posterior_samples=False):
#     generate_samples.generate_story_samples(model, data_source, args, corpus,
#                                             use_posterior_samples=use_posterior_samples,
#                                             samples_per_tile=1)

def generate_plot_samples(model, data_source, args, corpus):

    eos_id = corpus.dictionary.word2idx['<eos>']
    eot_id = corpus.dictionary.word2idx['<EOT>']
    ntokens = len(corpus.dictionary)
    input = torch.rand(1, 1).mul(ntokens).long()  # , volatile=True)
    if args.cuda:
        input = input.cuda()

    model.eval()
    with torch.no_grad():

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
                    # output = model.decoder_prior(output)
                    word_weights = output.squeeze().data.div(args.temperature).exp().cpu()
                    # print("word_weights : ", word_weights.size())
                    samples = torch.multinomial(word_weights, 5)
                    # print("samples = ", samples)
                    word_idx = samples[0]
                    # print("word_idx = ", word_idx)
                    input.data.fill_(word_idx)
                    output, hidden = model(input, hidden)
                    word = corpus.dictionary.idx2word[word_idx]
                    outf.write(word + ' ')
                    #if i == cond_length-1:
                    #    outf.write('\t >>\t')

                outf.write('\n')

                print('| Generated {} sentences'.format(nsent+1), file=sys.stderr)
                nsent += 1

            # gf.flush()
            outf.flush()
            print("Writing to ", args.outf)


def generate_story_samples(model, data_source, args, corpus, use_provided_plots=False, use_posterior_samples=False):

    eos_id = corpus.dictionary.word2idx['<eos>']
    eot_id = corpus.dictionary.word2idx['<EOT>']
    eol_id = corpus.dictionary.word2idx['<EOL>']
    beginning_of_line_idx = corpus.dictionary.word2idx['</s>']
    ntokens = len(corpus.dictionary)
    input = torch.rand(1, 1).mul(ntokens).long()  # , volatile=True)
    if args.cuda:
        input = input.cuda()

    model.eval()
    with torch.no_grad():

        with open(args.outf, 'w') as outf: #, open('gold_4sent.txt', 'w') as gf:
            data = data_source #corpus.tokenize(args.conditional_data, applyDict=True).tolist()
            # this is a list of ids corresponding to words from the word2idx dict
            nsent = 0

            while nsent < args.sents:

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

                ################# generate plot

                if use_provided_plots or args.sanity_check_title:
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
                    for i in range(5):
                        output = model.prior_model(output)
                        word_weights = output.squeeze().data.div(args.temperature).exp().cpu()
                        # print("word_weights : ", word_weights.size())
                        samples = torch.multinomial(word_weights, 5)
                        # print("samples = ", samples)
                        word_idx = samples[0]
                        # print("word_idx = ", word_idx)
                        input.data.fill_(word_idx)
                        output, hidden = model(input, hidden)
                        word = corpus.dictionary.idx2word[word_idx]
                        outf.write(word + ' ')
                        #if i == cond_length-1:
                        #    outf.write('\t >>\t')

                ################# generate story

                if not args.sanity_check_title:
                    # firtst add eol
                    input.data.fill_(eol_id)
                    output, hidden = model(input, hidden)
                    word = corpus.dictionary.idx2word[eol_id]
                    outf.write(word + ' ')

                cur_line_num = 0
                for j in range(args.words):

                    output = model.decoder(output)
                    word_weights = output.squeeze().data.div(args.temperature).exp().cpu()
                    # print("word_weights : ", word_weights.size())
                    samples = torch.multinomial(word_weights, 5)
                    # print("samples = ", samples)
                    word_idx = samples[0]
                    # print("word_idx = ", word_idx)

                    if word_idx == beginning_of_line_idx or (word_idx == eos_id and cur_line_num==5):
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





def evaluate(data_source): #, batch_size=10):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    if args.model == 'QRNN': model.reset()
    total_loss = 0
    total_elbo = 0
    total_loss_kl = 0
    total_loss_conditional = 0
    total_loss_conditional_per_token = 0
    total_valid_tokens_cnt = 0
    total_prior_loss_per_token = 0
    ntokens = len(corpus.dictionary)

    for i in range(0, len(data_source)):

        #################

        mode = 'title' #'prior'
        pad_prepend = True
        title_data = get_batch(data_source, i, args, seq_len=None, mode=mode,
                               dictionary=corpus.dictionary, pad_prepend=pad_prepend,
                               sanity_check_title_joined=args.sanity_check_title_joined or args.sanity_check_unconditional,
                               sanity_check_title=args.sanity_check_title or args.rake_model)
        _, bs = title_data.size()
        hidden_prior = model.init_hidden(bs) #args.batch_size)
        output_prior, hidden_prior = model(title_data, hidden_prior, return_h=False)
        prior_context = output_prior, hidden_prior #[-1]


        #################
        if not (args.sanity_check_title  or args.rake_model or args.pretrain_ae_mode or args.sanity_check_unconditional):
            sentences_data = get_batch(data_source, i, args, seq_len=None, mode='posterior', dictionary=corpus.dictionary)
            sample, _ = model.get_sample_from_posterior(sentences_data=sentences_data) # todo - add temperature here ?
                                                           # prior_context=prior_context)
            # sample: bs,5
            kl_loss = model.compute_kl_loss(sentences_data=sentences_data,
                                            prior_context=prior_context,
                                            num_samples=args.inference_nw_kl_num_samples) # bs
            ### pass 'sample' through model
            output, hidden = model(sample.t(), hidden_prior, return_h=False)

        elif args.pretrain_ae_mode: # skip computing KL
            assert False

        elif args.rake_model:
            sample = None
            mode='plot'
            data, targets_plot = get_batch(data_source,
                                      i,
                                      args,
                                      seq_len=None,
                                      mode=mode,
                                      dictionary=corpus.dictionary,
                                      sanity_check_title_joined=args.sanity_check_title_joined,
                                      sanity_check_title=args.sanity_check_title,
                                      print_more=False)
            output_plot, hidden = model(data, hidden_prior, return_h=False)
            seqlen,bs = data.size()
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
                loss_prior_b = criterion(output_flat, targetsi).sum() # ** NOTE: criterion
                prior_loss.append(loss_prior_b )  # NLL values
                prior_loss_per_token += loss_prior_b
            prior_loss = torch.stack(prior_loss)  # B
            prior_loss_per_token /= (bs*5.0)
            kl_loss = torch.zeros(title_data.size()[1])
            if args.cuda:
                kl_loss = kl_loss.cuda()

        else:
            assert False


        #################
        mode = 'conditional'
        data, targets = get_batch(data_source,
                                        i,
                                        args,
                                        seq_len=None,
                                        mode=mode,
                                        dictionary=corpus.dictionary,
                                        sanity_check_title=args.sanity_check_title,
                                        sanity_check_title_joined=args.sanity_check_title_joined,
                                        provided_kw = sample if args.new_decoder else None
                                  )
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
        cur_loss = negative_elbo # bs
        if args.rake_model:
            cur_loss = negative_elbo + prior_loss # bs
            total_valid_tokens_cnt += (bs*5)

        total_loss_conditional += cond_loss_vals.mean().data
        total_loss_conditional_per_token += cond_loss_vals_per_token.mean().data
        total_loss_kl += kl_loss.mean().data
        total_loss += cur_loss.sum().data # notice 'sum' here
        total_elbo += negative_elbo.mean().data
        if args.rake_model:
            total_prior_loss_per_token += prior_loss_per_token.data

        if args.debug_mode:
            break

    print("EVAL: total_loss_cond = ", total_loss_conditional,
          " ||| total_loss_kl = ", total_loss_kl,
          " || total_elbo = ", total_elbo)
    print("EVAL : loss_cond = " , total_loss_conditional / len(data_source),
          "loss_kl = " , total_loss_kl / len(data_source) ,
          "loss (total_loss_sum/ total_valid_tokens_cnt) = " , total_loss.item()/ total_valid_tokens_cnt,
          "loss_elbo = " , total_elbo.item()/ len(data_source),
          "prior_loss_per_token = " , total_prior_loss_per_token / len(data_source),
          "loss_cond_per_token = ", total_loss_conditional_per_token / len(data_source) )
    print("EVAL: total_valid_tokens_cnt = ", total_valid_tokens_cnt)
    return total_loss.item() * 1.0 / total_valid_tokens_cnt




###############################################################################
# Training code
###############################################################################



assert args.anneal_function in ['constant','logistic','linear']
def kl_anneal_function(step, anneal_function=args.anneal_function, k=args.anneal_k, x0=args.anneal_x0):
    if anneal_function == 'constant':
        return 1.0
    if anneal_function == 'logistic':
        return float(1/(1+np.exp(-k*(step-x0))))
    elif anneal_function == 'linear':
        return min(1, step/x0)

def temperature_anneal_function(step, temp_function=args.temp_function, k=args.temp_k, x0=args.temp_x0, tinf=args.temp_inf):
    if temp_function == 'constant':
        return 1.0
    elif temp_function == 'logistic':
        return max(tinf, 1.0-float(1/(1+np.exp(-k*(step-x0)))) )
    elif temp_function == 'linear':
        return max(tinf, 1.0 - step/x0)

num_train_steps = 0
def train():
    global moving_avg_baseline, num_train_steps, moving_avg_baseline_priorreg
    # Turn on training mode which enables dropout.
    if args.model == 'QRNN': model.reset()
    total_loss = 0
    total_loss_kl = 0
    total_loss_conditional = 0
    total_loss_conditional_per_token = 0
    total_elbo = 0
    total_reward = 0
    # total_prior_loss = 0
    total_prior_loss_per_token = 0.0
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    batch= 0
    num_batches = len(train_data) #//bsz
    beta_prior = args.beta_prior > 0

    while batch < num_batches: # - 1 - 1:

        model.train()
        optimizer.zero_grad()

        print_more = False
        if args.debug_mode or np.random.rand()<0.001 or batch%100==0:
            print_more = True

        #######-------title
        mode = 'title' #'prior'
        pad_prepend = True
        title_data = get_batch(train_data,
                               batch,
                               args,
                               seq_len=None,
                               mode=mode,
                               dictionary=corpus.dictionary,
                               pad_prepend=pad_prepend,
                               sanity_check_title_joined=args.sanity_check_title_joined or args.sanity_check_unconditional,
                               sanity_check_title=args.sanity_check_title or args.rake_model, #**
                               print_more=print_more)
        hidden_prior = model.init_hidden(args.batch_size)
        output_prior, hidden_prior = model(title_data, hidden_prior, return_h=False)
        prior_context = output_prior, hidden_prior #[-1]

        step_temperature=1.0
        if not (args.sanity_check_title or args.rake_model or args.pretrain_ae_mode or args.sanity_check_unconditional):
            #######-------Sample
            sentences_data = get_batch(train_data, batch, args, seq_len=None, mode='posterior',
                                       dictionary=corpus.dictionary, print_more=print_more)
            step_temperature = temperature_anneal_function(num_train_steps)
            sample, vals = model.get_sample_from_posterior(sentences_data=sentences_data,
                                                           print_more=print_more,
                                                           temperature=step_temperature)
            logprobs_sample = vals['logprobs'] # bs,5
            output, hidden = model(sample.t(), hidden_prior, return_h=False)
            distribs_distr = vals['distrib']
            scores = vals['scores'].view(logprobs_sample.size()[0],5,-1) # bs,5
            if print_more:
                distribs = vals['distrib'].probs.view(logprobs_sample.size()[0], 5, -1)  # bs,5
                for i in range(len(sample)):
                    datai = sample[i]
                    tokens = [corpus.dictionary.idx2word[idx] for idx in datai.data.cpu().numpy()]
                    # sentences_datai = [corpus.dictionary.idx2word[idx] for idx in sentences_data[i].data.cpu().numpy()]
                    print("[SAMPLE PASSING] sampli = ", tokens)
                    # print("[SAMPLE PASSING] sentences_datai = ", sentences_datai)
                    print("[SAMPLE PASSING] distribi = ", distribs[i])
            #######-------KL
            kl_loss = model.compute_kl_loss(sentences_data=sentences_data,
                                            prior_context=prior_context,
                                            num_samples=args.inference_nw_kl_num_samples,
                                            beta_prior=beta_prior)
            if beta_prior:
                kl_loss, rnn_hs_prior_fromkl = kl_loss
            # kl_loss: bs

        elif args.pretrain_ae_mode: # skip computing KL
            assert False

        elif args.rake_model:
            mode='plot'
            data, targets_plot = get_batch(train_data,
                                      batch,
                                      args,
                                      seq_len=None,
                                      mode=mode,
                                      dictionary=corpus.dictionary,
                                      sanity_check_title_joined=args.sanity_check_title_joined,
                                      sanity_check_title=args.sanity_check_title,
                                      print_more=print_more)
            output_plot, hidden, _, _= model(data, hidden_prior, return_h=True)
            seqlen,bs = data.size()
            prior_loss = []
            prior_loss_per_token = 0.0
            output_plot = output_plot.view(seqlen, bs, -1)
            for bsi in range(bs):
                outputi = output_plot[:, bsi, :]
                targetsi = targets_plot[:, bsi]
                idx = (targetsi != pad_id)
                valid_tokens_cnt = torch.sum(idx).data.item()
                assert valid_tokens_cnt == 5,  targetsi
                outputi = model.prior_model(outputi)
                output_flat = outputi.view(-1, ntokens)
                loss_prior_b = criterion(output_flat, targetsi).sum() #
                prior_loss.append(loss_prior_b )  # NLL values
                prior_loss_per_token += loss_prior_b
            prior_loss = torch.stack(prior_loss)  # B
            prior_loss_per_token /= (bs*5.0)
            kl_loss = torch.zeros(title_data.size()[1])
            if args.cuda:
                kl_loss = kl_loss.cuda()
            sample = None

        else:
            assert False


        ######-------Conditional
        # logprobs_sample: 5,bs
        mode = 'conditional'
        if args.sanity_check_title: #
            mode = 'conditional_title'
        data, targets = get_batch(train_data,
                                        batch,
                                        args,
                                        seq_len=None,
                                        mode=mode,
                                        dictionary=corpus.dictionary,
                                        sanity_check_title_joined=args.sanity_check_title_joined,
                                        sanity_check_title=args.sanity_check_title,
                                        print_more=print_more,
                                        provided_kw=sample if args.new_decoder else None
                                  )
        output, hidden, rnn_hs, dropped_rnn_hs = model(data, hidden, return_h=True)
        seqlen, bs = targets.size()
        output = output.view(seqlen, bs, -1)
        cond_loss_vals = []
        cond_loss_vals_per_token = []
        for bsi in range(bs):
            outputi = output[:,bsi,:]
            targetsi = targets[:,bsi]
            idx = (targetsi != pad_id)
            valid_tokens_cnt = torch.sum(idx).data.item()
            outputi = model.decoder(outputi)
            output_flat = outputi.view(-1, ntokens)
            loss_b = criterion(output_flat, targetsi).sum()
            cond_loss_vals.append(loss_b) # NLL values
            cond_loss_vals_per_token.append(loss_b/valid_tokens_cnt)
        cond_loss_vals = torch.stack(cond_loss_vals) # B
        cond_loss_vals_per_token = torch.stack(cond_loss_vals_per_token) # B

        if not (args.sanity_check_title or args.rake_model or args.sanity_check_unconditional): # or args.pretrain_ae_mode):
            #####------Posterior reward computation
            rewards = -cond_loss_vals.detach() # detaching since don't want to update reard tensor through prior loss term
            factor = 0.1
            rewards = rewards * factor
            if moving_avg_baseline is None:
                moving_avg_baseline = rewards.detach().mean()
            cur_mean_reward = rewards.detach().mean()
            total_reward += rewards.mean().data
            if print_more:
                print("rewards = ", rewards)
            rewards -= moving_avg_baseline.detach()
            rewards = rewards.unsqueeze(1).repeat(1, 5) # bs -> bs,5
            rewards[sample == pad_id] -= 300.0*factor

            if print_more:
                # print("sample == pad_id ", sample == pad_id)
                print("moving_avg_baseline = ", moving_avg_baseline)
                print("rewards after baseline = ", rewards)
                print("logprobs_sample = ", logprobs_sample)
            to_maximize = logprobs_sample * rewards # B,5

            posterior_loss =  -to_maximize.mean()
            assert len(logprobs_sample.size()) == len(rewards.size())
            assert logprobs_sample.size()[0] == rewards.size()[0]
            ratio = args.moving_avg_ratio # 0.1
            if print_more:
                print("moving_avg_baseline = ", moving_avg_baseline,
                      " ratio*rewards.detach().mean()= ", cur_mean_reward,
                      "(1.0-ratio)*moving_avg_baseline + ratio*cur_mean_reward.detach()=",
                      (1.0-ratio)*moving_avg_baseline + ratio*cur_mean_reward.detach() )
            moving_avg_baseline = (1.0 - ratio) * moving_avg_baseline + ratio * cur_mean_reward.detach()


        ###################
        prior_reg_loss = 0.0
        prior_reg_logprobs = []
        if args.prior_regularize:
            hidden_prior_reg = model.init_hidden(bs)  # args.batch_size)
            # printsamples_regi("title_data : ", title_data.size())
            output_prior_reg, hidden_prior_reg = model(title_data, hidden_prior_reg, return_h=False)
            output_prior_reg = output_prior_reg.view(-1, bs, output_prior_reg.size()[1])[-1, :, :]
            samples_reg = []
            for i in range(5):
                output_prior_reg = model.prior_model(output_prior_reg)
                word_weights = output_prior_reg
                distrib = torch.distributions.Categorical(logits=word_weights.squeeze())
                word_idx = distrib.sample()
                if args.cuda:
                    word_idx = word_idx.cuda()
                output_prior_reg, hidden_prior_reg = model(word_idx.view(1,-1), hidden_prior_reg)
                samples_reg.append(word_idx.unsqueeze(0))
                prior_reg_logprobs.append( distrib.log_prob(word_idx).unsqueeze(0) )
            samples_reg = torch.cat(samples_reg) # 5, bs
            prior_reg_logprobs = torch.cat(prior_reg_logprobs) # len,bs
            samples_reg = samples_reg.t() # bs,len
            prior_reg_logprobs = prior_reg_logprobs.t() # bs,len
            prior_reg_rewards = torch.zeros_like(prior_reg_logprobs) # bs, len
            if print_more:
                print("samples_reg = ", samples_reg)
                for samples_regi in samples_reg:
                    tokens = [corpus.dictionary.idx2word[idx] for idx in samples_regi.data.cpu().numpy()]
                    print("samples_regi = ", tokens)
            for idx in range(1,5):
                matches = samples_reg[:,idx].unsqueeze(1) == samples_reg[:,:idx]
                matches = matches.sum(dim=1).float()
                assert prior_reg_rewards.size()[0] == matches.size()[0], matches.size()
                prior_reg_rewards[:,idx] -= matches # penalize matches
                if print_more:
                    print("samples_reg[:,idx].unsqueeze(1) == samples_reg[:,:idx] = ",
                          (samples_reg[:,idx].unsqueeze(1) == samples_reg[:,:idx]).sum(dim=1).float() )


        #####---------loss computation and backprop
        kl_weight = kl_anneal_function(step=num_train_steps)
        loss = elbo = cond_loss_vals + kl_weight*kl_loss # B and B -> B
        elbo = elbo.mean() #1
        if args.rake_model:
            loss += prior_loss  # supervised model: for given z, maximize: log p(z) + log p(x|z)
        if args.prior_regularize:
            if args.use_moving_avg_baseline_priorreg:
                if moving_avg_baseline_priorreg is not None:
                    prior_reg_rewards = prior_reg_rewards - moving_avg_baseline_priorreg.detach()
                    moving_avg_baseline_priorreg = 0.9 * moving_avg_baseline_priorreg + 0.1 * prior_reg_rewards.mean().detach()
                else:
                    moving_avg_baseline_priorreg = prior_reg_rewards.mean().detach()
            if print_more:
                print("loss before = ", loss)
                print("prior_reg_logprobs = ", prior_reg_logprobs)
                print("prior_reg_rewards = ", prior_reg_rewards)
            prior_reg_loss = -prior_reg_rewards * prior_reg_logprobs # to minimize (- reward * probs)
            prior_reg_loss = prior_reg_loss.mean(dim=1) # bs,5 -> bs
            loss += args.prior_reg_loss_fact * prior_reg_loss #.mean()
            if print_more:
                print("prior_reg_loss.mean(dim=1) = ", prior_reg_loss)
                print("loss after = ", loss)
        if args.entropy:
            # print("distribs_distr = ", distribs_distr)
            entropy = distribs_distr.entropy() # we want to penalize very low entropy # B
            entropy = entropy.view(bs,5).sum(dim=1)
            if print_more:
                print("entropy = ", entropy) # todo -> need to avoid including pad
                # print("max(0.1, min(100./num_train_steps,1.) )*(-entropy) = ",
                #       max(0.1, min(100./(1+num_train_steps),1.) )*(-entropy))
                print("loss before = ", loss)
                print("loss after = ",  (loss-max(args.entropy_min, min(3000./(1+num_train_steps),args.entropy_max) )*(entropy)))
            # loss += max(0.1, min(100./(1+num_train_steps),1.) )*(-entropy)
            if args.latent_plot_typ == 'emotion':
                loss +=  0.0*(-entropy)
            else:
                loss +=  max(args.entropy_min, min(3000./(1+num_train_steps),args.entropy_max) )*(-entropy)
        if args.position_entropy:
            assert False
        loss = loss.mean()
        if args.sanity_check_title or args.rake_model or args.sanity_check_unconditional: # or args.pretrain_ae_mode:
            posterior_loss = 0 * loss.detach()
            total_reward = 0 * loss.detach()
        loss += args.posterior_loss_wt*posterior_loss

        ## Regularization
        # Activiation Regularization
        if args.alpha: loss = loss + sum(args.alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])
        # Temporal Activation Regularization (slowness)
        if args.beta: loss = loss + sum(args.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])
        if beta_prior:
            if print_more:
                print("beta prior: lossbefore = ", loss)
            loss = loss + args.beta_prior * (rnn_hs_prior_fromkl[1:] - rnn_hs_prior_fromkl[:-1]).pow(2).mean()
            if print_more:
                print("beta prior: lossafter = ", loss)
        if args.gamma: loss += args.gamma * model.get_l2_loss()
        if args.gamma_scores_l2:
            # print(scores.size())
            # print(scores.pow(2).mean())
            if print_more:
                print("gamma_scores_l2: lossbefore = ", loss)
            loss += args.gamma_scores_l2 * scores.pow(2).mean()
            if print_more:
                print("gamma_scores_l2: lossbefore = ", loss)

        loss.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        if args.clip: torch.nn.utils.clip_grad_norm_(params, args.clip)
        optimizer.step()
        num_train_steps += 1


        #####---------tracking
        total_loss += loss.data
        total_loss_conditional += cond_loss_vals.mean().data
        total_loss_conditional_per_token += cond_loss_vals_per_token.mean().data
        total_loss_kl += kl_loss.mean().data
        total_elbo += elbo.data  # here we are tracking elbo ( per sentence )
        if args.rake_model:
            total_prior_loss_per_token += prior_loss_per_token.data

        # total_prior_loss += prior_loss.data
        if batch % args.log_interval == 0: # and batch > 0:
            cur_loss = total_loss.item() / args.log_interval
            cur_loss_conditional = total_loss_conditional.item() / args.log_interval
            cur_loss_conditional_per_token = total_loss_conditional_per_token.item() / args.log_interval
            cur_loss_kl = total_loss_kl.item() / args.log_interval
            cur_elbo_loss = total_elbo.item() / args.log_interval
            cur_reward = (total_reward).item() / args.log_interval
            cur_prior_loss_per_token = total_prior_loss_per_token / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:05.5f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | cur_loss_conditional_per_token {:8.2f} | bpc {:8.3f} | cur_loss_conditional {:8.3f} |'
                  ' cur_loss_kl {:8.3f} | cur_elbo_loss {:8.3f} | cur_reward {:8.3f} | '
                  ' cur_prior_loss_per_token {:8.3f}'.format(
                epoch, batch, len(train_data), optimizer.param_groups[0]['lr'],
                elapsed * 1000 / args.log_interval, cur_loss, cur_loss_conditional_per_token,
                cur_loss / math.log(2), cur_loss_conditional, cur_loss_kl, cur_elbo_loss, cur_reward,
                cur_prior_loss_per_token ))
            print('| epoch {:3d} |  run_name={} | anneal func {} | kl weight {:8.3f} | temperature {:8.3f} '
                  .format(epoch, args.run_name , args.anneal_function, kl_anneal_function(num_train_steps),
                          step_temperature) )
            total_loss = 0
            total_loss_kl = 0
            total_loss_conditional = 0
            total_loss_conditional_per_token = 0
            total_elbo = 0
            total_reward = 0
            total_prior_loss_per_token = 0
            start_time = time.time()

        batch += 1

        if args.debug_mode:
            break


# Loop over epochs.
lr = args.lr
best_val_loss = []
stored_loss = 100000000

# At any point you can hit Ctrl + C to break out of training early.
try:
    optimizer = None
    # Ensure the optimizer is optimizing params, which includes both the model's weights as well as the criterion's weight (i.e. Adaptive Softmax)
    optimizer_changed = False
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params, lr=args.lr, weight_decay=args.wdecay)
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.wdecay)
    # verify params here again
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        if args.train_inference_nw_only and epoch == 1:
            val_loss2 = evaluate(val_data)
            print('-' * 89)
            print('| BEGINNING of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                  'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
                epoch, (time.time() - epoch_start_time), val_loss2, math.exp(val_loss2), val_loss2 / math.log(2)))
            print('-' * 89)
            sys.stdout.flush()
        train()
        if 't0' in optimizer.param_groups[0]:
            tmp = {}
            for prm in model.parameters():
                tmp[prm] = prm.data.clone()
                if not optimizer_changed: #####**new
                    prm.data = optimizer.state[prm]['ax'].clone()

            val_loss2 = evaluate(val_data)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
                    epoch, (time.time() - epoch_start_time), val_loss2, math.exp(val_loss2), val_loss2 / math.log(2)))
            print('-' * 89)
            sys.stdout.flush()

            if val_loss2 < stored_loss:
                model_save(args.save)
                print('Saving Averaged!')
                stored_loss = val_loss2

            for prm in model.parameters():
                prm.data = tmp[prm].clone()

        else:
            val_loss = evaluate(val_data) #, eval_batch_size)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
              epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss), val_loss / math.log(2)))
            print('-' * 89)

            if val_loss < stored_loss:
                model_save(args.save)
                print('Saving model (new best validation)')
                stored_loss = val_loss

            if args.optimizer == 'sgd' and 't0' not in optimizer.param_groups[0] and (len(best_val_loss)>args.nonmono and val_loss > min(best_val_loss[:-args.nonmono])):
                print('Switching to ASGD')
                optimizer_changed = True
                optimizer = torch.optim.ASGD(model.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)

            if epoch in args.when:
                print('Saving model before learning rate decreased')
                model_save('{}.e{}'.format(args.save, epoch))
                print('Dividing learning rate by 10')
                optimizer.param_groups[0]['lr'] /= 10.

            best_val_loss.append(val_loss)

        if epoch in [30,35,40,45]:
            model_save('{}.epoch{}'.format(args.save, epoch))
        if not (args.pretrain_ae_mode or args.sanity_check_title ):
            args.outf = 'tmp/outputs/'+args.run_name+'.epoch'+str(epoch)+'.plotsamples'
            args.sents = 21
            generate_plot_samples(model, test_data, args, corpus)
        args.outf = 'tmp/outputs/' + args.run_name + '.epoch' + str(epoch) + '.storysamples'
        args.sents = 21
        args.words = 100
        generate_story_samples(model, test_data, args, corpus)
        # args.outf = 'tmp/outputs/' + args.run_name + '.epoch' + str(epoch) + '.storysamplesfromposteriorplotsamples'
        # args.sents = 21
        # args.words = 100
        # generate_story_samples(model, test_data, args, corpus, use_posterior_samples=True)

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
model_load(args.save)

# Run on test data.
test_loss = evaluate(test_data) #, test_batch_size)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f} | test bpc {:8.3f}'.format(
    test_loss, math.exp(test_loss), test_loss / math.log(2)))
print('=' * 89)
