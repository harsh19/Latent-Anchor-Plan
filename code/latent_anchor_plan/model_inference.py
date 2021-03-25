import torch
import torch.nn as nn

from embed_regularize import embedded_dropout
from locked_dropout import LockedDropout
from weight_drop import WeightDrop
import numpy as np
from constants import stopwords
import utils
import math
#emotion_states

ignore_type_tokens_lists = utils.ignore_type_tokens_lists
tanh = nn.Tanh()


class InferenceNW(nn.Module):
    def __init__(self, typ, ntoken, ninp,
                 encoder=None,
                 prior_numstates=None,
                 use_cuda=False,
                 arch_type='linear1',
                 skip_first_token=False,
                 ignore_token_type='none', #'default', # default ignores only pad,
                 dictionary=None,
                 nw_frozen=False,
                 use_dual_contextualizer=False,
                 emotion_vocab_list=None,
                 uniform_distribution=False,
                 first_word_distribution=False,
                 emotion_type='basic',
                 new_mask_defn=False
                 ):
        super(InferenceNW, self).__init__()
        self.typ = typ
        self.arch_type = arch_type
        self.skip_first_token = skip_first_token
        self.ignore_token_type = ignore_token_type
        self.use_cuda = use_cuda
        self.nw_frozen = nw_frozen
        self.uniform_distribution = uniform_distribution
        self.first_word_distribution = first_word_distribution
        self.ninp = ninp
        self.use_dual_contextualizer = use_dual_contextualizer
        self.emotion_type = emotion_type
        self.dictionary = dictionary
        if uniform_distribution:
            assert False
            # nw_frozen
            # assert not first_word_distribution
        if first_word_distribution:
            assert False
            #assert not uniform_distribution
        ignore_tokens = [ dictionary.word2idx[token] for token in ignore_type_tokens_lists[ignore_token_type]
                          if token in dictionary.word2idx ]
        self.pad_id = dictionary.word2idx['<pad>']
        print("*** self.pad_id = ", self.pad_id)
        self.ignore_tokens = ignore_tokens
        self.new_mask_defn = new_mask_defn
        print("self.ignore_tokens = ", self.ignore_tokens)
        assert typ in ['kw'] #,'states','emotion']
        if typ=='kw':
            assert arch_type in ['linear1','linear1tanh','linear1no','linear2','linear2no','linear2tanh','lstm','bilstm']
            if encoder is not None:
                print("InferenceNW : encoder is shared ")
                self.encoder = encoder
            else:
                print("InferenceNW : encoder is NOT shared ")
                self.encoder = nn.Embedding(ntoken, ninp)
            if arch_type.count('lstm')>0:
                if arch_type == 'lstm':
                    self.contextualizer = nn.LSTM(ninp, ninp)
                    self.nh = ninp
                    if self.use_dual_contextualizer:
                        self.scorer = nn.Linear(2 * self.nh, 1)
                    else:
                        self.scorer = nn.Linear(self.nh, 1)
                elif arch_type == 'bilstm':
                    self.contextualizer = nn.LSTM(ninp, ninp, bidirectional=True) # ** corrected
                    self.nh = 2*ninp
                    if self.use_dual_contextualizer:
                        self.scorer = nn.Linear(3 * ninp, 1)
                    else:
                        self.scorer = nn.Linear(2*ninp, 1)
                    print("****  self.scorer = ",  self.scorer)
            self.softmax = nn.Softmax(dim=1)

        else:
            raise NotImplementedError

        print("InferenceNW : nw_frozen = ", nw_frozen)


    def init_weights(self, initrange, pretrained_model_load_path=None, pretrained_model_path_extractinference=False):
        if pretrained_model_load_path is None:
            if self.arch_type.count('lstm') > 0:
                self.scorer.bias.data.fill_(0)
                self.scorer.weight.data.uniform_(-initrange, initrange)
                # todo - init lstm
            else:
                self.scorer[0].bias.data.fill_(0)
                self.scorer[0].weight.data.uniform_(-initrange, initrange)
                if self.arch_type.count('linear2')>0:
                    self.scorer[1].bias.data.fill_(0)
                    self.scorer[1].weight.data.uniform_(-initrange, initrange)
        else:
            if pretrained_model_path_extractinference:
                print("===>>>> LOADING Inference NW from ", pretrained_model_load_path)
                mdl, _, _ = torch.load(pretrained_model_load_path)
                state_dict = mdl.inference_nw.state_dict()
                print("===>>>> LOADING Inference NW: pretrained_model_load_path.keys() =",
                      state_dict.keys())
                self.load_state_dict(state_dict)
            else:
                print("===>>>> LOADING Inference NW from ", pretrained_model_load_path)
                state_dict = torch.load(pretrained_model_load_path)
                print("===>>>> LOADING Inference NW: pretrained_model_load_path.keys() =",
                      state_dict.keys())
                print("****  self.scorer = ", self.scorer)
                self.load_state_dict(state_dict)
                # raise NotImplementedError

    def get_l2_loss(self):
        if self.typ == 'states':
            raise NotImplementedError
        elif self.typ == 'emotion':
            l2 = torch.mean(torch.pow(self.scorer[0].weight,2))
            return l2
        else:
            l2 = torch.mean(torch.pow(self.scorer[0].weight,2))
            return l2

    def forward(self, sentences, return_vals=False, print_more=False, temperature=1.0):
        bs,ln = sentences.size() # bs=b*5
        data = sentences # bs,ln,emsize
        if self.arch_type.count('lstm')>0:
            ct = 1
            nh=self.nh
            if self.arch_type.count('bilstm') > 0:
                ct = 2
                nh = self.nh // 2
            hidden = (torch.zeros(ct, bs, nh ),
                      torch.zeros(ct, bs, nh))
            if self.use_cuda:
                hidden = ( hidden[0].cuda(), hidden[1].cuda() )
            token_embs = self.encoder(data.t())  # ln,bs,emsize
            embs, _ = self.contextualizer(token_embs, hidden) # ln,bs,h
            embs = embs.permute(1, 0, 2) # bs,ln,h
            if self.use_dual_contextualizer:
                embs = torch.cat([embs,token_embs.permute(1,0,2)],dim=2) # ln,bs,2h
            if self.new_mask_defn:
                assert False
                pass #embs = tanh(embs)
        else:
            embs = self.encoder(data)  # bs,ln,emsize
        scores = self.scorer(embs) # bs,ln,1
        mask = torch.ones(scores.size())
        if self.use_cuda:
            mask = mask.cuda()
        for ignore_token in self.ignore_tokens:
            mask[sentences == ignore_token] = 0
        assert len(scores.size()) == 3
        scores = scores.view(bs, ln)
        if self.uniform_distribution:
            assert False
        else:
            if self.nw_frozen:
                scores = scores.detach()
            if self.new_mask_defn:
                scores = 2*(1.0 + tanh(scores))
            distrib = self.softmax(scores/temperature)
        if return_vals:
            return distrib, scores, {}
        return distrib


class PriorModel(nn.Module):
    def __init__(self,
                 typ,
                 ntoken,
                 ninp,
                 nhid,
                 tie_weights,
                 prior_numstates,
                 decoder=None,
                 cuda=False,
                 emotion_vocab_list=None,
                 use_fixed_uniform_prior=False
                 ):
        super(PriorModel, self).__init__()
        self.typ = typ
        self.use_cuda = cuda
        if self.typ == 'kw':
            assert decoder is not None # sharing decoder
            self.decoder_prior = decoder
        else:
            raise NotImplementedError

        if use_fixed_uniform_prior:
            raise NotImplementedError

    def init_weights(self, initrange):
        self.decoder_prior.bias.data.fill_(0)
        self.decoder_prior.weight.data.uniform_(-initrange, initrange)

    def forward(self, backbone_output):
        if self.typ == 'kw':
            ret = self.decoder_prior(backbone_output)
        else:
            raise NotImplementedError
        return ret


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, dropouth=0.5, dropouti=0.5, dropoute=0.1, wdrop=0, tie_weights=False, prior_numstates=5, use_fixed_uniform_prior=False, cuda=False, dictionary=None, latent_plot_typ='kw', infer_nw_arch_type='linear1', inference_pretrained_model_path=None, infer_nw_skip_first_token=False,
  infer_nw_ignore_token_type = 'default', infer_nw_share_encoder=True, inference_nw_frozen=False, inference_nw_uniform_distribution=False, emotion_type='basic', inference_pretrained_model_path_extractinference=False, inference_nw_first_word_distribution=False):

        super(RNNModel, self).__init__()
        self.lockdrop = LockedDropout()
        self.idrop = nn.Dropout(dropouti)
        self.hdrop = nn.Dropout(dropouth)
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.use_cuda = cuda
        assert rnn_type in ['LSTM', 'QRNN', 'GRU'], 'RNN type is not supported'
        if rnn_type == 'LSTM':
            self.rnns = [torch.nn.LSTM(ninp if l == 0 else nhid, nhid if l != nlayers - 1 \
                    else (ninp if tie_weights else nhid), 1, dropout=0) for l in range(nlayers)]
            if wdrop:
                self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop) for rnn in self.rnns]
        if rnn_type == 'GRU':
            self.rnns = [torch.nn.GRU(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else ninp, 1, dropout=0)
                         for l in range(nlayers)]
            if wdrop:
                self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop) for rnn in self.rnns]
        elif rnn_type == 'QRNN':
            from torchqrnn import QRNNLayer
            self.rnns = [QRNNLayer(input_size=ninp if l == 0 else nhid, hidden_size=nhid if l != nlayers - 1 \
                else (ninp if tie_weights else nhid), save_prev_x=True, zoneout=0, window=2 \
                if l == 0 else 1, output_gate=True) for l in range(nlayers)]
            for rnn in self.rnns:
                rnn.linear = WeightDrop(rnn.linear, ['weight'], dropout=wdrop)
        print(self.rnns)
        self.rnns = torch.nn.ModuleList(self.rnns)
        self.decoder = nn.Linear(nhid, ntoken)
        self.inference_pretrained_model_path_extractinference = inference_pretrained_model_path_extractinference
        # self.decoder_nw_frozen = decoder_nw_frozen

        #################### WEIGHT TIEING
        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            #if nhid != ninp:
            #    raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight
            # self.decoder_prior.weight = self.encoder.weight
            print("******* tied weights ******")
        print("**self.decoder = ", self.decoder)


        #################### PRIOR
        self.latent_plot_typ = latent_plot_typ
        provided_prior_decoder = None
        emotion_vocab_list = None
        self.emotion_type = emotion_type
        if latent_plot_typ == 'kw':
            provided_prior_decoder = self.decoder
        elif latent_plot_typ == 'emotion':
            self.emotion_vocab_list  = emotion_vocab_list = self.get_emotion_vocab_list(dictionary)
        self.prior_model = PriorModel(
                                typ=latent_plot_typ,
                                ntoken=ntoken,
                                ninp=ninp,
                                nhid=nhid,
                                tie_weights=tie_weights,
                                prior_numstates=None,
                                decoder=provided_prior_decoder,
                                cuda=False,
                                emotion_vocab_list=emotion_vocab_list,
                                use_fixed_uniform_prior = use_fixed_uniform_prior
                            )

        #################### INFERENCE NETWORK
        self.infer_nw_share_encoder = infer_nw_share_encoder
        self.inference_nw_uniform_distribution = inference_nw_uniform_distribution
        self.inference_nw_first_word_distribution = inference_nw_first_word_distribution
        if infer_nw_share_encoder:
            infer_nw_encoder = self.encoder
        else:
            infer_nw_encoder = None
        self.inference_nw = InferenceNW(typ=latent_plot_typ,
                                        ninp=ninp,
                                        ntoken=ntoken,
                                        encoder=infer_nw_encoder,
                                        prior_numstates=prior_numstates,
                                        use_cuda=cuda,
                                        arch_type=infer_nw_arch_type,
                                        skip_first_token=infer_nw_skip_first_token,
                                        dictionary=dictionary,
                                        ignore_token_type=infer_nw_ignore_token_type,
                                        nw_frozen=inference_nw_frozen,
                                        emotion_vocab_list=emotion_vocab_list,
                                        uniform_distribution=inference_nw_uniform_distribution,
                                        first_word_distribution=inference_nw_first_word_distribution,
                                        emotion_type=emotion_type)
        self.inference_pretrained_model_path = inference_pretrained_model_path # will call in init_weights
        self.inference_nw_frozen = inference_nw_frozen


        #################### INIT AND SAVE HYPERPARAMS
        self.init_weights()
        self.rnn_type = rnn_type
        self.ninp = ninp
        self.nhid = nhid
        self.nlayers = nlayers
        self.dropout = dropout
        self.dropouti = dropouti
        self.dropouth = dropouth
        self.dropoute = dropoute
        self.tie_weights = tie_weights
        self.wdrop = wdrop
        self.use_fixed_uniform_prior = use_fixed_uniform_prior


    def get_emotion_vocab_list(self, dictionary):
        return utils.get_emotion_vocab_list(dictionary=dictionary,
                                                emotion_type=self.emotion_type,
                                                use_cuda=self.use_cuda)

    def reset(self):
        if self.rnn_type == 'QRNN': [r.reset() for r in self.rnns]

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)
        self.inference_nw.init_weights(initrange,
                                       self.inference_pretrained_model_path,
                                       self.inference_pretrained_model_path_extractinference)
        self.prior_model.init_weights(initrange)

    def get_l2_loss(self):
        ret = self.inference_nw.get_l2_loss()
        return ret

    def forward(self, input, hidden, return_h=False):
        emb = embedded_dropout(self.encoder, input, dropout=self.dropoute if self.training else 0)
        emb = self.lockdrop(emb, self.dropouti)
        # for rnn in self.rnns:
        #     if self.wdrop:
        #         tmp = rnn.module
        #     else:
        #         tmp = rnn
        #     tmp.flatten_parameters()
        raw_output = emb
        new_hidden = []
        raw_outputs = []
        outputs = []
        for l, rnn in enumerate(self.rnns):
            current_input = raw_output
            raw_output, new_h = rnn(raw_output, hidden[l])
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            if l != self.nlayers - 1:
                #self.hdrop(raw_output)
                raw_output = self.lockdrop(raw_output, self.dropouth)
                outputs.append(raw_output)
        hidden = new_hidden
        output = self.lockdrop(raw_output, self.dropout)
        outputs.append(output)
        result = output.view(output.size(0)*output.size(1), output.size(2))
        if return_h:
            return result, hidden, raw_outputs, outputs
        return result, hidden


    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return [(weight.new(1, bsz, self.nhid if l != self.nlayers - 1
                else (self.ninp if self.tie_weights else self.nhid)).zero_(),
                weight.new(1, bsz, self.nhid if l != self.nlayers - 1
                else (self.ninp if self.tie_weights else self.nhid)).zero_())
                    for l in range(self.nlayers)]
        elif self.rnn_type == 'QRNN' or self.rnn_type == 'GRU':
            return [weight.new(1, bsz, self.nhid if l != self.nlayers - 1
                else (self.ninp if self.tie_weights else self.nhid)).zero_()
                    for l in range(self.nlayers)]


    def get_sample_from_posterior(self,
                                  sentences_data,
                                  print_more=False,
                                  temperature=1.0,
                                  use_argmax=False,
                                  completion_mode=False): # posterior_dist_logits: B,5,K
        # sentences_data: bs * 5 * ln
        # Bnew=bs*5
        bs, nums, ln = sentences_data.size()
        assert nums == 5 or completion_mode
        data = sentences_data.view(bs * nums, ln)
        probs, scores, _ = self.inference_nw(data, return_vals=True,
                                             print_more=print_more,
                                             temperature=temperature)  # bs*5,ln
        # distrib = torch.distributions.Categorical(logits=word_weights)
        if print_more:
            print(" [get_sample_from_posterior] probs = ", probs)
        distrib = torch.distributions.Categorical(probs=probs)
        if use_argmax:
            _, position_indices = probs.max(dim=1)  # Bnew
        else:
            # print("********")
            position_indices = distrib.sample()  # Bnew
        log_probs = distrib.log_prob(position_indices)  # Bnew
        if self.latent_plot_typ == 'kw':
            vocab_indices = data.gather(dim=1, index=position_indices.unsqueeze(1)).squeeze(1)  # Bnew
            vocab_indices = vocab_indices.view(bs, nums)  # bs,5
        elif self.latent_plot_typ == 'emotion':
            raise NotImplementedError
        else:
            raise NotImplementedError
        return vocab_indices, {'logprobs': log_probs.view(bs, nums), 'distrib': distrib, 'scores': scores}
        # log_probs will be used to update posterior via rewards


    def get_sample_from_posterior_ae(self,
                                  sentences_data,
                                  print_more=False,
                                  temperature=1.0,
                                  use_argmax=False): # posterior_dist_logits: B,5,K
        # sentences_data: bs * 5 * ln
        # Bnew=bs*5
        # print("sentences_data : ", sentences_data.size())
        bs,ln = sentences_data.size()
        data = sentences_data
        probs, scores , _ = self.inference_nw(data, return_vals=True,
                                         print_more=print_more,
                                         temperature=temperature) # bs*5,ln
        # distrib = torch.distributions.Categorical(logits=word_weights)
        if print_more:
            print(" [get_sample_from_posterior] probs = ", probs)
        distrib = torch.distributions.Categorical(probs=probs)
        if use_argmax:
            raise NotImplementedError
        else:
            position_indices = distrib.sample()  # Bnew
        log_probs = distrib.log_prob(position_indices) # Bnew
        if self.latent_plot_typ == 'kw':
            vocab_indices = data.gather(dim=1,index=position_indices.unsqueeze(1) ).squeeze(1) # Bnew
            vocab_indices = vocab_indices.view(bs) # bs,5
        # elif self.latent_plot_typ == 'emotion':
        else:
            raise NotImplementedError
        return vocab_indices, {'logprobs':log_probs.view(bs), 'distrib':distrib, 'scores':scores}
        # log_probs will be used to update posterior via rewards

    def get_posterior_dist(self,
                          sentences_data,
                          print_more=False,
                          temperature=1.0): # posterior_dist_logits: B,5,K
        # sentences_data: bs * 5 * ln
        # Bnew=bs*5
        bs,nums,ln = sentences_data.size()
        assert nums == 5
        data = sentences_data.view(bs*nums,ln)
        probs, _ , _ = self.inference_nw(data, return_vals=True,
                                         print_more=print_more,
                                         temperature=temperature) # bs*5,ln
        return probs

    def get_entropy_regularizer(self, distrib):  # compute by sampling
        return distrib.entropy()

    def compute_kl_loss(self, sentences_data, prior_context, num_samples=1, beta_prior=False ): # compute by sampling

        kl = 0.0
        use_new_kl_defn = True

        bs,nums,ln = sentences_data.size()
        data = sentences_data.view(bs * nums, ln)
        probs, _, _ = self.inference_nw(data, return_vals=True)  # bs*5,ln
        distrib = torch.distributions.Categorical(probs=probs)
        probs_batchwise = probs.view(bs,nums,-1) # q(z): bs,nums,ln

        if beta_prior:
            assert num_samples == 1
            rnnhs = []

        for k in range(num_samples):

            output, hidden = prior_context
            # output = output.view(6,20,-1) -- output: seqlen, bs, rnnsize
            position_indices = distrib.sample()  # Bnew
            log_probs = distrib.log_prob(position_indices)  # Bnew
            log_probs = log_probs.view(bs, nums)  # bs,5

            # kw
            if self.latent_plot_typ == 'kw':
                vocab_indices = data.gather(dim=1, index=position_indices.unsqueeze(1)).squeeze(1)  # Bnew
                vocab_indices = vocab_indices.view(bs, nums)  # bs,5
            else:
                raise NotImplementedError

            cur = 0.0

            for i in range(5):
                output_pre = output.view(-1,bs,output.size()[1])
                output_pre = output_pre[-1,:,:] # select last output state
                word_weights = self.prior_model(output_pre) # bs,vocab
                prior_dist = torch.distributions.Categorical(logits=word_weights)
                if use_new_kl_defn:
                    # sentences_data: bs,nums,ln
                    sentences_datai = sentences_data[:,i,:] # bs,ln
                    logprob_prior = prior_dist.log_prob(sentences_datai.t()).t() # log p(z|x): bs,ln
                    logprob_prior[torch.isnan(logprob_prior)] = math.log(0.0000000000001)
                    # probs_batchwise: posterior probs of size: bs,5,ln
                    probs_batchwisei = probs_batchwise[:, i, :] # q(zi): bs,ln
                    j=-1
                    kli = []
                    for probs_batchwiseij in probs_batchwisei:
                        # probs_batchwiseij: q(zi^{jth_data_point}): ln
                        j+=1
                        idxj = probs_batchwiseij>0
                        assert torch.sum(idxj)>0
                        probs_batchwiseij = probs_batchwiseij[idxj] # select q[zi^{(j)}]>0
                        logprob_posteriorij = torch.log(probs_batchwiseij) # log q(zi^{j}) : ln
                        logprob_priorj_idx = logprob_prior[j][idxj] # log p(z): bs,ln; select jth datapoint: log p(z^{j}): ln
                        # then select using idxj those for which q[zi^{(j)}]>0
                        # next, since there are only ln terms, we can exactly compute KL(q_zi||p(z_i|z_<i))
                        logprob_priorj_idx = torch.max(logprob_priorj_idx,
                                                          torch.zeros_like(logprob_priorj_idx)+math.log(0.00000001)  )
                        klij = probs_batchwiseij * (logprob_posteriorij - logprob_priorj_idx) # ln
                        kli.append(klij.sum().unsqueeze(0)) # 1,1
                        # \sum_z q(zi)logq(zi)/p(zi)
                    kli = torch.cat(kli, dim=0).squeeze(0) # bs
                    cur += kli # bs
                else:
                    raise NotImplementedError

                word_idx = vocab_indices[:,i]
                input = word_idx.unsqueeze(0)
                if beta_prior:
                    output, hidden, rnnhsi, _ = self.forward(input, hidden, return_h=True)
                    rnnhs.append(rnnhsi[-1]) #
                else:
                    output, hidden = self.forward(input, hidden)

            kl += ( (1.0/num_samples)* cur)
            # \sum_z q(z) log q(z)|p(z)
            # \sum_{z1,z2,z3,z4,z5}   q(z) log q(z)|p(z)
            # E_{z1~q(z1)} E_{z2~q(z2)} ..   [ log q(z1)/p(z1) + log q(z2)/p(z2|z1). + .. ]
            # 1st term =  E_{z1~q(z1)} E_{z2~q(z2)} .. [ log q(z1)/p(z1) ] = E_{z1~q(z1)} [ log q(z1)/p(z1) ] = KL(q(z_1)||p(z1))
            # and so on
            # \sum_{z1,z2,z3,z4,z5}   q(z1) log q(z1)/p(z1)  + q(z2) log q(z2)/p(z2|z1) + ...
            # \sum_z1 KL(q(z1)||p(z1)) + z1~sampled,\sum_z2 KL()

        if beta_prior:
            assert len(rnnhs) == 5
            rnnhs = torch.cat(rnnhs, dim=0) # 5, 20, 1000
            return kl, rnnhs
        else:
            return kl


    def compute_kl_loss_ae(self, sentences_data, prior_context, num_samples=1 ): # compute by sampling
        raise NotImplementedError
