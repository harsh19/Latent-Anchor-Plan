import json
import nltk
import diversity_evals
from collections import Counter

def load_file(fname = 'language-model/tmp/completion_outputs/series4_traindecoderprior_arch_bilstm_ignr_non_content_set2_sharedencfalse_singleaepretrained.encoder_analysis.test.out'):
    ret = []
    for line in open(fname,'r').readlines():
        t = json.loads(line.strip())
        ret.append(t)
    return ret

def process_story(storylines):
    storylines = [line.replace('</s>','').strip() for line in storylines]
    return storylines

def analyze_pos(data):
    # ctr = 0
    cnt  = {}
    total = 0
    not_found_cnt = 0
    positions = []
    for row in data:
        for sent,kw in zip(row['sentences_postags'],row['keywords']):
            found = False
            idx = 0
            for w,tag in sent:
                if w == kw:
                    cnt[tag] = cnt.get(tag,0) + 1
                    total += 1
                    found = True
                    positions.append(idx)
                    break
                idx+=1
            if not found:
                #print(kw,row['keywords'],row['sentences_postags'])
                not_found_cnt+=1
    # print(total)
    return cnt,total,not_found_cnt,Counter(positions)

def analyze(data):
    print("len(data) = ", len(data))
    print("data[0] = ", data[0])
    processed_data = [{'sentences': process_story(row['sentences']),
                       'keywords': row['keywords']}
                      for row in data
                      ]
    for row in processed_data:
        row['sentences_tokens'] = [ nltk.word_tokenize(sentence) for sentence in row['sentences'] ]
        row['sentences_postags'] = [ nltk.pos_tag(sentence) for sentence in row['sentences_tokens'] ]
    print("processed_data[0] = ", processed_data[0])
    postag_cnts, total_cnt, not_found_cnt, position_distr = analyze_pos(processed_data)
    print(sorted(list(postag_cnts.items()), key=lambda x: -x[1]))
    vals = sorted(list(postag_cnts.items()), key=lambda x: -x[1])
    vals_norm = [[val[0], val[1] / total_cnt] for val in vals]
    cntvb = (postag_cnts.get('VBD',0) + postag_cnts.get('VB',0) + postag_cnts.get('VBG',0) + postag_cnts.get('VBN',0) + postag_cnts.get('VBZ',0) +
     postag_cnts['VBP']) / total_cnt
    cntnn = (postag_cnts['NN'] + postag_cnts['NNS'] + postag_cnts.get('NNP',0))/total_cnt
    cntnns = (postag_cnts['NN'] + postag_cnts['NNS'])/total_cnt
    cntnnp = (postag_cnts.get('NNP',0))/total_cnt
    cntjj = ( postag_cnts.get('JJ',0)+postag_cnts.get('JJR',0)+postag_cnts.get('JJS',0))/total_cnt
    return {
        'cntvb':cntvb,
        'cntnn': cntnn,
        'cntjj': cntjj,
        'cntnnp': cntnnp,
        'cntnns': cntnns,
        'not_found_cnt':not_found_cnt,
        'position_distr':position_distr,
        'total_cnt':total_cnt
    }


if __name__ == '__main__':
    import sys
    fname = sys.argv[1]
    mode = sys.argv[2]
    assert mode in ['data','model']
    if mode == 'data':
        import diversity_evals
        data = diversity_evals.read_file(fname, 'plot', one_per_title=False) #, reorder=False)
        print("---->>> data: ", len(data))
        for row in data:
            row['sentences'] = row['storylines']
            row['keywords'] = row['plot'].strip().split()
        # print("data[0] = ", data[0])
    else:
        data = load_file(fname)
        # if mode == 'model_nonmonotonic':
        #     for row in data:
        #         row['sentences'] = [diversity_evals.reorder_line(line) for line in row['sentences']]
        print("---->>> data: ", len(data))
        print("data[0] = ", data[0])
    print(analyze(data))





