import scipy.stats
import numpy as np
import json

def reorder_line(line):
    words = line.strip().split()
    if '<eos>' in words:
        idx = words.index('<eos>')
    else:
        idx = len(words)
    left = [w for w in reversed(words[:idx])]
    right = words[idx + 1:]
    return ' '.join(left + right)


def read_file(fname, typ, one_per_title=False):
    assert typ in ['plot','plot_nonmonodecoder','titleonly']
    reorder = False
    if typ == 'plot_nonmonodecoder':
        reorder = True
    print("Reading from ", fname)
    f= open(fname,'r').readlines()
    ret = []
    titles_dct = {}
    err_cnt = 0
    for j,line in enumerate(f):
        cur = {}
        notes = line.strip().split()
        if len(notes) == 0:
            continue
        try:
            idx = notes.index('<EOT>')
        except:
            continue
        title = notes[:idx]
        cur['title'] = ' '.join(title)
        if one_per_title and cur['title'] in titles_dct:
            continue
        titles_dct[cur['title']] = 1
        notes = notes[idx+1:] # remove title and
        if typ in ['plot','plot_nonmonodecoder']:
            #assert line.count('EOL') > 0
            if line.count('EOL') == 0:
                err_cnt+=1
                notes = notes[:5] + ['<EOL>'] + notes[5:] # assuming 5 kws
            idx = notes.index('<EOL>')
            plot = ' '.join(notes[:idx])
            notes = notes[idx+1:] # remove title and
            cur['plot'] = plot
        else:
            assert line.count('EOL') == 0, "line = " + line
        cur['story'] = ' '.join(notes)
        cur['storylines'] = [linej.strip() for linej in ' '.join(notes).strip().split('</s>')[1:] ]
        if reorder:
            cur['storylines'] = list(map(reorder_line, cur['storylines']))
            cur['story'] = '</s> ' + ' </s> '.join(cur['storylines'])
        if len(cur['storylines']) != 5:
            print("**** story lines != 5 ***********")
        ret.append(cur)
        if j==0:
            print("j = 0 : ", json.dumps(cur))
    if typ in ['plot', 'plot_nonmonodecoder']:
        print("***** plot mode was used. But EOL not found in err_cnt = ",err_cnt, " instances")
    print("#instances =", len(ret))
    return ret


def diversity_entropy(data, use_plots=False):
    all_notes_cnt = {}
    all_binotes_cnt = {}
    all_trinotes_cnt = {}

    unigram_repetition_cnt = 0
    total_token_cnt = 0

    for row in data:

        if use_plots:
            story = row['plot'].strip().split()
            # print("story = ", story)
        else:
            story = ' '.join(row['storylines']).strip().split()
        # todo - remove punctuations and </s> symbols
        notes = story
        if use_plots:
            unigram_repetition_cnt += (len(notes) - len(set(notes)))
        total_token_cnt += len(notes)

        for j, e in enumerate(story):
            all_notes_cnt[e] = all_notes_cnt.get(e, 0) + 1
            if j > 0:
                all_binotes_cnt[' '.join(notes[j - 1:j + 1])] = all_binotes_cnt.get(' '.join(notes[j - 1:j + 1]), 0) + 1
            if j > 1:
                all_trinotes_cnt[' '.join(notes[j - 2:j + 1])] = all_trinotes_cnt.get(' '.join(notes[j - 2:j + 1]),
                                                                                      0) + 1

    ret = {}
    all_entropy = []
    print("-------unigrams")
    print(len(all_notes_cnt))
    print(sorted(all_notes_cnt.items(), key=lambda x: -x[1])[:11])
    all_vals = np.array(list(all_notes_cnt.values()))
    all_probs = all_vals / all_vals.sum()
    print("entropy = ", scipy.stats.entropy(all_probs))
    all_entropy.append(scipy.stats.entropy(all_probs))
    ret['uni_entropy'] = scipy.stats.entropy(all_probs)
    ret['uni_uniqcnt'] = len(all_notes_cnt)
    print("-------bigrams")
    print(len(all_binotes_cnt))
    print(sorted(all_binotes_cnt.items(), key=lambda x: -x[1])[:11])
    all_vals = np.array(list(all_binotes_cnt.values()))
    all_probs = all_vals / all_vals.sum()
    all_entropy.append(scipy.stats.entropy(all_probs))
    print("entropy = ", scipy.stats.entropy(all_probs))
    ret['bi_entropy'] = scipy.stats.entropy(all_probs)
    ret['bi_uniqcnt'] = len(all_binotes_cnt)
    print("-------trigrams")
    print(len(all_trinotes_cnt))
    print(sorted(all_trinotes_cnt.items(), key=lambda x: -x[1])[:11])
    all_vals = np.array(list(all_trinotes_cnt.values()))
    all_probs = all_vals / all_vals.sum()
    print("entropy = ", scipy.stats.entropy(all_probs))
    all_entropy.append(scipy.stats.entropy(all_probs))
    ret['tri_entropy'] = scipy.stats.entropy(all_probs)
    ret['tri_uniqcnt'] = len(all_trinotes_cnt)
    print("-------gm")
    print("GM = ", scipy.stats.mstats.gmean(np.array(all_entropy)))
    ret['geom_mean_allentropy'] = scipy.stats.mstats.gmean(np.array(all_entropy))
    if use_plots:
        ret['unigram_repetition_cnt'] = unigram_repetition_cnt
    ret['total_token_cnt'] = total_token_cnt
    ret['data_cnt'] = len(data)
    return ret


def diversity_inter(data, num_instances):
    from pycocotools.coco import COCO
    from pycocoevalcap.eval import COCOEvalCap
    preds_lst = [' '.join(row['storylines']) for row in data]
    preds_lst = [{'image_id': i, 'caption': prediction, 'id': i} for i, prediction in enumerate(preds_lst)]
    sz = len(preds_lst)
    gt_lst = []
    for i in range(sz):
        for j in range(sz):
            if j != i:
                gt_lst.append({'image_id': i, 'caption': preds_lst[j]['caption'], 'id': i})
    preds_lst = preds_lst[:num_instances]
    # diversity_inter(all_data[mname][:1000], 100) # compute for 100 istances only. with 999 references.
    print("[diversity_inter len(gt_lst = ", len(gt_lst))
    print("[diversity_inter] gt_lst[0] = ", gt_lst[0])
    images_list = [i for i in range(len(preds_lst))]
    gt_lst_cocoformat = {'annotations': gt_lst, 'images': [{'id': i} for i in images_list],
                         'type': 'captions', 'info': '', 'licenses': ''}
    coco = COCO(dataset=gt_lst_cocoformat)  # gt_file
    cocoRes = coco.loadRes(anns=preds_lst)  # pred_file)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()
    print("[diversity_inter]: ", cocoEval.eval.items())
    ret = {}
    for metric, score in cocoEval.eval.items():
        ret[metric] = score
    return ret


if __name__ == '__main__':
    import sys
    fname = sys.argv[1]
    typ = sys.argv[2]
    assert typ in ['plot','plot_nonmonodecoder','titleonly']
    # use_plots = {'plot':True, 'story':False}[sys.argv[3]]
    data = read_file(fname, typ, one_per_title=True)
    diversity_evals_d = diversity_entropy(data)
    if typ == 'plot':
        diversity_evals_plots = diversity_entropy(data, use_plots=True)
    print("="*33)
    print("="*33)
    print("--->>> diversity_evals: generate_story_samples: stories---> ", diversity_evals_d)
    if typ == 'plot':
        print("--->>> diversity_evals: diversity_evals_plots: plots---> ", diversity_evals_plots)
    print("-"*33)
    diversity_interstory = diversity_inter(data[:1000], 1000)
    # diversity_interstory = diversity_inter(data[:1000], 200)
    print("--->>> diversity_evals - interstory bleu ---> ", diversity_interstory)
    print("="*33)
    print("="*33)


