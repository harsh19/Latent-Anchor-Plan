from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
ps = PorterStemmer()
# # choose some words to be stemmed
# words = ["program", "programs", "programer", "programing", "programers"]
# for w in words:
#     print(w, " : ", ps.stem(w))


def control_usage(data, use_stemming=False, log_mismatches_fname=None):
    cntp = 0
    cntn = 0
    if log_mismatches_fname:
        fw_log_mismatches_fname = open(log_mismatches_fname,'w')
    for row in data:
        plot = row['plot'].strip().split()
        story = row['storylines']
        for p ,sent in zip(plot ,story):
            story_words = [w.strip() for w in sent.lower().strip().split()]
            if use_stemming:
                p = ps.stem(p.strip().lower())
                story_words = [ps.stem(w) for w in story_words]
            if p.lower().strip() in story_words:
                cntp += 1
            else:
                cntn += 1
                if log_mismatches_fname:
                    fw_log_mismatches_fname.write('\t'.join([p,sent,row['title'],row['plot'],row['storylines']]))
    print("cntp = ", cntp, (cntp +cntn), cntp /(cntp +cntn))
    print("cntn = ", cntn, (cntp +cntn), cntn /(cntp +cntn))
    if log_mismatches_fname:
        fw_log_mismatches_fname.close()
    return cntp, cntn



def control_usage_anywhere(data, use_stemming=False):
    cntp = 0
    cntn = 0
    for row in data:
        plot = row['plot'].strip().split()
        story = row['storylines']
        story_words_all = []
        p_all = []
        for p ,sent in zip(plot ,story):
            story_words = [w.strip() for w in sent.lower().strip().split()]
            if use_stemming:
                p = ps.stem(p.strip().lower())
                story_words = [ps.stem(w) for w in story_words]
            story_words_all.extend(story_words)
            p_all.append(p.lower().strip())
        for p in p_all:
            if p in story_words_all:
                cntp += 1
            else:
                cntn += 1
    print("cntp = ", cntp, (cntp +cntn), cntp /(cntp +cntn))
    print("cntn = ", cntn, (cntp +cntn), cntn /(cntp +cntn))
    return cntp, cntn



if __name__ == '__main__':
    import sys
    fname = sys.argv[1]
    typ = sys.argv[2]
    assert typ in ['plot','plot_nonmonodecoder']
    import diversity_evals
    data = diversity_evals.read_file(fname, typ, one_per_title=True)
    ctrl_evals = control_usage(data)
    ctrl_evals_stem = control_usage(data, use_stemming=True) #, log_mismatches_fname=fname+'.controlevals.mismatches.txt')
    ctrl_evals_any = control_usage_anywhere(data)
    print("="*33)
    print("="*33)
    print("--->>> ctrl_evals:  = ", ctrl_evals)
    print("--->>> ctrl_evals_stem:  = ", ctrl_evals_stem)
    print("--->>> ctrl_evals_any:  = ", ctrl_evals_any)
    print("="*33)
    print("="*33)

'''
the pet bug <EOT> cat dog food eat run <EOL> </s> cat is a cat . </s> dog is an animal . </s> they eat food </s> they feed </s> and run .
the pet bugs <EOT> cat dog food eat run <EOL> </s> cat is a cat . </s> dog is an animal . </s> they take food </s> they eat </s> and run .
the pets <EOT> cat dog food eat run <EOL> </s> cat is a cat . </s> dog is an animal . </s> they take food </s> they eat </s> and walk .
'''
'''
Expected o/p:
- 2 exact match misses. 1st line 4th sent and 3rd line last sentence. so (13,2)
- for 'any' mode, word 'food' is present elsewhere in first line. so (14,1)
Observed o/p:
--->>> ctrl_evals:  =  (13, 2)
--->>> ctrl_evals_any:  =  (14, 1)
'''