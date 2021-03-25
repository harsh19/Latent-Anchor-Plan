#!/usr/bin/env bash

TMPDIR='tmp/'

### Training

# Pretraining
ARCH=bilstm
PRESAVED_VOCAB_FILE=tmp/vocabs/vocab.pkl
MODEL_NAME=pretraininfnw
echo $MODEL_NAME
CUDA_VISIBLE_DEVICES=0 python latent_anchor_plan/pretraining.py  --batch_size 20 --dropouti 0.4  --dropouth 0.25 --seed 141  --save $TMPDIR/models/$MODEL_NAME.pt --vocab-file $PRESAVED_VOCAB_FILE --applyDict true --run_name $MODEL_NAME --cuda --debug_mode false | tee logs/$MODEL_NAME.train.log


# 2 Phase model training:
PRESAVED_VOCAB_FILE=$TMPDIR/vocabs/vocab.pkl
inference_pretrained_model_path=$TMPDIR/models/pretraininfnw.pt
#inference_pretrained_model_path=$TMPDIR/models/model_newexps_singleaepretraining_arch_bilstm_ignr_non_content_set2_sharedencfalse_exp2.pt
MODEL_NAME=lap
echo $MODEL_NAME
CUDA_VISIBLE_DEVICES=0 python latent_anchor_plan/main.py  --batch_size 20 --dropouti 0.4  --dropouth 0.25 --seed 141  --emsize 1000 --nhid 1000 --save $TMPDIR/models/$MODEL_NAME.pt --vocab-file $PRESAVED_VOCAB_FILE --applyDict true  --inference_nw_frozen true --run_name $MODEL_NAME --debug_mode false --cuda --inference_pretrained_model_path $inference_pretrained_model_path --inference_pretrained_model_path_extractinference true  | tee logs/$MODEL_NAME.train.log
# Fine-tuning - unlcok and train all
PRESAVED_VOCAB_FILE=tmp/vocabs/vocab.pkl
load_pretrained_model_path=tmp/models/lap.pt
#
MODEL_NAME=lapfinal2
echo $MODEL_NAME
CUDA_VISIBLE_DEVICES=3 python latent_anchor_plan/main_finetune.py  --batch_size 20 --dropouti 0.4  --dropouth 0.25 --seed 141  --emsize 1000 --nhid 1000 --save tmp/models/$MODEL_NAME.pt --vocab-file $PRESAVED_VOCAB_FILE --applyDict true --run_name $MODEL_NAME --debug_mode false --cuda --load_pretrained_model_path $load_pretrained_model_path  --prior_regularize true --prior_reg_loss_fact 100.0   --init_eval true --finetune true  --epochs 4  | tee logs/$MODEL_NAME.train.log



######### Evals
PRESAVED_VOCAB_FILE=tmp/vocabs/vocab.pkl
MODEL_NAME=lap #model_pretrained #*****
MODEL_NAME=lapfinal
MODEL_NAME=lapfina2
PATH_TO_PRETRAINED_MODEL=tmp/models/$MODEL_NAME.pt


### ELBO
ROC_DATA=rocstory_plan_write/ROCStories_all_merge_tokenize.titlesepkeysepstory.test
CUDA_VISIBLE_DEVICES=0 python latent_anchor_plan/eval_inference.py   --checkpoint $PATH_TO_PRETRAINED_MODEL  --vocab $PRESAVED_VOCAB_FILE --task test_eval --test-data  $ROC_DATA --cuda  | tee "$TMPDIR"/evals/$MODEL_NAME.test_eval.test.log

ROC_DATA=rocstory_plan_write/ROCStories_all_merge_tokenize.titlesepkeysepstory.dev
CUDA_VISIBLE_DEVICES=0 python latent_anchor_plan/eval_inference.py   --checkpoint $PATH_TO_PRETRAINED_MODEL  --vocab $PRESAVED_VOCAB_FILE --task test_eval  --test-data  $ROC_DATA --cuda | tee "$TMPDIR"/evals/$MODEL_NAME.test_eval.dev.log


### PPL using IW-NLL
ROC_DATA=rocstory_plan_write/ROCStories_all_merge_tokenize.titlesepkeysepstory.test
CUDA_VISIBLE_DEVICES=0 python latent_anchor_plan/eval_inference.py   --checkpoint $PATH_TO_PRETRAINED_MODEL  --vocab $PRESAVED_VOCAB_FILE --task evaluate_iwnll  --test-data  $ROC_DATA --cuda | tee "$TMPDIR"/evals/$MODEL_NAME.evaluate_iwnll.test.log

ROC_DATA=rocstory_plan_write/ROCStories_all_merge_tokenize.titlesepkeysepstory.dev
CUDA_VISIBLE_DEVICES=0 python latent_anchor_plan/eval_inference.py   --checkpoint $PATH_TO_PRETRAINED_MODEL  --vocab $PRESAVED_VOCAB_FILE --task evaluate_iwnll  --test-data  $ROC_DATA --cuda | tee "$TMPDIR"/evals/$MODEL_NAME.evaluate_iwnll.dev.log


#Samples
ROC_DATA=rocstory_plan_write/ROCStories_all_merge_tokenize.titlesepkeysepstory.test
OUTF="$TMPDIR"/samples/$MODEL_NAME.storysamples.out
NUM_DATA_POINTS=9816
CUDA_VISIBLE_DEVICES=0 python latent_anchor_plan/eval_inference.py   --checkpoint $PATH_TO_PRETRAINED_MODEL  --vocab $PRESAVED_VOCAB_FILE --task story_samples  --test-data  $ROC_DATA --cuda --outf $OUTF --sents $NUM_DATA_POINTS --print-cond-data --temperature 1.0 --top_p 0.6


### Diversity and Control Evals
OUTF="$TMPDIR"/samples/$MODEL_NAME.storysamples.out
python diversity_evals.py $OUTF plot plot > "$TMPDIR"/evals/$MODEL_NAME.diversity_analysis
python control_evals.py $OUTF plot > "$TMPDIR"/evals/$MODEL_NAME.control_analysis


# Plan POS analysis
ROC_DATA=rocstory_plan_write/ROCStories_all_merge_tokenize.titlesepkeysepstory.test
OUTF="$TMPDIR"/evals/$MODEL_NAME.encoderanalysis.out
NUM_DATA_POINTS=9816
CUDA_VISIBLE_DEVICES=0 python latent_anchor_plan/eval_inference.py   --checkpoint $PATH_TO_PRETRAINED_MODEL  --vocab $PRESAVED_VOCAB_FILE --task encoder_analysis  --test-data  $ROC_DATA --cuda --sents $NUM_DATA_POINTS --print-cond-data --inference_use_argmax true --outf $OUTF
#
python inferencenw_pos_analysis.py $OUTF model > "$TMPDIR"/evals/$MODEL_NAME.encoderanalysis



