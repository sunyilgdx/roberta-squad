# roberta-squad
roBERTa training for SQuAD 


Observations:
1. Decayed learning rates on finetuning seems to make it more robust (?)



## Experiment 1
### Run on SQuAD 2.0 Dev Set

```c
lr_rate_decay=1.0        
TOTAL_NUM_UPDATES=5430   # Number of training steps.
WARMUP_UPDATES=326       # Linearly increase LR over this many steps.
LR=1.5e-05               # Peak LR for fixed LR scheduler.
MAX_SENTENCES=3          # Batch size per GPU.
UPDATE_FREQ=2            # Accumulate gradients to simulate training on 8 GPUs.
DATA_DIR=qa_records_squad_q
ROBERTA_PATH=/home/paramihk/roberta.large/model.pt

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3.5 ./fairseq_train.py $DATA_DIR \
    --restore-file $ROBERTA_PATH \
    --reset-optimizer --reset-dataloader --reset-meters \
    --no-epoch-checkpoints --no-last-checkpoints --no-save-optimizer-state \
    --task squad2 \
    --max-positions 512 \
    --arch roberta_qa_large \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
    --criterion squad2 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-06 \
    --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $LR \
    --fp16 --fp16-init-scale 4 --threshold-loss-scale 1 --fp16-scale-window 128 --memory-efficient-fp16 \
    --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_NUM_UPDATES \
    --max-sentences $MAX_SENTENCES \
    --required-batch-size-multiple 1 \
    --update-freq $UPDATE_FREQ \
    --max-update $TOTAL_NUM_UPDATES \
    --lr_rate_decay $lr_rate_decay \
    --ddp-backend=no_c10d \
    --num-workers=0
```

```json
{
  "exact": 83.4329992419776,
  "f1": 86.7448817152165,
  "total": 11873,
  "HasAns_exact": 82.86099865047234,
  "HasAns_f1": 89.49426123562206,
  "HasAns_total": 5928,
  "NoAns_exact": 84.00336417157276,
  "NoAns_f1": 84.00336417157276,
  "NoAns_total": 5945,
  "best_exact": 85.21014065526826,
  "best_exact_thresh": -1.6142578125,
  "best_f1": 88.297090749954,
  "best_f1_thresh": -1.572265625
}
```





## Experiment 2
### Run on SQuAD 2.0 Dev Set
```c

lr_rate_decay=0.75
TOTAL_NUM_UPDATES=8144 # Number of training steps.
WARMUP_UPDATES=489     # Linearly increase LR over this many steps.
LR=1.5e-05               # Peak LR for fixed LR scheduler.
MAX_SENTENCES=4        # Batch size per GPU.
UPDATE_FREQ=1          # Accumulate gradients to simulate training on 8 GPUs.
DATA_DIR=qa_records_squad_q
ROBERTA_PATH=/home/paramihk/roberta.large/model.pt

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3.5 ./fairseq_train.py $DATA_DIR \
    --restore-file $ROBERTA_PATH \
    --save-dir checkpoints_q_decay_0.75 \
    --reset-optimizer --reset-dataloader --reset-meters \
    --no-epoch-checkpoints --no-last-checkpoints --no-save-optimizer-state \
    --task squad2 \
    --max-positions 512 \
    --arch roberta_qa_large \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
    --criterion squad2 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-06 \
    --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $LR \
    --fp16 --fp16-init-scale 4 --threshold-loss-scale 1 --fp16-scale-window 128 --memory-efficient-fp16 \
    --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_NUM_UPDATES \
    --max-sentences $MAX_SENTENCES \
    --required-batch-size-multiple 1 \
    --update-freq $UPDATE_FREQ \
    --max-update $TOTAL_NUM_UPDATES \
    --lr_rate_decay $lr_rate_decay \
    --ddp-backend=no_c10d \
    --num-workers=0
 
```

```json
{
  "exact": 84.03941716499621,
  "f1": 87.29093171231531,
  "total": 11873,
  "HasAns_exact": 83.02968960863697,
  "HasAns_f1": 89.54204322205142,
  "HasAns_total": 5928,
  "NoAns_exact": 85.04625735912532,
  "NoAns_f1": 85.04625735912532,
  "NoAns_total": 5945,
  "best_exact": 85.5217720879306,
  "best_exact_thresh": -1.921875,
  "best_f1": 88.56228638211618,
  "best_f1_thresh": -1.765625
}  
```









## Experiment 3
### Run on SQuAD 2.0 Dev Set
```c

lr_rate_decay=0.75
TOTAL_NUM_UPDATES=8144 # Number of training steps.
WARMUP_UPDATES=489     # Linearly increase LR over this many steps.
LR=2e-05               # Peak LR for fixed LR scheduler.
MAX_SENTENCES=4        # Batch size per GPU.
UPDATE_FREQ=1          # Accumulate gradients to simulate training on 8 GPUs.
DATA_DIR=qa_records_squad_q
ROBERTA_PATH=/home/paramihk/roberta.large/model.pt

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3.5 ./fairseq_train.py $DATA_DIR \
    --restore-file $ROBERTA_PATH \
    --save-dir checkpoints_q_decay_0.75 \
    --reset-optimizer --reset-dataloader --reset-meters \
    --no-epoch-checkpoints --no-last-checkpoints --no-save-optimizer-state \
    --task squad2 \
    --max-positions 512 \
    --arch roberta_qa_large \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
    --criterion squad2 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-06 \
    --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $LR \
    --fp16 --fp16-init-scale 4 --threshold-loss-scale 1 --fp16-scale-window 128 --memory-efficient-fp16 \
    --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_NUM_UPDATES \
    --max-sentences $MAX_SENTENCES \
    --required-batch-size-multiple 1 \
    --update-freq $UPDATE_FREQ \
    --max-update $TOTAL_NUM_UPDATES \
    --lr_rate_decay $lr_rate_decay \
    --ddp-backend=no_c10d \
    --num-workers=0
 
```

```json
{
  "exact": 84.89850922260591,
  "f1": 88.04700949753051,
  "total": 11873,
  "HasAns_exact": 83.62010796221323,
  "HasAns_f1": 89.92613761204144,
  "HasAns_total": 5928,
  "NoAns_exact": 86.17325483599663,
  "NoAns_f1": 86.17325483599663,
  "NoAns_total": 5945,
  "best_exact": 86.07765518403099,
  "best_exact_thresh": -1.859375,
  "best_f1": 88.99848000856761,
  "best_f1_thresh": -1.611328125
}
```






## Experiment 4
### Run on SQuAD 2.0 Dev Set
```c
lr_rate_decay=0.75
TOTAL_NUM_UPDATES=8144 # Number of training steps.
WARMUP_UPDATES=489     # Linearly increase LR over this many steps.
LR=2.5e-05               # Peak LR for fixed LR scheduler.
MAX_SENTENCES=4        # Batch size per GPU.
UPDATE_FREQ=1          # Accumulate gradients to simulate training on 8 GPUs.
DATA_DIR=qa_records_squad_q
ROBERTA_PATH=/home/paramihk/roberta.large/model.pt

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3.5 ./fairseq_train.py $DATA_DIR \
    --restore-file $ROBERTA_PATH \
    --save-dir checkpoints_q_decay_0.75 \
    --reset-optimizer --reset-dataloader --reset-meters \
    --no-epoch-checkpoints --no-last-checkpoints --no-save-optimizer-state \
    --task squad2 \
    --max-positions 512 \
    --arch roberta_qa_large \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
    --criterion squad2 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-06 \
    --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $LR \
    --fp16 --fp16-init-scale 4 --threshold-loss-scale 1 --fp16-scale-window 128 --memory-efficient-fp16 \
    --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_NUM_UPDATES \
    --max-sentences $MAX_SENTENCES \
    --required-batch-size-multiple 1 \
    --update-freq $UPDATE_FREQ \
    --max-update $TOTAL_NUM_UPDATES \
    --lr_rate_decay $lr_rate_decay \
    --ddp-backend=no_c10d \
    --num-workers=0
 
```

```json
{
  "exact": 85.42070243409417,
  "f1": 88.5973743793479,
  "total": 11873,
  "HasAns_exact": 83.83940620782727,
  "HasAns_f1": 90.20185998751667,
  "HasAns_total": 5928,
  "NoAns_exact": 86.99747687132044,
  "NoAns_f1": 86.99747687132044,
  "NoAns_total": 5945,
  "best_exact": 86.41455403015244,
  "best_exact_thresh": -1.5517578125,
  "best_f1": 89.47730538540738,
  "best_f1_thresh": -1.328125
}
```




