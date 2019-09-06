from torch import nn
import argparse
from fairseq.data import Dictionary
from fairseq.models.roberta import RobertaModel, RobertaEncoder, roberta_large_architecture
from fairseq.optim.fp16_optimizer import MemoryEfficientFP16Optimizer
import torch
from tokenizer.roberta import RobertaTokenizer, MASKED, NOT_MASKED, IS_MAX_CONTEXT, NOT_IS_MAX_CONTEXT
from glob import glob
import numpy as np
from torch.nn import functional as F
from torch.nn import CrossEntropyLoss
from ranger import Ranger
from ranger import Adam
import json
from tokenizer.validate import validate
from copy import deepcopy
from time import time
from multiprocessing import Pool
import multiprocessing
import gc
import random
from tqdm import tqdm
import os

roberta_directory = './roberta.large'


max_seq_length   = 512
max_query_length = 192
doc_stride       = 192

default_choices = ['Yes','No']
get_tokenizer = lambda: RobertaTokenizer(config_dir=roberta_directory)

tk = tokenizer =  get_tokenizer()


#Data Utilities


def init():
    global tokenizer, tk
    import gc
    tokenizer = tk = get_tokenizer()
    

def data_from_path(train_dir):
    index = 0
    for fn in glob(train_dir):
        with open(fn, "r") as f:
            entries = [e for e in json.load(f)["data"] for e in e['paragraphs']]


        print("%-40s : %s contexts"%(fn.split('/')[-1],len(entries)))
        for e in entries:
            c = e['context']
            yield index, c, e['qas']
            index += 1

def char_anchors_to_tok_pos(r):
    if len(r.char_anchors) == 2:
        a,b = r.char_anchors
    else:
        return -1,-1
    a = r.char_to_tok_offset[a]
    b = r.char_to_tok_offset[b]
    return a, b

def read(dat):
    uid, inp, start, end, p_mask, unanswerable = marshal.loads(dat)
    inp = np.frombuffer(inp, dtype=np.uint16).astype(np.int32)
    p_mask = np.frombuffer(p_mask, dtype=np.bool).astype(np.float32)
    return uid, inp, start, end, p_mask, unanswerable

def fread(f):
    uid, inp, start, end, p_mask, unanswerable = marshal.load(f)
    inp = np.frombuffer(inp, dtype=np.uint16).astype(np.int32)
    p_mask = np.frombuffer(p_mask, dtype=np.bool).astype(np.float32)
    return uid, inp, start, end, p_mask, unanswerable
            
def gen(paths):

    for i,context,qas in data_from_path(paths):
        for q in qas:
            if len(q['question']) < 5 or ('choices' in q and ''.join(q['choices']) == ''):
                continue
            if '\1' in q['question']:
                q['question'] = q['question'].replace('\1', '___')
        yield i,context, qas
        
        
        
def ids_equal_no_ans(inp,start,end):
  return inp[start_position] == 440 and  inp[end_position] == 1948
        
import marshal
def work(ss, debug=False):
    global unique_index, \
     context, \
     qas, \
     is_training, \
     return_feature, rss, start_position, end_position
    
    unique_index, \
     context, \
     qas, \
     is_training, \
     return_feature = ss
    
    rss = tokenizer.merge_cq(context.replace('<eop> ','\n'), 
                             qas,
                             max_seq_length = max_seq_length,
                             max_query_length = max_query_length,
                             doc_stride = doc_stride,
                             default_choices = default_choices,
                             unique_index=unique_index,
                             is_training=is_training,
                             debug = debug
                           )
    o = 0
    results = []
    for rs in rss:
        q = qas[o]
        o += 1
        for r in rs:
            inp = tk.convert_tokens_to_ids(r.all_doc_tokens)
            start_position,end_position = char_anchors_to_tok_pos(r)
            p_mask = r.p_mask
            uid = r.unique_index[0]*1000 + r.unique_index[1]
            if start_position == -1 and end_position == -1:
                start_position = 0
                end_position = 0
                
            
            no_ans = start_position == 0
            
            #if no_ans:
            #    print(q['answer_text'], '>>', r.all_doc_tokens[start_position:end_position+1])
            assert start_position >= 0 and end_position >= 0 and start_position < len(inp) and end_position < len(inp)
            assert len(inp) <= max_seq_length
            record = marshal.dumps(
                (
                uid,
                np.array(inp,dtype=np.uint16).tobytes(),
                start_position,
                end_position,
                np.array(p_mask,dtype=np.bool).tobytes(),
                int(no_ans)
                )
            )
            
            if return_feature:
                results.append((record, no_ans,r))
            else:
                results.append((record, no_ans))


    
    return results




def generate_tfrecord(data_dir,
                      write_fn=None, 
                      is_training=False,
                      return_feature=False,
                      parallel_process=False,
                      debug=False):
    global count

    if return_feature:
        rs = []

    i = 0
    
    if parallel_process:
        cpu_count = multiprocessing.cpu_count()
    
        pool = Pool(cpu_count-1,initializer=init)
        
    tokenizer = get_tokenizer()
        
    tot_num_no_ans = 0
    
    
        
    records = []
    
        
    num_no_ans = 0
    i += 1

    jobs = ((i, c, q, is_training, return_feature) for i, c, q in gen(data_dir))
    t0 = time()
    results = pool.imap_unordered(work,jobs) if parallel_process else tqdm(iter(work(e, debug=debug) for e in jobs))
    c = 0
    for e in results:
        for record in e:
            if return_feature:
                record, no_ans, r = record
                rs.append(r)
            else:
                record, no_ans = record


            records.append(record)

            if no_ans:
                num_no_ans += 1
            c += 1
            if c % 2500 == 0:
                t1 = time()
                uid, inp, start, end, p_mask, unanswerable = read(record)
                # print(uid, tk.convert_ids_to_tokens(inp) , start, end, p_mask)
                print('%d features (%d no ans) extracted (time: %.2f s)'%(c, num_no_ans, t1-t0))

    if not return_feature:
        random.shuffle(records)
        with open(write_fn, 'wb') as f:
            for record in records:
                f.write(record)
                f.write(b'\n')
    tot_num_no_ans = num_no_ans

    print('num has ans / num no ans : %d / %d'%(c - tot_num_no_ans, tot_num_no_ans))
    
    
    if return_feature:
        return records, rs
    


def chunks(l, n):
    if type(l) == type((e for e in range(1))):
        it = iter(l)
        while True:
            out = []
            try:
                for _ in range(n):
                    out.append(next(it))
            except StopIteration:
                yield out
                break

            yield out
    else:
    
        for i in range(0, len(l), n):
            yield l[i:i + n]

def pad(list_of_tokens, 
        dtype=np.long,
        torch_tensor=None,
        pad_idx=1):
    k = np.empty((len(list_of_tokens),max_seq_length), dtype=dtype)
    k.fill(pad_idx)
    i = 0
    for tokens in list_of_tokens:
        k[i,:len(tokens)] = tokens
        i += 1
    return k if torch_tensor is None else torch_tensor(k)

  
def from_records(records, batch_size = 48, half=False, shuffle=True):
    if half:
      float = torch.HalfTensor
    else:
      float = torch.FloatTensor
  
  
    fn_style = isinstance(records,str)
    if fn_style:
      def from_file(fn):
        with open(fn, 'rb') as f:
            while True:
                try:
                    record = fread(f)
                    yield record
                except EOFError:
                    break
      records = from_file(records)

    if shuffle:
      records = list(records)
      random.shuffle(records)
    for record_samples in chunks(records,batch_size):
        uid, inp, start, end, p_mask, unanswerable = zip(*record_samples) if fn_style else zip(*(read(record) for record in record_samples))
        start = torch.LongTensor(start)
        end = torch.LongTensor(end)
        unanswerable = float(unanswerable)
        inp = pad(inp,dtype=np.long, torch_tensor=torch.LongTensor)
        p_mask = pad(p_mask,dtype=np.float32, torch_tensor=float)

        yield inp, p_mask, start, end, unanswerable


# Train Utilities


import math
from functools import wraps
import warnings


class fairseq_LRScheduler(object):
    def __init__(self, optimizer, last_epoch=-1):
        self.optimizer = optimizer
        if last_epoch == -1:
            for group in optimizer.wrapped_optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
            last_epoch = 0
        else:
            for i, group in enumerate(optimizer.wrapped_optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.wrapped_optimizer.param_groups))
        self.last_epoch = last_epoch

        # Following https://github.com/pytorch/pytorch/issues/20124
        # We would like to ensure that `lr_scheduler.step()` is called after
        # `optimizer.step()`
        def with_counter(func, opt):
            @wraps(func)
            def wrapper(*args, **kwargs):
                opt._step_count += 1
                return func(*args, **kwargs)
            wrapper._with_counter = True
            return wrapper

        self.optimizer.step = with_counter(self.optimizer.step, self.optimizer)
        self.optimizer._step_count = 0
        self._step_count = 0
        self.step(last_epoch)

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_lr(self):
        raise NotImplementedError

    def step(self, epoch=None):
        # Raise a warning if old pattern is detected
        # https://github.com/pytorch/pytorch/issues/20124
        if self._step_count == 1:
            if not hasattr(self.optimizer.step, "_with_counter"):
                warnings.warn("Seems like `optimizer.step()` has been overridden after learning rate scheduler "
                              "initialization. Please, make sure to call `optimizer.step()` before "
                              "`lr_scheduler.step()`. See more details at "
                              "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate", UserWarning)

            # Just check if there were two first lr_scheduler.step() calls before optimizer.step()
            elif self.optimizer._step_count < 1:
                warnings.warn("Detected call of `lr_scheduler.step()` before `optimizer.step()`. "
                              "In PyTorch 1.1.0 and later, you should call them in the opposite order: "
                              "`optimizer.step()` before `lr_scheduler.step()`.  Failure to do this "
                              "will result in PyTorch skipping the first value of the learning rate schedule."
                              "See more details at "
                              "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate", UserWarning)
        self._step_count += 1

        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.wrapped_optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
            
            
            
class DelayedCosineAnnealingLR(fairseq_LRScheduler):

    def __init__(self, optimizer, T_max, delayed_steps=0, eta_min=0, last_epoch=-1):
        self.T_max = T_max - delayed_steps
        self.eta_min = eta_min
        self.delayed_steps = delayed_steps
        super(DelayedCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self.eta_min + (base_lr - self.eta_min) *
                (1 + math.cos(math.pi * (self.last_epoch - self.delayed_steps) / self.T_max)) / 2
                for base_lr in self.base_lrs] if self.last_epoch > self.delayed_steps else \
               [base_lr for base_lr in self.base_lrs]

            
class LinearAnnealingLRWithWarmUp(fairseq_LRScheduler):

    def __init__(self, optimizer, T_max, warmup_steps=0, last_epoch=-1):
        self.T_max = T_max - warmup_steps
        self.warmup_steps = warmup_steps
        super(LinearAnnealingLRWithWarmUp, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * (1 - (self.last_epoch - self.warmup_steps) / self.T_max)
                for base_lr in self.base_lrs] if self.last_epoch > self.warmup_steps else \
               [base_lr * (self.last_epoch/self.warmup_steps) for base_lr in self.base_lrs]

# Model Utilities

MAX_FLOAT16 = 2**15
MIN_FLOAT16 = 6e-5
MAX_FLOAT32 = 1e30
MIN_FLOAT32 = 1e-12
max_float = MAX_FLOAT16
min_float = MIN_FLOAT16
max_float = MAX_FLOAT32
min_float = MIN_FLOAT32

class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        #inlining this saves 1 second per epoch (V100 GPU) vs having a temp x and then returning x(!)
        return x *( torch.tanh(F.softplus(x)))


class PoolerAnswerClass(nn.Module):
    """ Compute SQuAD 2.0 answer class from classification and start tokens hidden states. """
    def __init__(self, hidden_size):
        super(PoolerAnswerClass, self).__init__()
        self.dense_0 = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh() #Mish() # nn.Tanh()
        #self.dropout = nn.Dropout(p=dropout)
        self.dense_1 = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden_states, cls_index=None):
        """
        Args:
            One of ``start_states``, ``start_positions`` should be not None.
            If both are set, ``start_positions`` overrides ``start_states``.
            **cls_index**: torch.LongTensor of shape ``(batch_size,)``
                position of the CLS token. If None, take the last token.

            note(Original repo):
                no dependency on end_feature so that we can obtain one single `cls_logits`
                for each sample
        """

        if cls_index is not None:
            cls_index = cls_index[:, None, None].expand(-1, -1, hsz) # shape (bsz, 1, hsz)
            cls_token_state = hidden_states.gather(-2, cls_index).squeeze(-2) # shape (bsz, hsz)
        else:
            cls_token_state = hidden_states[:, 0, :] # shape (bsz, hsz)


        x = self.dense_0(cls_token_state)
        x = self.activation(x)
        x = self.dense_1(x).squeeze(-1)


        return x

class RobertaQA(torch.nn.Module):
    def __init__(self, 
                 roberta_path='roberta.large',
                 checkpoint_file='model.pt',
                 start_n_top = 5,
                 end_n_top = 5,
                 use_ans_class = False,
                 strict = False):
        super(RobertaQA, self).__init__()
        
        
        state = torch.load(os.path.join(roberta_path, checkpoint_file))
        
        args = state['args']
        roberta_large_architecture(args)

        if not hasattr(args, 'max_positions'):
            args.max_positions = args.tokens_per_sample

        self.dictionary = dictionary = Dictionary.load(os.path.join(roberta_path, 'dict.txt'))
        dictionary.add_symbol('<mask>')

        model = RobertaModel(args, RobertaEncoder(args, dictionary))
        self.args = args
        
        self.roberta = model
        
        hs = args.encoder_embed_dim
        self.span_logits =  nn.Linear(hs, 2)
        self.answer_class = PoolerAnswerClass(hs)
        
        self.start_n_top = start_n_top
        self.end_n_top = end_n_top
        self.use_ans_class = use_ans_class
        
        print('loading from checkpoint...')
        print('use_ans_class:', use_ans_class)
        print('roberta_path:', roberta_path)
        print('checkpoint_file:', checkpoint_file)
        print('strict:', strict)
        self.load_state_dict(state['model'], strict=strict)

    def extract_features(self, tokens: torch.LongTensor, return_all_hiddens: bool = False) -> torch.Tensor:
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        if tokens.size(-1) > self.roberta.max_positions():
            raise ValueError('tokens exceeds maximum length: {} > {}'.format(
                tokens.size(-1), self.roberta.max_positions()
            ))
        features, extra = self.roberta(
            tokens,
            features_only=True,
            return_all_hiddens=return_all_hiddens,
        )
        if return_all_hiddens:
            # convert from T x B x C -> B x T x C
            inner_states = extra['inner_states']
            return [inner_state.transpose(0, 1) for inner_state in inner_states]
        else:
            return features  # just the last layer's features

    def forward(self, x, start_positions=None, end_positions=None, unanswerable=None, cls_index=None): 
        use_ans_class = self.use_ans_class
        hidden_states = self.extract_features(x)  # [bs, seq_len, hs]
        
        start_logits, end_logits = self.span_logits(hidden_states).split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        if use_ans_class:
            # Predict answerability from the representation of CLS and START
            cls_logits = self.answer_class(hidden_states, cls_index=cls_index)
            

        outputs = (start_logits, end_logits, cls_logits)  # Keep mems, hidden states, attentions if there are in it

            
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, let's remove the dimension added by batch splitting
            for x in (start_positions, end_positions, unanswerable):
                if x is not None and x.dim() > 1:
                    x.squeeze_(-1)

            loss_fct = CrossEntropyLoss()
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2


            loss_fct_cls = nn.BCEWithLogitsLoss()
            cls_loss = loss_fct_cls(cls_logits, unanswerable)
            
            total_loss += cls_loss * 0.5

            outputs = (total_loss,)

        return outputs


class FnnLayer(nn.Module):
    """ Compute SQuAD start_logits from sequence hidden states. """
    def __init__(self, hidden_size, dropout=0.2):
        super(FnnLayer, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = Mish()
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, hidden_states):

        return self.dropout(self.activation(self.dense(hidden_states).squeeze(-1))) + hidden_states

class RobertaQAEmbed(torch.nn.Module):
    def __init__(self, 
                 roberta_path='roberta.large',
                 checkpoint_file='model.pt',
                 strict = False):
        super(RobertaQA, self).__init__()
        
        
        state = torch.load(os.path.join(roberta_path, checkpoint_file))
        
        args = state['args']
        roberta_large_architecture(args)

        if not hasattr(args, 'max_positions'):
            args.max_positions = args.tokens_per_sample

        self.dictionary = dictionary = Dictionary.load(os.path.join(roberta_path, 'dict.txt'))
        dictionary.add_symbol('<mask>')

        model = RobertaModel(args, RobertaEncoder(args, dictionary))
        self.args = args
        
        self.roberta = model
        
        hs = args.encoder_embed_dim
        self.log_softmax = torch.nn.LogSoftmax(dim=1)
        self.q_fnn_layer = FnnLayer(hs)
        self.a_fnn_layer = FnnLayer(hs)
        
        print('loading from checkpoint...')
        self.load_state_dict(state['model'], strict=strict)

    def extract_features(self, tokens: torch.LongTensor, return_all_hiddens: bool = False) -> torch.Tensor:
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        if tokens.size(-1) > self.roberta.max_positions():
            raise ValueError('tokens exceeds maximum length: {} > {}'.format(
                tokens.size(-1), self.roberta.max_positions()
            ))
        features, extra = self.roberta(
            tokens,
            features_only=True,
            return_all_hiddens=return_all_hiddens,
        )
        if return_all_hiddens:
            # convert from T x B x C -> B x T x C
            inner_states = extra['inner_states']
            return [inner_state.transpose(0, 1) for inner_state in inner_states]
        else:
            return features  # just the last layer's features

    def forward(self, q=None, a=None, normalize=False, return_loss=False): 
        
        if q and a:
          assert q.shape[0] == a.shape[0]
          q_hs, a_hs = self.extract_features(torch.cat([q,a],dim=0)).mean(1).split(q.size(0))
        elif q:
          q_hs = self.extract_features(q).mean(1)  # [bs, hs]
        elif a:
          a_hs = self.extract_features(a).mean(1)  # [bs, hs]
        if q:
          q_embed = self.q_fnn_layer(q_hs)
        if a:
          a_embed = self.a_fnn_layer(q_hs)

        outputs = () 

        if return_loss:
            if not (q and a):
              raise Exception('Cannot calculate loss without both q and a')
            q_embed_norm = q_embed / q_embed.norm(dim=1)[:,None]
            a_embed_norm = a_embed / a_embed.norm(dim=1)[:,None]
            loss = -(torch.eye(q_hs.shape[0]) * log_softmax(torch.mm(q_embed_norm,a_embed_norm.t()) )).sum()
            outputs = (total_loss,)

        else:
            if q:
              if normalize:
                q_embed = q_embed / q_embed.norm(dim=1)[:,None]
              outputs = outputs + (q_embed,)
              
            if a:
              if normalize:
                a_embed = a_embed / a_embed.norm(dim=1)[:,None]
              outputs = outputs + (a_embed,)
              
        # return start_top_log_probs, start_top_index, end_top_log_probs, end_top_index, cls_logits
        # or (if labels are provided) (total_loss,)
        return outputs


def get_decayed_param_groups(roberta, num_layers, lr=3e-5, lr_rate_decay=0.908517):
  lr_factors = []
  prefix = 'module.roberta.decoder.sentence_encoder.layers.'

  for k, v in roberta.named_parameters():
      factor = 1
      if 'sentence_encoder.layers' in k:
          layer = int(k[len(prefix):].split('.')[0])
          factor = lr_rate_decay**(num_layers-layer)

      elif 'embed_tokens.weight' in k or 'embed_positions' in k:
          layer = 0
          factor = lr_rate_decay**(num_layers-layer)

      lr_factors.append({
          'params': v,
          'lr': lr * factor,
      })
  return lr_factors
      
# Eval Utilities

import collections
_PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
    "PrelimPrediction",
    ["feature_index", "start_index", "end_index",
    "start_log_prob", "end_log_prob", "this_paragraph_text",
    "cur_null_score"])
_NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
    "NbestPrediction", ["text", "start_log_prob", "end_log_prob","cur_null_score"])

import math
def _compute_softmax(scores):
  """Compute softmax probability over raw logits."""
  if not scores:
    return []

  max_score = None
  for score in scores:
    if max_score is None or score > max_score:
      max_score = score

  exp_scores = []
  total_sum = 0.0
  for score in scores:
    x = math.exp(score - max_score)
    exp_scores.append(x)
    total_sum += x

  probs = []
  for score in exp_scores:
    probs.append(score / total_sum)
  return probs


# Model Init


from time import time

roberta_single = RobertaQA(roberta_path=roberta_directory, checkpoint_file='roberta_qa_squad_24.pt', strict=True)



log_steps = 500
num_epochs = 2
max_seq_length = 512
num_cores = torch.cuda.device_count() # 8
effective_batch_size = 64             # 8  bs per device
update_freq = 1                       # 4  bs per device
fp16 = True
class args:
  update_freq=update_freq
  fp16_scale_window=128
  distributed_world_size=1
  fp16_init_scale=4
  fp16_scale_tolerance=0
  threshold_loss_scale=1
  min_loss_scale=1e-4
  
  

use_gpu = None

assert effective_batch_size % update_freq == 0

batch_size = effective_batch_size // update_freq



if num_cores > 1:
  roberta = nn.DataParallel(roberta_single)

  
print("Let's use", num_cores, "GPUs!")

use_gpu = torch.cuda.is_available() if use_gpu is None else use_gpu

device = torch.device("cuda:0" if use_gpu else "cpu")


if not use_gpu:
  fp16 = False


roberta.to(device)

if fp16:
  max_float = MAX_FLOAT16
  min_float = MIN_FLOAT16
  roberta.half()
  
roberta.eval()
  
  
eval_dir = 'test-v2.SQuAD.json'

orig_data = {} 
for e in gen(eval_dir):
  for q in e[2]:
    orig_data[q['id']] = q
  
records, rs = generate_tfrecord(eval_dir, is_training=False, parallel_process=False, return_feature=True)

records = records
rs = rs


batches = list(zip(from_records(records,batch_size, half=fp16, shuffle=False), chunks(rs,batch_size)))

prediction_by_qid = {}
with torch.no_grad():
  for e, rs in tqdm(batches):
    inp, p_mask, start, end, _ = e
    result_tuples = roberta(inp.to(device=device), p_mask.to(device=device))
    
    for result, r in zip(zip(*result_tuples), rs):
      qid = r.qid
      if qid not in prediction_by_qid:
        prediction_by_qid[qid] = []
      prediction_by_qid[qid].append((result, r))


from squad_evaluation import compute_f1

def handle_prediction_by_qid(self, 
                             prediction_by_qid, 
                             n_best_size = 5,
                             threshold = -1.5,
                             max_answer_length = 48,
                             debug = False):
  global prelim_predictions
  use_ans_class = self.use_ans_class
  all_predictions = {}
  scores_diff_json = {}
  score = 0
  for qid, predictions in prediction_by_qid.items():
    q = orig_data[qid]
    ri = 0
    prelim_predictions = []
    for result, r in predictions:
      paragraph_text = r.original_text
      original_s, original_e = r.original_text_span # exclusive

      this_paragraph_text = paragraph_text[original_s:original_e]
      
      
      cur_null_score = -1e6
      
      sub_prelim_predictions = []

      if use_ans_class:
        start_top_log_probs, end_top_log_probs, cls_logits = result
        cur_null_score = cls_logits.tolist()
        
      else:
        start_top_log_probs, end_top_log_probs = result
        
      if True:
        start_top_log_probs = start_top_log_probs.cpu().detach().numpy()
        end_top_log_probs = end_top_log_probs.cpu().detach().numpy()
        start_top_index = start_top_log_probs.argsort()[-self.start_n_top:][::-1].tolist()
        end_top_index = end_top_log_probs.argsort()[-self.end_n_top:][::-1].tolist()
        start_top_log_probs = start_top_log_probs.to_list()
        end_top_log_probs = end_top_log_probs.tolist()
        for start_index in start_top_index:
            for end_index in end_top_index:
              if start_index == 0 or end_index == 0:
                continue
              if end_index < start_index:
                continue
              if start_index >= len(r.segments) or end_index >= len(r.segments):
                continue
              seg_s = r.segments[start_index]
              seg_e = r.segments[end_index]

              if seg_s != seg_e:
                continue

              if r.is_max_context[start_index] == 0 :
                continue

              length = end_index - start_index + 1
              if length > max_answer_length:
                continue
                
                

              start_log_prob = start_top_log_probs[start_index]
              end_log_prob = end_top_log_probs[end_index]


              sub_prelim_predictions.append(
                  _PrelimPrediction(
                      feature_index=ri,
                      start_index=start_index,
                      end_index=end_index,
                      start_log_prob=start_log_prob,
                      end_log_prob=end_log_prob,
                      this_paragraph_text=this_paragraph_text,
                      cur_null_score=cur_null_score
                  ))
              
        
      prelim_predictions.extend(sub_prelim_predictions)
      ri += 1

    prelim_predictions = sorted(
        prelim_predictions,
        key=(lambda x: (x.start_log_prob + x.end_log_prob))),
        reverse=True)

    seen_predictions = {}
    nbest = []
    for pred in prelim_predictions:
      if len(nbest) >= n_best_size:
          break

      r = predictions[pred.feature_index][1]
      
      cur_null_score = pred.cur_null_score

      this_paragraph_text = pred.this_paragraph_text

      s,e = pred.start_index, pred.end_index  # e is inclusive

      char_s  = r.tok_to_char_offset[s]
      char_e  = r.tok_to_char_offset[e]  # inclusive
      char_e += r.length_at_char[char_e]


      final_text = r.text[char_s:char_e].strip() # this_paragraph_text[char_s:char_e]

        
      if False:
        print(final_text, '>>', r.all_text_tokens[s:e+1])

      if final_text in seen_predictions:
          continue
          

      seen_predictions[final_text] = True

      nbest.append(
        _NbestPrediction(
            text=final_text,
            start_log_prob=pred.start_log_prob,
            end_log_prob=pred.end_log_prob,
            cur_null_score=cur_null_score))



    if len(nbest) == 0:
        nbest.append(
          _NbestPrediction(text="", start_log_prob=-1e6,
          end_log_prob=-1e6,
          cur_null_score=-1e6))

    total_scores = []
    best_non_null_entry = None
    best_null_score = None
    best_score_no_ans = None
    for entry in nbest:
      total_scores.append(entry.start_log_prob + entry.end_log_prob)
      if not best_non_null_entry:
        best_non_null_entry = entry
        best_null_score = entry.cur_null_score if use_ans_class else -(entry.start_log_prob + entry.end_log_prob)
        best_score_no_ans = entry.cur_null_score

    probs = _compute_softmax(total_scores)

    nbest_json = []
    for (i, entry) in enumerate(nbest):
      output = collections.OrderedDict()
      output["text"] = entry.text
      output["probability"] = probs[i]
      output["start_log_prob"] = entry.start_log_prob
      output["end_log_prob"] = entry.end_log_prob
      nbest_json.append(output)

    if debug:
      ans = best_non_null_entry.text if best_null_score < threshold else '*No answer*'
      truth = q['answer_text'] or '*No answer*'
      print('Q:', q['question'])
      print('A:', ans, '(',best_null_score,')',  '[',best_score_no_ans,']', )
      print('Truth:', truth)
      print('')
      score += compute_f1(truth, ans)

    assert len(nbest_json) >= 1
    assert best_non_null_entry is not None


    all_predictions[qid] = best_non_null_entry.text
    scores_diff_json[qid] = best_null_score
  
  
  if debug:
    print('score: ', score, '/', len(all_predictions), '=', score / len(all_predictions))
  
  
  return nbest_json, all_predictions, scores_diff_json

nbest_json, all_predictions, scores_diff_json = handle_prediction_by_qid(roberta_single, prediction_by_qid, debug=False)



from squad_evaluation import evaluate
with open(eval_dir, "r") as f:
  predict_data = json.load(f)["data"]
result, exact_raw, f1_raw, wrongs = evaluate(predict_data, 
                                             all_predictions, 
                                             na_probs=scores_diff_json, 
                                             na_prob_thresh=0, 
                                             out_file=None, 
                                             out_image_dir=None)

import code
code.interact(local=locals())
