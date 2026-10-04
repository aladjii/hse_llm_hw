"""Completed homework skeleton. See experiment.ipynb for the full workflow."""
import os
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import argparse
import gc
import hashlib
import json
import math
import signal
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (AutoTokenizer, Qwen3Config, Qwen3ForCausalLM,
                          Trainer, TrainingArguments, TrainerCallback,
                          default_data_collator, set_seed)

# Fixed assignment parameters from the supplied skeleton.
MAX_TRAINING_TIME_SECONDS = 60 * 15
MAX_LENGTH = 512
INPUT_IDS, ATTENTION_MASK, LABELS = 'input_ids', 'attention_mask', 'labels'
TOKENIZER_NAME = 'ai-forever/rugpt3small_based_on_gpt2'
OUTPUT_DIR = './output_dir'
NUM_SHARDS = 32
VALIDATION_SIZE = 5000
SEED = 42

TRAINING_CONFIG = {
    'output_dir': f'{OUTPUT_DIR}/qwen3-1b-russian',
    'max_steps': 1_000_000_000,  # TimeoutCallback determines training duration.
    'per_device_train_batch_size': 16, 'gradient_accumulation_steps': 4,
    'per_device_eval_batch_size': 4,
    'learning_rate': 3e-4, 'lr_scheduler_type': 'cosine', 'warmup_ratio': 0.05,
    'optim': 'adamw_torch_fused', 'weight_decay': 0.1,
    'adam_beta1': 0.9, 'adam_beta2': 0.95, 'max_grad_norm': 1.0,
    'bf16': True, 'tf32': True, 'torch_compile': False,
    'gradient_checkpointing': False, 'logging_steps': 5, 'logging_first_step': True,
    'logging_nan_inf_filter': False, 'save_strategy': 'no', 'eval_strategy': 'no',
    'prediction_loss_only': True, 'dataloader_num_workers': 2,
    'dataloader_pin_memory': True, 'remove_unused_columns': False,
    'report_to': 'none', 'seed': SEED, 'data_seed': SEED,
}
EXPERIMENTS = {
    'base': {},
    'batch8': {'per_device_train_batch_size': 8, 'gradient_accumulation_steps': 8},
    'batch32': {'per_device_train_batch_size': 32, 'gradient_accumulation_steps': 2},
    'lr1e4': {'learning_rate': 1e-4},
    'lr6e4': {'learning_rate': 6e-4},
    'constant': {'lr_scheduler_type': 'constant_with_warmup'},
    'linear': {'lr_scheduler_type': 'linear'},
    'compile': {'torch_compile': True},
    'adamw': {'optim': 'adamw_torch'},
    'no_tf32': {'tf32': False},
    'fp32': {'bf16': False},
}
PROMPTS = ['Москва — столица', 'В 1961 году Юрий Гагарин',
           'Искусственный интеллект — это', 'Вода состоит из', 'Российская империя была']


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def get_training_config(experiment='base', **overrides):
    if experiment not in EXPERIMENTS:
        raise ValueError(f'Unknown experiment: {experiment}')
    return {**TRAINING_CONFIG, **EXPERIMENTS[experiment], **overrides}


class TimeoutCallback(TrainerCallback):
    """Stop after 900 wall seconds at an optimizer-step boundary, as in the skeleton."""
    def __init__(self, timeout_seconds):
        self.timeout_seconds = timeout_seconds
        self.start_time = self.end_time = None
        self.timed_out = False
        self.trainer = None

    @property
    def elapsed(self):
        if self.start_time is None:
            return 0.0
        return max(0.0, (self.end_time if self.end_time is not None else time.monotonic())-self.start_time)

    def on_train_begin(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start_time = time.monotonic()
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if self.elapsed >= self.timeout_seconds:
            self.end_time = time.monotonic()
            self.timed_out = True
            control.should_training_stop = control.should_log = True
            print(f'Timeout: {self.elapsed:.2f}s at optimizer step {state.global_step}', flush=True)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        if self.end_time is None:
            self.end_time = time.monotonic()
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        row = dict(logs or {}, step=state.global_step, wall_seconds=self.elapsed)
        if self.trainer:
            row.update(samples=self.trainer.samples_seen, target_tokens=int(self.trainer.target_tokens_seen))
        for key, value in list(row.items()):
            if isinstance(value, float) and not math.isfinite(value):
                row[key] = None
                row[key + '_nonfinite'] = str(value)
        with open(Path(args.output_dir)/'metrics.jsonl', 'a') as f:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
        for key in ('loss', 'grad_norm'):
            if logs and key in logs and not math.isfinite(float(logs[key])):
                raise FloatingPointError(f'Non-finite {key}; inspect LR/dtype/data.')
        return control


def prepare_tokenizer(source=TOKENIZER_NAME):
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=source != TOKENIZER_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    if tokenizer.eos_token_id is None or len(tokenizer) != tokenizer.vocab_size:
        raise ValueError('Tokenizer vocabulary/special tokens changed.')
    return tokenizer


def tokenize_function(examples, tokenizer):
    encoded = tokenizer(examples['text'], truncation=True, padding='max_length',
                        max_length=MAX_LENGTH, return_token_type_ids=False)
    # Keep every source row: filtering would change the first-5000 eval split.
    # A single content token gets EOS; an empty article gets BOS/EOS boundaries.
    # Both cases need at least one target after the causal shift.
    for ids, mask in zip(encoded[INPUT_IDS], encoded[ATTENTION_MASK]):
        count = sum(mask)
        if count < 2:
            if count == 0:
                ids[0] = (tokenizer.bos_token_id if tokenizer.bos_token_id is not None
                          else tokenizer.eos_token_id)
            ids[1] = tokenizer.eos_token_id
            mask[:2] = [1, 1]
    # Mask padding by attention_mask; real EOS is kept even though pad_token == EOS.
    encoded[LABELS] = [[t if m else -100 for t, m in zip(ids, mask)]
                      for ids, mask in zip(encoded[INPUT_IDS], encoded[ATTENTION_MASK])]
    return dict(encoded)


def save_as_parquets(ds, output_dir=OUTPUT_DIR, num_shards=NUM_SHARDS):
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    shards = []
    for index in range(num_shards):
        # contiguous=True is essential: source order and first 5000 articles remain intact.
        shard = ds.shard(num_shards=num_shards, index=index, contiguous=True)
        path = directory/f'{index:05d}.parquet'
        shard.to_parquet(str(path))
        shards.append({'file': path.name, 'rows': len(shard), 'sha256': sha256(path)})
    return shards


def prepare_dataset(data_dir=OUTPUT_DIR, num_proc=16):
    directory = Path(data_dir)
    if directory.exists():
        raise FileExistsError(f'{directory} exists; reuse it or choose another data_dir.')
    dataset = load_dataset('wikimedia/wikipedia', '20231101.ru', split='train')
    if len(dataset) <= VALIDATION_SIZE:
        raise ValueError('Unexpectedly small Wikipedia split.')
    tokenizer = prepare_tokenizer()
    tokenized = dataset.map(tokenize_function, batched=True, batch_size=256, num_proc=num_proc,
            fn_kwargs={'tokenizer': tokenizer}, remove_columns=dataset.column_names)
    tmp = directory.with_name(directory.name+'.preparing')
    shards = save_as_parquets(tokenized, tmp)
    tokenizer.save_pretrained(tmp/'tokenizer')
    manifest = {'dataset': 'wikimedia/wikipedia', 'config': '20231101.ru', 'split': 'train',
                'tokenizer': TOKENIZER_NAME, 'max_length': MAX_LENGTH,
                'validation_size': VALIDATION_SIZE, 'rows': len(dataset), 'shards': shards,
                'source_fingerprint': dataset._fingerprint,
                'short_article_policy': 'Keep source rows: one token -> token/EOS; zero tokens -> BOS/EOS',
                'boundary_ids': {str(i): dataset[i].get('id') for i in (0,4999,5000,len(dataset)-1)}}
    save_json(tmp/'manifest.json', manifest)
    tmp.replace(directory)
    return manifest


def load_tokenized_dataset(data_dir=OUTPUT_DIR, verify_hashes=False):
    directory = Path(data_dir)
    meta = read_json(directory/'manifest.json')
    expected = {'dataset':'wikimedia/wikipedia','config':'20231101.ru','split':'train',
                'tokenizer':TOKENIZER_NAME,'max_length':MAX_LENGTH,'validation_size':VALIDATION_SIZE}
    if any(meta.get(k) != v for k,v in expected.items()):
        raise ValueError('Dataset/tokenizer/split differs from assignment.')
    files = sorted(directory.glob('*.parquet'))
    if len(files) != NUM_SHARDS or [p.name for p in files] != [s['file'] for s in meta['shards']]:
        raise ValueError('Wrong parquet shard set.')
    if verify_hashes and any(sha256(p) != s['sha256'] for p,s in zip(files,meta['shards'])):
        raise ValueError('Parquet checksum mismatch.')
    dataset = load_dataset('parquet', data_files=[str(p) for p in files], split='train')
    if len(dataset) != meta['rows'] or set(dataset.column_names) != {INPUT_IDS,ATTENTION_MASK,LABELS}:
        raise ValueError('Wrong row count or columns.')
    return dataset


def split_dataset(dataset, validation_size=VALIDATION_SIZE):
    if validation_size != VALIDATION_SIZE or len(dataset) <= validation_size:
        raise ValueError('Eval must contain exactly the first 5000 source articles.')
    train_dataset = dataset.select(range(validation_size,len(dataset)))
    eval_dataset = dataset.select(range(validation_size))
    print(f'Training samples: {len(train_dataset)}; validation samples: {len(eval_dataset)}')
    return train_dataset, eval_dataset


def validate_dataset(dataset, tokenizer):
    positions = sorted(set([0,1,4999,5000,len(dataset)-1] +
                          [i*(len(dataset)-1)//127 for i in range(128)]))
    for i in positions:
        row = dataset[i]
        ids, mask, labels = row[INPUT_IDS], row[ATTENTION_MASK], row[LABELS]
        if len(ids)!=512 or len(mask)!=512 or len(labels)!=512:
            raise ValueError(f'Wrong sequence length at row {i}')
        if any(t<0 or t>=tokenizer.vocab_size for t in ids) or sum(mask)<2:
            raise ValueError(f'Invalid tokens at row {i}')
        if any(m not in (0,1) for m in mask) or mask!=sorted(mask,reverse=True):
            raise ValueError(f'Invalid padding mask at row {i}')
        if labels != [t if m else -100 for t,m in zip(ids,mask)]:
            raise ValueError(f'Invalid labels at row {i}')
    return len(positions)


def create_model(tokenizer, attention='flash_attention_2', bf16=True):
    # Don't change these parameters: copied exactly from the supplied skeleton.
    MODEL_CONFIG = {
        'hidden_size': 2048,
        'num_hidden_layers': 12,
        'num_attention_heads': 16,
        'num_key_value_heads': 8,
        'intermediate_size': 8192,
        'head_dim': 128,
        'hidden_act': 'silu',
        'initializer_range': 0.02,
        'scale_attn_weights': True,
        'use_cache': True,
    }
    config = Qwen3Config(vocab_size=tokenizer.vocab_size,
            bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id, **MODEL_CONFIG)
    model = Qwen3ForCausalLM._from_config(config, attn_implementation=attention,
            torch_dtype=torch.bfloat16 if bf16 else torch.float32)
    print(f'Total params: {sum(p.numel() for p in model.parameters()):,}', flush=True)
    return model


def lr_multiplier(elapsed, scheduler, warmup_ratio=.05, budget=900):
    warmup = budget*warmup_ratio
    if elapsed < warmup:
        return max(1.,elapsed)/warmup
    p = min(1.,max(0.,(elapsed-warmup)/(budget-warmup)))
    if scheduler=='constant_with_warmup': return 1.
    if scheduler=='linear': return 1.-p
    if scheduler=='cosine': return (1.+math.cos(math.pi*p))/2
    raise ValueError(scheduler)


class TimedTrainer(Trainer):
    """Correct accumulated causal-token normalization and LR schedule over 900 seconds."""
    def __init__(self, *args, timeout, **kwargs):
        self.timeout = timeout
        self.samples_seen = self.target_tokens_seen = 0
        super().__init__(*args, **kwargs)
        timeout.trainer = self

    def create_scheduler(self, num_training_steps, optimizer=None):
        if self.lr_scheduler is None:
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer or self.optimizer,
                lambda step: lr_multiplier(self.timeout.elapsed, self.args.lr_scheduler_type.value,
                                            self.args.warmup_ratio, self.timeout.timeout_seconds))
            self._created_lr_scheduler = True
        return self.lr_scheduler

    def get_batch_samples(self, iterator, num_batches, device):
        batches, count = super().get_batch_samples(iterator,num_batches,device)
        if count is not None:
            count = sum((b[LABELS][:,1:]!=-100).sum() for b in batches).to(device)
        return batches, count

    def training_step(self, model, inputs, *args, **kwargs):
        self.samples_seen += inputs[INPUT_IDS].shape[0]
        self.target_tokens_seen += (inputs[LABELS][:,1:]!=-100).sum().detach()
        return super().training_step(model,inputs,*args,**kwargs)


def check_gpu():
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError('The process must see exactly one CUDA GPU via CUDA_VISIBLE_DEVICES.')
    if int(os.environ.get('WORLD_SIZE','1'))!=1:
        raise RuntimeError('Use independent jobs, not torchrun/DDP.')
    gpu = torch.cuda.get_device_properties(0)
    return {'gpu':gpu.name,'memory_gib':gpu.total_memory/2**30,
            'a100_80gb':'A100' in gpu.name and gpu.total_memory/2**30>74,
            'torch':torch.__version__,'cuda':torch.version.cuda,
            'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
            'cpu_affinity':sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None}


def check_attention_backend(attention='flash_attention_2'):
    """Run a small real CUDA forward/backward, including masked padding."""
    environment=check_gpu()
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('GPU does not support bf16.')
    if attention=='flash_attention_2':
        try:
            import flash_attn
        except ImportError as error:
            raise RuntimeError('Install flash-attn==2.7.3 using the uv commands in experiment.ipynb.') from error
        environment['flash_attn']=flash_attn.__version__
    config=Qwen3Config(vocab_size=32,hidden_size=128,intermediate_size=256,
            num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,
            head_dim=64,pad_token_id=2,eos_token_id=2,use_cache=False)
    model=Qwen3ForCausalLM._from_config(config,attn_implementation=attention,
            torch_dtype=torch.bfloat16).cuda()
    ids=torch.tensor([[1,3,4,5,2,2],[1,6,7,8,9,2]],device='cuda')
    mask=torch.tensor([[1,1,1,1,0,0],[1,1,1,1,1,1]],device='cuda')
    labels=ids.masked_fill(mask==0,-100)
    loss=model(input_ids=ids,attention_mask=mask,labels=labels).loss
    loss.backward()
    if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all()
                                       for p in model.parameters()):
        raise RuntimeError('CUDA attention check produced nonfinite loss/gradients.')
    torch.cuda.synchronize()
    environment.update(attention=attention,check_loss=loss.item(),forward_backward='passed')
    del model,loss
    torch.cuda.empty_cache()
    return environment


def generate_examples(model, tokenizer, directory, name, max_new_tokens=100, prompts=PROMPTS):
    model.eval()
    tokenizer.padding_side='left'
    previous=model.config.use_cache
    model.config.use_cache=True
    examples=[]
    try:
        for i,prompt in enumerate(prompts):
            inputs=tokenizer(prompt,return_tensors='pt',truncation=True,
                             max_length=MAX_LENGTH-max_new_tokens,
                             return_token_type_ids=False).to(model.device)
            for mode in ('greedy','sample'):
                set_seed(SEED+i)
                options={'max_new_tokens':max_new_tokens,'do_sample':mode=='sample',
                         'use_cache':True,
                         'pad_token_id':tokenizer.pad_token_id,'eos_token_id':tokenizer.eos_token_id}
                if mode=='sample':options.update(temperature=.8,top_p=.95,top_k=50)
                with torch.inference_mode():output=model.generate(**inputs,**options)
                examples.append({'prompt':prompt,'mode':mode,'seed':SEED+i,'settings':options,
                      'generation':tokenizer.decode(output[0,inputs[INPUT_IDS].shape[1]:],skip_special_tokens=True)})
    finally:
        model.config.use_cache=previous
        tokenizer.padding_side='right'
    save_json(Path(directory)/f'{name}.json',examples)
    return examples


def train_model(data_dir=OUTPUT_DIR, output_root='runs', experiment='base', attention='flash_attention_2',
                smoke_steps=None, skip_baseline=False, run_name=None):
    environment=check_gpu()
    cfg=get_training_config(experiment)
    if cfg['bf16'] and not torch.cuda.is_bf16_supported():
        raise RuntimeError('GPU does not support bf16.')
    if attention=='flash_attention_2' and not cfg['bf16']:
        raise ValueError('Use sdpa for FP32.')
    if smoke_steps is not None and smoke_steps<2:
        raise ValueError('Smoke requires >=2 steps.')
    directory=Path(output_root)/(run_name or (f'smoke_{experiment}' if smoke_steps else experiment))
    directory.mkdir(parents=True,exist_ok=False)
    cfg['output_dir']=str(directory)
    cfg['dataloader_persistent_workers']=cfg['dataloader_num_workers']>0
    if cfg['torch_compile']:
        os.environ['TORCHINDUCTOR_CACHE_DIR']=str((directory/'.inductor_cache').resolve())
        os.environ['TRITON_CACHE_DIR']=str((directory/'.triton_cache').resolve())
        os.environ['TORCHINDUCTOR_FX_GRAPH_CACHE']='0'
        os.environ['TORCHINDUCTOR_AUTOGRAD_CACHE']='0'
    dataset=load_tokenized_dataset(data_dir)
    tokenizer=prepare_tokenizer(str(Path(data_dir)/'tokenizer'))
    validate_dataset(dataset,tokenizer)
    train_dataset,eval_dataset=split_dataset(dataset)
    set_seed(SEED)
    model=create_model(tokenizer,attention,cfg['bf16'])
    model.config.use_cache=False
    timeout=TimeoutCallback(timeout_seconds=MAX_TRAINING_TIME_SECONDS)
    if smoke_steps:
        cfg.update(max_steps=smoke_steps,logging_steps=1)
    trainer=TimedTrainer(model=model,args=TrainingArguments(**cfg),train_dataset=train_dataset,
            eval_dataset=eval_dataset.select(range(32)) if smoke_steps else eval_dataset,
            data_collator=default_data_collator,processing_class=tokenizer,
            callbacks=[timeout],timeout=timeout)
    save_json(directory/'run_config.json',{'experiment':experiment,'config':cfg,'attention':attention,
              'seed':SEED,'environment':environment,'budget_seconds':900,
              'parameter_count':sum(p.numel() for p in model.parameters()),
              'dataset_manifest_sha256':sha256(Path(data_dir)/'manifest.json'),
              'model_config':model.config.to_dict(),
              'scheduler_basis':'wall-clock; 5% warmup; compile inside budget; eval/save outside'})
    initial=None
    try:
        if not smoke_steps and not skip_baseline:
            # Use a separate uncompiled evaluator: compile warm-up belongs inside the timer.
            initial_args=TrainingArguments(output_dir=str(directory/'initial_eval'),
                    per_device_eval_batch_size=4,prediction_loss_only=True,report_to='none',
                    bf16=cfg['bf16'],tf32=cfg['tf32'],dataloader_num_workers=2)
            initial_trainer=Trainer(model=model,args=initial_args,eval_dataset=eval_dataset,
                                   data_collator=default_data_collator,processing_class=tokenizer)
            initial=initial_trainer.evaluate(metric_key_prefix='initial_eval')
            save_json(directory/'initial_eval.json',initial)
            del initial_trainer
            gc.collect()
            generate_examples(model,tokenizer,directory,'generations_before',max_new_tokens=64)
        set_seed(SEED)
        tracked=model.lm_head.weight[:2,:32].detach().float().cpu().clone() if smoke_steps else None
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        trainer.train()
        if not smoke_steps and not timeout.timed_out:
            raise RuntimeError('Training stopped before 900 seconds; invalid submission run.')
        if smoke_steps and torch.equal(tracked,model.lm_head.weight[:2,:32].detach().float().cpu()):
            raise AssertionError('Smoke: optimizer did not change weights.')
        summary={'status':'smoke_only' if smoke_steps else 'trained','initial_eval':initial,
                 'training_seconds':timeout.elapsed,'steps':trainer.state.global_step,
                 'samples_seen':trainer.samples_seen,'target_tokens_seen':int(trainer.target_tokens_seen),
                 'target_tokens_per_second':int(trainer.target_tokens_seen)/timeout.elapsed,
                 'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30}
        save_json(directory/'summary.json',summary)
        # Right-padded loss evaluation must never create a generation KV cache.
        model.config.use_cache=False
        if not smoke_steps:
            trainer.save_model(str(directory/'model'))
            tokenizer.save_pretrained(directory/'model')
        trainer.save_state()
        print('Running final evaluation...',flush=True)
        final=trainer.evaluate()
        if not math.isfinite(final['eval_loss']):raise FloatingPointError('Non-finite eval_loss.')
        summary.update(eval_loss=final['eval_loss'],perplexity=math.exp(final['eval_loss']) if final['eval_loss']<700 else None,
                       final_eval=final,status='smoke_only' if smoke_steps else 'evaluated')
        save_json(directory/'summary.json',summary)
        if not smoke_steps:
            generate_examples(model,tokenizer,directory,'generations_after')
            summary['status']='complete'
            save_json(directory/'summary.json',summary)
        trainer.save_state()
        return summary
    except BaseException as error:
        save_json(directory/'failure.json',{'error':type(error).__name__,'message':str(error),
                  'training_seconds':timeout.elapsed,'step':trainer.state.global_step})
        trainer.save_state()
        raise


def generate_saved(run_dir, prompt, max_new_tokens=100):
    check_gpu()
    if not 1 <= max_new_tokens < MAX_LENGTH:
        raise ValueError('max_new_tokens must be in [1,511].')
    folder=Path(run_dir)
    saved=read_json(folder/'run_config.json')
    cfg=saved['config']
    tokenizer=prepare_tokenizer(str(folder/'model'))
    model=Qwen3ForCausalLM.from_pretrained(folder/'model',local_files_only=True,
          attn_implementation=saved['attention'],
          torch_dtype=torch.bfloat16 if cfg['bf16'] else torch.float32).cuda()
    return generate_examples(model,tokenizer,folder,'generations_custom',max_new_tokens,[prompt])


def can_finish_run(run_dir):
    folder=Path(run_dir)
    return ((folder/'summary.json').exists() and (folder/'run_config.json').exists()
            and (folder/'model/config.json').exists()
            and any((folder/'model').glob('*.safetensors'))
            and read_json(folder/'summary.json').get('training_seconds',0)>=MAX_TRAINING_TIME_SECONDS)


def finish_saved_run(run_dir, data_dir=OUTPUT_DIR):
    """Finish full eval/generations from an already trained checkpoint; no optimizer steps."""
    environment=check_gpu()
    folder=Path(run_dir)
    if not can_finish_run(folder):
        raise ValueError('Recovery needs saved weights and a recorded >=900-second training run.')
    saved=read_json(folder/'run_config.json')
    if saved['dataset_manifest_sha256']!=sha256(Path(data_dir)/'manifest.json'):
        raise ValueError('Recovery dataset differs from the original run.')
    cfg=saved['config']
    tokenizer=prepare_tokenizer(str(folder/'model'))
    dataset=load_tokenized_dataset(data_dir)
    validate_dataset(dataset,tokenizer)
    _,eval_dataset=split_dataset(dataset)
    model=Qwen3ForCausalLM.from_pretrained(folder/'model',local_files_only=True,
            attn_implementation=saved['attention'],
            torch_dtype=torch.bfloat16 if cfg['bf16'] else torch.float32).cuda()
    # Old checkpoints may have been saved with use_cache=True; override for loss.
    model.config.use_cache=False
    args=TrainingArguments(output_dir=str(folder),per_device_eval_batch_size=cfg['per_device_eval_batch_size'],
            prediction_loss_only=True,report_to='none',bf16=cfg['bf16'],tf32=cfg['tf32'],
            dataloader_num_workers=cfg['dataloader_num_workers'],seed=SEED,data_seed=SEED)
    trainer=Trainer(model=model,args=args,eval_dataset=eval_dataset,
            data_collator=default_data_collator,processing_class=tokenizer)
    summary=read_json(folder/'summary.json')
    try:
        final=trainer.evaluate()
        if not math.isfinite(final['eval_loss']):
            raise FloatingPointError('Non-finite eval_loss.')
        summary.update(eval_loss=final['eval_loss'],
                perplexity=math.exp(final['eval_loss']) if final['eval_loss']<700 else None,
                final_eval=final,status='evaluated',recovered_finalization=True,
                finalization_environment=environment,finalization_torch_compile=False)
        save_json(folder/'summary.json',summary)
        generate_examples(model,tokenizer,folder,'generations_after')
        summary['status']='complete'
        save_json(folder/'summary.json',summary)
        return summary
    except BaseException as error:
        save_json(folder/'finish_failure.json',{'error':type(error).__name__,'message':str(error)})
        raise


def gpu_ids_from_env(gpu_ids=None):
    visible=os.environ.get('CUDA_VISIBLE_DEVICES')
    allowed=None if visible is None else [x.strip() for x in visible.split(',') if x.strip()]
    if gpu_ids is None:
        gpu_ids=allowed if allowed is not None else subprocess.check_output(
                ['nvidia-smi','--query-gpu=index','--format=csv,noheader'],text=True).splitlines()
    ids=[str(x).strip() for x in gpu_ids]
    if not ids or len(ids)!=len(set(ids)) or '-1' in ids:
        raise ValueError('Provide nonempty unique GPU IDs.')
    if allowed is not None and any(x not in allowed for x in ids):
        raise ValueError(f'GPU IDs must respect CUDA_VISIBLE_DEVICES={visible}.')
    return ids


def run_experiments(experiments, gpu_ids=None, data_dir=OUTPUT_DIR, output_root='runs',
                    smoke=False, skip_baseline=False, skip_completed=True, cpu_cores=8,
                    attention='flash_attention_2', recover_failed=False):
    """Up to one process per GPU; each process trains independently from scratch."""
    ids=gpu_ids_from_env(gpu_ids)
    names=list(experiments)
    if not names or len(names)!=len(set(names)) or any(n not in EXPERIMENTS for n in names):
        raise ValueError('Experiment names must be unique and known.')
    if attention not in ('flash_attention_2','sdpa'):
        raise ValueError('Unknown attention backend.')
    if attention=='flash_attention_2' and any(not get_training_config(n)['bf16'] for n in names):
        raise ValueError('FlashAttention-2 requires BF16/FP16. Run fp32 separately with attention="sdpa".')
    root=Path(output_root).resolve()
    root.mkdir(parents=True,exist_ok=True)
    if (root/'.queue.lock').exists():
        raise FileExistsError('A queue is active in this output_root; interrupt it before starting recovery.')
    logs=root/'logs'
    logs.mkdir(exist_ok=True)
    pending=[]
    commands={}
    if smoke and recover_failed:
        raise ValueError('Recovery applies only to full training runs.')
    for name in names:
        folder=root/(f'smoke_{name}' if smoke else name)
        if folder.exists():
            if not smoke and skip_completed and (folder/'summary.json').exists() and (folder/'run_config.json').exists():
                saved=read_json(folder/'run_config.json')
                expected=get_training_config(name)
                if (read_json(folder/'summary.json').get('status')=='complete'
                    and saved['dataset_manifest_sha256']==sha256(Path(data_dir)/'manifest.json')
                    and saved.get('attention')==attention
                    and all(saved['config'].get(k)==v for k,v in expected.items() if k!='output_dir')):
                    print('SKIP completed:',name,flush=True)
                    continue
            if not recover_failed:
                raise FileExistsError(f'{folder} already exists; choose another output_root for retry.')
            if (folder/'run_config.json').exists():
                saved=read_json(folder/'run_config.json')
                expected=get_training_config(name)
                if (saved['dataset_manifest_sha256']!=sha256(Path(data_dir)/'manifest.json')
                    or saved.get('attention')!=attention
                    or any(saved['config'].get(k)!=v for k,v in expected.items() if k!='output_dir')):
                    raise ValueError(f'{name}: saved settings differ; use the original settings to recover.')
            if can_finish_run(folder):
                commands[name]='finish'
                print('FINISH saved checkpoint:',name,flush=True)
            else:
                # Preserve incomplete attempts before starting a fresh 900-second run.
                archived=root/'_interrupted_runs'/f'{name}_{time.time_ns()}'
                archived.parent.mkdir(exist_ok=True)
                folder.rename(archived)
                if (logs/f'{name}.log').exists():
                    (logs/f'{name}.log').rename(archived/'process.log')
                print('RESTART incomplete run; previous attempt saved at:',archived,flush=True)
        pending.append(name)
    cores=sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else list(range(os.cpu_count() or 1))
    width=min(cpu_cores,len(cores)//len(ids))
    if width<1:raise ValueError('Insufficient allocated CPU cores.')
    queued=deque(pending)
    active={}
    finished=[]
    lock=root/'.queue.lock'
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    os.close(fd)
    def state():
        save_json(root/'queue.json',{'queued':list(queued),'active':[v[2] for v in active.values()],'finished':finished})
    try:
        while queued or active:
            for index,gpu in enumerate(ids):
                if gpu in active or not queued:continue
                name=queued.popleft()
                env=os.environ.copy()
                env.update(CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
                           OPENBLAS_NUM_THREADS='1',TORCHINDUCTOR_COMPILE_THREADS='2',PYTHONUNBUFFERED='1')
                env.setdefault('HF_DATASETS_CACHE',str(root/'.dataset_cache'))
                command=['uv','run','--no-project','--python',sys.executable,'python',str(Path(__file__).resolve()),
                         commands.get(name,'smoke' if smoke else 'train'),'--data-dir',str(Path(data_dir).resolve()),
                         '--output-root',str(root),'--experiment',name,'--attention',attention]
                if commands.get(name)=='finish':command.extend(['--run-dir',str(root/name)])
                if skip_baseline:command.append('--skip-baseline')
                import shutil
                if shutil.which('taskset'):
                    command=['taskset','--cpu-list',','.join(map(str,cores[index*width:(index+1)*width])),*command]
                log=open(logs/f'{name}.log','a' if commands.get(name)=='finish' else 'w')
                try:process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                except BaseException:log.close();raise
                action=commands.get(name,'smoke' if smoke else 'train')
                active[gpu]=(process,log,{'experiment':name,'gpu':gpu,'pid':process.pid,'action':action})
                print(f'START {name} [{action}] -> GPU {gpu}',flush=True)
                state()
            failed=False
            for gpu,(process,log,record) in list(active.items()):
                code=process.poll()
                if code is None:continue
                log.close()
                finished.append(dict(record,exit_code=code))
                del active[gpu]
                print(f'END {record["experiment"]} GPU {gpu}: exit={code}',flush=True)
                if code:
                    print('\n'.join((logs/f'{record["experiment"]}.log').read_text(errors='replace').splitlines()[-12:]))
                    failed=True
                state()
            if failed and smoke:raise RuntimeError('Smoke failed; inspect logs before pretraining.')
            if active:time.sleep(.5)
        return finished
    except BaseException:
        for process,log,record in active.values():
            if process.poll() is None:os.killpg(process.pid,signal.SIGTERM)
        for process,log,record in active.values():
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
            log.close()
            finished.append(dict(record,exit_code=process.returncode))
        active.clear()
        state()
        raise
    finally:lock.unlink(missing_ok=True)


def self_check():
    """Small CPU check of Qwen3/loss/backward/save/load, without downloading anything."""
    import tempfile
    torch.set_num_threads(2)
    config=Qwen3Config(vocab_size=32,hidden_size=32,intermediate_size=64,num_hidden_layers=1,
            num_attention_heads=2,num_key_value_heads=1,head_dim=16,pad_token_id=2,eos_token_id=2,use_cache=False)
    model=Qwen3ForCausalLM(config)
    ids=torch.tensor([[1,3,4,2,2,2]])
    mask=torch.tensor([[1,1,1,0,0,0]])
    labels=ids.clone();labels[mask==0]=-100
    out=model(input_ids=ids,attention_mask=mask,labels=labels)
    expected=torch.nn.functional.cross_entropy(out.logits[0,:2].float(),ids[0,1:3])
    assert torch.allclose(out.loss,expected,atol=1e-6)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
    before=model.lm_head.weight.detach().clone()
    out.loss.backward();optimizer.step()
    assert not torch.equal(before,model.lm_head.weight)
    with tempfile.TemporaryDirectory() as tmp:
        model.save_pretrained(tmp)
        loaded=Qwen3ForCausalLM.from_pretrained(tmp,local_files_only=True).eval()
        with torch.inference_mode():
            output=loaded.generate(input_ids=ids[:,:2],attention_mask=mask[:,:2],max_new_tokens=4,pad_token_id=2)
        assert 2<output.shape[1]<=6
    assert lr_multiplier(45,'cosine')==1 and lr_multiplier(900,'cosine')==0
    assert lr_multiplier(900,'linear')==0 and lr_multiplier(900,'constant_with_warmup')==1
    print('CPU self-check passed: causal loss, masking, backward, optimizer, save/load, generation, scheduler.')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['prepare','check','train','smoke','self-check','generate','finish'])
    parser.add_argument('--data-dir',default=OUTPUT_DIR)
    parser.add_argument('--output-root',default='runs')
    parser.add_argument('--experiment',choices=EXPERIMENTS,default='base')
    parser.add_argument('--num-proc',type=int,default=16)
    parser.add_argument('--attention',choices=['sdpa','flash_attention_2'],default='flash_attention_2')
    parser.add_argument('--skip-baseline',action='store_true')
    parser.add_argument('--run-dir')
    parser.add_argument('--prompt',default=PROMPTS[0])
    parser.add_argument('--max-new-tokens',type=int,default=100)
    args=parser.parse_args()
    if args.command=='prepare':result=prepare_dataset(args.data_dir,args.num_proc)
    elif args.command=='check':result=check_attention_backend(args.attention)
    elif args.command=='self-check':result=self_check()
    elif args.command=='generate':
        if not args.run_dir:parser.error('--run-dir is required for generate')
        result=generate_saved(args.run_dir,args.prompt,args.max_new_tokens)
    elif args.command=='finish':
        if not args.run_dir:parser.error('--run-dir is required for finish')
        result=finish_saved_run(args.run_dir,args.data_dir)
    else:result=train_model(args.data_dir,args.output_root,args.experiment,args.attention,
                           4 if args.command=='smoke' else None,args.skip_baseline)
    if result is not None:print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
