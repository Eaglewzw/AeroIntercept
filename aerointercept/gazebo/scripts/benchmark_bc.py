"""Measure cached-real-batch BC throughput in isolated GPU subprocesses.

Includes host-to-device transfer, forward, backward, clipping and AdamW.
Excludes dataset decoding/collation: this is not an end-to-end epoch benchmark.
No trained checkpoint is written or modified.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def trial(args):
    import torch
    from torch.utils.data import default_collate
    from aerointercept.config import DotDict
    from aerointercept.end_to_end.data import EpisodeSequenceDataset
    from aerointercept.end_to_end.policy import EndToEndActorCritic
    from aerointercept.end_to_end.optimization import (
        adamw_with_backbone_lr, keep_backbone_batch_norm_eval)
    from aerointercept.gazebo.config import load_gazebo_config
    from aerointercept.gazebo.checkpoint import validate_task_checkpoint, load_model_weights
    from aerointercept.training.train_e2e_bc import compute_losses

    cfg = load_gazebo_config(args.config)
    cfg['end_to_end']['model']['encoder_chunk_size'] = args.chunk
    cfg['end_to_end']['auxiliary']['action_coef'] = 20.
    torch.manual_seed(0)
    device = torch.device('cuda:0')
    if not torch.cuda.is_available():
        raise RuntimeError('GPU unavailable; CPU fallback is not a valid benchmark')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    validate_task_checkpoint(checkpoint, cfg)
    paths = [Path(args.data)/'episodes'/p for p in checkpoint['dataset_split']['train'][:2]]
    dataset = EpisodeSequenceDataset(paths, 16, 2, cache_size=2,
                                    self_state_dim=6, camera_supervision=True)
    batch = default_collate([dataset[i] for i in range(args.batch)])
    batch = {key: value.pin_memory() for key, value in batch.items()}
    model_cfg = DotDict(dict(cfg.end_to_end.model))
    model_cfg['pretrained_weights'] = None
    model = EndToEndActorCritic(model_cfg).to(device)
    load_model_weights(model, checkpoint, dict(cfg.end_to_end.model))
    model.actor.train()
    keep_backbone_batch_norm_eval(model)
    optimizer = adamw_with_backbone_lr(model, 1e-4, 1e-5, actor_only=True)
    initial = model.actor.action_head[-1].weight.detach().clone()

    def step():
        optimizer.zero_grad(set_to_none=True)
        losses = compute_losses(model.actor, batch, device, cfg.end_to_end.auxiliary)
        if not bool(torch.isfinite(losses.total)):
            raise RuntimeError('non-finite loss')
        losses.total.backward()
        torch.nn.utils.clip_grad_norm_(model.actor.parameters(), 1.)
        optimizer.step()

    free, total = torch.cuda.mem_get_info()
    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    durations = []
    for _ in range(args.steps):
        started = time.perf_counter()
        step()
        torch.cuda.synchronize()
        durations.append(time.perf_counter()-started)
    elapsed = sum(durations)
    delta = float((model.actor.action_head[-1].weight-initial).detach().abs().max())
    if delta <= 0:
        raise RuntimeError('optimizer did not change Actor weights')
    return {
        'status': 'ok', 'batch_size': args.batch, 'encoder_chunk_size': args.chunk,
        'sequence_length': 16, 'history_frames': 2, 'precision': 'float32',
        'backbone_trainable': True, 'batch_norm_eval': True,
        'timed_steps': args.steps, 'warmup_steps': args.warmup,
        'mean_step_seconds': elapsed/args.steps,
        'step_seconds': durations,
        'decision_samples_per_second': args.batch*16*args.steps/elapsed,
        'image_encodings_per_second': args.batch*16*2*args.steps/elapsed,
        'peak_allocated_mib': torch.cuda.max_memory_allocated()/2**20,
        'peak_reserved_mib': torch.cuda.max_memory_reserved()/2**20,
        'free_before_mib': free/2**20, 'total_mib': total/2**20,
        'parameter_delta': delta, 'gpu': torch.cuda.get_device_name(0),
        'torch': torch.__version__, 'hip': torch.version.hip,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/gazebo_feedback.yaml')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data', default='artifacts/data/noncontact_geometry_corrective')
    p.add_argument('--output', required=True)
    p.add_argument('--batch-sizes', nargs='+', type=int, default=[2, 4, 8])
    p.add_argument('--chunk-sizes', nargs='+', type=int, default=[4, 16])
    p.add_argument('--steps', type=int, default=5)
    p.add_argument('--warmup', type=int, default=2)
    p.add_argument('--batch', type=int)
    p.add_argument('--chunk', type=int)
    args = p.parse_args()
    if args.steps < 1 or args.warmup < 1 or any(v < 1 for v in args.batch_sizes+args.chunk_sizes):
        p.error('steps, warmup, batch sizes and chunk sizes must be positive')
    if args.batch is not None and (args.batch < 1 or args.chunk is None or args.chunk < 1):
        p.error('an isolated trial requires positive --batch and --chunk')
    if args.chunk is not None and args.batch is None:
        p.error('--chunk requires --batch')
    if args.batch is not None:
        import torch
        try:
            result = trial(args)
        except torch.cuda.OutOfMemoryError as exc:
            result = {'status': 'out_of_memory', 'batch_size': args.batch,
                      'encoder_chunk_size': args.chunk, 'error': str(exc)}
        print('BC_BENCHMARK='+json.dumps(result), flush=True)
        return
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {'scope': 'cached real batch, transfer + full training step; excludes input pipeline',
              'checkpoint': args.checkpoint, 'trials': [], 'complete': False}
    for chunk in args.chunk_sizes:
        for batch in args.batch_sizes:
            cmd = [sys.executable, '-m', 'aerointercept.gazebo.scripts.benchmark_bc',
                   '--config', args.config, '--checkpoint', args.checkpoint,
                   '--data', args.data, '--output', args.output,
                   '--batch', str(batch), '--chunk', str(chunk),
                   '--steps', str(args.steps), '--warmup', str(args.warmup)]
            try:
                proc = subprocess.run(cmd, text=True, capture_output=True, timeout=300)
                lines = [s for s in proc.stdout.splitlines() if s.startswith('BC_BENCHMARK=')]
                if proc.returncode or not lines:
                    raise RuntimeError(f'exit={proc.returncode}; {proc.stderr[-2000:]}')
                result = json.loads(lines[-1].split('=', 1)[1])
            except Exception as exc:
                result = {'status': 'failed', 'batch_size': batch,
                          'encoder_chunk_size': chunk, 'error': repr(exc)}
            report['trials'].append(result)
            temporary = output.with_suffix('.tmp')
            temporary.write_text(json.dumps(report, indent=2))
            temporary.replace(output)
            print(json.dumps(result), flush=True)
    report['complete'] = True
    output.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
