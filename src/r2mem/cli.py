import argparse
import json

from .training import run


def main(stage):
    parser = argparse.ArgumentParser(description=f'R²Mem {stage}: user-owned data/backbone adapter')
    parser.add_argument('--config',required=True)
    parser.add_argument('--out',required=True)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--resume',action='store_true')
    if stage == 'stage2':
        parser.add_argument('--memory-init',required=True)
    args = parser.parse_args()
    result = run(stage,args.config,args.out,memory_init=getattr(args,'memory_init',None),
                 device=args.device,resume=args.resume)
    print(json.dumps(result,indent=2),flush=True)
    if stage == 'stage1' and not result['validation']['passed']:
        parser.exit(2,'Memory weights saved, but held-out reconstruction did not pass the configured gain threshold.\n')
