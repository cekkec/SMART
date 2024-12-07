import os
from os import path
from argparse import ArgumentParser
import shutil
import sys
import pandas as pd
import time
from collections import Counter

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
from inference.data.test_datasets import DAVISTestDataset, YouTubeVOSTestDataset, TAOTestDataset
from inference.data.mask_mapper import MaskMapper, MaskMapper_TAO
from model.network import XMem
from inference.inference_core import InferenceCore
from progressbar import progressbar

from eval_D17.davis2017.evaluation import DAVISEvaluation

try:
    import hickle as hkl
except ImportError:
    print('Failed to import hickle. Fine if not using multi-scale testing.')

parser = ArgumentParser()
parser.add_argument('--model')

# Datasets Path
parser.add_argument('--d17_path', help='DAVIS Path', default='../DAVIS/2017')
parser.add_argument('--y18_path', help='YouTube2018 Path', default='../YouTube2018')
parser.add_argument('--y19_path', help='YouTube2019 Path', default='../YouTube')
parser.add_argument('--tao_path', help='TAO Path', default='../TAO_VOS')
parser.add_argument('--generic_path')

# Val & Test
parser.add_argument('--dataset', help='D17/Y18/Y19/TAO', default='D17')
parser.add_argument('--split', help='val/test', default='val')
parser.add_argument('--output', default='./Results')
parser.add_argument('--save_all', action='store_true', 
            help='Save all frames. Useful only in YouTubeVOS/long-time video', )
parser.add_argument('--benchmark', action='store_true', help='enable to disable amp for FPS benchmarking')
        
# Long-term memory options in XMem
parser.add_argument('--disable_long_term', action='store_true')
parser.add_argument('--max_mid_term_frames', help='T_max in paper, decrease to save memory', type=int, default=10)
parser.add_argument('--min_mid_term_frames', help='T_min in paper, decrease to save memory', type=int, default=5)
parser.add_argument('--max_long_term_elements', help='LT_max in paper, increase if objects disappear for a long time', 
                                                type=int, default=10000)
parser.add_argument('--num_prototypes', help='P in paper', type=int, default=128)

parser.add_argument('--top_k', type=int, default=30)
parser.add_argument('--mem_every', help='r in paper. Increase to improve running speed.', type=int, default=5)
parser.add_argument('--deep_update_every', help='Leave -1 normally to synchronize with mem_every', type=int, default=-1)

# Multi-scale options
parser.add_argument('--save_scores', action='store_true')
parser.add_argument('--flip', action='store_true')
parser.add_argument('--size', default=480, type=int, 
            help='Resize the shorter side to this size. -1 to use original resolution. ')

# Need for evaluation
parser.add_argument('--update', action='store_true', help='Recompute the performance results.')
parser.add_argument('--task', type=str, help='Task to evaluate the results', default='semi-supervised')

args = parser.parse_args()
config = vars(args)
config['enable_long_term'] = not config['disable_long_term']

if args.output is None:
    args.output = f'../output/{args.dataset}_{args.split}'
    print(f'Output path not provided. Defaulting to {args.output}')

SET = args.split
TASK = args.task

proj_name = f'XMem_{args.model}'
print('proj_name : ', proj_name)

if args.dataset == 'D17':
    if args.split == 'val':
        os.makedirs(os.path.join(args.output, 'D17_val'), exist_ok=True)
        args.output = os.path.join(args.output, 'D17_val', proj_name)
        out_path = args.output
        
    elif args.split == 'test':
        os.makedirs(os.path.join(args.output, 'D17_test'), exist_ok=True)
        args.output = os.path.join(args.output, 'D17_test', proj_name)
        out_path = args.output

elif args.dataset == 'Y18':
    os.makedirs(os.path.join(args.output, 'Y18'), exist_ok=True)
    args.output = os.path.join(args.output, 'Y18', proj_name)
    out_path = args.output
    
elif args.dataset == 'Y19':
    os.makedirs(os.path.join(args.output, 'Y19'), exist_ok=True)
    args.output = os.path.join(args.output, 'Y19', proj_name)
    out_path = args.output

elif args.dataset == 'TAO':
    os.makedirs(os.path.join(args.output, 'TAO'), exist_ok=True)
    args.output = os.path.join(args.output, 'TAO', proj_name)
    out_path = args.output
    
"""
Data preparation
"""
is_youtube = args.dataset.startswith('Y')
is_tao = args.dataset.startswith('T')
is_davis = args.dataset.startswith('D')

if is_youtube or args.save_scores:
    out_path = path.join(args.output, 'Annotations')
else:
    out_path = args.output

if is_youtube:
    if args.dataset == 'Y18':
        print('### Start Evaluation on YouTube18 ###')
        yv_path = args.y18_path
    elif args.dataset == 'Y19':
        print('### Start Evaluation on YouTube19 ###')
        yv_path = args.y19_path

    if args.split == 'val':
        args.split = 'valid'
        meta_dataset = YouTubeVOSTestDataset(data_root=yv_path, split='valid', size=args.size)
    elif args.split == 'test':
        meta_dataset = YouTubeVOSTestDataset(data_root=yv_path, split='test', size=args.size)
    else:
        raise NotImplementedError

elif is_tao:
    print('### Start Evaluation on TAO-VOS ###')
    tao_path = args.tao_path
    if args.split == 'val':
        args.split = 'valid'
        meta_dataset = TAOTestDataset(data_root=tao_path, split='valid', size=args.size)
    elif args.split == 'test':
        meta_dataset = TAOTestDataset(data_root=tao_path, split='test', size=args.size)
    else:
        raise NotImplementedError

elif is_davis:
    print('### Start Evaluation on DAVIS ###')
    if args.split == 'val':
        meta_dataset = DAVISTestDataset(path.join(args.d17_path, 'trainval'), imset='2017/val.txt', size=args.size)
    elif args.split == 'test':
        meta_dataset = DAVISTestDataset(path.join(args.d17_path, 'test-dev'), imset='2017/test-dev.txt', size=args.size)
    else:
        raise NotImplementedError

else:
    raise NotImplementedError

torch.autograd.set_grad_enabled(False)

meta_loader = meta_dataset.get_datasets()
network = XMem(config, args.model).cuda().eval()
if args.model is not None:
    model_weights = torch.load(args.model)
    network.load_weights(model_weights, init_as_zero_if_needed=True)
else:
    print('No model loaded.')

total_process_time = 0
total_frames = 0

# Start eval
for vid_reader in progressbar(meta_loader, max_value=len(meta_dataset), redirect_stdout=True):

    loader = DataLoader(vid_reader, batch_size=1, shuffle=False, num_workers=2)
    vid_name = vid_reader.vid_name
    vid_length = len(loader)

    config['enable_long_term_count_usage'] = (
        config['enable_long_term'] and
        (vid_length
            / (config['max_mid_term_frames']-config['min_mid_term_frames'])
            * config['num_prototypes'])
        >= config['max_long_term_elements']
    )

    if is_tao:
        mapper = MaskMapper_TAO()
    else:
        mapper = MaskMapper()
        
    processor = InferenceCore(network, config=config)
    first_mask_loaded = False

    for ti, data in enumerate(loader):
        with torch.cuda.amp.autocast(enabled=not args.benchmark):
            rgb = data['rgb'].cuda()[0]
            msk = data.get('mask')
            info = data['info']
            frame = info['frame'][0]
            shape = info['shape']
            need_resize = info['need_resize'][0]

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

            if not first_mask_loaded:
                if msk is not None:
                    first_mask_loaded = True
                else:
                    continue

            if args.flip:
                rgb = torch.flip(rgb, dims=[-1])
                msk = torch.flip(msk, dims=[-1]) if msk is not None else None

            if msk is not None:
                msk, labels = mapper.convert_mask(msk[0].numpy())
                msk = torch.Tensor(msk).cuda()
                if need_resize:
                    msk = vid_reader.resize_mask(msk.unsqueeze(0))[0]
                processor.set_all_labels(list(mapper.remappings.values()))
            else:
                labels = None

            prob = processor.step(rgb, msk, labels, end=(ti==vid_length-1))

            if need_resize:
                prob = F.interpolate(prob.unsqueeze(1), shape, mode='bilinear', align_corners=False)[:,0]

            end.record()
            torch.cuda.synchronize()
            total_process_time += (start.elapsed_time(end)/1000)
            total_frames += 1

            if args.flip:
                prob = torch.flip(prob, dims=[-1])

            out_mask = torch.max(prob, dim=0).indices
            out_mask = (out_mask.detach().cpu().numpy()).astype(np.uint8)

            if args.save_scores:
                prob = (prob.detach().cpu().numpy()*255).astype(np.uint8)

            # Save the mask
            if args.save_all or info['save'][0]:
                this_out_path = path.join(out_path, vid_name)
                os.makedirs(this_out_path, exist_ok=True)
                out_mask = mapper.remap_index_mask(out_mask)
                out_img = Image.fromarray(out_mask)
                if vid_reader.get_palette() is not None:
                    out_img.putpalette(vid_reader.get_palette())
                out_img.save(os.path.join(this_out_path, frame[:-4]+'.png'))

            if args.save_scores:
                np_path = path.join(args.output, 'Scores', vid_name)
                os.makedirs(np_path, exist_ok=True)
                if ti==len(loader)-1:
                    hkl.dump(mapper.remappings, path.join(np_path, f'backward.hkl'), mode='w')
                if args.save_all or info['save'][0]:
                    hkl.dump(prob, path.join(np_path, f'{frame[:-4]}.hkl'), mode='w', compression='lzf')

if is_tao:
### Eval
from TAO_metric import evaluate_tao_results
# These should be adjusted to your actual file paths
IMGSET_FILE = "../TAO_VOS/valid/ImageSets/valid_except.txt"
GT_PATH = "../TAO_VOS/valid/Annotations/"
out_path = os.path.join(out_path, '')
result_path = out_path
# Evaluate the results
results = evaluate_tao_results(result_path, IMGSET_FILE, GT_PATH, parallel=True, processes=12)

# You can use the results dictionary for further processing or printing
print("Evaluation Results:")
print("Path:", results["result_path"])
print("Number of objects:", results["number_of_objects"])
print("J-measure (IoU):", results["J-measure"])
print("F-measure:", results["F-measure"])
print("J&F measure:", results["J&F measure"])

# Save the results to a CSV file
csv_file_path = os.path.join(result_path, 'evaluation_results.csv')

# Define the header and the data
header = ["Path", "Number of objects", "J-measure (IoU)", "F-measure", "J&F measure"]
# data = [results["result_path"], results["number_of_objects"], results["J-measure"], results["F-measure"], results["J&F measure"]]
data = [
    results["result_path"],
    results["number_of_objects"],
    f"{results['J-measure']:.3f}",
    f"{results['F-measure']:.3f}",
    f"{results['J&F measure']:.3f}"
]

# Write to CSV
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)
    writer.writerow(data)

print(f"Results saved to {csv_file_path}")
