#!/usr/bin/env python3
from os import environ, makedirs, listdir
from os.path import join
from sys import argv, exit
from subprocess import run
import csv
import logging
import os
import json
import numpy as np
from time import sleep
from json import dumps, load, loads, dump

DEVICE = "cuda"
BATCH_SIZE = 8
NUM_WORKERS = 8

SCORE_METRICS = ["Gain Score", "Quality Score", "SRCC"]



def run_single_test(data_dir, checkpoint_dir, output_dir):
    cmd = f'python3 test.py --data_dir {data_dir} --output_dir {output_dir} --checkpoint_dir {checkpoint_dir} --device {DEVICE} --batch_size {BATCH_SIZE} --num_workers {NUM_WORKERS} --score_detail_level simple'

    ret_code = run(cmd, shell=True).returncode

    baseline_scores = load(open(join(data_dir, 'scores.json'), "r"))
    scores = load(open(join(output_dir, 'scores.json'), "r"))

    result = {
        f"Difference in {metric}": scores[metric]-baseline_scores[metric]
        for metric in SCORE_METRICS
    }

    with open(join(output_dir,'results.json'), 'w') as f:
        dump(result, f)

    if ret_code != 0:
        exit(ret_code)


def check_test(data_dir):
    # {data_dir}---output/results.csv
    #          \---gt/gt.csv
    scores = load(open(join(data_dir, 'output/results.json'), "r"))

    if environ.get('CHECKER'):
        print(dumps(scores))

    return scores
    

def grade(output_dir):
    # {data_dir}---results.json

    results = load(open(join(output_dir, 'results.json'), "r"))
    ok_count = 0
    
    res = {
        f"Difference in {metric}": []
        for metric in SCORE_METRICS
    }
    
    for result in results:
        if "Time limit" in result['status'] or "Runtime error" in result['status']:
            continue
        
        ok_count += 1
        metrics = loads(result['status'])
        
        for metric in res:
            res[metric].append(metrics[metric])
    
    res = {'description': {metric: sum(res[metric])/len(res[metric]) for metric in res}, 'mark': ok_count}
    if environ.get('CHECKER'):
        print(dumps(res))
    return res


if __name__ == '__main__':
    if environ.get('CHECKER'):
        if len(argv) != 4:
            print(f'Usage: {argv[0]} mode data_dir output_dir')
            exit(0)

        mode = argv[1]
        data_dir = argv[2]
        output_dir = argv[3]

        if mode == 'run_single_test':
            # Run each test        
            data_dir = "/data"
            checkpoint_dir = "/weights"
            run_single_test(data_dir, checkpoint_dir, output_dir)
        elif mode == 'check_test':
            # Put a mark for each test result
            check_test(data_dir)
        elif mode == 'grade':
            # Put overall mark
            grade(data_dir)
    else:
        # cmd = f'python3 test.py --data_dir 00_test_file_input --output_dir 00_test_file_input --device {DEVICE} --batch_size {BATCH_SIZE} --num_workers {NUM_WORKERS} --score_detail_level simple'
        cmd = f'python3 test.py --data_dir splitted/val --output_dir splitted/val --device {DEVICE} --batch_size {BATCH_SIZE} --num_workers {NUM_WORKERS} --score_detail_level simple'
        ret_code = run(cmd, shell=True, text=True).returncode
        if ret_code != 0:
            exit(ret_code)