#!/usr/bin/env python3

from json import dumps, load
from os import environ, makedirs, listdir
from os.path import join
from sys import argv, exit
from subprocess import run
import csv
import pandas as pd


def run_single_test(data_dir, output_dir):

    with open(join(data_dir, 'test.cfg')) as f:
        preset = f.read()

    if preset == 'ITER':
        cmd = f'python3 test.py --attack_type iterative --csv_results_dir {output_dir} --device cpu --dataset_path {data_dir}/images --model_weights ./weights/RoIPoolModel.pth',
    elif preset == 'UAP':
        cmd = f'python3 test.py --attack_type uap --uap_train_path ./pretrained_uap_paq2piq.png --csv_results_dir {output_dir} --device cpu --dataset_path {data_dir}/images --model_weights ./weights/RoIPoolModel.pth'

    ret_code = run(cmd, shell=True).returncode
    if ret_code != 0:
        exit(ret_code)


def check_test(data_dir):

    with open(join(data_dir, 'output/results.csv')) as f:
        csvreader = csv.reader(f)
        result = list(csvreader)

    if environ.get('CHECKER'):
        print(result)
    return result
    


def grade(data_dir):

    results = load(open(join(data_dir, 'results.json')))
    ok_count = 0
    for result in results:
        if 'eps' in str(result) and 'score' in str(result):
            ok_count += 1
    mark = ok_count
    
    res = {'description': '', 'mark': mark}
    if environ.get('CHECKER'):
        print(dumps(res))
    return res


if __name__ == '__main__':
    if environ.get('CHECKER'):
        # Script is running in testing system
        if len(argv) != 4:
            print(f'Usage: {argv[0]} mode data_dir output_dir')
            exit(0)

        mode = argv[1]
        data_dir = argv[2]
        output_dir = argv[3]

        if mode == 'run_single_test':
            # Run each test
            run_single_test(data_dir, output_dir)
        elif mode == 'check_test':
            # Put a mark for each test result
            check_test(data_dir)
        elif mode == 'grade':
            # Put overall mark
            grade(data_dir)
    else:
        # Script is running locally, run on dir with tests
        cmds = [
            f'python3 test.py --attack_type iterative --csv_results_dir ./ --device cuda --dataset_path {" ".join(argv[1:])} --model_weights ./weights/RoIPoolModel.pth',
            # f'python3 test.py --attack_type uap --uap_train_path ./uap_trained_data/pretrained_uap_paq2piq.png --csv_results_dir ./ --device cuda --dataset_path {" ".join(argv[1:])} --model_weights ./weights/RoIPoolModel.pth'
        ]

        for cmd in cmds:
            ret_code = run(cmd, shell=True).returncode
            if ret_code != 0:
                exit(ret_code)