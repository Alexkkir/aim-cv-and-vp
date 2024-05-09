#!/usr/bin/env python3
import os
from glob import glob
from json import load, dump, dumps
from os import environ
from os.path import basename, join, exists, splitext
from sys import argv

NUM_OBSERVERS_TO_CHECK = 3

def check_test(data_dir):

    from utils import calculate_single_observer_metrics

    test_output_path = join(data_dir, 'output')
    gt_path = join(data_dir, 'gt', 'test')
    
    res = calculate_single_observer_metrics(test_output_path, gt_path, num_observers=NUM_OBSERVERS_TO_CHECK)

    res = f'Ok, CC: {res["cc"]:.4f}, NSS: {res["nss"]:.4f}, SIM: {res["sim"]:.4f}'

    if environ.get('CHECKER'):
        print(res)

    return res


def grade(data_path):
    results = load(open(join(data_path, 'results.json')))
    result = results[-1]['status']

    if not result.startswith('Ok'):
        res = {'description': '', 'mark': 0}
    else:
        res = {'description': result[4:], 'mark': 1}

    if environ.get('CHECKER'):
        print(dumps(res))

    return res


def run_single_test(data_dir, output_dir):

    from saliency_single_observer_baseline import SingleObserverSaliencyEvaluator
    from os.path import abspath, dirname, join

    video_folders = join(data_dir, 'test')
    code_dir = dirname(abspath(__file__))

    for video_name in sorted(os.listdir(video_folders)):
        
        observers = sorted(os.listdir(os.path.join(video_folders, video_name, 'observers')))
        observers = [x for x in observers if '.DS_Store' not in x]

        for observer_id in observers[:NUM_OBSERVERS_TO_CHECK]:

            evaluator = SingleObserverSaliencyEvaluator(join(code_dir, 'saliency_single_observer.pth'))
            video_frames_path = os.path.join(video_folders, video_name, 'frames')
            observer_data_path = os.path.join(video_folders, video_name, 'observers', observer_id)
            output_saliency_path = os.path.join(output_dir, video_name, 'observer_' + str(observer_id))
            os.makedirs(output_saliency_path, exist_ok=True)
            
            evaluator.evaluate(video_frames_path, observer_data_path, output_saliency_path)


if __name__ == '__main__':
    if environ.get('CHECKER'):
        # Script is running in testing system
        if len(argv) != 4:
            print('Usage: %s mode data_dir output_dir' % argv[0])
            exit(0)

        mode = argv[1]
        data_dir = argv[2]
        output_dir = argv[3]

        if mode == 'run_single_test':
            run_single_test(data_dir, output_dir)
        elif mode == 'check_test':
            check_test(data_dir)
        elif mode == 'grade':
            grade(data_dir)
    else:
        # Script is running locally, run on dir with tests
        if len(argv) != 2:
            print('Usage: %s tests_dir' % argv[0])
            exit(0)

        from glob import glob
        from json import dump
        from re import sub
        from time import time
        from traceback import format_exc
        from os import makedirs
        from os.path import basename, exists
        from shutil import copytree

        tests_dir = argv[1]

        results = []
        for input_dir in sorted(glob(join(tests_dir, '[0-9][0-9]_*_input'))):
            output_dir = sub('input$', 'check', input_dir)
            run_output_dir = join(output_dir, 'output')
            makedirs(run_output_dir, exist_ok=True)
            gt_src = sub('input$', 'gt', input_dir)
            gt_dst = join(output_dir, 'gt')
            if not exists(gt_dst):
                copytree(gt_src, gt_dst)

            try:
                start = time()
                run_single_test(input_dir, run_output_dir)
                end = time()
                running_time = end - start
            except:
                status = 'Runtime error'
                traceback = format_exc()
            else:
                try:
                    status = check_test(output_dir)
                except:
                    status = 'Checker error'
                    traceback = format_exc()

            test_num = basename(input_dir)[:2]
            if status == 'Runtime error' or status == 'Checker error':
                print(test_num, status, '\n', traceback)
                results.append({'status': status})
            else:
                print(test_num, '%.2fs' % running_time, status)
                results.append({
                    'time': running_time,
                    'status': status})

        dump(results, open(join(tests_dir, 'results.json'), 'w'))
        res = grade(tests_dir)
        print('Mark:', res['mark'], res['description'])