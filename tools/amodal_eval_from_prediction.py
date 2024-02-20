'''
evaluate from saved prediction
ref: https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
'''

import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from aistron.evaluation import AmodalCOCOeval
import numpy as np
import json
import argparse
from aistron.evaluation import AmodalInstanceEvaluator

from collections.abc import Mapping

def print_csv_format(results):
    """
    Print main metrics in a format similar to Detectron,
    so that they are easy to copypaste into a spreadsheet.

    Args:
        results (OrderedDict[dict]): task_name -> {metric -> score}
            unordered dict can also be printed, but in arbitrary order
    """
    for task, res in results.items():
        if isinstance(res, Mapping):
            # Don't print "AP-category" metrics since they are usually not tracked.
            important_res = [(k, v) for k, v in res.items() if "-" not in k]
            print("copypaste: Task: {}".format(task))
            print("copypaste: " + ",".join([k[0] for k in important_res]))
            print("copypaste: " + ",".join(["{0:.4f}".format(k[1]) for k in important_res]))
        else:
            print(f"copypaste: {task}={res}")


def amodal_eval_from_prediction(
        annFile,
        resFile,
        dataset_name,  # "d2sa_val", "cocoa_test"
        tasks="amodal_segm" # "amodal_segm", "visible_segm", "occluding_segm", "occluded_segm"
    ):
    
    with open(resFile, 'r') as f:
        predictions = json.load(f)

    cocoGt=COCO(annFile)
    cocoDt = cocoGt.loadRes(predictions)

    cocoEval = AmodalCOCOeval(cocoGt, cocoDt, tasks)
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    amodal_evaluator = AmodalInstanceEvaluator(dataset_name=dataset_name, tasks=tasks)
    amodal_evaluator.reset()
    res = amodal_evaluator._derive_coco_results(cocoEval, tasks, class_names=amodal_evaluator._metadata.get("thing_classes"))
    print_csv_format(res)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=""
    )
    parser.add_argument("--annFile", required=True, help="JSON file produced by the model")
    parser.add_argument("--resFile", required=True, help="output directory")
    parser.add_argument("--dataset_name", help="name of the dataset", default="d2sa_val")
    parser.add_argument("--tasks", default="occluding_segm", type=str, help="amodal, visible, occluding, or occluded")
    args = parser.parse_args()
    amodal_eval_from_prediction(
        annFile=args.annFile,
        resFile=args.resFile,
        dataset_name=args.dataset_name,
        tasks=args.tasks
    )
