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

def print_csv_format(res, task):
    """
    Print main metrics in a format similar to Detectron,
    so that they are easy to copypaste into a spreadsheet.

    Args:
        results (OrderedDict[dict]): task_name -> {metric -> score}
            unordered dict can also be printed, but in arbitrary order
    """
    for ap_type, value in res.items():
        print(f"copypaste: {ap_type}={value:.4f}")

    if isinstance(res, Mapping):
        # Don't print "AP-category" metrics since they are usually not tracked.
        important_res = [(k, v) for k, v in res.items() if "-" not in k]
        print("copypaste: Task: {}".format(task))
        print("copypaste: " + ",".join([k[0] for k in important_res]))
        print("copypaste: " + ",".join(["{0:.4f}".format(k[1]) for k in important_res]))


def eval_iou_only(annFile, resFile):
    """
    this is when the resFile using the bounding box ground truth
    and the prediction is just the instance mask inside that box
    now computing ap would be pointless, so we just compute iou
    to reflect more accurately the quality of the mask
    """
    
    # first assert that the number of instances in the resFile is the same as in the annFile
    with open(resFile, 'r') as f:
        predictions = json.load(f)
    with open(annFile, 'r') as f:
        gt = json.load(f)

    assert len(predictions) == len(gt), "number of instances in the prediction and ground truth should be the same"
    # TODO: implement the iou computation
    pass


def amodal_eval_from_prediction(
        annFile,
        resFile,
        dataset_name, 
        task
    ):
    
    with open(resFile, 'r') as f:
        predictions = json.load(f)

    cocoGt=COCO(annFile)
    cocoDt = cocoGt.loadRes(predictions)

    cocoEval = AmodalCOCOeval(cocoGt, cocoDt, task)
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    amodal_evaluator = AmodalInstanceEvaluator(dataset_name=dataset_name, tasks=task)
    amodal_evaluator.reset()
    res = amodal_evaluator._derive_coco_results(cocoEval, task, class_names=amodal_evaluator._metadata.get("thing_classes"))
    print_csv_format(res, task)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=""
    )
    parser.add_argument("--annFile", required=True, help="JSON file produced by the model")
    parser.add_argument("--resFile", required=True, help="output directory")
    parser.add_argument("--dataset_name", help="name of the dataset", default="d2sa_val")
    parser.add_argument("--task", default="occluding_segm", type=str, help="amodal, visible, occluding, or occluded")
    args = parser.parse_args()
    amodal_eval_from_prediction(
        annFile=args.annFile,
        resFile=args.resFile,
        dataset_name=args.dataset_name,
        task=args.task
    )
