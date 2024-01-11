import sys, os, json
from detectron2.utils.logger import setup_logger
from prepare_amodal_instances import process_occluder_gt_and_misc, convert_to_aistron_cocolike_dict



def convert_cocoa_json_to_aistron_format(cocoa_ann_file):
    cocoa_ann_dict = json.load(open(cocoa_ann_file, 'r'))
    for ann in cocoa_ann_dict['annotations']:
        ann['visible_segm'] = ann['amodal_segm'] = ann.pop('segmentation')
        ann['amodal_bbox'] = ann['visible_bbox'] = ann.pop('bbox')

    tmp_json_fname = os.path.dirname(cocoa_ann_file)
    tmp_json_fname = os.path.join(tmp_json_fname, 'tmp.json')
    with open(tmp_json_fname, 'w') as f:
        json.dump(cocoa_ann_dict, f)

    return tmp_json_fname, cocoa_ann_dict['categories']



if __name__ == "__main__":
    """
    """  
    logger = setup_logger(name=__name__)

    json_ann_path = sys.argv[1]
    tmp_json_fname, categories = convert_cocoa_json_to_aistron_format(json_ann_path)
    # TODO: make these steps less redundant
    # current preprocess steps utilize BCNet preprocess code
    d2_dicts = process_occluder_gt_and_misc(tmp_json_fname) # the processed dict is in det2 obj
    coco_dict = convert_to_aistron_cocolike_dict(d2_dicts, categories) # convert det2 obj to cocolike dict

    output_file = os.path.splitext(sys.argv[1].split('/')[-1])[0] + '_aistron.json'
    ann_path = os.path.dirname(sys.argv[1])
    output_file = os.path.join(ann_path, output_file)

    with open(output_file, "w") as json_file:
        logger.info(f"Caching annotations in new format: {output_file}")
        json.dump(coco_dict, json_file)
