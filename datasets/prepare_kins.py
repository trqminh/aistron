import sys, os, json
from detectron2.utils.logger import setup_logger
from prepare_amodal_instances import load_coco_json, convert_to_aistron_cocolike_dict



def convert_kins2020_to_aistron_cocolike_dict(kins2020_ann_file):
    kins2020_ann_dict = json.load(open(kins2020_ann_file, 'r'))
    for ann in kins2020_ann_dict['annotations']:
        ann['amodal_segm'] = ann.pop('a_segm')
        ann['amodal_bbox'] = ann.pop('a_bbox')
        ann['visible_segm'] = ann.pop('i_segm')
        ann['visible_bbox'] = ann.pop('i_bbox')
    
    tmp_json_fname = os.path.dirname(kins2020_ann_file)
    tmp_json_fname = os.path.join(tmp_json_fname, 'tmp.json')
    with open(tmp_json_fname, 'w') as f:
        json.dump(kins2020_ann_dict, f)

    return tmp_json_fname, kins2020_ann_dict['categories']



if __name__ == "__main__":
    """
    """  
    logger = setup_logger(name=__name__)

    tmp_json_fname, categories = convert_kins2020_to_aistron_cocolike_dict(sys.argv[1])
    dicts = load_coco_json(tmp_json_fname, sys.argv[2])
    
    logger.info("Done loading {} samples.".format(len(dicts)))
    coco_dict = convert_to_aistron_cocolike_dict(dicts, categories)

    output_file = os.path.splitext(sys.argv[1].split('/')[-1])[0] + '_aistron.json'
    ann_path = os.path.dirname(sys.argv[1])
    output_file = os.path.join(ann_path, output_file)
    import pdb;pdb.set_trace()

    with open(output_file, "w") as json_file:
        logger.info(f"Caching annotations in new format: {output_file}")
        json.dump(coco_dict, json_file)

