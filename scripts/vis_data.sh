export AISTRON_DATASETS=../data/datasets/

config_file=configs/Base-RCNN-FPN-KINS2020.yaml
output_dir=../data/outtest/aistron_vis_gt_test_kins2020/
mkdir ${output_dir}


python visualize_data.py --config-file ${config_file} \
    --output-dir ${output_dir} \
    --source "annotation" \
    --split "test" \
    --option "visible"
