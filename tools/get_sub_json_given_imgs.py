import json
import os

#json_path = '../data/datasets/KINS/annotations/update_test_2020_aistron.json'
#selected_imgs_path = '../data/datasets/KINS/selected_images'
#selected_imgs = os.listdir(selected_imgs_path)
#selected_imgs = ['001002.png']

json_path = '../data/datasets/D2SA/d2s_amodal_annotations_v1/D2S_amodal_validation_aistron.json'
selected_imgs = ['D2S_025813.jpg']

def main():
    with open(json_path, 'r') as f:
        data = json.load(f)


    new_data = {'images': [], 'annotations': []}
    selected_img_ids = []
    for img in data['images']:
        if img['file_name'] in selected_imgs:
            new_data['images'].append(img)
            selected_img_ids.append(img['id'])

    for ann in data['annotations']:
        if ann['image_id'] in selected_img_ids:
            new_data['annotations'].append(ann)

    new_data['categories'] = data['categories']
    new_data['info'] = data['info']
    new_data['licenses'] = data['licenses']

    # with open('../data/datasets/KINS/annotations/update_test_2020_aistron_selected.json', 'w') as f:
    #     json.dump(new_data, f)

    with open('../data/datasets/D2SA/d2s_amodal_annotations_v1/D2S_amodal_validation_aistron_selected.json', 'w') as f:
        json.dump(new_data, f)

main()
