"""
D2SA (D2S Amodal)
register d2sa dataset as a coco-like instance segmentation dataset

"""

import os
from os.path import join
from .coco_amodal import register_aistron_cocolike_instances


D2SA_CATEGORIES = [
    {
        "supercategory": "water",
        "id": 1,
        "name": "adelholzener_alpenquelle_classic_075",
        "color": [231, 16, 52],
    },
    {
        "supercategory": "water",
        "id": 2,
        "name": "adelholzener_alpenquelle_naturell_075",
        "color": [13, 200, 10],
    },
    {
        "supercategory": "apple spritzer",
        "id": 3,
        "name": "adelholzener_classic_bio_apfelschorle_02",
        "color": [163, 111, 2],
    },
    {
        "supercategory": "water",
        "id": 4,
        "name": "adelholzener_classic_naturell_02",
        "color": [135, 240, 87],
    },
    {
        "supercategory": "blue glass",
        "id": 5,
        "name": "adelholzener_gourmet_mineralwasser_02",
        "color": [76, 137, 127],
    },
    {
        "supercategory": "brown glass",
        "id": 6,
        "name": "augustiner_lagerbraeu_hell_05",
        "color": [44, 54, 189],
    },
    {
        "supercategory": "brown glass",
        "id": 7,
        "name": "augustiner_weissbier_05",
        "color": [228, 193, 49],
    },
    {
        "supercategory": "coca cola",
        "id": 8,
        "name": "coca_cola_05",
        "color": [254, 239, 92],
    },
    {
        "supercategory": "coca cola",
        "id": 9,
        "name": "coca_cola_light_05",
        "color": [206, 69, 107],
    },
    {
        "supercategory": "canned soft drink",
        "id": 10,
        "name": "suntory_gokuri_limonade",
        "color": [6, 116, 2],
    },
    {
        "supercategory": "brown glass",
        "id": 11,
        "name": "tegernseer_hell_03",
        "color": [94, 9, 211],
    },
    {
        "supercategory": "packaged cereal bars",
        "id": 12,
        "name": "corny_nussvoll",
        "color": [242, 179, 143],
    },
    {
        "supercategory": "textured packaging",
        "id": 13,
        "name": "corny_nussvoll_single",
        "color": [140, 200, 192],
    },
    {
        "supercategory": "packaged cereal bars",
        "id": 14,
        "name": "corny_schoko_banane",
        "color": [30, 173, 24],
    },
    {
        "supercategory": "textured packaging",
        "id": 15,
        "name": "corny_schoko_banane_single",
        "color": [13, 66, 13],
    },
    {
        "supercategory": "cereals",
        "id": 16,
        "name": "dr_oetker_vitalis_knuspermuesli_klassisch",
        "color": [109, 180, 180],
    },
    {
        "supercategory": "cereals",
        "id": 17,
        "name": "koelln_muesli_fruechte",
        "color": [8, 3, 75],
    },
    {
        "supercategory": "cereals",
        "id": 18,
        "name": "koelln_muesli_schoko",
        "color": [1, 53, 132],
    },
    {
        "supercategory": "textured packaging",
        "id": 19,
        "name": "caona_kakaohaltiges_getraenkepulver",
        "color": [31, 149, 239],
    },
    {
        "supercategory": "textured packaging",
        "id": 20,
        "name": "cocoba_fruehstueckskakao_mit_honig",
        "color": [54, 62, 69],
    },
    {
        "supercategory": "textured packaging",
        "id": 21,
        "name": "cafe_wunderbar_espresso",
        "color": [183, 50, 84],
    },
    {
        "supercategory": "textured packaging",
        "id": 22,
        "name": "douwe_egberts_professional_kaffee_gemahlen",
        "color": [186, 223, 250],
    },
    {
        "supercategory": "textured packaging",
        "id": 23,
        "name": "gepa_bio_caffe_crema",
        "color": [220, 190, 38],
    },
    {
        "supercategory": "textured packaging",
        "id": 24,
        "name": "gepa_italienischer_bio_espresso",
        "color": [90, 203, 60],
    },
    {
        "supercategory": "cardboard tray",
        "id": 25,
        "name": "apple_braeburn_bundle",
        "color": [215, 133, 134],
    },
    {
        "supercategory": "apples",
        "id": 26,
        "name": "apple_golden_delicious",
        "color": [18, 168, 191],
    },
    {
        "supercategory": "apples",
        "id": 27,
        "name": "apple_granny_smith",
        "color": [12, 117, 5],
    },
    {
        "supercategory": "apples",
        "id": 28,
        "name": "apple_roter_boskoop",
        "color": [108, 255, 107],
    },
    {"supercategory": "avocados", "id": 29, "name": "avocado", "color": [243, 15, 152]},
    {
        "supercategory": "bananas",
        "id": 30,
        "name": "banana_bundle",
        "color": [242, 240, 237],
    },
    {
        "supercategory": "bananas",
        "id": 31,
        "name": "banana_single",
        "color": [116, 139, 187],
    },
    {"supercategory": "nets", "id": 32, "name": "clementine", "color": [98, 246, 156]},
    {
        "supercategory": "orange citrus",
        "id": 33,
        "name": "clementine_single",
        "color": [54, 229, 93],
    },
    {
        "supercategory": "plastic tray",
        "id": 34,
        "name": "grapes_green_sugraone_seedless",
        "color": [19, 173, 143],
    },
    {
        "supercategory": "plastic tray",
        "id": 35,
        "name": "grapes_sweet_celebration_seedless",
        "color": [219, 206, 154],
    },
    {"supercategory": "kiwis", "id": 36, "name": "kiwi", "color": [241, 40, 90]},
    {
        "supercategory": "orange citrus",
        "id": 37,
        "name": "orange_single",
        "color": [26, 20, 186],
    },
    {"supercategory": "nets", "id": 38, "name": "oranges", "color": [7, 91, 69]},
    {"supercategory": "pears", "id": 39, "name": "pear", "color": [185, 49, 12]},
    {
        "supercategory": "pasta",
        "id": 40,
        "name": "pasta_reggia_elicoidali",
        "color": [184, 151, 172],
    },
    {
        "supercategory": "pasta",
        "id": 41,
        "name": "pasta_reggia_fusilli",
        "color": [249, 152, 237],
    },
    {
        "supercategory": "pasta",
        "id": 42,
        "name": "pasta_reggia_spaghetti",
        "color": [146, 40, 232],
    },
    {
        "supercategory": "board eraser",
        "id": 43,
        "name": "franken_tafelreiniger",
        "color": [41, 64, 192],
    },
    {
        "supercategory": "ink cartridge",
        "id": 44,
        "name": "pelikan_tintenpatrone_canon",
        "color": [109, 79, 240],
    },
    {
        "supercategory": "tea",
        "id": 45,
        "name": "ethiquable_gruener_tee_ceylon",
        "color": [178, 105, 255],
    },
    {
        "supercategory": "tea",
        "id": 46,
        "name": "gepa_bio_und_fair_fencheltee",
        "color": [13, 86, 160],
    },
    {
        "supercategory": "tea",
        "id": 47,
        "name": "gepa_bio_und_fair_kamillentee",
        "color": [228, 164, 196],
    },
    {
        "supercategory": "tea",
        "id": 48,
        "name": "gepa_bio_und_fair_kraeuterteemischung",
        "color": [35, 76, 90],
    },
    {
        "supercategory": "tea",
        "id": 49,
        "name": "gepa_bio_und_fair_pfefferminztee",
        "color": [85, 15, 62],
    },
    {
        "supercategory": "tea",
        "id": 50,
        "name": "gepa_bio_und_fair_rooibostee",
        "color": [57, 34, 164],
    },
    {
        "supercategory": "tea",
        "id": 51,
        "name": "kilimanjaro_tea_earl_grey",
        "color": [134, 38, 163],
    },
    {
        "supercategory": "cucumbers",
        "id": 52,
        "name": "cucumber",
        "color": [62, 154, 162],
    },
    {"supercategory": "carrots", "id": 53, "name": "carrot", "color": [164, 15, 99]},
    {
        "supercategory": "plastic tray",
        "id": 54,
        "name": "feldsalat",
        "color": [202, 255, 54],
    },
    {"supercategory": "salad", "id": 55, "name": "lettuce", "color": [118, 190, 216]},
    {
        "supercategory": "cardboard tray",
        "id": 56,
        "name": "rispentomaten",
        "color": [83, 145, 7],
    },
    {
        "supercategory": "cardboard tray",
        "id": 57,
        "name": "roma_rispentomaten",
        "color": [231, 229, 0],
    },
    {
        "supercategory": "plastic tray",
        "id": 58,
        "name": "rucola",
        "color": [121, 208, 194],
    },
    {
        "supercategory": "foil",
        "id": 59,
        "name": "salad_iceberg",
        "color": [109, 100, 102],
    },
    {"supercategory": "zucchinis", "id": 60, "name": "zucchini", "color": [0, 249, 58]},
]


def _get_d2sa_instances_meta(cat_list):
    thing_ids = [k["id"] for k in cat_list]
    thing_colors = [k["color"] for k in cat_list]
    # assert len(thing_ids) == 7, len(thing_ids)
    # Mapping from the incontiguous category id to an id in [0, 6]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in cat_list]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret


def register_d2sa(root):
    register_aistron_cocolike_instances(
        "d2sa_train",
        _get_d2sa_instances_meta(D2SA_CATEGORIES),
        join(root, "D2SA/d2s_amodal_annotations_v1/D2S_amodal_training_rot0_aistron.json"),
        join(root, "D2SA/d2s_amodal_images_v1/images/"),
    )

    register_aistron_cocolike_instances(
        "d2sa_train_aug",
        _get_d2sa_instances_meta(D2SA_CATEGORIES),
        join(root, "D2SA/d2s_amodal_annotations_v1/D2S_amodal_augmented_aistron.json"),
        join(root, "D2SA/d2s_amodal_images_v1/images/"),
    )

    register_aistron_cocolike_instances(
        "d2sa_val",
        _get_d2sa_instances_meta(D2SA_CATEGORIES),
        join(root, "D2SA/d2s_amodal_annotations_v1/D2S_amodal_validation_aistron.json"),
        join(root, "D2SA/d2s_amodal_images_v1/images/"),
    )


_root = os.getenv("AISTRON_DATASETS", "datasets")
register_d2sa(_root)
