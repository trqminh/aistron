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
    "color": [
      0,
      255,
      140
    ],
  },
  {
    "supercategory": "water",
    "id": 2,
    "name": "adelholzener_alpenquelle_naturell_075",
    "color": [
      255,
      140,
      0
    ],
  },
  {
    "supercategory": "apple spritzer",
    "id": 3,
    "name": "adelholzener_classic_bio_apfelschorle_02",
    "color": [
      255,
      140,
      0
    ],
  },
  {
    "supercategory": "water",
    "id": 4,
    "name": "adelholzener_classic_naturell_02",
    "color": [
      255,
      255,
      0
    ],
  },
  {
    "supercategory": "blue glass",
    "id": 5,
    "name": "adelholzener_gourmet_mineralwasser_02",
    "color": [
      255,
      0,
      0
    ],
  },
  {
    "supercategory": "brown glass",
    "id": 6,
    "name": "augustiner_lagerbraeu_hell_05",
    "color": [
      255,
      140,
      0
    ],
  },
  {
    "supercategory": "brown glass",
    "id": 7,
    "name": "augustiner_weissbier_05",
    "color": [
      255,
      0,
      140
    ],
  },
  {
    "supercategory": "coca cola",
    "id": 8,
    "name": "coca_cola_05",
    "color": [
      0,
      0,
      255
    ],
  },
  {
    "supercategory": "coca cola",
    "id": 9,
    "name": "coca_cola_light_05",
    "color": [
      57,
      197,
      187
    ],
  },
  {
    "supercategory": "canned soft drink",
    "id": 10,
    "name": "suntory_gokuri_limonade",
    "color": [
      0,
      255,
      140
    ],
  },
  {
    "supercategory": "brown glass",
    "id": 11,
    "name": "tegernseer_hell_03",
    "color": [
      57,
      197,
      187
    ],
  },
  {
    "supercategory": "packaged cereal bars",
    "id": 12,
    "name": "corny_nussvoll",
    "color": [
      0,
      255,
      0
    ],
  },
  {
    "supercategory": "textured packaging",
    "id": 13,
    "name": "corny_nussvoll_single",
    "color": [
      255,
      255,
      0
    ],
  },
  {
    "supercategory": "packaged cereal bars",
    "id": 14,
    "name": "corny_schoko_banane",
    "color": [
      0,
      255,
      0
    ],
  },
  {
    "supercategory": "textured packaging",
    "id": 15,
    "name": "corny_schoko_banane_single",
    "color": [
      0,
      255,
      255
    ],
  },
  {
    "supercategory": "cereals",
    "id": 16,
    "name": "dr_oetker_vitalis_knuspermuesli_klassisch",
    "color": [
      0,
      255,
      0
    ],
  },
  {
    "supercategory": "cereals",
    "id": 17,
    "name": "koelln_muesli_fruechte",
    "color": [
      180,
      0,
      255
    ],
  },
  {
    "supercategory": "cereals",
    "id": 18,
    "name": "koelln_muesli_schoko",
    "color": [
      255,
      0,
      0
    ],
  },
  {
    "supercategory": "textured packaging",
    "id": 19,
    "name": "caona_kakaohaltiges_getraenkepulver",
    "color": [
      255,
      0,
      140
    ],
  },
  {
    "supercategory": "textured packaging",
    "id": 20,
    "name": "cocoba_fruehstueckskakao_mit_honig",
    "color": [
      140,
      255,
      0
    ],
  },
  {
    "supercategory": "textured packaging",
    "id": 21,
    "name": "cafe_wunderbar_espresso",
    "color": [
      255,
      140,
      0
    ],
  },
  {
    "supercategory": "textured packaging",
    "id": 22,
    "name": "douwe_egberts_professional_kaffee_gemahlen",
    "color": [
      255,
      0,
      140
    ],
  },
  {
    "supercategory": "textured packaging",
    "id": 23,
    "name": "gepa_bio_caffe_crema",
    "color": [
      57,
      197,
      187
    ],
  },
  {
    "supercategory": "textured packaging",
    "id": 24,
    "name": "gepa_italienischer_bio_espresso",
    "color": [
      255,
      0,
      255
    ],
  },
  {
    "supercategory": "cardboard tray",
    "id": 25,
    "name": "apple_braeburn_bundle",
    "color": [
      180,
      0,
      255
    ],
  },
  {
    "supercategory": "apples",
    "id": 26,
    "name": "apple_golden_delicious",
    "color": [
      255,
      0,
      0
    ],
  },
  {
    "supercategory": "apples",
    "id": 27,
    "name": "apple_granny_smith",
    "color": [
      255,
      0,
      0
    ],
  },
  {
    "supercategory": "apples",
    "id": 28,
    "name": "apple_roter_boskoop",
    "color": [
      180,
      0,
      255
    ],
  },
  {
    "supercategory": "avocados",
    "id": 29,
    "name": "avocado",
    "color": [
      255,
      0,
      255
    ],
  },
  {
    "supercategory": "bananas",
    "id": 30,
    "name": "banana_bundle",
    "color": [
      255,
      0,
      255
    ],
  },
  {
    "supercategory": "bananas",
    "id": 31,
    "name": "banana_single",
    "color": [
      0,
      255,
      0
    ],
  },
  {
    "supercategory": "nets",
    "id": 32,
    "name": "clementine",
    "color": [
      0,
      0,
      255
    ],
  },
  {
    "supercategory": "orange citrus",
    "id": 33,
    "name": "clementine_single",
    "color": [
      255,
      0,
      140
    ],
  },
  {
    "supercategory": "plastic tray",
    "id": 34,
    "name": "grapes_green_sugraone_seedless",
    "color": [
      180,
      0,
      255
    ],
  },
  {
    "supercategory": "plastic tray",
    "id": 35,
    "name": "grapes_sweet_celebration_seedless",
    "color": [
      0,
      0,
      255
    ],
  },
  {
    "supercategory": "kiwis",
    "id": 36,
    "name": "kiwi",
    "color": [
      255,
      140,
      0
    ],
  },
  {
    "supercategory": "orange citrus",
    "id": 37,
    "name": "orange_single",
    "color": [
      255,
      255,
      0
    ],
  },
  {
    "supercategory": "nets",
    "id": 38,
    "name": "oranges",
    "color": [
      255,
      0,
      140
    ],
  },
  {
    "supercategory": "pears",
    "id": 39,
    "name": "pear",
    "color": [
      57,
      197,
      187
    ],
  },
  {
    "supercategory": "pasta",
    "id": 40,
    "name": "pasta_reggia_elicoidali",
    "color": [
      255,
      140,
      0
    ],
  },
  {
    "supercategory": "pasta",
    "id": 41,
    "name": "pasta_reggia_fusilli",
    "color": [
      255,
      255,
      0
    ],
  },
  {
    "supercategory": "pasta",
    "id": 42,
    "name": "pasta_reggia_spaghetti",
    "color": [
      0,
      255,
      255
    ],
  },
  {
    "supercategory": "board eraser",
    "id": 43,
    "name": "franken_tafelreiniger",
    "color": [
      0,
      255,
      140
    ],
  },
  {
    "supercategory": "ink cartridge",
    "id": 44,
    "name": "pelikan_tintenpatrone_canon",
    "color": [
      0,
      255,
      140
    ],
  },
  {
    "supercategory": "tea",
    "id": 45,
    "name": "ethiquable_gruener_tee_ceylon",
    "color": [
      180,
      0,
      255
    ],
  },
  {
    "supercategory": "tea",
    "id": 46,
    "name": "gepa_bio_und_fair_fencheltee",
    "color": [
      0,
      255,
      255
    ],
  },
  {
    "supercategory": "tea",
    "id": 47,
    "name": "gepa_bio_und_fair_kamillentee",
    "color": [
      0,
      255,
      140
    ],
  },
  {
    "supercategory": "tea",
    "id": 48,
    "name": "gepa_bio_und_fair_kraeuterteemischung",
    "color": [
      255,
      255,
      0
    ],
  },
  {
    "supercategory": "tea",
    "id": 49,
    "name": "gepa_bio_und_fair_pfefferminztee",
    "color": [
      255,
      0,
      255
    ],
  },
  {
    "supercategory": "tea",
    "id": 50,
    "name": "gepa_bio_und_fair_rooibostee",
    "color": [
      57,
      197,
      187
    ],
  },
  {
    "supercategory": "tea",
    "id": 51,
    "name": "kilimanjaro_tea_earl_grey",
    "color": [
      57,
      197,
      187
    ],
  },
  {
    "supercategory": "cucumbers",
    "id": 52,
    "name": "cucumber",
    "color": [
      57,
      197,
      187
    ],
  },
  {
    "supercategory": "carrots",
    "id": 53,
    "name": "carrot",
    "color": [
      255,
      0,
      140
    ],
  },
  {
    "supercategory": "plastic tray",
    "id": 54,
    "name": "feldsalat",
    "color": [
      255,
      0,
      255
    ],
  },
  {
    "supercategory": "salad",
    "id": 55,
    "name": "lettuce",
    "color": [
      180,
      0,
      255
    ],
  },
  {
    "supercategory": "cardboard tray",
    "id": 56,
    "name": "rispentomaten",
    "color": [
      255,
      0,
      255
    ],
  },
  {
    "supercategory": "cardboard tray",
    "id": 57,
    "name": "roma_rispentomaten",
    "color": [
      0,
      255,
      0
    ],
  },
  {
    "supercategory": "plastic tray",
    "id": 58,
    "name": "rucola",
    "color": [
      0,
      255,
      140
    ],
  },
  {
    "supercategory": "foil",
    "id": 59,
    "name": "salad_iceberg",
    "color": [
      140,
      255,
      0
    ],
  },
  {
    "supercategory": "zucchinis",
    "id": 60,
    "name": "zucchini",
    "color": [
      255,
      255,
      0
    ],
  }
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

    register_aistron_cocolike_instances(
        "d2sa_selected",
        _get_d2sa_instances_meta(D2SA_CATEGORIES),
        join(root, "D2SA/d2s_amodal_annotations_v1/D2S_amodal_validation_aistron.json"),
        join(root, "D2SA/selected_images/"),
    )


_root = os.getenv("AISTRON_DATASETS", "datasets")
register_d2sa(_root)
