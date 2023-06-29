from .dataset_mapper import AmodalDatasetMapper

from .datasets import (
    register_aistron_cocolike_instances,
    load_aistron_cocolike_json,
    KINS2020_CATEGORIES,
    KINS_CATEGORIES,
    D2SA_CATEGORIES,
    COCOA_CATEGORIES,
    COCOA_cls_CATEGORIES,
)

# register built-in datasets
from . import datasets

