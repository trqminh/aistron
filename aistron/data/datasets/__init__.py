from . import register_kins
from . import register_d2sa
from . import register_cocoa
from . import register_coco

from .coco_amodal import (
    load_aistron_cocolike_json,
    register_aistron_cocolike_instances
)

from .register_kins import KINS2020_CATEGORIES, KINS_CATEGORIES
from .register_d2sa import D2SA_CATEGORIES
from .register_cocoa import COCOA_CATEGORIES, COCOA_cls_CATEGORIES
from .register_coco import COCO_CATEGORIES

