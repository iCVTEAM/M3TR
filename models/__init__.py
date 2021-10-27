from .M3TR import M3TR
import torchvision
from .vit_utils.multi_modal_transformer import *
from .vit_utils.utils import *

model_dict = {'M3TR': M3TR}


def get_model(num_classes, args):
    vit = vit_base_patch16_384(img_size=448, pretrained=True)
    res101 = torchvision.models.resnet101(pretrained=True)
    model = model_dict[args.model_name](vit, res101, num_classes)
    return model
