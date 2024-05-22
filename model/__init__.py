from .cnn import RecFieldCNN
from .vit import RecFieldViT
from .fno import FNO2d
from .rfno import RFNO
from .utils import timer

model_dict = {
    'cnn': RecFieldCNN,
    'vit': RecFieldViT,
    'fno': FNO2d,
    'rfno': RFNO
}