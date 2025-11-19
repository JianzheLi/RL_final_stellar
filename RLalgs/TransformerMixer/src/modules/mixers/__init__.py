from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
from modules.mixers.transformer_qmix import TransformerQMixer

REGISTRY = {}

REGISTRY["vdn"] = VDNMixer
REGISTRY["qmix"] = QMixer
REGISTRY["transformer_qmix"] = TransformerQMixer

