from collections import OrderedDict

import torch.nn as nn
from transformers import AutoTokenizer

from general.config import cfg

from . import bert_model, clip_model
from .clip_model import CLIPTransformer
from .hfpt_tokenizer import HFPTTokenizer

# from .simple_tokenizer import SimpleTokenizer

# from .hfpt_tokenizer import HFPTTokenizer
# from . import word_utils


def build_bert():
    body = bert_model.BertEncoder()
    model = nn.Sequential(OrderedDict([("body", body)]))
    return model


def build_clip():
    body = clip_model.CLIPTransformer()
    model = nn.Sequential(OrderedDict([("body", body)]))
    return model


def build_lm():
    """returns a language model"""

    LM_ARCH = {"bert-base-uncased": build_bert, "clip": build_clip}
    return LM_ARCH[cfg.MODEL.LANG.BODY]()


def build_tokenizer():
    """returns a language tokenizer"""

    if cfg.MODEL.LANG.TOKENIZER == "CLIP":

        MLM = cfg.MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS

        print("Reuse token 'ðŁĴĳ</w>' (token_id = 49404) for mask token!")
        url = "openai/clip-vit-base-patch32"
        kwargs = {"mask_token": "ðŁĴĳ</w>"} if MLM else {}
        return CLIPTokenizerFast.from_pretrained(url, from_slow=True, **kwargs)

    else:
        return AutoTokenizer.from_pretrained(cfg.MODEL.LANG.TOKENIZER)
