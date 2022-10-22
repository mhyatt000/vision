from . import bert_model, clip_model
from .clip_model import CLIPTransformer
from .hfpt_tokenizer import HFPTTokenizer

# from .simple_tokenizer import SimpleTokenizer

# from .hfpt_tokenizer import HFPTTokenizer
# from . import word_utils


def build_bert(cfg):
    body = bert_model.BertEncoder(cfg)
    model = nn.Sequential(OrderedDict([("body", body)]))
    return model


def build_clip(cfg):
    body = clip_model.CLIPTransformer(cfg)
    model = nn.Sequential(OrderedDict([("body", body)]))
    return model


def build_lm(cfg):
    """returns a language model"""

    LM_ARCH = {"bert-base-uncased": build_bert, "clip": build_clip}
    return LM_ARCH[cfg.MODEL.LANGUAGE_BACKBONE.MODEL_TYPE](cfg)


def build_tokenizer(cfg):
    """returns a language tokenizer"""

    if cfg.MODEL.LANGUAGE_BACKBONE.TOKENIZER_TYPE == "clip":

        MLM = cfg.MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS

        print("Reuse token 'ðŁĴĳ</w>' (token_id = 49404) for mask token!")
        url = "openai/clip-vit-base-patch32"
        kwargs = {"mask_token": "ðŁĴĳ</w>"} if MLM else {}
        return CLIPTokenizerFast.from_pretrained(url, from_slow=True, **kwargs)

    else:
        return AutoTokenizer.from_pretrained(cfg.MODEL.LANGUAGE_BACKBONE.TOKENIZER_TYPE)


"""
def build_tokenizer(tokenizer_name):
    "returns a language tokenizer"

    if tokenizer_name == "clip":
        return SimpleTokenizer()
    elif "hf_" in tokenizer_name:
        return HFPTTokenizer(pt_name=tokenizer_name[3:])
    elif "hfc_" in tokenizer_name:
        return HFPTTokenizer(pt_name=tokenizer_name[4:])
    else:
        raise ValueError("Unknown tokenizer")
"""
