import re
import pandas as pd 
from sentence_transformers import models, SentenceTransformer

class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class EmbedderModel(object,metaclass=Singleton):
    def __init__(self) -> None:
        self.model_name_or_path = 'm3hrdadfi/bert-fa-base-uncased-wikinli-mean-tokens'
        self.model = self.load_st_model()

    def load_st_model(self):
        word_embedding_model = models.Transformer(self.model_name_or_path)
        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=True,
            pooling_mode_cls_token=False,
            pooling_mode_max_tokens=False)
        
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        return model



def clean_caption(caption):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    caption = emoji_pattern.sub(r' ', caption)
    # taking the whole caption and remove extra characters and returning split caption(cleaned_caption)
    if pd.isna(caption):
        return []
    caption = caption.replace('آ', 'ا')
    caption = caption.replace(':', ' ')
    caption = caption.replace('=', ' ')
    caption = caption.replace('.',' ')
    caption = caption.lower()
    caption = re.sub('#(_*[آ-ی0-9a-z_]*_*\s*)', '', caption)
    caption = re.sub('[^A-Za-z0-9آ-ی #@\n/_.]+', '', caption)
    caption = " " + caption + " "

    punc='!"”“#$%&\'()*+,./:;<=>?@[\\]^_`{|}~-0123456789۰۱۲۳۴۵۶۷۸۹abcdefghijklmnopqrstuvwxyz'
    for ch in punc:
        caption = caption.replace(ch, '')
    caption=caption.replace('‌','')
    return caption.strip()


def run(prompt):
    # Load the Sentence-Transformer
    embedder = EmbedderModel().model
    data = []
    for d in prompt:
        data.append(clean_caption(d))
    corpus_embeddings = embedder.encode(data, show_progress_bar=True)
    return corpus_embeddings