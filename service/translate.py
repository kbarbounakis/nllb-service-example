from typing import List
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from collections import namedtuple
from jsonschema import validate

TranslateRequestSchema = {
    'type': 'object',
    'properties': {
        'text': {'type': 'string'},
    },
    'required': ['text'],
}

TranslateRequestBase = namedtuple('TranslateRequest', [ 'text' ])

TranslateResponse = namedtuple('TranslateResponse', [ 'result' ])

class TranslateRequest(TranslateRequestBase):

    @staticmethod
    def validate(instance: dict):
        return validate(instance=instance, schema=TranslateRequestSchema)


tokenizer = AutoTokenizer.from_pretrained(
    "facebook/nllb-200-distilled-600M", src_lang="ell_Latn"
)
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")

def translate(text: str) -> str:
    inputs = tokenizer(text, return_tensors="pt")
    translated_tokens = model.generate(
        **inputs, forced_bos_token_id=tokenizer.lang_code_to_id["eng_Latn"], max_length=30
    )
    return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]