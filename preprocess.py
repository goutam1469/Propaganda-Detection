import pandas as pd


#Extract <BOS> and <EOS> spans
def extract_bos_eos(text):
    if '<BOS>' in text and '<EOS>' in text:
        _, rest = text.split('<BOS>', 1)
        span, _ = rest.split('<EOS>', 1)
        return span.strip()
    return text.strip()
    

