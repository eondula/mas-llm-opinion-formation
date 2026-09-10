import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


SENTIMENT_MODEL_NAME = (
    "distilbert-base-uncased-finetuned-sst-2-english"
)


def get_sentiment_device():
    if torch.cuda.is_available():
        return "cuda"

    if torch.backends.mps.is_available():
        return "mps"

    return "cpu"


sentiment_device = get_sentiment_device()

_loaded = False

def load_sentiment():
    global _loaded, sentiment_tokenizer, sentiment_model
    if _loaded:
        return
    print(f"Using sentiment device: {sentiment_device}")
    print(f"Loading sentiment model: {SENTIMENT_MODEL_NAME}")
    
    
    sentiment_tokenizer = AutoTokenizer.from_pretrained(
        SENTIMENT_MODEL_NAME
    )
    
    sentiment_model = AutoModelForSequenceClassification.from_pretrained(
        SENTIMENT_MODEL_NAME
    )
    
    sentiment_model.to(sentiment_device)
    sentiment_model.eval()
    
    _loaded = True


def measure_sentiment(text):
    """
    Measure affective valence of a generated public message.

    Returns:
        p_negative
        p_positive
        sentiment_score = p_positive - p_negative

    sentiment_score ranges from -1 to +1:
        negative -> more negative affect
        positive -> more positive affect

    This is NOT a stance measure.
    """

    load_sentiment()
    inputs = sentiment_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    inputs = {
        key: value.to(sentiment_device)
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        outputs = sentiment_model(**inputs)

    probabilities = torch.softmax(
        outputs.logits,
        dim=-1,
    )[0]

    id2label = sentiment_model.config.id2label

    scores = {}

    for index in range(len(probabilities)):
        label = id2label[index].lower()
        scores[label] = probabilities[index].item()

    p_negative = scores["negative"]
    p_positive = scores["positive"]

    sentiment_score = p_positive - p_negative

    return {
        "p_negative": p_negative,
        "p_positive": p_positive,
        "sentiment_score": sentiment_score,
    }