import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import re
import string
import joblib
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

class DisasterTweetPredictor:
    """
    Real-Time Scalable Disaster Tweet Predictor Engine.
    Supports both Hugging Face DistilBERT Transformer and Soft Voting Ensemble engines.
    Processes raw tweets, extracts metaphor/slang/sentiment features,
    and returns real-time disaster predictions with confidence scores.
    """
    def __init__(self, model_dir='processed_data', default_model='distilbert'):
        self.default_model = default_model
        
        # Dynamically locate model directory to handle root, nested, or case variations on Linux
        self.model_dir = self._resolve_model_dir(model_dir)
        
        # 1. Initialize Hugging Face DistilBERT Transformer
        self.distilbert_path = os.path.join(self.model_dir, "distilbert_disaster")
        self.distilbert_model = None
        self.distilbert_tokenizer = None
        self.distilbert_threshold = 0.40  # Peak F1 threshold for DistilBERT
        
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch
            if os.path.exists(self.distilbert_path):
                self.distilbert_tokenizer = AutoTokenizer.from_pretrained(self.distilbert_path)
                self.distilbert_model = AutoModelForSequenceClassification.from_pretrained(self.distilbert_path)
                self.distilbert_model.eval()
        except Exception as e:
            print(f"Notice: DistilBERT engine unavailable ({e}). Falling back to Ensemble.")
            self.default_model = 'ensemble'

        # 2. Initialize Soft Voting Ensemble
        model_path = os.path.join(self.model_dir, 'disaster_ensemble_model.pkl')
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Cannot find 'disaster_ensemble_model.pkl'. Looked in '{self.model_dir}'. "
                f"Please ensure the 5 .pkl files are uploaded to your GitHub repository."
            )
        self.model = joblib.load(model_path)
        
        # Ensure cross-version compatibility for scikit-learn
        for est in getattr(self.model, 'estimators_', []):
            if hasattr(est, '__class__') and est.__class__.__name__ == 'LogisticRegression':
                if not hasattr(est, 'multi_class'):
                    setattr(est, 'multi_class', 'auto')
        for name, est in getattr(self.model, 'named_estimators_', {}).items():
            if hasattr(est, '__class__') and est.__class__.__name__ == 'LogisticRegression':
                if not hasattr(est, 'multi_class'):
                    setattr(est, 'multi_class', 'auto')

        self.tfidf = joblib.load(os.path.join(self.model_dir, 'tfidf_vectorizer_adv.pkl'))
        self.scaler = joblib.load(os.path.join(self.model_dir, 'scaler_adv.pkl'))
        self.kw_risk_dict = joblib.load(os.path.join(self.model_dir, 'keyword_risk_dict.pkl'))
        self.config = joblib.load(os.path.join(self.model_dir, 'model_config.pkl'))
        self.best_threshold = self.config.get('best_threshold', 0.46)

        self.sia = SentimentIntensityAnalyzer()
        self.stop_words = set(ENGLISH_STOP_WORDS)
        self.global_mean = 0.4297
        
        self.SLANG_WORDS = set([
            'lol', 'loool', 'lmao', 'lmfao', 'haha', 'hahaha', 'hah', 'omg',
            'rt', 'im', 'gonna', 'wanna', 'gotta', 'fuck', 'shit', 'cool', 'cooool'
        ])
        
        self.METAPHORICAL_KEYWORDS = set([
            'aftershock', 'body bags', 'body bag', 'ruin', 'blazing', 'electrocute',
            'screaming', 'traumatised', 'blew up', 'panicking', 'blight', 'wrecked', 'explode', 'panic'
        ])
        
        self.DISASTER_URGENCY_WORDS = set([
            'fire', 'wildfire', 'earthquake', 'flood', 'flooding', 'tornado', 'storm',
            'tsunami', 'bomb', 'bombing', 'debris', 'derailment', 'evacuation', 'evacuated',
            'killed', 'dead', 'fatalities', 'casualty', 'disaster', 'emergency', 'spill', 'outbreak'
        ])

    def _resolve_model_dir(self, model_dir):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        cwd = os.getcwd()
        target = 'disaster_ensemble_model.pkl'

        # Candidate locations where files might be stored
        candidates = [
            model_dir,
            os.path.join(script_dir, model_dir),
            os.path.join(cwd, model_dir),
            os.path.join(script_dir, 'processed_data'),
            os.path.join(script_dir, 'Processed_data'),
            os.path.join(cwd, 'processed_data'),
            os.path.join(cwd, 'Processed_data'),
            script_dir,   # If files were uploaded to repository root
            cwd           # If files are in current working directory
        ]

        for path in candidates:
            if os.path.isfile(os.path.join(path, target)):
                return path

        # Search recursively in script directory and working directory
        for search_root in [script_dir, cwd]:
            for root, dirs, files in os.walk(search_root):
                if target in files:
                    return root

        return model_dir

    def _clean_text(self, text):
        if not isinstance(text, str):
            return ""
        txt = re.sub(r'https?://\S+|www\.\S+', '', text)
        txt = re.sub(r'<.*?>', '', txt)
        txt = re.sub(r'[^a-zA-Z0-9\s]', '', txt).lower()
        return re.sub(r'\s+', ' ', txt).strip()

    def _extract_features(self, raw_tweets, keywords=None):
        clean_texts = [self._clean_text(t) for t in raw_tweets]
        tokens_list = [[w for w in t.split() if w not in self.stop_words and len(w) > 1] for t in clean_texts]
        joined_texts = [" ".join(toks) for toks in tokens_list]
        
        # TF-IDF matrix
        X_text = self.tfidf.transform(joined_texts).toarray()
        
        meta_features = []
        for i, text in enumerate(raw_tweets):
            clean_txt = clean_texts[i]
            kw = keywords[i] if (keywords and i < len(keywords)) else None
            
            kw_risk = self.kw_risk_dict.get(kw, self.global_mean) if kw else self.global_mean
            is_meta = 1 if (kw and kw in self.METAPHORICAL_KEYWORDS) else 0
            slang_cnt = sum(1 for w in clean_txt.split() if w in self.SLANG_WORDS)
            urgency_cnt = sum(1 for w in clean_txt.split() if w in self.DISASTER_URGENCY_WORDS)
            
            char_len = len(text)
            word_cnt = len(text.split())
            avg_word_len = char_len / (word_cnt + 1e-5)
            url_cnt = len(re.findall(r'https?://\S+|www\.\S+', text))
            hashtag_cnt = len(re.findall(r'#\w+', text))
            mention_cnt = len(re.findall(r'@\w+', text))
            punc_cnt = sum(1 for c in text if c in string.punctuation)
            upper_cnt = sum(1 for c in text if c.isupper())
            
            sent_scores = self.sia.polarity_scores(text)
            sent_comp = sent_scores['compound']
            sent_neg = sent_scores['neg']
            
            detected_slang = [w for w in clean_txt.split() if w in self.SLANG_WORDS]
            detected_urgency = [w for w in clean_txt.split() if w in self.DISASTER_URGENCY_WORDS]
            
            row_meta = [
                kw_risk, is_meta, slang_cnt, urgency_cnt,
                char_len, word_cnt, url_cnt, hashtag_cnt, mention_cnt,
                punc_cnt, upper_cnt, sent_comp, sent_neg
            ]
            meta_features.append(row_meta)
            
        X_meta_scaled = self.scaler.transform(np.array(meta_features))
        return np.hstack([X_text, X_meta_scaled]), meta_features

    def predict(self, raw_tweets, keywords=None, threshold=None, model_type=None):
        if isinstance(raw_tweets, str):
            raw_tweets = [raw_tweets]
            if keywords and isinstance(keywords, str):
                keywords = [keywords]
                
        engine = model_type if model_type is not None else self.default_model
        
        # Determine threshold
        if threshold is not None:
            thresh = threshold
        elif engine == 'distilbert':
            thresh = self.distilbert_threshold
        else:
            thresh = self.best_threshold

        # Feature extraction for metadata
        _, meta_features = self._extract_features(raw_tweets, keywords)

        # Generate probabilities based on chosen engine
        if engine == 'distilbert' and self.distilbert_model is not None and self.distilbert_tokenizer is not None:
            import torch
            inputs = self.distilbert_tokenizer(raw_tweets, padding=True, truncation=True, max_length=128, return_tensors="pt")
            with torch.no_grad():
                logits = self.distilbert_model(**inputs).logits
                probs = torch.softmax(logits, dim=1)[:, 1].numpy()
        else:
            X, _ = self._extract_features(raw_tweets, keywords)
            probs = self.model.predict_proba(X)[:, 1]

        preds = (probs >= thresh).astype(int)
        
        results = []
        for i in range(len(raw_tweets)):
            txt = raw_tweets[i]
            meta = meta_features[i]
            clean_t = self._clean_text(txt)
            slang_found = [w for w in clean_t.split() if w in self.SLANG_WORDS]
            urgency_found = [w for w in clean_t.split() if w in self.DISASTER_URGENCY_WORDS]
            
            results.append({
                'tweet': txt,
                'engine': engine,
                'is_disaster': bool(preds[i]),
                'disaster_probability': float(round(probs[i], 4)),
                'threshold_used': float(thresh),
                'label': 'REAL DISASTER [ALERT]' if preds[i] == 1 else 'NON-DISASTER / INFORMAL [CLEAR]',
                'sentiment_compound': float(round(meta[11], 3)),
                'sentiment_neg': float(round(meta[12], 3)),
                'urgency_words': urgency_found,
                'slang_words': slang_found,
                'char_length': int(meta[4]),
                'word_count': int(meta[5]),
                'url_count': int(meta[6]),
                'hashtag_count': int(meta[7]),
                'mention_count': int(meta[8])
            })
        return results

if __name__ == '__main__':
    predictor = DisasterTweetPredictor()
    test_samples = [
        "13,000 people receive #wildfires evacuation orders in California",
        "Haha South Tampa is getting flooded hah- WAIT A SECOND I LIVE IN SOUTH TAMPA WHAT AM I GONNA DO FVCK #flooding",
        "My math test was a total disaster today, I got body bagged by question 4 loool",
        "Heavy rain causes flash flooding of streets in Manitou, Colorado Springs areas!"
    ]
    predictions = predictor.predict(test_samples)
    print("\n--- Real-Time Disaster Tweet Predictor Output ---")
    for res in predictions:
        print(f"\nTweet  : {res['tweet']}")
        print(f"Prob   : {res['disaster_probability']:.4f}")
        print(f"Result : {res['label']}")
