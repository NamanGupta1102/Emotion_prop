import pickle
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from transformers import pipeline

if __name__ == "__main__":
    classifier = pipeline("sentiment-analysis", model="michellejieli/emotion_text_classifier", custom_cache_dir="./cache")
    # Your other code here...
    print(classifier("You are good "))
    with open('model.pkl', 'wb') as f:
        pickle.dump(classifier, f)
    
    print("DEBUGGG:  downloaded model")
    with open('model.pkl', 'rb') as f:
        data_loaded = pickle.load(f)
    
    print(classifier("You rock "))
    