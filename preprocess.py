# @title
# from nltk.corpus import stopwords
import pandas as pd
from math import floor
import nltk
# nltk.download('stopwords')
# nltk.download('wordnet')
# from nltk.corpus import stopwords
# stop_words = set(stopwords.words('english')) 
import numpy as np

import regex as re
from nltk.stem import WordNetLemmatizer

def lemmatization(text):
    lemmatizer= WordNetLemmatizer()

    text = text.split()

    text=[lemmatizer.lemmatize(y) for y in text]

    return " ".join(text)

# def remove_stop_words(text):

#     Text=[i for i in str(text).split() if i not in stop_words]
#     return " ".join(Text)

def Removing_numbers(text):
    text=''.join([i for i in text if not i.isdigit()])
    return text

def lower_case(text):

    text = text.split()

    text=[y.lower() for y in text]

    return " ".join(text)


def Removing_punctuations(text):
    ## Remove punctuations
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)
    text = text.replace('؛',"", )

    ## remove extra whitespace
    text = re.sub('\s+', ' ', text)
    text =  " ".join(text.split())
    return text.strip()

def Removing_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)


def del_username(text):
    text = re.sub(r"@\w+", "", text).strip()
    # print(text)
    return text


def remove_small_sentences(df):
    for i in range(len(df)):
        if len(df.line.iloc[i].split()) < 3:
            df.line.iloc[i] = np.nan



def normalize_text(df):
    df.line=df.line.apply(lambda text : del_username(text))
    df.line=df.line.apply(lambda text : lower_case(text))
    # df.line=df.line.apply(lambda text : remove_stop_words(text))
    df.line=df.line.apply(lambda text : Removing_numbers(text))
    df.line=df.line.apply(lambda text : Removing_punctuations(text))
    df.line=df.line.apply(lambda text : Removing_urls(text))
    df.line=df.line.apply(lambda text : lemmatization(text))

    return df


def normalized_sentence(sentence):
    sentence= del_username(sentence)
    sentence= lower_case(sentence)
    # sentence= remove_stop_words(sentence)
    sentence= Removing_numbers(sentence)
    sentence= Removing_punctuations(sentence)
    sentence= Removing_urls(sentence)
    sentence= lemmatization(sentence)
    # print(sentence)

    # print(sentence)
    return sentence


# df = normalize_text(df)
# print("pre_process done")
# print(df.head())

def normalize_yt(transcript):
    # Convert data into a DataFrame
    df = pd.DataFrame(transcript)

    # Add a minute column (flooring the start time divided by 60)
    df['minute'] = df['start'].apply(lambda x: floor(x / 60))

    # Group by minute and concatenate text
    merged_df = df.groupby('minute')['text'].apply(' '.join).reset_index()

    # Rename columns for clarity
    merged_df.drop(['minute'],axis=1,inplace=True)
    merged_df.columns = ['line']

    return merged_df