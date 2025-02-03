import pandas as pd
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

# import swifter

from nltk.stem import PorterStemmer

# Initialize the Porter Stemmer
stemmer = PorterStemmer()

# Function to apply stemming to each bigram
def stem_bigrams(bigrams):
    stemmed_bigrams = []
    for bigram in bigrams:
        # Split the bigram into individual words
        words = bigram.split()

        # Stem each word and join back into a bigram
        stemmed_bigram = ' '.join([stemmer.stem(word) for word in words])
        stemmed_bigrams.append(stemmed_bigram)

    return stemmed_bigrams

def identify_important_bigrams(dataframe, column_name):
    # Initialize TF-IDF Vectorizer for bigrams
    vectorizer = TfidfVectorizer(ngram_range=(1, 1), stop_words='english', max_df=0.95, min_df=0, max_features=300)

    # Apply the vectorizer to the specified column
    tfidf_matrix = vectorizer.fit_transform(dataframe[column_name])

    # Extract feature names (bigrams)
    feature_names = vectorizer.get_feature_names_out()

    # Identify the most important bigram for each document
    important_bigrams = []
    for row in tfidf_matrix.toarray():
        # Get the index of the max TF-IDF score
        max_index = row.argmax()
        
        # Get the corresponding bigram
        important_bigram = feature_names[max_index]
        
        # Append to the list
        important_bigrams.append(important_bigram)

    # important_bigrams = ", ".join(important_bigrams)
    return important_bigrams

df =pd.read_json('/home/mourad/topicGPT/data/output/assignment_corrected.jsonl', lines=True)
df = df[['id', 'responses']]
df = df.rename({'responses': 'cause_topic', 'id': 'cluster_id'}, axis=1)

news_data = pd.read_parquet('/data/mourad/narratives/topic/data.parquet')
news_data = df.merge(news_data, left_on='cluster_id', right_on='cause_cluster')
news_data = news_data.drop('cluster_id', axis=1)

df = df.rename({'cause_topic': 'effect_topic'}, axis=1)

news_data = df.merge(news_data, left_on='cluster_id', right_on='effect_cluster')
news_data = news_data.drop('cluster_id', axis=1)
df = news_data

df['narrative'] = df.cause_cluster.astype(str) + '-' + df.effect_cluster.astype(str)

df['main_cause'] = df.cause_topic.apply(lambda x: x.split(':')[0])
df['main_effect'] = df.effect_topic.apply(lambda x: x.split(':')[0])

df['cause_topic'] = df.cause_topic.apply(lambda x: x.split(':')[1].split('\n')[0])
df['effect_topic'] = df.effect_topic.apply(lambda x: x.split(':')[1].split('\n')[0])

df['cause_keywords'] = identify_important_bigrams(df, 'cause_topic')
df['effect_keywords'] = identify_important_bigrams(df, 'effect_topic')

df['cause_keywords'] = stem_bigrams(df['cause_keywords'])
df['effect_keywords'] = stem_bigrams(df['effect_keywords'])
# breakpoint()


grouped = df.groupby(['main_cause', 'main_effect'])['cause', 'effect', 'text', 'cause_topic', 'effect_topic', 'month']
for i, (name, group) in enumerate(grouped):
    print(name)
    if len(group) < 5: continue
    
    for id, row in group.iterrows(): 
        cause_desc = row.cause_topic.split(':')
        if len(cause_desc) > 1:
            cause_desc = cause_desc[1].split('(')[0]
        else: cause_desc = cause_desc[0]
        effect_desc = row.effect_topic.split(':')
        if len(effect_desc) > 1:
            effect_desc = effect_desc[1].split('(')[0]
        else: effect_desc = effect_desc[0]
        print(f"{cause_desc} - {effect_desc}")
        print(f"{row.cause} --------- {row.effect}\n\n")
    print(Counter(group.month))
    # plt.xticks(range(12), [str(m) for m in range(12)])
    # sns.countplot(x=group.month.sort_values(ascending=False))
    # plt.title(' - '.join(name))
    
    # plt.savefig(f'/data/mourad/narratives/topics_over_time/{i}.png')
    # plt.clf()
    # if i == 35: breakpoint()

    # fig, ax = plt.subplots(figsize=(6, 4))
    # sns.histplot(group.month.sort_values(ascending=False), bins=range(0, 13), kde=False, ax=ax, color="blue", discrete=True)
    # ax.set_xlim([0, 12])
    # ax.set_xticks(range(0, 13))
    # ax.set_xlabel('Months')
    # ax.set_ylabel('Frequency')
    # ax.set_title(' - '.join(name))
    # plt.savefig(f'/data/mourad/narratives/topics_over_time/{i}.png')
    # plt.clf()

    breakpoint()