from InstructorEmbedding import INSTRUCTOR
from sklearn.metrics.pairwise import cosine_similarity
import sklearn.cluster
import pandas as pd
import utils.utils as utils
import os
import numpy as np
import termplotlib as tpl
from tqdm.auto import tqdm
from sklearn.metrics import silhouette_score
import hdbscan
from umap import UMAP
import plotly.express as px
import seaborn as sns
import argparse

np.random.seed(42)

def prep_for_topic(df):
    df.to_parquet('/data/mourad/narratives/topic/data.parquet')

    causes = df.groupby('cause_cluster')['cause'].agg(list).to_frame()
    effects = df.groupby('effect_cluster')['effect'].agg(list).to_frame()

    causes.cause = causes.cause.swifter.apply(lambda x: '. '.join(x) + '. ')
    effects.effect = effects.effect.swifter.apply(lambda x: '. '.join(x) + '. ')
    docs = causes.merge(effects, left_index=True, right_index=True, how='outer')
    breakpoint()
    docs = docs.fillna('')

    # cause_docs = df.groupby('cause_cluster')['cause'].transform(lambda x: '. '.join(x)).to_frame()
    # effect_docs = df.groupby('effect_cluster')['effect'].transform(lambda x: '. '.join(x)).to_frame()
    # docs = cause_docs.merge(effect_docs, left_index=True, right_index=True, how='outer')
    docs['text'] = docs['cause'] + '. ' + docs['effect']
    docs = docs.reset_index()
    docs = docs.rename({'index':'id'}, axis=1)
    docs['label'] = ''
    docs[['id', 'text', 'label']].to_json('/home/mourad/topicGPT/data/input/data.jsonl', orient='records', lines=True)
    # breakpoint(c)

def plot(df, cluster_assignments, embedding, kind=None):
    reducer = UMAP(n_components=3, n_neighbors=10)# min_dist=0)
    embedding = reducer.fit_transform(embedding)

    # Step 3: Plot the results using plotly express
    fig = px.scatter_3d(
        embedding, x=0, y=1, z=2, color=cluster_assignments,
        labels={'0': 'UMAP 1', '1': 'UMAP 2', '2': 'UMAP 3'},
        color_continuous_scale=px.colors.qualitative.Set1,
        title='UMAP Projection with HDBSCAN Clusters',
        hover_data={'text': df['cause'].tolist()+df['effect'].tolist()}
    )

    fig.write_html(f"/data/mourad/narratives/plots/result.html")

    
def embed(text):
    instructions = 'Represent the news sentences for clustering: '
    emb = model.encode([[instructions, text]])
    return emb[0]

def cluster(df):
    # for kind in ['cause', 'effect']:
    # print('Clustering ', kind)
    # Convert the embeddings Series to a list of arrays
    # list_of_embeddings = [np.array(x) for x in df[f'{kind}_emb']]

    print('Clustering')
    list_of_embeddings = [np.array(x) for x in df['cause_emb']] + [np.array(x) for x in df['effect_emb']]
    # Convert this list to a 2D numpy array
    embeddings_array = np.vstack(list_of_embeddings)

    reduced_embeddings = UMAP(n_neighbors=20, min_dist=0, n_components=10).fit_transform(embeddings_array)


    # Fit the model
    # clustering_model = sklearn.cluster.MiniBatchKMeans(n_clusters=250)
    clustering_model = hdbscan.HDBSCAN(min_samples=1, min_cluster_size=7, cluster_selection_method='leaf', cluster_selection_epsilon=0.01)

    clustering_model.fit(reduced_embeddings)
    cluster_assignment = clustering_model.labels_
    df[f'cause_cluster'] = cluster_assignment[:len(df)]
    df[f'effect_cluster'] = cluster_assignment[len(df):]
    # ax = clustering_model.condensed_tree_.plot(select_clusters=True, selection_palette=sns.color_palette('deep',  np.unique(cluster_assignment).shape[0]))
    # ax.figure.savefig(f'/data/mourad/narratives/plots/{kind}_condensed_tree.png')
    # ax.clear()
    # df[f'{kind}_cluster'] = cluster_assignment
    print(pd.Series(cluster_assignment).value_counts().head(40))
    plot(df, cluster_assignment, embeddings_array)
    # breakpoint()
    # return None
    return df

def silhouette_cluster(df):
    # Convert the embeddings Series to a list of arrays
    list_of_embeddings = [np.array(x) for x in df.cause_emb]

    # Convert this list to a 2D numpy array
    embeddings_array = np.vstack(list_of_embeddings)
    k_values = range(380, 480, 20)  # Assuming you want to check k from 2 to 10
    sil_scores = []

    for k in tqdm(k_values):
        # Fit the model
        clustering_model = sklearn.cluster.MiniBatchKMeans(n_clusters=k)
        clustering_model.fit(embeddings_array)
        cluster_assignment = clustering_model.labels_
        sil_score = silhouette_score(embeddings_array, cluster_assignment)
        sil_scores.append(sil_score)

    fig = tpl.figure()
    fig.plot(k_values, sil_scores)
    # plt.xlabel('Number of clusters')
    # plt.ylabel('Silhouette Score')
    # plt.title('Silhouette Score vs. Number of Clusters')
    fig.show()

    optimal_k = k_values[np.argmax(sil_scores)]
    print(f"Optimal number of clusters: {optimal_k}")
    return None

def group_narratives(df):
    # TODO groupby sentence to get narratives from different sentences
    # df = df['cause_cluster', 'effect_cluster', 'cause', 'effect']]
    # df = df[(df.cause_cluster != -1) & (df.effect_cluster != -1)]
    df['narrative_cluster'] = df.apply(lambda x: f"{x.cause_cluster}-{x.effect_cluster}", axis=1)
    print(df.narrative_cluster.value_counts().head(50))
    variety = df.groupby('cause_cluster').effect_cluster.nunique() /  df.groupby('cause_cluster').size()
    filtered = variety[df.groupby('cause_cluster').size() > 10].sort_values(ascending=False).head(40)
    print(filtered)

    variety = df.groupby('effect_cluster').cause_cluster.nunique() /  df.groupby('effect_cluster').size()
    filtered = variety[df.groupby('effect_cluster').size() > 10].sort_values(ascending=False).head(40)
    print(filtered)

    breakpoint()
    #df[df.narrative_cluster == '68-73'][['cause', 'effect']]        
    # narratives = df.groupby(['narrative_cluster'])[['cause', 'effect']].agg(list)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument("-y", "--year", type=int, required=False, default=None, help="Year to filter on")
    parser.add_argument("-s", "--sample", type=int, required=False, default=None, help="Random sample size")
    parser.add_argument("-rw", "--rewrite", action='store_true', help='rewrite the gpt4 output? cached results will still be used')
    parser.add_argument("-t", "--topic", type=str, default='inflation', choices=["inflation", "recession"], help='what economic topic is being studied')
    parser.add_argument('--model', choices=['claude', 'gpt'], default='claude')
    # Parse the arguments
    args = parser.parse_args()

    # sample = 5000
    # rewrite=False
    embed_path = f'/data/mourad/narratives/{args.topic}/{args.model}/instructor/{args.sample if args.sample else args.year}.parquet'
    if not os.path.exists(embed_path) or args.rewrite:
        model = INSTRUCTOR('hkunlp/instructor-large')  
        df = pd.read_parquet(f"/data/mourad/narratives/{args.topic}/{args.model}/{'sample' if args.sample else 'year'}/{args.sample if args.sample else args.year}.parquet")
        df = df.dropna()
        # df = utils.create_cause_effect_cols(df)

        df['cause_emb'] = df.cause.swifter.apply(embed)
        df['effect_emb'] = df.effect.swifter.apply(embed)
        df.to_parquet(embed_path)


    df = pd.read_parquet(embed_path)
    # breakpoint()
    # silhouette_cluster(df)
    df.year = df.year.astype(int)
    df = df[df.year < 2020]
    print(len(df))
    df = df.drop_duplicates(['cause', 'effect'])
    df = df[~df.cause.str.contains('inflation-adjusted')]
    df = df[~df.effect.str.contains('inflation-adjusted')]
    print(len(df))
    df = cluster(df)
    df = df[(df.cause_cluster != -1) & (df.effect_cluster != -1)]
    prep_for_topic(df)
    group_narratives(df)

