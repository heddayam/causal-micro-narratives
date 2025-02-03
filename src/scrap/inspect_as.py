import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

dir_path = "/data/mourad/narratives/inflation/americanstories_sentences"

dfs = []
for file in os.listdir(dir_path):
    df = pd.read_csv(os.path.join(dir_path, file), sep="\t")
    df.text = df.text.str.replace("\n", " ")
    df['year'] = int(file.split(".")[0])
    dfs.append(df)

df = pd.concat(dfs)
df = df[df.mention_flag == 1]

sns.histplot(data=df, x='year', binwidth=1)
plt.savefig('../../data/plots/americanstories/as_filtered_hist.png', dpi=300, bbox_inches='tight')

# for text in df.text.sample(50):
    # print(text)
    # print()
breakpoint()
    # print(df.head())
    