import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
# Load data set.
github_data = pd.read_csv('archive/github_dataset.csv')
#github_data = pd.read_csv('archive/ml-10M100K/movies.dat')
language_counts = pd.DataFrame(github_data['language'].value_counts()).sort_values(by='language',ascending=True).rename(columns={'count': 'language_count'})


def regularit(df):
    new_df = pd.DataFrame(index=df.index)
    columns = ['stars_count', 'forks_count', 'issues_count', 'pull_requests', 'contributors']
    for c in columns:
        d = df[c]
        MAX = d.max()
        MIN = d.min()
        new_df[c] = ((d - MIN) / (d - MAX))
    return new_df
def delete():
    github_data_cleaned = github_data.dropna()
    print(github_data_cleaned)
def HFreplace():
    language_HF = language_counts.index[-1]
    github_data_HF_replaced = github_data.replace(np.nan, language_HF)
    print(github_data_HF_replaced)

def REreplace():
    df_coded = pd.get_dummies(github_data, columns=['language'], dummy_na=True, drop_first=True)
    plt.figure(figsize=(40, 40))
    sns.heatmap(df_coded.corr(method='spearman'), cmap='YlGnBu', annot=True)
    plt.title('Correlation Analysis')
    github_data_attr_corr = github_data
    df1 = github_data_attr_corr.groupby('language').agg(avg=('stars_count', 'mean'))
    for i in range(len(github_data_attr_corr)):
        if github_data_attr_corr['language'].iloc[i] == 'NAN':
            rate = github_data_attr_corr['stars_count'].iloc[i]
            dist = []
            for j in range(len(df1)):
                dist.append(abs(df1.iloc[j]['avg'] - rate))
            idx = dist.index(min(dist))
            github_data_attr_corr['language'].iloc[i] = df1.index[idx]
    github_data_attr_corr['language'].value_counts()
    language_count_attr_corr = language_counts
    language_count_attr_corr['language_count_attr_corr'] = [0] * len(language_counts)

    for level in list(language_counts.index):
        if level in list(github_data_attr_corr['language'].value_counts().index):
            language_count_attr_corr.loc[[level], ['language_count_attr_corr']] = \
            github_data_attr_corr['language'].value_counts().loc[[level]].values[0]

    plt.figure(figsize=(40, 40))
    plt.yticks(fontsize=24)
    plt.xticks(fontsize=24)
    plt.barh(list(range(len(language_count_attr_corr))), tick_label=language_count_attr_corr.index,
             width=language_count_attr_corr['language_count'], label='language_count', height=0.4)
    plt.barh([d + 0.42 for d in list(range(len(language_count_attr_corr)))], tick_label=language_count_attr_corr.index,
             width=language_count_attr_corr['language_count_attr_corr'], label='language_count_attr_corr', height=0.4)
    plt.ylabel('language', fontsize=24)
    plt.xlabel('', fontsize=24)
    # plt.title('Number of movies for each appropriation-level?', fontsize=32, loc='center')
    plt.legend(fontsize=32, loc='lower right')
    plt.show()

def RSreplace():
    github_data_sample_corr = github_data
    normal_github_data = regularit(github_data_sample_corr)
    normal_language = pd.concat([normal_github_data, github_data_sample_corr['language']], axis=1)
    infos = []
    for i in range(len(normal_language)):
        info = []
        star = normal_language['stars_count'].iloc[i]
        fork = normal_language['forks_count'].iloc[i]
        issue = normal_language['issues_count'].iloc[i]
        pull = normal_language['pull_requests'].iloc[i]
        contributor = normal_language['contributors'].iloc[i]
        info.append(star)
        info.append(fork)
        info.append(issue)
        info.append(pull)
        info.append(contributor)
        infos.append(info)
    for i in range(len(normal_language)):
        if normal_language['language'].iloc[i] == 'NAN':
            dists = []
            for j in len(infos):
                dist = np.sqrt(np.sum(np.square(infos[i] - infos[j])))
                dists.append(dist)
            idx = dists.index(min(dists))
            github_data_sample_corr['language'].iloc[i] = github_data_sample_corr['language'].iloc[idx]
    github_data_sample_corr['language'].value_counts()




delete()
#HFreplace()
#REreplace()
#RSreplace()
