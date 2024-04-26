import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
# Load data set.
#github_data = pd.read_csv('archive/github_dataset.csv')
github_data = pd.read_csv('archive/ml-10M100K/movies.dat',sep='::', header = None, names =['mID','title','genres'], engine = 'python')

# Check the columns.
#print(github_data.columns)
def fluency():
   for item in github_data:
        print(github_data[item].value_counts())

def describe():
   numeric_github_data = pd.DataFrame(github_data, columns=github_data.columns)
   print(numeric_github_data.describe())
   NaN_counts = github_data.isna().sum()
   NaN_counts = pd.DataFrame(NaN_counts, columns=['NaN_counts']).T
   print(NaN_counts)

def visual1():
    language_counts = pd.DataFrame(github_data['language'].value_counts()).sort_values(by='language',ascending=True).rename(columns={'count': 'language_count'})
    plt.figure(figsize=(40, 40))
    plt.yticks(fontsize=24)
    plt.xticks(fontsize=24)
    plt.barh(language_counts.index, width=language_counts['language_count'])
    plt.show()

def visual2():
    sns.set_style("whitegrid")
    stars_count_box = sns.catplot(data=github_data, kind='box', y='language', x='stars_count', height=20,
                                  palette='rocket', sym='')
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)
    plt.ylabel('language', fontsize=20, )
    plt.xlabel('stars_count', fontsize=20)
    plt.show()

def visual3():
    language_counts = pd.DataFrame(github_data['genres'].value_counts()).sort_values(by='genres',
                                                                                     ascending=True).rename(
        columns={'count': 'genres_count'})

    plt.figure(figsize=(40, 40))

    # 设置字体大小
    plt.yticks(fontsize=3)
    plt.xticks(fontsize=3)

    # 增加左边距
    plt.subplots_adjust(left=0.3)

    # 绘制条形图
    plt.barh(language_counts.index, language_counts['genres_count'])

    plt.show()


def visual4():
    sns.set_style("whitegrid")
    stars_count_box = sns.catplot(data=github_data, kind='box', y='genres', x='mID', height=20,
                                  palette='rocket', sym='')
    plt.yticks(fontsize=3)
    plt.xticks(fontsize=3)
    plt.ylabel('genres', fontsize=3, )
    plt.xlabel('mID', fontsize=3)
    plt.show()


fluency()
#describe()
#visual1()#用于github dataset数据集
#visual2()#用于github dataset数据集
#visual3()#用于movie数据集
#visual4()#用于movie数据集
