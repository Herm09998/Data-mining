import os
import tempfile
import shutil
import urllib
import zipfile
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from spmf import Spmf
import pathlib

# Temporary folder for data we need during execution of this notebook (we'll clean up
# at the end, we promise)
temp_dir = os.path.join(tempfile.gettempdir(), 'mind')
os.makedirs(temp_dir, exist_ok=True)

# The dataset is split into training and validation set, each with a large and small version.
# The format of the four files are the same.
# For demonstration purpose, we will use small version validation set only.
base_url = 'https://mind201910small.blob.core.windows.net/release'
training_small_url = f'{base_url}/MINDsmall_train.zip'
validation_small_url = f'{base_url}/MINDsmall_dev.zip'
training_large_url = f'{base_url}/MINDlarge_train.zip'
validation_large_url = f'{base_url}/MINDlarge_dev.zip'


def download_url(url,
                 destination_filename=None,
                 progress_updater=None,
                 force_download=False,
                 verbose=True):
    """
    Download a URL to a temporary file
    """
    if not verbose:
        progress_updater = None
    # This is not intended to guarantee uniqueness, we just know it happens to guarantee
    # uniqueness for this application.
    if destination_filename is None:
        url_as_filename = url.replace('://', '_').replace('/', '_')
        destination_filename = \
            os.path.join(temp_dir,url_as_filename)
    if (not force_download) and (os.path.isfile(destination_filename)):
        if verbose:
            print('Bypassing download of already-downloaded file {}'.format(
                os.path.basename(url)))
        return destination_filename
    if verbose:
        print('Downloading file {} to {}'.format(os.path.basename(url),
                                                 destination_filename),
              end='')
    urllib.request.urlretrieve(url, destination_filename, progress_updater)
    assert (os.path.isfile(destination_filename))
    nBytes = os.path.getsize(destination_filename)
    if verbose:
        print('...done, {} bytes.'.format(nBytes))
    return destination_filename

# For demonstration purpose, we will use small version validation set only.
# This file is about 30MB.
zip_path = download_url(validation_small_url, verbose=True)
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(temp_dir)

os.listdir(temp_dir)

# The behaviors.tsv file contains the impression logs and users' news click histories.
# It has 5 columns divided by the tab symbol:
# - Impression ID. The ID of an impression.
# - User ID. The anonymous ID of a user.
# - Time. The impression time with format "MM/DD/YYYY HH:MM:SS AM/PM".
# - History. The news click history (ID list of clicked news) of this user before this impression.
# - Impressions. List of news displayed in this impression and user's click behaviors on them (1 for click and 0 for non-click).
behaviors_path = os.path.join(temp_dir, 'behaviors.tsv')
pd.read_table(
    behaviors_path,
    header=None,
    names=['impression_id', 'user_id', 'time', 'history', 'impressions'])

# The news.tsv file contains the detailed information of news articles involved in the behaviors.tsv file.
# It has 7 columns, which are divided by the tab symbol:
# - News ID
# - Category
# - Subcategory
# - Title
# - Abstract
# - URL
# - Title Entities (entities contained in the title of this news)
# - Abstract Entities (entities contained in the abstract of this news)
news_path = os.path.join(temp_dir, 'news.tsv')
pd.read_table(news_path,
              header=None,
              names=[
                  'id', 'category', 'subcategory', 'title', 'abstract', 'url',
                  'title_entities', 'abstract_entities'
              ])

# The entity_embedding.vec file contains the 100-dimensional embeddings
# of the entities learned from the subgraph by TransE method.
# The first column is the ID of entity, and the other columns are the embedding vector values.
entity_embedding_path = os.path.join(temp_dir, 'entity_embedding.vec')
entity_embedding = pd.read_table(entity_embedding_path, header=None)
entity_embedding['vector'] = entity_embedding.iloc[:, 1:101].values.tolist()
entity_embedding = entity_embedding[[0,
                                     'vector']].rename(columns={0: "entity"})
entity_embedding
# The relation_embedding.vec file contains the 100-dimensional embeddings
# of the relations learned from the subgraph by TransE method.
# The first column is the ID of relation, and the other columns are the embedding vector values.
relation_embedding_path = os.path.join(temp_dir, 'relation_embedding.vec')
relation_embedding = pd.read_table(relation_embedding_path, header=None)
relation_embedding['vector'] = relation_embedding.iloc[:,
                                                       1:101].values.tolist()
relation_embedding = relation_embedding[[0, 'vector'
                                         ]].rename(columns={0: "relation"})
relation_embedding
shutil.rmtree(temp_dir)

behaviors_df['clicked_news'] = behaviors_df['impressions'].apply(lambda x: [news.split('-')[0] for news in x.split() if news.endswith('-1')])

# 创建交易数据
te = TransactionEncoder()
te_ary = te.fit(behaviors_df['clicked_news']).transform(behaviors_df['clicked_news'])
df = pd.DataFrame(te_ary, columns=te.columns_)

# 使用Apriori算法找出支持度至少为0.01的频繁项集
frequent_itemsets = apriori(df, min_support=0.01, use_colnames=True)

# 寻找关联规则
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
print(rules[['antecedents', 'consequents', 'support', 'confidence']])
def prepare_data_for_spmf(sequences):
    """将数据转换成SPMF可接受的格式"""
    with open('sequential_data.txt', 'w') as f:
        for sequence in sequences:
            line = ' -1 '.join(sequence) + ' -2\n'
            f.write(line)

# 准备数据
prepare_data_for_spmf(behaviors_df['clicked_news'])

# 运行SPMF的PrefixSpan算法
spmf = Spmf("PrefixSpan", input_filename="sequential_data.txt",
            output_filename="output.txt", arguments=[0.02], spmf_bin_location_path=str(pathlib.Path().absolute()))
spmf.run()
print(spmf.to_pandas_dataframe(pickle=True))
spmf.to_csv("output.csv")

frequent_itemsets = pd.DataFrame({
    'antecedents': [('N100',), ('N102',), ('N104', 'N105'), ('N107',)],
    'consequents': [('N101',), ('N103',), ('N106',), ('N108', 'N109')],
    'support': [0.015, 0.013, 0.012, 0.010],
    'confidence': [0.700, 0.650, 0.750, 0.500]
})

sequence_patterns = pd.DataFrame({
    'pattern': [['N200', 'N201'], ['N202', 'N203', 'N204'], ['N205', 'N206', 'N207']],
    'support': [0.020, 0.018, 0.015]
})

# 为频繁项集命名
frequent_itemsets['pattern_name'] = frequent_itemsets.apply(
    lambda row: "News Co-Click" if len(row['antecedents']) == 1 else "News Bundle Click", axis=1
)

# 为序列模式命名
sequence_patterns['pattern_name'] = sequence_patterns.apply(
    lambda row: "Sequential News Journey" if len(row['pattern']) > 2 else "Simple Click-Through", axis=1
)

# 输出命名结果
print("Frequent Itemsets with Names:")
print(frequent_itemsets[['antecedents', 'consequents', 'support', 'confidence', 'pattern_name']])
print("\nSequence Patterns with Names:")
print(sequence_patterns[['pattern', 'support', 'pattern_name']])
