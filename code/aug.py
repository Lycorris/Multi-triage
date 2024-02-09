import pandas as pd # loadingData
from nltk.stem import PorterStemmer # func1,cleanData
from nltk.corpus import stopwords # func1
import regex as re # func1
import nlpaug.augmenter.word as naw #func5 CreateOversamplingWithDataAugmentation
import gc # func5
import numpy as np # (*func7*) one_hot_encode
from nltk import PorterStemmer, word_tokenize
from nltk.corpus import stopwords
import nlpaug.augmenter.word as naw
import tqdm
# CreateOversamplingWithDataAugmentation
# 将少数列的行数通过bert上下文替换增加到和多数列一样多

DataAugmentation, DataAugThreshold = True, 30000
DataFilePath, DataFileName, FileType = "autodl-tmp/Multitriage/Data/powershell", "Issueazure-powershellWebScrap", ".csv"
MAX_SEQUENCE_LENGTH, EMBEDDING_DIM = 300, 100
LoadDataAugFromFile = False
LearningRate = 0.001
VALIDATION_SPLIT = 0.2
data = pd.read_csv('Data/powershell/C_uA_Train.csv')

# /*Get Contextual Word Embedding Model*/
TRANSFORMERS_OFFLINE=1
aug = naw.ContextualWordEmbsAug(model_path='bert-base-uncased', action="substitute") # aug是上下文替换器
# /*Get Developer Bug Count/
dfcountbybug = data.groupby(["Name", "FixedByID"], as_index=True)["FixedByID"].size().reset_index(name="count")
dfcountbybug.to_csv(DataFilePath + 'list.csv')
# /*Get Majority Class List*/
majoritycount = dfcountbybug[(dfcountbybug['FixedByID'] != 'unknown') & (dfcountbybug['Name'] != 'unknown')][
    'count'].max()
# /*Get Minority Class Count*/
minoritylist = dfcountbybug[(dfcountbybug['count'] != dfcountbybug['count'].max())]
# 42 643 27006
print(majoritycount, len(minoritylist), majoritycount * len(minoritylist))
estimatetotalnoofdataaugrecord = majoritycount * len(minoritylist)
maxnoofaug = majoritycount
## Data Aug Record Count Validation, If over threshold reduce the majaritycount
if estimatetotalnoofdataaugrecord > DataAugThreshold: # 30000
    print('Overtheshold--')
    maxnoofaug = int((DataAugThreshold / estimatetotalnoofdataaugrecord) * majoritycount)

# /*Loop through Minor Class Group*/
for ind in tqdm.tqdm(minoritylist.index):
    # print(minoritylist['FixedByID'][ind], minoritylist['count'][ind])
    developer = minoritylist['FixedByID'][ind]
    bugtype = minoritylist['Name'][ind]
    minoritycount = minoritylist['count'][ind]
    data1 = data[(data['FixedByID'] == developer) & (data['Name'] == bugtype)]
    # print(len(data1), developer,bugtype)
    # print('minoritycount  --->',minoritycount, 'majoritycount--->',majoritycount, 'index --->', ind , 'out of ', len(minoritylist.index))
    # Create Sample Data until minority class count Match up with majority class count
    while minoritycount < maxnoofaug:
        # majoritycount:
        samplerow = data1.sample()
        oldbugdescription = str(samplerow['Title_Description'].values).strip('[]').replace("'", "")
        if oldbugdescription:
            first100words_aug = str(aug.augment(' '.join(oldbugdescription.split()[:100])))
            remainingwords = ' '.join(oldbugdescription.split()[100:])
            newbugdescription = first100words_aug + remainingwords
            new_row = {'RepoID': str(samplerow['RepoID'].values).strip('[]').replace("'", ""),
                        # 'PullRequestID' : str(samplerow['PullRequestID'].values).strip('[]').replace("'",""),
                       'IssueID': str(samplerow['IssueID'].values).strip('[]').replace("'", ""),
                       'Title_Description': newbugdescription,  # /*Data Augmentation : Title_Desciption*/
                       'AST': str(samplerow['AST'].values).strip('[]').replace("'", ""),
                       'FixedByID': str(samplerow['FixedByID'].values).strip('[]').replace("'", ""),
                       'Name': str(samplerow['Name'].values).strip('[]').replace("'", ""),
                       'CreatedDate': str(samplerow['CreatedDate'].values).strip('[]').replace("'", "")}
            data = data.append(new_row, ignore_index=True)
            minoritycount = minoritycount + 1
        gc.collect()

data.to_csv(DataFilePath + 'C_A_Train.csv')