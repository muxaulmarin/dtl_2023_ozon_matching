#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

df = pd.read_parquet('data/raw/test_data.parquet')


# In[2]:


df.head()


# In[3]:


df_train_pairs = pd.read_parquet('data/raw/test_pairs_wo_target.parquet')


# In[4]:


df['cat3'] = df['categories'].apply(lambda x : eval(x)['3'])


# In[11]:


df['color_parsed'] = df['color_parsed'].apply(lambda x: 'Colors: ' + ', '.join(x) if x is not None else '[UNK]')
df['text'] = df['name'] + ' ' + df['color_parsed']


# In[12]:


import tqdm

dataset = []
for i in tqdm.tqdm(range(len(df_train_pairs))):
    t1,t2 = df_train_pairs.iloc[i].variantid1, df_train_pairs.iloc[i].variantid2
    name1 = df.loc[df.variantid == t1,'text'].values[0]
    name2 = df.loc[df.variantid == t2,'text'].values[0]
    dataset.append((name1, name2))


# In[7]:


from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import tqdm
  
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
model1 = AutoModelForSequenceClassification.from_pretrained("bert-base-multilingual-cased", 
                                                           num_labels=2).cuda()
model1.load_state_dict(torch.load('colors_0_0.711_0.927.pth'))

model2 = AutoModelForSequenceClassification.from_pretrained("bert-base-multilingual-cased", 
                                                           num_labels=2).cuda()
model2.load_state_dict(torch.load('colors_1_0.714_0.926.pth'))
model3 = AutoModelForSequenceClassification.from_pretrained("bert-base-multilingual-cased", 
                                                           num_labels=2).cuda()
model3.load_state_dict(torch.load('colors_2_0.716_0.926.pth'))
model4 = AutoModelForSequenceClassification.from_pretrained("bert-base-multilingual-cased", 
                                                           num_labels=2).cuda()
model4.load_state_dict(torch.load('colors_3_0.727_0.93.pth'))
model5 = AutoModelForSequenceClassification.from_pretrained("bert-base-multilingual-cased", 
                                                           num_labels=2).cuda()
model5.load_state_dict(torch.load('colors_4_0.722_0.928.pth'))


# In[13]:


evaldf = []
ss = torch.nn.Softmax(dim=1)
for t in tqdm.tqdm(range(len(dataset))):
    name1, name2 = dataset[t]
    tks = tokenizer.encode_plus(name1[:600], name2[:600], max_length=512, pad_to_max_length=False,
                        return_attention_mask=True, return_tensors='pt', truncation=True)

    with torch.no_grad():
        score1 = ss(model1(tks['input_ids'].cuda(), 
                           attention_mask=tks['attention_mask'].cuda(),
                           token_type_ids=tks['token_type_ids'].cuda()).logits)[0][1].item()
        score2 = ss(model2(tks['input_ids'].cuda(), 
                           attention_mask=tks['attention_mask'].cuda(),
                           token_type_ids=tks['token_type_ids'].cuda()).logits)[0][1].item()
        score3 = ss(model3(tks['input_ids'].cuda(), 
                           attention_mask=tks['attention_mask'].cuda(),
                           token_type_ids=tks['token_type_ids'].cuda()).logits)[0][1].item()
        score4 = ss(model4(tks['input_ids'].cuda(), 
                           attention_mask=tks['attention_mask'].cuda(),
                           token_type_ids=tks['token_type_ids'].cuda()).logits)[0][1].item()
        score5 = ss(model5(tks['input_ids'].cuda(), 
                           attention_mask=tks['attention_mask'].cuda(),
                           token_type_ids=tks['token_type_ids'].cuda()).logits)[0][1].item()

        score = (score1 + score2 + score3 + score4 + score5) * 0.2
        
        #score = score1
        evaldf.append((df_train_pairs.iloc[t].variantid1,
                       df_train_pairs.iloc[t].variantid2,
                       score))
        
evaldf = pd.DataFrame(evaldf)
evaldf.columns = ["variantid1","variantid2","colorbert"]

nn_features = pd.read_parquet('data/preprocessed/nn_test.parquet')
nn_features = nn_features.merge(evaldf, on=["variantid1","variantid2"], how='left')
nn_features.to_parquet('data/preprocessed/nn_test.parquet')


# In[14]:


evaldf.to_parquet('test_colorbert.parquet')


# In[ ]:




