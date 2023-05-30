#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

df = pd.read_parquet('hackathon_files_for_participants_ozon/test_data.parquet')


# In[2]:


df.head()


# In[3]:


df_train_pairs = pd.read_parquet('hackathon_files_for_participants_ozon/test_pairs_wo_target.parquet')


# In[4]:


df['cat3'] = df['categories'].apply(lambda x : eval(x)['3'])


# In[5]:


import tqdm
def making_char_pairs(js1, js2):
    res_dist, res_similar = [], []
    try:
        js1 = eval(js1)
        js2 = eval(js2)
        jskeys = set(js1.keys()) & set(js2.keys())
    except:
        return res_dist, res_similar
    
    for k in jskeys:
        v1 = js1.get(k)
        v2 = js2.get(k)
        if v1 != v2:
            res_dist.append(k)
        if v1 == v2:
            res_similar.append(k)
    return res_dist, res_similar

dataset = []
for i in tqdm.tqdm(range(len(df_train_pairs))):
    t1,t2 = df_train_pairs.iloc[i].variantid1, df_train_pairs.iloc[i].variantid2
    target = 0
    category = df.loc[df.variantid == t1].cat3.values[0]
    name1 = df.loc[df.variantid == t1].name.values[0]
    name2 = df.loc[df.variantid == t2].name.values[0]
    cat_groupped = "rest"
    res_dist, res_similar = making_char_pairs(df.loc[df.variantid == t1].characteristic_attributes_mapping.values[0],
                                              df.loc[df.variantid == t2].characteristic_attributes_mapping.values[0]
                                             )
    dataset.append((category, name1, name2, ', '.join(res_dist), ', '.join(res_similar), target, cat_groupped))


# In[42]:


from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import tqdm
  
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
model1 = AutoModelForSequenceClassification.from_pretrained("bert-base-multilingual-cased", 
                                                           num_labels=2).cuda()
model1.load_state_dict(torch.load('chstic_0_0.762_0.937.pth'))

model2 = AutoModelForSequenceClassification.from_pretrained("bert-base-multilingual-cased", 
                                                           num_labels=2).cuda()
model2.load_state_dict(torch.load('chstic_1_0.767_0.937.pth'))
model3 = AutoModelForSequenceClassification.from_pretrained("bert-base-multilingual-cased", 
                                                           num_labels=2).cuda()
model3.load_state_dict(torch.load('chstic_2_0.764_0.937.pth'))
model4 = AutoModelForSequenceClassification.from_pretrained("bert-base-multilingual-cased", 
                                                           num_labels=2).cuda()
model4.load_state_dict(torch.load('chstic_3_0.762_0.937.pth'))
model5 = AutoModelForSequenceClassification.from_pretrained("bert-base-multilingual-cased", 
                                                           num_labels=2).cuda()
model5.load_state_dict(torch.load('chstic_4_0.763_0.936.pth'))


# In[43]:


evaldf = []
for t in tqdm.tqdm(range(len(dataset))):
    category, name1, name2, dist, sim, target, cat_groupped = dataset[t]
    s = category + '[SEP]' + name1 + '[SEP]' + name2 + '[SEP]'  + dist #+ '[SEP]' + sim
    tks = tokenizer.encode_plus(s[:1200], max_length=512, pad_to_max_length=False,
                        return_attention_mask=True, return_tensors='pt', truncation=True)

    with torch.no_grad():
        score1 = torch.sigmoid(model1(tks['input_ids'].cuda(), 
                      attention_mask=tks['attention_mask'].cuda(),
                      token_type_ids=tks['token_type_ids'].cuda()).logits[0][1]).item()
        score2 = torch.sigmoid(model2(tks['input_ids'].cuda(), 
                      attention_mask=tks['attention_mask'].cuda(),
                      token_type_ids=tks['token_type_ids'].cuda()).logits[0][1]).item()
        score3 = torch.sigmoid(model3(tks['input_ids'].cuda(), 
                      attention_mask=tks['attention_mask'].cuda(),
                      token_type_ids=tks['token_type_ids'].cuda()).logits[0][1]).item()
        score4 = torch.sigmoid(model4(tks['input_ids'].cuda(), 
                      attention_mask=tks['attention_mask'].cuda(),
                      token_type_ids=tks['token_type_ids'].cuda()).logits[0][1]).item()
        score5 = torch.sigmoid(model5(tks['input_ids'].cuda(), 
                      attention_mask=tks['attention_mask'].cuda(),
                      token_type_ids=tks['token_type_ids'].cuda()).logits[0][1]).item()
        score = (score1 + score2 + score3 + score4 + score5) * 0.2
        
        #score = score1
        evaldf.append((df_train_pairs.iloc[t].variantid1,
                       df_train_pairs.iloc[t].variantid2,
                       score))
        
evaldf = pd.DataFrame(evaldf)
evaldf.columns = ["variantid1","variantid2","chstic"]
evaldf.to_parquet('test_chstic.parquet')

nn_features = pd.read_parquet('test_mbert.parquet')
nn_features = nn_features.merge(evaldf, on=["variantid1","variantid2"], how='left')
nn_features.to_parquet('nn/test.parquet')


# In[ ]:




