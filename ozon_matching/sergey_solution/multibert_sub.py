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


def colors(s):
    if (s is None) or (len(s) == 0):
        return 'нет'
    return ', '.join(eval(str(s)))

def desc(s):
    if s is None:
        return 'нет'
    return str(eval(s))

df['text'] = df['name'] #+ ' цвета: ' + df['color_parsed'].apply(colors)# + ' описание: ' + \
#df['characteristic_attributes_mapping'].apply(desc)


# In[5]:


import gc
df = df[['variantid','text','categories']]
gc.collect()


# In[6]:


df_train_pairs.head()


# In[7]:


from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import tqdm
  
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
model1 = AutoModelForSequenceClassification.from_pretrained("bert-base-multilingual-cased", 
                                                           num_labels=2).cuda()
model1.load_state_dict(torch.load('deep_model_0_0.69.pth'))
model2 = AutoModelForSequenceClassification.from_pretrained("bert-base-multilingual-cased", 
                                                           num_labels=2).cuda()
model2.load_state_dict(torch.load('deep_model_1_0.698.pth'))
model3 = AutoModelForSequenceClassification.from_pretrained("bert-base-multilingual-cased", 
                                                           num_labels=2).cuda()
model3.load_state_dict(torch.load('deep_model_2_0.694.pth'))
model4 = AutoModelForSequenceClassification.from_pretrained("bert-base-multilingual-cased", 
                                                           num_labels=2).cuda()
model4.load_state_dict(torch.load('deep_model_3_0.712.pth'))
model5 = AutoModelForSequenceClassification.from_pretrained("bert-base-multilingual-cased", 
                                                           num_labels=2).cuda()
model5.load_state_dict(torch.load('deep_model_4_0.695.pth'))


# In[9]:


evaldf = []
for t in tqdm.tqdm(range(len(df_train_pairs))):
    text1 = df.loc[df.variantid == df_train_pairs.iloc[t].variantid1].text.values[0]
    text2 = df.loc[df.variantid == df_train_pairs.iloc[t].variantid2].text.values[0]
    tks = tokenizer.encode_plus(text1[:254], text2[:254], max_length=500, pad_to_max_length=False, 
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
        evaldf.append((df_train_pairs.iloc[t].variantid1,
                       df_train_pairs.iloc[t].variantid2,
                       score))
        
evaldf = pd.DataFrame(evaldf)
evaldf.columns = ["variantid1","variantid2","mbert"]
evaldf.to_parquet('test_mbert.parquet')


# In[9]:


evaldf.to_csv('sub_scores.csv', index=False)


# In[ ]:




