#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

df = pd.read_parquet('data/raw/train_data.parquet')


# In[2]:


df.head()


# In[3]:


df_train_pairs = pd.read_parquet('data/raw/train_pairs.parquet')


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


import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, auc


def pr_auc_macro(
    df: pd.DataFrame,
    prec_level: float = 0.75
) -> float:
        
    y_true = df["target"]
    y_pred = df["scores"]
    categories = df["categories"]
    
    weights = []
    pr_aucs = []

    unique_cats, counts = np.unique(categories, return_counts=True)

    for i, category in enumerate(unique_cats):
        cat_idx = np.where(categories == category)[0]
        y_pred_cat = y_pred[cat_idx]
        y_true_cat = y_true[cat_idx]

        y, x, thr = precision_recall_curve(y_true_cat, y_pred_cat)
        gt_prec_level_idx = np.where(y >= prec_level)[0]

        try:
            pr_auc_prec_level = auc(x[gt_prec_level_idx], y[gt_prec_level_idx])
            if not np.isnan(pr_auc_prec_level):
                pr_aucs.append(pr_auc_prec_level)
                weights.append(counts[i] / len(categories))
        except ValueError as err:
            pr_aucs.append(0)
            weights.append(0)
    return np.average(pr_aucs, weights=weights)


# In[ ]:


from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import tqdm
from transformers import get_linear_schedule_with_warmup

kf = KFold(n_splits=5, shuffle=True, random_state=239)

ifold = 0
oof = np.zeros(len(dataset))
for tr, va in kf.split(df_train_pairs):
    train = df_train_pairs.loc[tr].reset_index(drop=True)
    valid = df_train_pairs.loc[va].reset_index(drop=True)
    
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-multilingual-cased", 
                                                               num_labels=2).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    epochs = 2
    total_steps = len(train) * epochs // 32
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 0, 
                                                num_training_steps = total_steps)
    
    i = 0
    train_idxs = np.arange(len(train))
    train_losses = []
    for ep in range(epochs):
        np.random.shuffle(train_idxs)
        optimizer.zero_grad()
        losses = []
        for t in train_idxs:
            text1 = df.loc[df.variantid == train.iloc[t].variantid1].text.values[0]
            text2 = df.loc[df.variantid == train.iloc[t].variantid2].text.values[0]
            target = train.iloc[t].target

            tks = tokenizer.encode_plus(text1[:254], text2[:254], max_length=500, pad_to_max_length=False, 
                                        return_attention_mask=True, return_tensors='pt', truncation=True)
            out = model(tks['input_ids'].cuda(), 
                        attention_mask=tks['attention_mask'].cuda(),
                        token_type_ids=tks['token_type_ids'].cuda(),
                        labels = torch.tensor([[1.0-target, target]]).float().cuda()
                       )
            #print(out.logits[0][1].item(), target)
            losses.append(out.loss)
            
            i += 1
            if i % 32 == 0:
                loss = sum(losses) / 32.0
                loss.backward()
                losses = []
                train_losses.append(loss.item())
                optimizer.step() 
                optimizer.zero_grad()
                scheduler.step()
            
            if i % 32000 == 0:
                evaldf = []
                for t in range(len(valid)):
                    text1 = df.loc[df.variantid == valid.iloc[t].variantid1].text.values[0]
                    text2 = df.loc[df.variantid == valid.iloc[t].variantid2].text.values[0]
                    target = valid.iloc[t].target
                    category = df.loc[df.variantid == valid.iloc[t].variantid2].categories.values[0]
                    tks = tokenizer.encode_plus(text1[:254], text2[:254], max_length=500, pad_to_max_length=False, 
                                        return_attention_mask=True, return_tensors='pt', truncation=True)

                    with torch.no_grad():
                        score = torch.sigmoid(model(tks['input_ids'].cuda(), 
                                      attention_mask=tks['attention_mask'].cuda(),
                                      token_type_ids=tks['token_type_ids'].cuda()).logits[0][1].item()
                    evaldf.append((target, score, category))
                evaldf = pd.DataFrame(evaldf)
                evaldf.columns = ["target", "scores", "categories"]
                m = pr_auc_macro(evaldf)
                m2 = roc_auc_score(evaldf.target.values, evaldf.scores.values)
                print('fold', ifold, 'epoch', ep, 'prauc(0.75)', round(m,3), 'rocauc', round(m2,3))
                torch.save(model.state_dict(), f'deep_model_{ifold}_{round(m,3)}.pth')
                np.save(f'train_losses_{ifold}.npy', np.array(train_losses))
                oof[va] = evaldf.scores.values
    ifold += 1
    
df_oof = pd.read_parquet('hackathon_files_for_participants_ozon/train_pairs.parquet')
df_oof['mbert'] = oof
df_oof.to_parquet('oof_mbert.parquet')


# In[ ]:




