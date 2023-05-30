#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

df = pd.read_parquet('hackathon_files_for_participants_ozon/train_data.parquet')


# In[2]:


df.head()


# In[3]:


df_train_pairs = pd.read_parquet('hackathon_files_for_participants_ozon/train_pairs.parquet')


# In[4]:


df['cat3'] = df['categories'].apply(lambda x : eval(x)['3'])


# In[5]:


df['cat3']


# In[6]:


df['cat4'] = df['categories'].apply(lambda x : eval(x)['4'])
df['color_parsed'] = df['color_parsed'].apply(lambda x: 'Colors: ' + ', '.join(x) if x is not None else '[UNK]')


# In[7]:


cat3_counts = df.cat3.value_counts().to_dict()
df["cat3_groupped"] = df["cat3"].apply(lambda x: x if cat3_counts[x] > 1000 else "rest")


# In[8]:


df['text'] = df['name'] + ' ' + df['color_parsed']


# In[9]:


import tqdm

dataset = []
for i in tqdm.tqdm(range(len(df_train_pairs))):
    t1,t2 = df_train_pairs.iloc[i].variantid1, df_train_pairs.iloc[i].variantid2
    target = df_train_pairs.iloc[i].target
    name1 = df.loc[df.variantid == t1,'text'].values[0]
    name2 = df.loc[df.variantid == t2,'text'].values[0]
    cat_groupped = df.loc[df.variantid == t1].cat3_groupped.values[0]
    dataset.append((name1, name2, target, cat_groupped))


# In[10]:


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

    # calculate metric for each big category
    for i, category in enumerate(unique_cats):
        # take just a certain category
        cat_idx = np.where(categories == category)[0]
        y_pred_cat = y_pred[cat_idx]
        y_true_cat = y_true[cat_idx]

        # if there is no matches in the category then PRAUC=0
        if sum(y_true_cat) == 0:
            pr_aucs.append(0)
            weights.append(counts[i] / len(categories))
            continue
        
        # get coordinates (x, y) for (recall, precision) of PR-curve
        y, x, _ = precision_recall_curve(y_true_cat, y_pred_cat)
        
        # reverse the lists so that x's are in ascending order (left to right)
        y = y[::-1]
        x = x[::-1]
        
        # get indices for x-coordinate (recall) where y-coordinate (precision) 
        # is higher than precision level (75% for our task)
        good_idx = np.where(y >= prec_level)[0]
        
        # if there are more than one such x's (at least one is always there, 
        # it's x=0 (recall=0)) we get a grid from x=0, to the rightest x 
        # with acceptable precision
        if len(good_idx) > 1:
            gt_prec_level_idx = np.arange(0, good_idx[-1] + 1)
        # if there is only one such x, then we have zeros in the top scores 
        # and the curve simply goes down sharply at x=0 and does not rise 
        # above the required precision: PRAUC=0
        else:
            pr_aucs.append(0)
            weights.append(counts[i] / len(categories))
            continue
        
        # calculate category weight anyway
        weights.append(counts[i] / len(categories))
        # calculate PRAUC for all points where the rightest x 
        # still has required precision 
        try:
            pr_auc_prec_level = auc(x[gt_prec_level_idx], y[gt_prec_level_idx])
            if not np.isnan(pr_auc_prec_level):
                pr_aucs.append(pr_auc_prec_level)
        except ValueError:
            pr_aucs.append(0)
            
    return np.average(pr_aucs, weights=weights)


# In[11]:


from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import tqdm
from transformers import get_linear_schedule_with_warmup

kf = KFold(n_splits=5, shuffle=True, random_state=239)
ds_indexes = np.arange(len(dataset))

ifold = 0
oof = np.zeros(len(dataset))
for tr, va in kf.split(ds_indexes):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-multilingual-cased", 
                                                               num_labels=2).cuda()
    #model.load_state_dict(torch.load(models_by_folds[ifold]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
    epochs = 1
    total_steps = (1 + len(tr) // 32) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 0, 
                                                num_training_steps = total_steps)
    ss = torch.nn.Softmax(dim=1)
    i = 0
    train_losses = []
    for ep in range(epochs):
        np.random.shuffle(tr)
        optimizer.zero_grad()
        losses = []
        for t in tr:
            name1, name2, target, _ = dataset[t]
            tks = tokenizer.encode_plus(name1[:600], name2[:600], max_length=512, pad_to_max_length=False, 
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
                
        if len(losses) > 0:
            loss = sum(losses) / 32.0
            loss.backward()
            losses = []
            train_losses.append(loss.item())
            optimizer.step() 
            optimizer.zero_grad()
            scheduler.step()
            
        evaldf = []
        for t in va:
            name1, name2, target, cat_groupped = dataset[t]
            tks = tokenizer.encode_plus(name1[:600], name2[:600], max_length=512, pad_to_max_length=False,
                                return_attention_mask=True, return_tensors='pt', truncation=True)

            with torch.no_grad():
                score = ss(model(tks['input_ids'].cuda(), 
                                 attention_mask=tks['attention_mask'].cuda(),
                                 token_type_ids=tks['token_type_ids'].cuda()).logits)[0][1].item()
            evaldf.append((target, score, cat_groupped))
        evaldf = pd.DataFrame(evaldf)
        evaldf.columns = ["target", "scores", "categories"]
        m = pr_auc_macro(evaldf)
        m2 = roc_auc_score(evaldf.target.values, evaldf.scores.values)
        print('fold', ifold, 'epoch', ep, 'prauc(0.75)', round(m,3), 'rocauc', round(m2,3))
        torch.save(model.state_dict(), f'colors_{ifold}_{round(m,3)}_{round(m2,3)}.pth')
        oof[va] = evaldf.scores.values
    ifold += 1


# In[16]:


df_oof = pd.read_parquet('hackathon_files_for_participants_ozon/train_pairs.parquet')
df_oof['colorbert'] = oof
df_oof.to_parquet('oof_colorbert.parquet')

nn_features = pd.read_parquet('nn/train.parquet')["variantid1","variantid2","mbert","chstic"]
nn_features = nn_features.merge(df_oof, on=["variantid1","variantid2"], how='left')
nn_features.to_parquet('nn/train.parquet')

print(roc_auc_score(df_oof['target'], df_oof['colorbert']))
df_oof.head()


# In[ ]:




