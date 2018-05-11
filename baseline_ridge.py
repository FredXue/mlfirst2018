## This is a baseline ridge regression model for 
## web traffic prediction. 
## author Jiayi Gao

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#read in data
train = pd.read_csv("train_2.csv")
key = pd.read_csv("key_2.csv")
submission = pd.read_csv("sample_submission_2.csv")


# In[38]:


# downsize_train = pd.read_csv("Downsample.csv")


# #### shuffle data

# In[39]:


sampled = train.sample(frac=1,random_state=2)


# In[41]:


sampled.reset_index(drop=True,inplace=True)


# In[42]:


sampled


# In[98]:


sampled = sampled.fillna(0)


# In[102]:


df_train = sampled.iloc[:,:-60]
df_target = sampled.iloc[:,-60:]


# In[128]:


df_target.head()


# #### feature engineering


# In[107]:


df_train['agent'] = df_train['Page'].str.split('_').str[-1]
df_train['access'] = df_train['Page'].str.split('_').str[-2]
df_train['project'] = df_train['Page'].str.split('_').str[-3]
df_train['language'] = df_train['project'].str.split('.').str[0]


# In[108]:


page = df_train['Page']
data = df_train.drop('Page',axis = 1)


# In[122]:


data.head()


# #### one-hot encoding categorical features

# In[110]:


agent_dummy = pd.get_dummies(data['agent'])
access_dummy = pd.get_dummies(data['access'])
project_dummy = pd.get_dummies(data['project'])
language_dummy = pd.get_dummies(data['language'])


# In[111]:


dummies = pd.concat([agent_dummy,access_dummy,project_dummy,language_dummy],axis=1)


# In[112]:


dummies.head()


# #### scale features to 0-1 range

# In[127]:


scaler_train = MinMaxScaler()
scaler_target = MinMaxScaler()


# In[129]:


scaled_train = scaler_train.fit_transform(data.iloc[:,:743])
scaled_target = scaler_target.fit_transform(df_target.values)


# In[130]:


scaled_train = pd.DataFrame(scaled_train,columns = data.columns.values[:743])
scaled_target = pd.DataFrame(scaled_target,columns=df_target.columns.values)


# In[138]:


processed_data = pd.concat([scaled_train,dummies,scaled_target],axis=1)


# #### train validation test split

# In[144]:


n_sample,col = processed_data.shape


# In[149]:


# percentage
train_size = int(0.7 * n_sample)
val_size = int(0.2 * n_sample)
test_size = int(0.1 * n_sample)


# In[150]:


X_train = processed_data.iloc[:train_size,:-60]
X_val = processed_data.iloc[train_size:(train_size+val_size),:-60]
X_test = processed_data.iloc[-test_size:,:-60]


# In[155]:


Y_train = processed_data.iloc[:train_size,-60:]
Y_val = processed_data.iloc[train_size:(train_size+val_size),-60:]
Y_test = processed_data.iloc[-test_size:,-60:]


# #### evaluation metric

# In[135]:


# Approximated differentiable SMAPE for one prediction
def differentiable_smape(true, predicted):
    epsilon = 0.1
    true_o = true
    pred_o = predicted
    summ = np.maximum(np.abs(true_o) + np.abs(pred_o) + epsilon, 0.5 + epsilon)
    smape = (np.abs(pred_o - true_o) / summ) * 2
    return smape


# In[169]:


def compute_mean_smape(pred,y_test):
    smape = []
    for i in range(len(pred)):
        fore = pred[i]
        actural = y_test[i]
        sm = kaggle_smape(actural,fore)
        smape.append(sm)
    mean_sm = np.asarray(smape).mean()
    return mean_sm


# In[168]:


# SMAPE as Kaggle calculates it
def kaggle_smape(true, predicted):
    true_o = true
    pred_o = predicted
    summ = np.abs(true_o) + np.abs(pred_o)
    smape = np.where(summ==0, 0, 2*np.abs(pred_o - true_o) / summ)
    return smape     


# #### fit ridge regression

# In[159]:


ridge = Ridge()
ridge.fit(X_train, Y_train)


# In[161]:


pred = ridge.predict(X_val)


# In[165]:


type(Y_val)


# #### tuning hyperparameter alpha

# In[160]:


alpha = [10**-9,10**-8,10**-7,10**-6,10**-5,10**-4,10**-3,10**-2,10**-1,1,10,100,1000,10**4]


# In[166]:


len(alpha)


# In[173]:


loss_hist = []
for a in alpha:
    model = Ridge(alpha=a)
    print('fitting Ridge with alpha='+str(a))
    model.fit(X_train,Y_train)
    preds = model.predict(X_val)
    revert_preds = scaler_target.inverse_transform(preds)
    revert_yvalue = scaler_target.inverse_transform(Y_val.values)
    summ = 0
    for i in range(preds.shape[0]):
        loss = compute_mean_smape(revert_preds[i,:],revert_yvalue[i,:])
        summ += loss
    loss_hist.append(summ/preds.shape[0])


fig = plt.figure(1,figsize=(11,7))
ax = plt.subplot(111)

ax.semilogx(alpha, loss_hist)
ax.grid()
ax.set_title("Validation Performance vs L2 Regularization")
ax.set_xlabel("L2-Penalty Regularization Parameter")
ax.set_ylabel("Mean SMAPE")
plt.show()


# In[175]:


np.argmin(loss_hist)


# #### retrain model using train+val, evaluate test

# In[179]:


X_tv = pd.concat([X_train,X_val])
Y_tv = pd.concat([Y_train,Y_val])


# In[181]:


model = Ridge(alpha=1)
model.fit(X_tv,Y_tv)


# In[182]:


pred = model.predict(X_test)
revert_pred = scaler_target.inverse_transform(pred)
revert_ytest = scaler_target.inverse_transform(Y_test.values)
summ = 0
for i in range(pred.shape[0]):
    loss = compute_mean_smape(revert_pred[i,:],revert_ytest[i,:])
    summ += loss
avg = summ/pred.shape[0]


# In[183]:


avg

