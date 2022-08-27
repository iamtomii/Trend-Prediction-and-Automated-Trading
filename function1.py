from sklearn.preprocessing import MinMaxScaler 
from sklearn.model_selection import train_test_split
import numpy as np
from numpy import concatenate
from sklearn.metrics import mean_squared_error # for model evaluation metrics
from datetime import date
from datetime import datetime
from numpy import concatenate
import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler 
from numpy import concatenate
import streamlit as st
import math
from datetime import datetime
import datetime as dt

def add_label(df):
    idx = len(df.columns)
    col=df['RSI_15']
    new_col=[]
    for i in range(len(col)):
      if col[i]<=30:
        new_col.append(1)
      elif col[i]>=70:
        new_col.append(0)
      else:
        new_col.append(2)
    df.insert(loc=idx, column='signal', value=new_col)
def prep_data(datain,n, time_step):
    # 1. Y-array
    # tạo một mảng dựa trên time_step
    y_indices = np.arange(start=time_step, stop=len(datain), step=time_step)
     # tạo mảng y dựa trên y_indices
    y_tmp = datain[y_indices]

    rows_X = len(y_tmp)
    X_tmp = datain[range(time_step*rows_X)]
       # reshape X_tmp về shape(rows_X,time_step,2)
    X_tmp = np.reshape(X_tmp, (rows_X, time_step, n))
    return X_tmp, y_tmp

def prediction_test(data_trad,date,n,model,attributes,time_step):
  scaler = MinMaxScaler()
  days = np.datetime64(date)
  day_30m=np.datetime64(days) -np.timedelta64(60, 'm')
  datatest=data_trad.loc[day_30m:days]
  datatest=datatest[attributes]
  datatest_scale=scaler.fit_transform(datatest)
  data_testlist=[]
  data_test30m=[]
  for i in datatest_scale:
    data_testlist.append(list(i))
  for i in range(0, len(data_testlist)-time_step):
      if i==0:
          data_test30m=data_testlist[i:i+time_step]
      else: 
          data_test30m.extend(data_testlist[i:i+time_step])
  X_test30m=[]
  for i in data_test30m:
    X_test30m.append(np.array(i))
  X_test30m=np.array(X_test30m)
  X_test30m=np.reshape(X_test30m, (math.floor(len(X_test30m)/time_step), time_step, len(attributes)))
  test30mp=X_test30m
  test30mp =test30mp.reshape((test30mp.shape[0], 30*len(attributes)))
  K= model.predict(X_test30m)
  inv_pred_test30m1 = concatenate((K, test30mp[:,-7:]), axis=1)
  inv_pred_test30m1=scaler.inverse_transform(inv_pred_test30m1)
  inv_pred_test30m1[0:30]=[0]*len(attributes)
  Close_predict=list(inv_pred_test30m1[:,0])
  Signal_predict=list(inv_pred_test30m1[:,-1])
  return days,int(Close_predict[-1]),int(Signal_predict[-1])
def autotrading(start,end,data,initial_holdings,take_profit,loss,model,time_step,attributes):
  total_holding_coin=0
  total_holding_USD=0
  index=end-start
  index=int(index.total_seconds()//60)
  for i in range(0,index):
    starts=np.datetime64(start)+np.timedelta64(i,'m')
    total_price=data[data['Date']==starts]['Close']
    total_date,price_predict,total_trend=prediction_test(data,starts,30,model,attributes,time_step)
    if total_price.empty==True:
      continue
    if int(total_trend)==1:#vi the mua
      if initial_holdings>int(total_price):
        
        total_holding_coin=initial_holdings//int(total_price)
        initial_holdings-=(total_holding_coin*int(total_price))
        total_holding_USD+=total_holding_coin*int(total_price)
        st.text('Thị trường có xu hướng tăng mua '+str(int(total_holding_coin))+' btc với giá '+str(total_holding_coin*int(total_price)))
        st.text('số tiền còn lại: '+str(initial_holdings) )
    elif int(total_trend)==0: 
        if total_holding_coin!=0:
          initial_holdings+=total_holding_coin*int(total_price)
          st.text('Thị trường có xu hướng giảm bán '+str(int(total_holding_coin))+' btc với giá '+str(total_holding_coin*int(total_price)))
          st.text('số tiền còn lại: '+str(initial_holdings) )
          total_holding_coin=0
          total_holding_USD=0
    if total_holding_coin!=0:
      if ((total_holding_coin*int(total_price))-total_holding_USD)>=(total_holding_USD*take_profit)/100:
        initial_holdings+=total_holding_coin*int(total_price)
        st.text('take profit bán '+str(int(total_holding_coin))+' btc với giá '+str(total_holding_coin*int(total_price)))
        st.text('số tiền còn lại: '+str(initial_holdings ))
        total_holding_coin=0
        total_holding_USD=0
      elif ((total_holding_coin*int(total_price))-total_holding_USD)<=((-total_holding_USD*loss)/100):
        initial_holdings+=total_holding_coin*int(total_price)
        st.text('loss bán '+str(int(total_holding_coin))+' btc với giá '+str(total_holding_coin*int(total_price)))
        st.text('số tiền còn lại: '+str(initial_holdings ))
        total_holding_coin=0
        total_holding_USD=0
  return initial_holdings+total_holding_USD




