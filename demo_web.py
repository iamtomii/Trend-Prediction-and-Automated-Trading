import streamlit as st
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.figure_factory as ff
from pandas.tseries.offsets import BDay
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from tensorflow import keras
import warnings
from function1 import *
from datetime import date
from datetime import datetime
import datetime as dt
import mplfinance as mpf
import tensorflow as tf
import pandas_ta as ta
from sklearn.model_selection import train_test_split
import numpy as np
# Khai báo thư viện MinMaxScaler
from sklearn.preprocessing import MinMaxScaler 
from numpy import concatenate
from sklearn.metrics import mean_squared_error # for model evaluation metrics
from datetime import timedelta
import math
plt.style.use("fivethirtyeight")
warnings.filterwarnings("ignore")



st.set_page_config(layout="wide", initial_sidebar_state="expanded")
st.markdown('# HỘI THI TÌM KIẾM TÀI NĂNG CÔNG NGHỆ THÔNG TIN 2022\n# Dự đoán xu hướng thị trường và ứng dụng trong trading tự động\n') # brand name
st.text('Phạm Tommy, Huỳnh Trọng Đạo')
st.text('Khoa Công nghệ thông tin, Trường Đại học Tôn Đức Thắng')
   
# ------ layout setting---------------------------

SYMB = "BTC"

if SYMB != "Another Choice":
# # # # ------------------------Plot stock linecharts--------------------
    tab1, tab2,tab3,tab4 = st.tabs(["📈 Chart", "🗃 Data", "📊Result and prediction","💳Trading"])
    
    data=pd.read_csv("BTCUSD_1m_2.csv")
    data.index = pd.DatetimeIndex(data['Date'])
    #converse 'Date' columns to datime types
    data['Date']=[datetime.strptime(str(d)[0:19], '%Y-%m-%d %H:%M:%S') for d in data['Date']]
    plt.rcParams.update({'font.size': 8})
    fig, ax1 = plt.subplots(figsize=(6,3))
    ax1.set_ylabel('Price in USDT')
    ax1.set_xlabel('Date')
    ax1.set_title('BTCUSDT')
    ax1.plot('Date','Close',data=data, label='Close', linewidth=0.3, color='b')
    with tab1:
        with st.container():
            c1,c2 = st.columns((1,1))
            with c1:
                st.title('Biểu đồ thể hiện giá đóng của '+SYMB)
                st.pyplot(fig)
            with c2:
                data1=data[['Open','Close','High','Low', 'Volume']]
                #*******
            #*******
            # RSI
            #RSI
                data['Diff'] = data['Close'].transform(lambda x: x.diff())
                data['Up'] = data['Diff']
                data.loc[(data['Up']<0), 'Up'] = 0
                data['Down'] = data['Diff']
                data.loc[(data['Down']>0), 'Down'] = 0 
                data['Down'] = abs(data['Down'])
                data['avg_5up'] = data['Up'].transform(lambda x: x.rolling(window=5).mean())
                data['avg_5down'] = data['Down'].transform(lambda x: x.rolling(window=5).mean())
                data['avg_15up'] = data['Up'].transform(lambda x: x.rolling(window=15).mean())
                data['avg_15down'] = data['Down'].transform(lambda x: x.rolling(window=15).mean())
                data['RS_5'] = data['avg_5up'] / data['avg_5down']
                data['RS_15'] = data['avg_15up'] / data['avg_15down']
                data['RSI_5'] = 100 - (100/(1+data['RS_5']))
                data['RSI_15'] = 100 - (100/(1+data['RS_15']))
                data['RSI_ratio'] = data['RSI_5']/data['RSI_15']
                #*******************
                st.title('Biểu đồ nến và và RSI của '+SYMB)
                today = dt.datetime.strptime('2022-08-08', '%Y-%m-%d')
                YESTERDAY = today - BDay(0)
                DEFAULT_START=today -  BDay(1)
                START = st.date_input("From", value=DEFAULT_START, max_value=YESTERDAY)
                END = st.date_input("To", value=YESTERDAY, max_value=YESTERDAY, min_value=START)
    
                selected_data = data1.loc[START:END,['Open','Close','High','Low', 'Volume']]
                selected_data2 = data.loc[START:END,['Open','Close','High','Low','RSI_15', 'Volume']]
                apd = mpf.make_addplot(selected_data2['RSI_15'],panel=2,color='lime',ylim=(10,90),secondary_y=True)
                fig3=mpf.plot(selected_data, # the dataframe containing the OHLC (Open, High, Low and Close) data
                    type='candle', # use candlesticks 
                    volume=True, # also show the volume
                    mav=(5,15), # use two different moving averages
                    figratio=(3,1), # set the ratio of the figure
                    addplot=apd, # RSI
                    style='yahoo',  # choose the yahoo style
                title='Bitcoin Daily (RSI)' # title
                )
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.pyplot(fig3) 
        with tab2:
            st.title(' Dữ liệu từ ngày 01/01/2020 đến ngày 08/08/2022 của '+SYMB)
            st.write(data1[:][:])
        with tab3:
            data['Average'] = data.ta.midprice(length=1) #midprice
            data['MA40'] = data.ta.sma(length=40)
            data['MA80'] = data.ta.sma(length=80)
            data['MA160'] = data.ta.sma(length=160)
            columns2Drop = ['Diff','Up','Down','avg_5up','avg_5down','avg_15up','avg_15down','RS_5','RS_15','RSI_5']
            data = data.drop(labels = columns2Drop, axis=1)
            add_label(data)  
            data=data.fillna(0)
            attributes=['Close','RSI_15','RSI_ratio','Average','MA40','MA80','MA160','signal']
            X = data[attributes]
            print(X)
             # tạo bộ scaler
            scaler = MinMaxScaler()
            # fit scaler và thực hiện scale
            X_scaled=scaler.fit_transform(X)
            #####- Tạo train test
            X_train,X_test = train_test_split(X_scaled,test_size=0.2, shuffle=False)
            time_step = 30
            X_train, y_train = prep_data(X_train,len(attributes),time_step)
            X_test, y_test = prep_data(X_test,len(attributes), time_step)
            RNN_model = tf.keras.models.load_model('RNN_model.h5')
            LSTM_M = tf.keras.models.load_model('LSTM_model.h5')
            with st.container():
                c31,c32 = st.columns((1,1))
                pred_test = RNN_model.predict(X_test)
                testp=X_test
                testp =testp.reshape((testp.shape[0], 30*len(attributes)))
    
                # invert scaling for forecast
                inv_pred_test1 = concatenate((pred_test, testp[:,-7:]), axis=1)
                MSERNN=mean_squared_error(y_test, inv_pred_test1)
                inv_pred_test1=scaler.inverse_transform(inv_pred_test1)
                df_pred=pd.DataFrame()
                for i in range(len(attributes)):
                    df_pred[attributes[i]]=list(inv_pred_test1[:,i])
                inv_test1=scaler.inverse_transform(y_test)
                df_ytest=pd.DataFrame()
                for i in range(len(attributes)):
                    df_ytest[attributes[i]]=list(inv_test1[:,i])
                count=0
                for i in range(len(df_ytest['signal'])):
                    if df_ytest['signal'][i]==df_pred['signal'][i]:
                        count+=1
                accuracyRNN=count/(len(df_ytest['signal']))*100
                
                # ****LSTM
                # Predict the result on test data
                pred_test_LSTM = LSTM_M.predict(X_test)
                testpLSTM=X_test
                testpLSTM =testpLSTM.reshape((testpLSTM.shape[0], 30*len(attributes)))
                inv_pred_testLSTM = concatenate((pred_test_LSTM, testpLSTM[:,-7:]), axis=1)
                MSELSTM=mean_squared_error(y_test, inv_pred_testLSTM )
                inv_pred_testLSTM=scaler.inverse_transform(inv_pred_testLSTM)
                df_predLSTM=pd.DataFrame()
                for i in range(len(attributes)):
                    df_predLSTM[attributes[i]]=list(inv_pred_testLSTM[:,i])
                count1=0
                for i in range(len(df_ytest['signal'])):
                    if df_ytest['signal'][i]==df_predLSTM['signal'][i]:
                        count1+=1
                accuracyLSTM=count1/(len(df_ytest['signal']))*100
                with c31:
                    st.title('Mô hình RNN')
                    df_pl=pd.DataFrame()
                    df_pl['Test']=df_ytest['Close']
                    df_pl['Predict']=df_pred['Close']
                    st.text('RNN close price test')
                    st.line_chart(df_pl[1000:])
                    bar_plot=df_ytest['signal'].value_counts()
                    st.text('RNN trend predict')
                    st.bar_chart(bar_plot)
                with c32:
                    st.title('Mô hình LSTM')
                    df_plLSTM=pd.DataFrame()
                    df_plLSTM['Test']=df_ytest['Close']
                    df_plLSTM['Predict']=df_predLSTM['Close']
                    st.text('RNN close price test')
                    st.line_chart(df_plLSTM[1000:])
                    bar_plot=df_predLSTM['signal'].value_counts()
                    st.text('RNN trend predict')
                    st.bar_chart(bar_plot)
                st.text('so sánh MSE và ACCURACY')    
                l_MSE=[]
                l_ACC=[]
                L_MODEL=['RNN','LSTM']
                l_MSE.extend([MSERNN,MSELSTM])
                l_ACC.extend([accuracyRNN,accuracyLSTM])
                dfmetrics=pd.DataFrame()
                dfmetrics['Model']=L_MODEL
                dfmetrics['MSE']=l_MSE
                dfmetrics['Accuracy']=l_ACC
                st.write(dfmetrics)
                attributesd=['Date']+attributes
                data_trad=data[attributesd]
                #*******TEST
                st.title('Thử nghiệm dự đoán xu hướng')
                with st.form(key='my_form'):
	                date_test = st.text_input(label='Enter date predict',value='2022-07-23 15:00:00')
	                submit_button = st.form_submit_button(label='Dự đoán')
                if submit_button:
                    date_test=datetime.strptime(date_test[0:19], '%Y-%m-%d %H:%M:%S')
                    n=30
                    days,Close_predict,Signal_predict=prediction_test(data_trad,date_test,n,RNN_model,attributes,time_step)
                    if Signal_predict==1:
                        trend_txt='Kết quả là xu hướng tăng'
                    elif Signal_predict==0:
                        trend_txt='Kết quả là xu hướng giảm'
                    else:
                        trend_txt='Kết quả là xu hướng ngang'
                    st.metric(label="Kết quả dự đoán", value=Signal_predict)
                    st.text(trend_txt)
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=[date_test], y=[Signal_predict], name="stock_open"))
                    fig.layout.update(title_text='Biểu đồ dự đoán xu hướng ', xaxis_rangeslider_visible=True)
                    st.plotly_chart(fig)
        with tab4:
            #*****TRADING
            with st.container():
                c41,c42,c43= st.columns((1,4,1))
                with c42:
                    st.title('Ứng dụng vào trading tự động')
                    with st.form(key='trad_form'):
                        date_trading_test = st.text_input(label='Nhập ngày bắt đầu: ',value='2022-07-23 15:00:00')
                        time_input= st.text_input('Nhập thời gian kết thúc sau(giờ):',value='3')
                        initial_holdings = st.text_input('Nhập số tiền đầu tư (USD):',value='30000')
                        take_profit = st.number_input('Nhập phần trăm mục tiêu lợi nhuận(%)', min_value=5, max_value=15, value=5, step=1)
                        loss = st.number_input('Nhập phần trăm cắt lỗ(%)', min_value=1, max_value=10, value=5, step=1)
                        submit_button = st.form_submit_button(label='Bắt đầu')
            
                    if submit_button:
                        date_trading_test=datetime.strptime(date_trading_test[0:19], '%Y-%m-%d %H:%M:%S')
                        date_end=np.datetime64(date_trading_test)+np.timedelta64(int(time_input)*60,'m')
                        date_end = str(date_end).split('T')[0]+' '+ str(date_end).split('T')[1]
                        date_end=datetime.strptime(date_end[0:19], '%Y-%m-%d %H:%M:%S')
                        profit=autotrading(date_trading_test,date_end,data_trad,int(initial_holdings),take_profit,loss,RNN_model,time_step,attributes)
                        st.success("Bạn đã giao dịch thành công số tiền hiện có là: "+str(profit)+'USD')

    
    
    
    
    
