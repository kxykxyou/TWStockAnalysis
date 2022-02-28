import pandas as pd
# import mplfinance as mpf

def print_intro():
    stock_nums = [3008, 2330, 2891, 2317, 2454]
    stock_name = ['大立光', '台積電', '中信金', '鴻海  ', '聯發科']
    for i in range(len(stock_nums)):
        print(stock_name[i], end='\t')
        print(stock_nums[i])
    print('若不知道要分析哪檔股票，可以參考以上幾檔股票；當然證交所網站上找得到的已上市股票都可以試試看！')
    print('不建議從今年開始分析，資料量可能不足而導致無法繪圖或進行勝率分析！')
    print('下載每個月的資料要花5秒（太頻繁會被證交所封鎖IP），建議可從2018開始分析，分析時間越久下載檔案會越久！')
    print('P.S. 如果檔案已存在，執行程式則會跳過下載部分直接分析')

def get_stock_prices(Stock_No, year, month):
    import urllib.request as req
    import bs4
    import pandas as pd
    url = 'https://www.twse.com.tw/exchangeReport/STOCK_DAY?response=html&date='+str(year)+str(month).rjust(2,'0')+'01&stockNo='+str(Stock_No)
    with req.urlopen(url) as response:
        data = response.read().decode('utf-8')
    soup = bs4.BeautifulSoup(data, 'html.parser')
    results = soup.find_all('td')
    dates = []
    vol = []
    turnover = []
    op = []
    high = []
    low = []
    close = []
    count = 0
    for line in results[9:]:
        if count%9==0:
            dates.append(line.string)
        elif count%9==1:
            vol.append(line.string)
        elif count%9==2:
            turnover.append(line.string)
        elif count%9==3:
            op.append(line.string)
        elif count%9==4:
            high.append(line.string)
        elif count%9==5:
            low.append(line.string)
        elif count%9==6:
            close.append(line.string)
        count+=1

    df = pd.DataFrame({
        'Date':dates,
        'Volume':vol,
        'Turnover':turnover,
        'Open':op,
        'High':high,
        'Low':low,
        'Close':close
    })
    return df

def download_stock(Stock_No, from_):
    import pandas as pd
    import time
    import os
    import datetime
    t1 = time.time()
    df_total = pd.DataFrame({})
    print('start downloading...., 抓一個月的資料要花五秒, 你抓越久以前的資料要等越久....')
    time_now = datetime.datetime.now()
    year_now = time_now.year
    month_now = time_now.month
    dir = str(Stock_No) + '_' + str(from_)
    if not os.path.exists(os.getcwd() + '/' + dir + '/' + str(Stock_No) + '_from_' + str(from_) + '.csv'):
        for year in range(from_, year_now):
            for month in range(1, 13):
                df = get_stock_prices(Stock_No, year, month)
                df_total = df_total.append(df)
                time.sleep(5)
        for month in range(1, month_now+1):
            df = get_stock_prices(Stock_No, year_now, month)
            df_total = df_total.append(df)
        if not os.path.exists(dir):
            os.mkdir(dir)
            df_total.to_csv(os.getcwd() + '/' + dir + '/' + str(Stock_No) + '_from_' + str(from_) + '.csv')
        elif os.path.exists(dir):
            df_total.to_csv(os.getcwd() + '/' + dir + '/' + str(Stock_No) + '_from_' + str(from_) + '.csv')
        else:
            pass
    t2 = time.time()
    print('downloaded')
    print('Time cost :', str(round(t2-t1, 4)), 'seconds')

def df_for_args(Stock_No, from_):
    # 獲得作圖需要的五個parameters（date、open、high、low、close）
    ###########學到了df['Name'].to_list()將pd之Column轉為list、依照list.index(e)找到index
    import pandas as pd
    import os
    dir = str(Stock_No) + '_' + str(from_)
    df = pd.read_csv(os.getcwd() + '/' + dir + '/' + str(Stock_No) + '_from_' + str(from_) + '.csv')
    row = 0
    dateslist = df['Date'].to_list()
    for x in dateslist:
        if str(from_-1911) in x:
            row = dateslist.index(x)
            break
    df = df[row:]
    df_new=pd.DataFrame({
        'Date':(df['Date']),
        'Volume':df['Volume'],
        'Open':df['Open'],
        'High':df['High'],
        'Low':df['Low'],
        'Close':df['Close']
    })
    # 解決數字上有coma的問題
    def replacecomma(a):
        if type(a) == float:
            return a
        elif type(a) != float:
            return float(a.replace(',', ''))
        else:
            pass
    df_new['Volume'] = df_new['Volume'].apply(replacecomma)
    df_new['Open'] = df_new['Open'].apply(replacecomma)
    df_new['High'] = df_new['High'].apply(replacecomma)
    df_new['Low'] = df_new['Low'].apply(replacecomma)
    df_new['Close'] = df_new['Close'].apply(replacecomma)

    return df_new

def timechange(df):
    # 民國年-->西元年-->timestamp轉換以符合mpl_finance.candlestick_ohpc作圖的時間所需
    # 將每個時間資料取出前三個作為民國年，再將民國年轉為西元年
    # 西元年yyyy/mm/dd使用datetime.datetime.strptime()格式化轉為yyyy-mm-dd
    # 使用mdate.date2num()(即為matplotlib.date.date2num())將上述格式化資料轉為timestamp
    import pandas as pd
    from matplotlib import dates as mdates
    import datetime
    newdates = []
    for t in df['Date']:
        t = str(int(t.split('/')[0])+1911)+t[3:]
        t1 = datetime.datetime.strptime(t, '%Y/%m/%d')
        newdates.append(t1)
    df['Date'] = newdates
    df_ohlc = pd.DataFrame({
        'Volume': df['Volume'],
        'Open': df['Open'],
        'High': df['High'],
        'Low': df['Low'],
        'Close': df['Close']
    })
    df_new = pd.merge(df['Date'], df_ohlc, left_index=True, right_index=True, how='left')
    df_new = df_new.set_index(pd.DatetimeIndex(df_new['Date']))
    return df_new

def add_moving_average(df, span):
    # 利用df['Name'].rolling(span).mena()找出移動平均
    # 利用df.fillna(0)將NaN轉為0
    # 利用pd.merge(df1, df2, left_index=True, right_index=True, how='left')合併兩個df
    df_copy = df.copy()
    df_copy['MA_'+str(span)] = df['Close'].rolling(span).mean().fillna(0)
    return pd.merge(df, df_copy['MA_'+str(span)], left_index=True, right_index=True, how='left')

def add_KD(df, span=9):
    # df['Name'].rolling(span).min/max()求移動最大最小值
    # 利用for index, element in enumerate(list)，在做迴圈的同時取出index以及element
    df_copy = df.copy()
    df_copy['min'] = df_copy['Low'].rolling(span).min()
    df_copy['max'] = df_copy['High'].rolling(span).max()
    df_copy['RSV'] = (df_copy['Close']-df_copy['min'])/(df_copy['max']-df_copy['min'])
    df_KD = pd.DataFrame({'Close':df['Close']})
    df_copy = df_copy.fillna(0)
    K_list = [0.5]
    for index, rsv in enumerate(list(df_copy['RSV'])):
        K_yesterday = K_list[index]
        K_today = 2/3*K_yesterday + 1/3*rsv
        K_list.append(K_today)
    df_KD['K'] = K_list[1:]
    D_list = [0.5]
    for index, K in enumerate(list(df_KD['K'])):
        D_yesterday = D_list[index]
        D_today = 2/3*D_yesterday + 1/3*K
        D_list.append(D_today)
    df_KD['D'] = D_list[1:]
    df_KD['K'] *= 100
    df_KD['D'] *= 100
    return pd.merge(df, df_KD[['K', 'D']], left_index=True, right_index=True, how='left')

def add_MACD(df, span1=12, span2=26, DIFspan=9):
    import pandas as pd
    df_copy = df.copy()

    df_copy['DI'] = (df_copy['High']+df_copy['Low']+2*df_copy['Close'])/4
    df_copy['EMA_' + str(span1)] = [float(0)]*len(df_copy['Close'])
    df_copy['EMA_' + str(span2)] = [float(0)] * len(df_copy['Close'])
    df_copy['DIF'] = [float(0)] * len(df_copy['Close'])
    df_copy['MACD'] = [float(0)] * len(df_copy['Close'])
    df_copy['DIF-MACD'] = [float(0)] * len(df_copy['Close'])
    df_copy['DIF-MACD_positive'] = [float(0)] * len(df_copy['Close'])
    df_copy['DIF-MACD_negtive'] = [float(0)] * len(df_copy['Close'])
    df_copy['EMA_' + str(span1)][span1 - 1] = sum(df_copy['DI'][:span1]) / span1
    df_copy['EMA_' + str(span2)][span2 - 1] = sum(df_copy['DI'][:span2]) / span2

    for index in range(len(df_copy)):
        if index >= span1:
            df_copy['EMA_' + str(span1)][index] = (df_copy['DI'][index]*2+df_copy['EMA_' + str(span1)][index-1]*(span1-1))/(span1+1)
        else:
            pass

    for index in range(len(df_copy)):
        if index >= span2:
            df_copy['EMA_' + str(span2)][index] = (df_copy['DI'][index]*2+df_copy['EMA_' + str(span2)][index-1]*(span2-1))/(span2+1)
        else:
            pass
    df_copy['DIF'][span2-1:] = df_copy['EMA_' + str(span1)][span2-1:] - df_copy['EMA_' + str(span2)][span2-1:]
    df_copy['MACD'][span2 + DIFspan-2] = sum(df_copy['DIF'][span2-1:span2+DIFspan-1]) / DIFspan

    for index in range(len(df_copy)):
        if index >= span2+DIFspan-1:
            df_copy['MACD'][index] = (df_copy['DIF'][index]*2+df_copy['MACD'][index-1]*(DIFspan-1))/(DIFspan+1)
        else:
            pass

    df_copy['DIF-MACD'][span2 + DIFspan-2:] = df_copy['DIF'][span2 + DIFspan-2:]-df_copy['MACD'][span2 + DIFspan-2:]
    def get_pos(a):
        if a >= 0:
            return a
        else:
            pass

    def get_neg(a):
        if a < 0:
            return a
        else:
            pass
    df_copy['DIF-MACD_positive'] = df_copy['DIF-MACD'].apply(get_pos)
    df_copy['DIF-MACD_negtive'] = df_copy['DIF-MACD'].apply(get_neg)
    df_return = pd.DataFrame({
        # 'EMA_12':df_copy['EMA_' + str(span1)],
        # 'EMA_26': df_copy['EMA_' + str(span2)],
        'DIF':df_copy['DIF'],
        'MACD': df_copy['MACD'],
        'DIF-MACD': df_copy['DIF-MACD'],
        'DIF-MACD_positive':df_copy['DIF-MACD_positive'],
        'DIF-MACD_negtive':df_copy['DIF-MACD_negtive']
    })
    return pd.merge(df, df_return, left_index=True, right_index=True, how='left')

def add_RSI(df, span):
    import numpy as np
    import pandas as pd
    df_copy = df.copy()
    df_copy = df_copy[['Close']]
    df_copy['Change'] = [np.nan] * len(df_copy)
    df_copy['avgGain'] = [np.nan] * len(df_copy)
    df_copy['avgLoss'] = [np.nan] * len(df_copy)
    for index in range(1, len(df_copy)):
        df_copy['Change'][index] =  df_copy['Close'][index]-df_copy['Close'][index-1]
    def get_gain(a):
        if a > 0:
            return a
        else:
            return 0
    def get_loss(a):
        if a < 0:
            return abs(a)
        else:
            return 0
    df_copy['Gain'] = df_copy['Change'].apply(get_gain)
    df_copy['Loss'] = df_copy['Change'].apply(get_loss)
    df_copy['avgGain'][span] = sum(df_copy['Gain'][1:span+1]) / span
    df_copy['avgLoss'][span] = sum(df_copy['Loss'][1:span+1]) / span
    for index in range(1, len(df_copy)):
        if index >= span+1:
            df_copy['avgGain'][index] = (df_copy['Gain'][index]+df_copy['avgGain'][index-1]*(span-1))/span
        else:
            pass
    for index in range(1, len(df_copy)):
        if index >= span+1:
            df_copy['avgLoss'][index] = (df_copy['Loss'][index]+df_copy['avgLoss'][index-1]*(span-1))/span
        else:
            pass
    df_copy['RS'] = df_copy['avgGain']/df_copy['avgLoss']
    def cal_RSI(RS):
        return (100*RS)/(RS+1)
    df_copy['RSI_' + str(span)] = df_copy['RS'].apply(cal_RSI)

    return pd.merge(df, df_copy['RSI_' + str(span)], left_index=True, right_index=True, how='left')

def add_MACDsignals(df):
    import numpy as np
    df_copy = df.copy()
    df_copy['MACD_GC'] = [np.nan] * len(df_copy)
    df_copy['MACD_DC'] = [np.nan] * len(df_copy)
    df_copy['MACD_buysignal'] = [np.nan] * len(df_copy)
    df_copy['MACD_sellsignal'] = [np.nan] * len(df_copy)
    for index in range(1, len(df_copy)):
        if df_copy['DIF'][index-1]-df_copy['MACD'][index-1] > 0 and df_copy['DIF'][index]-df_copy['MACD'][index] < 0 :
            df_copy['MACD_DC'][index] = df_copy['DIF'][index]
            if df_copy['MACD_DC'][index] > 0:
                df_copy['MACD_sellsignal'][index] = df_copy['MACD_DC'][index]
        elif df_copy['DIF'][index-1]-df_copy['MACD'][index-1] < 0 and df_copy['DIF'][index]-df_copy['MACD'][index] > 0 :
            df_copy['MACD_GC'][index] = df_copy['DIF'][index]
            if df_copy['MACD_GC'][index] < 0:
                df_copy['MACD_buysignal'][index] = df_copy['MACD_GC'][index]
        else:
            pass

    df_return = df_copy[[
        'MACD_GC',
        'MACD_DC',
        'MACD_buysignal',
        'MACD_sellsignal'
    ]]

    return pd.merge(df, df_return, left_index=True, right_index=True, how='left')

def analyze_MACD(df, daysafter):
    import numpy as np
    df_copy = df.copy()
    df_copy['MACD_buysignal_' +str(daysafter) +'_pricediff'] = [np.nan] * len(df_copy)
    # df_copy['MACD_buysignal_' +str(daysafter) +'_pricediff_ratio'] = [np.nan] * len(df_copy)
    df_copy['MACD_sellsignal_' +str(daysafter) +'_pricediff'] = [np.nan] * len(df_copy)
    # df_copy['MACD_sellsignal_' +str(daysafter) +'_pricediff_ratio'] = [np.nan] * len(df_copy)
    for index in range(0, len(df_copy)-daysafter):
        if df_copy['MACD_buysignal'][index] < 0:
            df_copy['MACD_buysignal_' +str(daysafter) +'_pricediff'][index] = df_copy['Close'][index+daysafter] - df_copy['Close'][index]
            # df_copy['MACD_buysignal_' +str(daysafter) +'_pricediff_ratio'][index] = (df_copy['Close'][index + daysafter] - df_copy['Close'][index])/df_copy['Close'][index]

        if df_copy['MACD_sellsignal'][index] > 0:
            df_copy['MACD_sellsignal_' +str(daysafter) +'_pricediff'][index] = df_copy['Close'][index+daysafter] - df_copy['Close'][index]
            # df_copy['MACD_sellsignal_' +str(daysafter) +'_pricediff_ratio'][index] = (df_copy['Close'][index+daysafter] - df_copy['Close'][index])/df_copy['Close'][index]

    df_return = df_copy[[
        'MACD_buysignal_' +str(daysafter) +'_pricediff',
        # 'MACD_buysignal_' +str(daysafter) +'_pricediff_ratio',
        'MACD_sellsignal_' +str(daysafter) +'_pricediff',
        # 'MACD_sellsignal_' +str(daysafter) +'_pricediff_ratio'
    ]]

    return pd.merge(df, df_return, left_index=True, right_index=True, how='left')

def analyze_KD(df, upper=80, lower=20):
    import numpy as np
    df_copy = df.copy()
    df_copy['KD_GC'] = [np.nan] * len(df_copy)
    df_copy['KD_DC'] = [np.nan] * len(df_copy)
    for index in range(0, len(df_copy)-1):
        if df_copy['K'][index]-df_copy['D'][index] > 0 and df_copy['K'][index+1]-df_copy['D'][index+1] < 0 :
            if df_copy['K'][index] > upper and df_copy['D'][index] > upper:
                df_copy['KD_DC'][index] = df_copy['K'][index]
            else:
                pass
        elif df_copy['K'][index]-df_copy['D'][index] < 0 and df_copy['K'][index+1]-df_copy['D'][index+1] > 0:
            if df_copy['K'][index] < lower and df_copy['D'][index] < lower:
                df_copy['KD_GC'][index] = df_copy['K'][index]
            else:
                pass
        else:
            pass
    return pd.merge(df, df_copy[['KD_GC', 'KD_DC']], left_index=True, right_index=True, how='left')

def add_KD_standard(df, upper, lower):
    df_copy = df.copy()
    df_copy['KD_upper_line'] = [upper]*len(df)
    df_copy['KD_lower_line'] = [lower]*len(df)
    return pd.merge(df, df_copy[['KD_upper_line', 'KD_lower_line']], left_index=True, right_index=True, how='left')

def analyze_RSI(df, upper, lower, span):
    import numpy as np
    df_copy = df.copy()
    df_copy['RSI_upperGC'] = [np.nan] * len(df_copy)
    df_copy['RSI_lowerGC'] = [np.nan] * len(df_copy)
    # df_copy['RSI_sell'] = [np.nan] * len(df_copy)
    # df_copy['RSI_buy'] = [np.nan] * len(df_copy)

    for index in range(0, len(df_copy) - 1):
        if df_copy['RSI_'+str(span)][index-1]-upper < 0 and df_copy['RSI_'+str(span)][index]-upper > 0:
            df_copy['RSI_upperGC'][index] = df_copy['RSI_'+str(span)][index]
        elif df_copy['RSI_'+str(span)][index-1]-lower > 0 and df_copy['RSI_'+str(span)][index]-lower < 0:
            df_copy['RSI_lowerGC'][index] = df_copy['RSI_'+str(span)][index]
        else:
            pass

    df_copy['RSI_peak'] = [np.nan] * len(df_copy)
    df_copy['RSI_lowest'] = [np.nan] * len(df_copy)

    for index in range(1, len(df_copy)-1):
        if df_copy['RSI_'+str(span)][index] > upper:
            if df_copy['RSI_' + str(span)][index]>df_copy['RSI_' + str(span)][index+1] and df_copy['RSI_' + str(span)][index]>df_copy['RSI_' + str(span)][index-1]:
                df_copy['RSI_peak'][index] = df_copy['RSI_' + str(span)][index]
    for index in range(1, len(df_copy) - 1):
        if df_copy['RSI_' + str(span)][index] < lower:
            if df_copy['RSI_' + str(span)][index] < df_copy['RSI_' + str(span)][index + 1] and df_copy['RSI_' + str(span)][index] < df_copy['RSI_' + str(span)][index - 1]:
                df_copy['RSI_lowest'][index] = df_copy['RSI_' + str(span)][index]

    # 以下為RSI過熱/過冷搭配隔日價格破五日均線所產生的buy/sell訊號，但缺乏可信度。
    # for index in range(0, len(df_copy)-1):
    #     if df_copy['RSI_upperGC'][index] > 0 and df_copy['Close'][index+1] < df_copy['MA_5'][index+1]:
    #         df_copy['RSI_sell'][index] = df_copy['RSI_upperGC'][index]
    #     else:
    #         pass
    #
    # for index in range(0, len(df_copy)-1):
    #     if df_copy['RSI_lowerGC'][index] > 0 and df_copy['Close'][index+1] > df_copy['MA_5'][index+1]:
    #         df_copy['RSI_buy'][index] = df_copy['RSI_lowerGC'][index]
    #     else:
    #         pass

    return pd.merge(df, df_copy[['RSI_peak','RSI_lowest']], left_index=True, right_index=True, how='left')

def add_RSI_standard(df, upper, lower):
    df_copy = df.copy()
    df_copy['RSI_upper_line'] = [upper]*len(df)
    df_copy['RSI_lower_line'] = [lower]*len(df)
    return pd.merge(df, df_copy[['RSI_upper_line', 'RSI_lower_line']], left_index=True, right_index=True, how='left')

def draw_MACDanalysis(df, Stock_No, from_):
    df = df[(df.index > str(from_))]
    apds_MA_MACD = [mpf.make_addplot(df['DIF-MACD_positive'],type='bar',width=0.7,panel=2,
                             color='#FF0000',alpha=1,secondary_y=False),
            mpf.make_addplot(df['DIF-MACD_negtive'], type='bar', width=0.7, panel=2,
                             color='#008800', alpha=1, secondary_y=False),
            mpf.make_addplot(df['DIF'],panel=2,color='fuchsia',secondary_y=False),
            mpf.make_addplot(df['MACD'],panel=2,color='#00FF00',secondary_y=False),
            mpf.make_addplot(df['MACD_buysignal'], type='scatter', markersize=100, marker='^', panel=2,
                                     color='#FF0000', secondary_y=False),
            mpf.make_addplot(df['MACD_sellsignal'], type='scatter', markersize=100, marker='v', panel=2,
                                     color='#006400', secondary_y=False),
            mpf.make_addplot(df['MA_5'], panel=0, color='#ADFF2F', secondary_y=False),
            mpf.make_addplot(df['MA_20'], panel=0, color='#1E90FF', secondary_y=False),
            mpf.make_addplot(df['MA_60'], panel=0, color='#006400', secondary_y=False)
            ]
    mpf.plot(df, type='candle', addplot=apds_MA_MACD, figscale=2, figratio=(12, 8), title='\n'+str(Stock_No)+'\n'+'MACD Analysis',
             style='charles', volume=True, volume_panel=1, panel_ratios=(6, 2, 3))

def draw_KDanalysis(df, Stock_No, from_):
    df = df[(df.index > str(from_))]
    apds_KD = [mpf.make_addplot(df['K'],panel=2,color='#FF4500',secondary_y=False),
               mpf.make_addplot(df['D'],panel=2,color='#0000CD',secondary_y=False),
               mpf.make_addplot(df['KD_GC'], type='scatter', markersize=100, marker='^', panel=2, color='#FF0000',
                          secondary_y=False),
               mpf.make_addplot(df['KD_DC'], type='scatter', markersize=100, marker='v', panel=2, color='#006400',
                          secondary_y=False),
               mpf.make_addplot(df['KD_upper_line'], panel=2, color='#880000', secondary_y=False),
               mpf.make_addplot(df['KD_lower_line'], panel=2, color='#880000', secondary_y=False),
               mpf.make_addplot(df['MA_5'], panel=0, color='#ADFF2F', secondary_y=False),
               mpf.make_addplot(df['MA_20'], panel=0, color='#1E90FF', secondary_y=False),
               mpf.make_addplot(df['MA_60'], panel=0, color='#006400', secondary_y=False)]
    mpf.plot(df, type='candle', addplot=apds_KD, figscale=2, figratio=(12, 8), title='\n'+str(Stock_No)+'\n'+'KD Analysis',
             style='charles', volume=True, volume_panel=1, panel_ratios=(6, 2, 3))

def draw_RSIanalysis(df, Stock_No, from_):
    df = df[(df.index > str(from_))]
    apds_RSI = [mpf.make_addplot(df['RSI_5'], panel=2, color='#00AAAA', secondary_y=False),
            mpf.make_addplot(df['RSI_upper_line'], panel=2, color='#880000', secondary_y=False),
            mpf.make_addplot(df['RSI_lower_line'], panel=2, color='#880000', secondary_y=False),
            mpf.make_addplot(df['RSI_peak'], type='scatter', markersize=100, marker='^', panel=2, color='#FF0000', secondary_y=False),
            mpf.make_addplot(df['RSI_lowest'], type='scatter', markersize=100, marker='^', panel=2, color='#FF0000', secondary_y=False),
            mpf.make_addplot(df['MA_5'], panel=0, color='#ADFF2F', secondary_y=False),
            mpf.make_addplot(df['MA_20'], panel=0, color='#1E90FF', secondary_y=False),
            mpf.make_addplot(df['MA_60'], panel=0, color='#006400', secondary_y=False)
            ]
    mpf.plot(df, type='candle', addplot=apds_RSI, figscale=2, figratio=(12, 8), title='\n'+str(Stock_No)+'\n'+'RSI Analysis',
             style='charles', volume=True, volume_panel=1, panel_ratios=(6, 2, 3))

def main_drawfig(df, Stock_No, from_):
    draw_MACDanalysis(df, Stock_No, from_)
    draw_KDanalysis(df, Stock_No, from_)
    draw_RSIanalysis(df, Stock_No, from_)

def savefig_MACDanalysis(df, Stock_No, from_, filename='MACD分析圖', filetype='png'):
    df = df[(df.index > str(from_))]
    apds_MA_MACD = [mpf.make_addplot(df['DIF-MACD_positive'],type='bar',width=0.7,panel=2,
                             color='#FF0000',alpha=1,secondary_y=False),
            mpf.make_addplot(df['DIF-MACD_negtive'], type='bar', width=0.7, panel=2,
                             color='#008800', alpha=1, secondary_y=False),
            mpf.make_addplot(df['DIF'],panel=2,color='fuchsia',secondary_y=False),
            mpf.make_addplot(df['MACD'],panel=2,color='#00FF00',secondary_y=False),
            mpf.make_addplot(df['MACD_buysignal'], type='scatter', markersize=100, marker='^', panel=2,
                                     color='#FF0000', secondary_y=False),
            mpf.make_addplot(df['MACD_sellsignal'], type='scatter', markersize=100, marker='v', panel=2,
                                     color='#006400', secondary_y=False),
            mpf.make_addplot(df['MA_5'], panel=0, color='#ADFF2F', secondary_y=False),
            mpf.make_addplot(df['MA_20'], panel=0, color='#1E90FF', secondary_y=False),
            mpf.make_addplot(df['MA_60'], panel=0, color='#006400', secondary_y=False)
            ]
    mpf.plot(df, type='candle', addplot=apds_MA_MACD, figscale=2, figratio=(12, 8), title='\n'+str(Stock_No)+'\n'+'MACD Analysis',
             style='charles', volume=True, volume_panel=1, panel_ratios=(6, 2, 3), savefig=str(filename)+'.'+str(filetype))

def savefig_KDanalysis(df, Stock_No, from_, filename='KD分析圖', filetype='png'):
    df = df[(df.index > str(from_))]
    apds_KD = [mpf.make_addplot(df['K'],panel=2,color='#FF4500',secondary_y=False),
            mpf.make_addplot(df['D'],panel=2,color='#0000CD',secondary_y=False),
            mpf.make_addplot(df['KD_GC'], type='scatter', markersize=100, marker='^', panel=2, color='#FF0000',
                          secondary_y=False),
            mpf.make_addplot(df['KD_DC'], type='scatter', markersize=100, marker='v', panel=2, color='#006400',
                          secondary_y=False),
            mpf.make_addplot(df['KD_upper_line'], panel=2, color='#880000', secondary_y=False),
            mpf.make_addplot(df['KD_lower_line'], panel=2, color='#880000', secondary_y=False),
            mpf.make_addplot(df['MA_5'], panel=0, color='#ADFF2F', secondary_y=False),
            mpf.make_addplot(df['MA_20'], panel=0, color='#1E90FF', secondary_y=False),
            mpf.make_addplot(df['MA_60'], panel=0, color='#006400', secondary_y=False)
            ]
    mpf.plot(df, type='candle', addplot=apds_KD, figscale=2, figratio=(12, 8), title='\n'+str(Stock_No)+'\n'+'KD Analysis',
                style='charles', volume=True, volume_panel=1, panel_ratios=(6, 2, 3), savefig=str(filename)+'.'+str(filetype))

def savefig_RSIanalysis(df, Stock_No, from_, filename='RSI分析圖', filetype='png'):
    df = df[(df.index > str(from_))]
    apds_RSI = [mpf.make_addplot(df['RSI_5'], panel=2, color='#00AAAA', secondary_y=False),
            # mpf.make_addplot(df['RSI_10'], panel=2, color='#0000AA', secondary_y=False),
            mpf.make_addplot(df['RSI_upper_line'], panel=2, color='#880000', secondary_y=False),
            mpf.make_addplot(df['RSI_lower_line'], panel=2, color='#880000', secondary_y=False),
                mpf.make_addplot(df['RSI_peak'], type='scatter', markersize=100, marker='^', panel=2, color='#FF0000',
                                 secondary_y=False),
                mpf.make_addplot(df['RSI_lowest'], type='scatter', markersize=100, marker='^', panel=2, color='#FF0000',
                                 secondary_y=False),
            mpf.make_addplot(df['MA_5'], panel=0, color='#ADFF2F', secondary_y=False),
            mpf.make_addplot(df['MA_20'], panel=0, color='#1E90FF', secondary_y=False),
            mpf.make_addplot(df['MA_60'], panel=0, color='#006400', secondary_y=False)
            ]
    mpf.plot(df, type='candle', addplot=apds_RSI, figscale=2, figratio=(12, 8), title='\n'+str(Stock_No)+'\n'+'RSI Analysis',
             style='charles', volume=True, volume_panel=1, panel_ratios=(6, 2, 3), savefig=str(filename)+'.'+str(filetype))

def main_savefig(df, Stock_No, from_):
    import os
    print('saving figures....')
    if not os.path.exists(str(Stock_No)+'_'+str(from_)+'_MACD.png'):
        savefig_MACDanalysis(df, Stock_No=Stock_No, from_=from_, filename=str(Stock_No)+'_'+str(from_)+'_MACD', filetype='png')
        print('MACD圖 saved!')
    else:
        print('MACD圖 已存在!')

    if not os.path.exists(str(Stock_No)+'_'+str(from_)+'_KD.png'):
        savefig_KDanalysis(df, Stock_No=Stock_No, from_=from_, filename=str(Stock_No)+'_'+str(from_)+'KD', filetype='png')
        print('KD圖 saved!')
    else:
        print('KD圖 已存在!')

    if not os.path.exists(str(Stock_No)+'_'+str(from_)+'_RSI.png'):
        savefig_RSIanalysis(df, Stock_No=Stock_No, from_=from_, filename=str(Stock_No)+'_'+str(from_)+'RSI', filetype='png')
        print('RSI圖 saved!')
    else:
        print('RSI圖 已存在!')

def get_MACD_result(df):
    # 都是價差
    sell_5 = []
    buy_5 = []
    sell_10 = []
    buy_10 = []
    for index in range(0, len(df)-5):
        if df['MACD_sellsignal'][index] >0 :
            sell_5.append(df['Close'][index+5]-df['Close'][index])
        if df['MACD_buysignal'][index] < 0 :
            buy_5.append(df['Close'][index+5]-df['Close'][index])
        else:
            pass
    for index in range(0, len(df) - 10):
        if df['MACD_sellsignal'][index] >0:
            sell_10.append(df['Close'][index+10] - df['Close'][index])
        if df['MACD_buysignal'][index] < 0:
            buy_10.append(df['Close'][index+10] - df['Close'][index])
        else:
            pass
    win_sell_5 = [i for i in sell_5 if i < 0]
    win_sell_10 = [i for i in sell_10 if i < 0]
    win_buy_5 = [i for i in buy_5 if i > 0]
    win_buy_10 = [i for i in buy_10 if i > 0]
    avg_sell_5 = sum(sell_5)/len(sell_5)
    avg_sell_10 = sum(sell_10) / len(sell_10)
    avg_buy_5 = sum(buy_5) / len(buy_5)
    avg_buy_10 = sum(buy_10) / len(buy_10)
    winrate_sell_5 = round(len(win_sell_5)/len(sell_5)*100, 2)
    winrate_sell_10 = round(len(win_sell_10) / len(sell_10) * 100, 2)
    winrate_buy_5 = round(len(win_buy_5) / len(buy_5)*100, 2)
    winrate_buy_10 = round(len(win_buy_10) / len(buy_10)*100, 2)
    print('MACD零軸上死亡交叉做空 5日勝率：', winrate_sell_5, '%', '操作次數：', len(sell_5), '次', '平均價格漲跌：', round(avg_sell_5, 2))
    print('MACD零軸上死亡交叉做空10日勝率：', winrate_sell_10, '%', '操作次數：', len(sell_10), '次', '平均價格漲跌：', round(avg_sell_10, 2))
    print('MACD零軸下黃金交叉做多 5日勝率：', winrate_buy_5, '%', '操作次數：', len(buy_5), '次', '平均價格漲跌：', round(avg_buy_5,2))
    print('MACD零軸下黃金交叉做多10日勝率：', winrate_buy_10, '%', '操作次數：', len(buy_10), '次', '平均價格漲跌：', round(avg_buy_10,2))

def get_RSI_result(df):
    sell_5 = []
    buy_5 = []
    sell_10 = []
    buy_10 = []
    for index in range(0, len(df)-5):
        if df['RSI_peak'][index] > 0 :
            sell_5.append(df['Close'][index+5]-df['Close'][index])
        if df['RSI_lowest'][index] >0 :
            buy_5.append(df['Close'][index+5]-df['Close'][index])
        else:
            pass
    for index in range(0, len(df) - 10):
        if df['RSI_peak'][index] > 0:
            sell_10.append(df['Close'][index+10] - df['Close'][index])
        if df['RSI_lowest'][index] > 0:
            buy_10.append(df['Close'][index+10] - df['Close'][index])
        else:
            pass

    win_sell_5 = [i for i in sell_5 if i < 0]
    win_sell_10 = [i for i in sell_10 if i < 0]
    win_buy_5 = [i for i in buy_5 if i > 0]
    win_buy_10 = [i for i in buy_10 if i > 0]
    avg_sell_5 = sum(sell_5)/len(sell_5)
    avg_sell_10 = sum(sell_10) / len(sell_10)
    avg_buy_5 = sum(buy_5) / len(buy_5)
    avg_buy_10 = sum(buy_10) / len(buy_10)
    winrate_sell_5 = round(len(win_sell_5)/len(sell_5)*100, 2)
    winrate_sell_10 = round(len(win_sell_10) / len(sell_10) * 100, 2)
    winrate_buy_5 = round(len(win_buy_5) / len(buy_5)*100, 2)
    winrate_buy_10 = round(len(win_buy_10) / len(buy_10)*100, 2)
    print('RSI大於90出現波峰做空 5日勝率：', winrate_sell_5, '%', '操作次數：', len(sell_5), '次', '平均價格漲跌：', round(avg_sell_5, 2))
    print('RSI大於90出現波峰做空10日勝率：', winrate_sell_10, '%', '操作次數：', len(sell_10), '次', '平均價格漲跌：', round(avg_sell_10, 2))
    print('RSI小於10出現波谷做多 5日勝率：', winrate_buy_5, '%', '操作次數：', len(buy_5), '次', '平均價格漲跌：', round(avg_buy_5,2))
    print('RSI小於10出現波谷做多10日勝率：', winrate_buy_10, '%', '操作次數：', len(buy_10), '次', '平均價格漲跌：', round(avg_buy_10,2))

def get_KD_result(df):
    sell_5 = []
    buy_5 = []
    sell_10 = []
    buy_10 = []
    for index in range(0, len(df)-5):
        if df['KD_DC'][index] > 0 :
            sell_5.append(df['Close'][index+5]-df['Close'][index])
        if df['KD_GC'][index] > 0 :
            buy_5.append(df['Close'][index+5]-df['Close'][index])
        else:
            pass
    for index in range(0, len(df) - 10):
        if df['KD_DC'][index] > 0:
            sell_10.append(df['Close'][index+10] - df['Close'][index])
        if df['KD_GC'][index] > 0:
            buy_10.append(df['Close'][index+10] - df['Close'][index])
        else:
            pass

    win_sell_5 = [i for i in sell_5 if i < 0]
    win_sell_10 = [i for i in sell_10 if i < 0]
    win_buy_5 = [i for i in buy_5 if i > 0]
    win_buy_10 = [i for i in buy_10 if i > 0]
    avg_sell_5 = sum(sell_5)/len(sell_5)
    avg_sell_10 = sum(sell_10) / len(sell_10)
    avg_buy_5 = sum(buy_5) / len(buy_5)
    avg_buy_10 = sum(buy_10) / len(buy_10)
    winrate_sell_5 = round(len(win_sell_5)/len(sell_5)*100, 2)
    winrate_sell_10 = round(len(win_sell_10) / len(sell_10) * 100, 2)
    winrate_buy_5 = round(len(win_buy_5) / len(buy_5)*100, 2)
    winrate_buy_10 = round(len(win_buy_10) / len(buy_10)*100, 2)
    print('KD大於80出現死亡交叉做空 5日勝率：', winrate_sell_5, '%', '操作次數：', len(sell_5), '次', '平均價格漲跌：', round(avg_sell_5, 2))
    print('KD大於80出現死亡交叉做空10日勝率：', winrate_sell_10, '%', '操作次數：', len(sell_10), '次', '平均價格漲跌：', round(avg_sell_10, 2))
    print('KD小於20出現黃金交叉做多 5日勝率：', winrate_buy_5, '%', '操作次數：', len(buy_5), '次', '平均價格漲跌：', round(avg_buy_5,2))
    print('KD小於20出現黃金交叉做多10日勝率：', winrate_buy_10, '%', '操作次數：', len(buy_10), '次', '平均價格漲跌：', round(avg_buy_10,2))

def get_main_data(Stock_No, from_):
    df_ori = df_for_args(Stock_No, from_)
    df = timechange(df_ori)
    addma = add_moving_average(df, 5)
    addma = add_moving_average(addma, 20)
    addma = add_moving_average(addma, 60)
    addmaKD = add_KD(addma)
    addmaKD = add_KD_standard(addmaKD, upper = 80, lower=20)
    addmaKDRSI = add_RSI(addmaKD, span=5)
    addmaKDRSI = analyze_RSI(addmaKDRSI, upper=90, lower=10, span=5)
    addmaKDRSI = add_RSI_standard(addmaKDRSI, upper=90, lower=10)
    addmaKDRSI_MACD = add_MACD(addmaKDRSI)
    addmaKDRSI_MACD = add_MACDsignals(addmaKDRSI_MACD)
    addmaKDRSI_MACD = analyze_MACD(addmaKDRSI_MACD, daysafter=5)
    addmaKDRSI_MACD = analyze_MACD(addmaKDRSI_MACD, daysafter=10)
    addmaKDRSI_MACD = analyze_MACD(addmaKDRSI_MACD, daysafter=20)
    addmaKDRSI_MACD_KD = analyze_KD(addmaKDRSI_MACD, upper=80, lower=20)
    return addmaKDRSI_MACD_KD

def main():
    print_intro()
    Stock_No = int(input('請輸入想分析的股票代碼：'))
    from_ = int(input('想從西元第幾年後開始分析？'))
    download_stock(Stock_No, from_)
    df = get_main_data(Stock_No, from_)
    # main_drawfig(df, Stock_No, from_)
    # main_savefig(df, Stock_No, from_)
    print('股票代碼：', str(Stock_No))
    print('MACD分析')
    get_MACD_result(df)
    print('KD分析')
    get_KD_result(df)
    print('RSI_5分析')
    get_RSI_result(df)
    print('********投資一定有風險，股票投資有賺有賠，申購前應詳閱公開說明書********')

main()