#!/usr/bin/env python
# coding: utf-8

# # 주식 섹터 정보와 예측 모형 설계를 통한 Y & Z 세대 분석 
# 
# ## Y & Z 세대, 그들은 정말 새로운가?
# 
# 디지털에 친숙한 Y세대와 날 때부터 스마트폰을 접한 Z 세대는 기존의 베이비붐 및 X 세대와 투자 행태 또한 다를 것으로 예상된다. 그러나 그들은 정말 새로울까? 
# 
# 여기서는 거래량, 투자 종목, 활용 기기 등을 통해 예측 모형을 형성함으로써 Y & Z 세대의 투자 성향을 알아보고 그를 기존 세대와 비교하고자 한다. 정말 새로운지, 그렇다면 어떤 점이 새로운지, 예측 모형을 설계해 해석하였다. 주식 섹터 구분은 FnGuide의 [Wise Sector Index (WI26)](http://www.wiseindex.com/Index/Index#/WI100)을 참고하였다. 
# 
# 
# ------------------------------------------
# 
# ## 목차 
# 
# ### 1. 탐색적 데이터 분석 
# 
# ### 2. 전처리 
# 
# ### 3. 모형 설계
# * 로지스틱 회귀분석
# * 의사결정나무 
# 
# ### 4. 해석 및 결론

# ### 1. 탐색적 데이터 분석

# 먼저 데이터를 불러온 뒤 세대를 구분할 수 있는 인덱스를 추가하였다. 세대 구분은 [한겨레 신문 기사](http://www.hani.co.kr/arti/science/future/911357.html)를 참조하여 가장 근접한 군집으로 설정하였다. 

# In[ ]:


# 한글 폰트 사용을 위한 글꼴 설정
import matplotlib.pyplot as plt
import matplotlib as mpl 
mpl.rcParams['axes.unicode_minus'] = False
import matplotlib.font_manager as fm 

basic_path = '/content/gdrive/MyDrive/Contest/profiling'

plt.rc('font', family=fm.FontProperties(fname = basic_path + '/NanumGothic.ttf', size=13).get_name())
get_ipython().system('apt-get update -qq')
get_ipython().system('apt-get install fonts-nanum* -qq')
fm._rebuild()


# In[ ]:


from google.colab import drive
drive.mount('/content/gdrive')


# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn')
sns.set(font_scale=1.3)


# In[ ]:


# WiseIndex
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np

sec26_cd_dict = {'에너지':'WI100', '화학':'WI110', 
               '비철금속': 'WI200', '철강':'WI210', '건설':'WI220','기계':'WI230', '조선':'WI240', '상사/자본재':'WI250', '운송':'WI260',
               '자동차':'WI300', '화장품/의류':'WI310', '호텔/레저':'WI320', '미디어/교육':'WI330', '소매(유통)':'WI340',
               '필수소비재':'WI400', '건강관리':'WI410', 
               '은행':'WI500', '증권':'WI510', '보험':'WI520', 
               '소프트웨어':'WI600','IT하드웨어':'WI610','반도체':'WI620','IT가전':'WI630','디스플레이':'WI640',
               '전기통신서비스':'WI700',
               '유틸리티':'WI800'}

sec_cd = []
sec_name = []
cd = []
name = []

for values in sec26_cd_dict.values():
  # http get request
  req = requests.get('http://www.wiseindex.com/Index/GetIndexComponets?ceil_yn=0&dt=20200630&sec_cd='+values)
  # get json obj
  jsn = req.json()

  for item in jsn['list']:
    sec_cd.append(item['IDX_CD'])
    sec_name.append(item['IDX_NM_KOR'].split()[1])
    cd.append(item['CMP_CD'])
    name.append(item['CMP_KOR'])

dat26 = pd.DataFrame()
dat26['sec_cd'] = sec_cd
dat26['sec_name'] = sec_name
dat26['cd'] = cd
dat26['name'] = name      
kr_sec_cd = dat26      


# In[ ]:


cus_info = pd.read_csv(basic_path+'/data/2_cus_info.csv')
act_info = pd.read_csv(basic_path+'/data/2_act_info.csv')
iem_info = pd.read_csv(basic_path+'/data/2_iem_info.csv')
trd_kr = pd.read_csv(basic_path+'/data/2_trd_kr.csv')
trd_oss = pd.read_csv(basic_path+'/data/2_trd_oss.csv')


# In[ ]:


# 나이기준 세대 구분 
print('definition of babyboom:',2020-1964+1,'to',2020-1946+1)
print('definition of x:',2020-1980+1,'to',2020-1965+1)
print('definition of y:',2020-2000+1,'to',2020-1981+1)
print('definition of z: to',2020-2001+1)


# In[ ]:


# 세대 index 생성
cus_info['gen'] = cus_info['cus_age'].apply(lambda x: (x >= 55 and 'babyboom') or 
                                            (x >= 40 and 'x') or 
                                            (x >= 20 and 'y') or
                                            ('z'))
cus_info['bigen'] = cus_info['cus_age'].apply(lambda x: (x >= 40 and 'old') or 
                                              ('young')) 


# In[ ]:


# generation index
f, ax = plt.subplots(1, 2, figsize=(18,8))

cus_info['gen'].value_counts().sort_index().plot.pie(explode=[0.01,0.01,0.01,0.01],autopct='%1.1f%%', colors=['#FF4500','#FF8C00','#FFA500','#FFD700'],
                                        wedgeprops={'width': 0.7, 'edgecolor': 'w', 'linewidth': 0.1}, labels = ['X', 'Y', 'Z', '베이비붐'], ax=ax[0])
cus_info['bigen'].value_counts().plot.pie(explode=[0.01,0.01],autopct='%1.1f%%',colors=['#FF4500','#FFA500'],
                                          wedgeprops={'width': 0.7, 'edgecolor': 'w', 'linewidth': 0.1}, labels = ['기성세대(X & 베이비붐)','신세대(Y & Z)'], ax=ax[1])

ax[0].set_title('세대별 비중 (X, 베이비붐, Y, Z )')
ax[0].set_ylabel('')
ax[1].set_title('세대별 비중 (기성세대, 신세대)')
ax[1].set_ylabel('')

plt.subplots_adjust(wspace=0.2)
plt.show()


# 데이터에서 각 세대가 차지하는 비중을 확인할 수 있다. 베이비붐과 X 세대 (이후 '기성세대'로 명명) 가 데이터의 63.7%, Y&Z 세대 (이후 '신세대'로 명명) 가 36.3% 를 차지한다. 그를 **old**와 **young**으로 나눠 이후 분석에서 활용한다. 

# In[ ]:


f, ax = plt.subplots(2,1,figsize=(12,8))
y = act_info['act_opn_ym'][act_info['cus_id'].isin(cus_info['cus_id'][cus_info['bigen']=='young'])]
o = act_info['act_opn_ym'][act_info['cus_id'].isin(cus_info['cus_id'][cus_info['bigen']=='old'])]
y.plot(kind='kde',xlim=(y.min()-1, 202011),ax=ax[0])
o.plot(kind='kde',xlim=(o[o>0].min()-1, 202011),ax=ax[1])

ax[0].set_xlabel('계좌 생성 년월')
ax[1].set_xlabel('계좌 생성 년월')
ax[0].set_title('신세대 계좌 생성 추이')
ax[1].set_title('기성세대 계좌 생성 추이')
ax[0].set_ylabel('')
ax[1].set_ylabel('')

plt.subplots_adjust(hspace=1)
# plt.xticks(rotation=45)
plt.show()


# 계좌를 생성한 신세대는 2020년에 급증하였다.

# ### 2. 전처리

# * cus_info
# 

# In[ ]:


# NaNs 처리 
# zip_ctp_cd
cus_info['zip_ctp_cd'].loc[cus_info['zip_ctp_cd']=='-'] = '41'
cus_info['zip_ctp_cd'].loc[cus_info['zip_ctp_cd']!='-'].shape[0]
# ivs_icn_cd
cus_info['ivs_icn_cd'].loc[cus_info['ivs_icn_cd']=='-'] = '00'


# In[ ]:


# ordinal variable to int 
cus_info['tco_cus_grd_cd'] = cus_info['tco_cus_grd_cd'].apply(lambda x: 7 if x=='01' else x) # 탑클래스 
cus_info['tco_cus_grd_cd'] = cus_info['tco_cus_grd_cd'].apply(lambda x: 6 if x=='02' else x) # 골드
cus_info['tco_cus_grd_cd'] = cus_info['tco_cus_grd_cd'].apply(lambda x: 5 if x=='03' else x) # 로얄
cus_info['tco_cus_grd_cd'] = cus_info['tco_cus_grd_cd'].apply(lambda x: 4 if x=='04' else x) # 그린
cus_info['tco_cus_grd_cd'] = cus_info['tco_cus_grd_cd'].apply(lambda x: 3 if x=='05' else x) # 블루
cus_info['tco_cus_grd_cd'] = cus_info['tco_cus_grd_cd'].apply(lambda x: 2 if x=='09' else x) # 등급없음
cus_info['tco_cus_grd_cd'] = cus_info['tco_cus_grd_cd'].apply(lambda x: 1 if x=='_ ' else x) # 기타(추후 값 변경 가능)


# In[ ]:


# one-hot encoding for categorical variables 
cus_info['sex_male'] = cus_info['sex_dit_cd'].apply(lambda x: 0 if x == 2 else x) # male:1, female:0
cus_info['zip_cap'] = cus_info['zip_ctp_cd'].apply(lambda x: 1 if x =='41' else (1 if x == '11' else 0)) # 서울/경기 
cus_info['zip_metro'] = cus_info['zip_ctp_cd'].apply(lambda x: (x=='26' and 1) or (x=='27' and 1) or (x=='28' and 1)                                                      or (x=='29' and 1) or (x=='30' and 1) or (x=='31' and 1) or (x=='36' and 1) or 0) # 광역시/특별시 
cus_info['ivs_icn_01'] = cus_info['ivs_icn_cd'].apply(lambda x: 1 if x=='01' else 0) # 안정형 
cus_info['ivs_icn_02'] = cus_info['ivs_icn_cd'].apply(lambda x: 1 if x=='02' else 0) # 안정추구형
cus_info['ivs_icn_03'] = cus_info['ivs_icn_cd'].apply(lambda x: 1 if x=='03' else 0) # 위험중립형 
cus_info['ivs_icn_04'] = cus_info['ivs_icn_cd'].apply(lambda x: 1 if x=='04' else 0) # 적극투자형
cus_info['ivs_icn_05'] = cus_info['ivs_icn_cd'].apply(lambda x: 1 if x=='05' else 0) # 공격투자형
cus_info['ivs_icn_09'] = cus_info['ivs_icn_cd'].apply(lambda x: 1 if x=='09' else 0) # 전문투자자형        

cus_info['bigen'] = cus_info['bigen'].apply(lambda x: 1 if x=='young' else 0) # Y&Z


# In[ ]:


# remove redundant variables
cus_info = cus_info.drop(['sex_dit_cd', 'zip_ctp_cd', 'ivs_icn_cd', 'gen'], axis=1)


# * act_info

# In[ ]:


# NaNs 확인 
# act_opn_ym
print('the Number of Na in act_info:', sum(act_info['act_opn_ym']==0))


# * cus_info x act_info

# In[ ]:


# 고객 당 계좌 수
dup_acc = pd.DataFrame({'num_acc':act_info['cus_id'].value_counts()}) # 고객 당 계좌 수


# In[ ]:


# cus_info와 병합 
cus_info = cus_info.merge(dup_acc,how = 'left', left_on='cus_id', right_index=True)
cus_info = cus_info.merge(act_info, how='left', on='cus_id')


# In[ ]:


# act_opn_ym에 결측값이 있는 고객의 세대 
# print(cus_info['bigen'].loc[cus_info['act_opn_ym']==0])

# 같은 세대 내 최빈값으로 act_opn_ym 대체 
cus_info['act_opn_ym'].loc[cus_info['act_opn_ym']==0]=cus_info['act_opn_ym'].loc[cus_info['bigen']==0].value_counts().index.values[0]


# In[ ]:


# 고객 평균 계좌 보유 기간 (단위: 월)
from datetime import datetime
temp_str = cus_info['act_opn_ym'].apply(lambda x: str(x))
cus_info['act_opn_ym'] = datetime.strptime('202011', '%Y%m') - temp_str.apply(lambda x: datetime.strptime(x, '%Y%m'))
cus_info['act_opn_ym'] = cus_info['act_opn_ym'].apply(lambda x: round(x.days/30))                               


# In[ ]:


temp = cus_info[['act_opn_ym', 'cus_id']].groupby(['cus_id']).max()
cus_info = cus_info.drop(['act_opn_ym'], axis=1)
cus_info = cus_info.merge(temp, left_on='cus_id', right_index=True)
# len(cus_info.act_opn_ym.unique()) # 421 to 408


# In[ ]:


# 계좌 보유 기간 대비 계좌 수 
cus_info['ad_num_acc'] = cus_info['num_acc']/(cus_info['act_opn_ym']) 


# In[ ]:


# 나이 대비 계좌 보유 기간 
cus_info['cus_age'].loc[cus_info['cus_age']==0] = 15 # avoid denominator 0
cus_info['ad_act_opn_ym'] = cus_info['act_opn_ym']/(cus_info['cus_age']) 


# * iem_info
# * kr_sec_cd

# In[ ]:


# print(len(iem_info.iem_cd[0])) # 12
kr_sec_cd['cd'] = kr_sec_cd['cd'].apply(lambda x: int(x))

# 형식변환
kr_sec_cd['iem_cd'] = kr_sec_cd['cd'].apply(lambda x: 'A{0:06d}'.format(x))
kr_sec_cd['iem_cd'] = kr_sec_cd['iem_cd'].apply(lambda x: x.ljust(12)) # 공백추가 

m_iem_info = pd.merge(iem_info, kr_sec_cd[['iem_cd','sec_cd', 'sec_name']], how='left', on='iem_cd')


# In[ ]:


# 보통주와 앞 5자리가 같은 우선주에 섹터 정보 부여 
pref_ind = m_iem_info[m_iem_info['sec_cd'].isnull()==True].index.values

pref_stc = m_iem_info['iem_cd'].iloc[pref_ind,].str.extract('(A\d{5}\s*)').apply(lambda x: x.str[0:6])
pref_stc = pref_stc[pref_stc.iloc[:,0].isnull()==False]
pref_stc = pref_stc.rename({0:'iem_cd'}, axis='columns')


# In[ ]:


n_pref_ind = m_iem_info['iem_cd'][m_iem_info['sec_cd'].isnull()==False].index.values

pref_stc['sec_cd']=np.repeat(np.nan, pref_stc.shape[0])
pref_stc['sec_name']=np.repeat(np.nan, pref_stc.shape[0])
for i in range(pref_stc.shape[0]):
  if sum(m_iem_info['iem_cd'].iloc[n_pref_ind,].str.extract('(A\d*\s*)').apply(lambda x: x.str.contains(pref_stc.iloc[i,0])).iloc[:,0])>0:
    pref_stc['sec_cd'].iloc[i,] =     m_iem_info['sec_cd'].iloc[n_pref_ind,][m_iem_info['iem_cd'].iloc[n_pref_ind,].str.extract('(A\d*\s*)').apply(lambda x: x.str.contains(pref_stc.iloc[i,0])).iloc[:,0]==1].values[0]
    pref_stc['sec_name'].iloc[i,] =     m_iem_info['sec_name'].iloc[n_pref_ind,][m_iem_info['iem_cd'].iloc[n_pref_ind,].str.extract('(A\d*\s*)').apply(lambda x: x.str.contains(pref_stc.iloc[i,0])).iloc[:,0]==1].values[0]
  else:
    pref_stc['sec_cd'].iloc[i,] = np.nan
    pref_stc['sec_name'].iloc[i,] = np.nan


# In[ ]:


m_iem_info['sec_cd'].iloc[pref_stc.index.values,] = pref_stc['sec_cd']
m_iem_info['sec_name'].iloc[pref_stc.index.values,] = pref_stc['sec_name']


# * trd_kr
# * trd_oss

# In[ ]:


# 환율 반영
trd_oss.orr_pr = trd_oss.orr_pr * trd_oss.trd_cur_xcg_rt


# * 데이터 셋 병합 

# In[ ]:


data = trd_kr.merge(m_iem_info[['iem_cd', 'sec_name', 'sec_cd']], how='left', on='iem_cd')
data = pd.concat([data, trd_oss.iloc[:,0:10]], axis=0)
data = data.merge(cus_info.drop(['cus_age', 'num_acc', 'act_opn_ym'], axis=1), how='left', on='act_id')
# print(data.isnull().sum()/data.shape[0])


# * 투자 성향을 나타내는 파생 변수 생성 
# 
# 고객의 계좌 보유 개수가 보통 두 개 이상이기에, 계좌가 많은 고객의 정보가 더 많이 반영되는 것을 방지하기 위해 계좌가 아닌 각 개인에 대해 일 평균 매도/매수 횟수, 수량 등의 파생 변수를 생성하였다. 

# In[ ]:


# 거래 횟수 
# 일평균 매도 횟수: 총 매도 횟수 / 매도 거래일 수 
sell_num = data['cus_id'].loc[data['sby_dit_cd']==1].value_counts().sort_index() /   data[['cus_id', 'orr_dt']].loc[data['sby_dit_cd']==1].drop_duplicates().value_counts().groupby(['cus_id']).sum().sort_index()

# 일평균 매수 횟수: 총 매수 횟수 / 매수 거래일 수  
buy_num = data['cus_id'].loc[data['sby_dit_cd']==2].value_counts().sort_index() /   data[['cus_id', 'orr_dt']].loc[data['sby_dit_cd']==2].drop_duplicates().value_counts().groupby(['cus_id']).sum().sort_index()


# In[ ]:


data['tco_cus_grd_cd'].loc[-data['cus_id'].isin(buy_num.index)].value_counts().plot.bar(rot=0)
plt.title('고객등급 \'해당 사항 없음\' 고객의 매수 기록')
plt.show()


# 위 코드에서 고객등급이 '해당 사항 없음'의 값을 가지는 고객은 기간 내 매수 기록이 없다. 따라서 해당 고객군에 매긴 등급 (가장 낮은 순위) 을 변경하지 않는다. 

# In[ ]:


# 거래 수량 
# 일평균 매도 수량: 총 매도 수량 / 매도 거래일 수 
sell_qty = data[['cns_qty', 'cus_id']].loc[data['sby_dit_cd']==1].groupby(['cus_id']).sum()['cns_qty'] /   data[['cus_id', 'orr_dt']].loc[data['sby_dit_cd']==1].drop_duplicates().value_counts().groupby(['cus_id']).sum().sort_index()

# 일평균 매수 수량 : 총 매수 수량 / 매수 거래일 수 
buy_qty = data[['cns_qty', 'cus_id']].loc[data['sby_dit_cd']==2].groupby(['cus_id']).sum()['cns_qty'] /   data[['cus_id', 'orr_dt']].loc[data['sby_dit_cd']==2].drop_duplicates().value_counts().groupby(['cus_id']).sum().sort_index()


# In[ ]:


# 거래 단가 (단위: 원화)  
# 일평균 매도 단가: 총 매도 단가 / 매도 거래일 수 
sell_pr = data[['orr_pr', 'cus_id']].loc[data['sby_dit_cd']==1].groupby(['cus_id']).sum()['orr_pr'] /   data[['cus_id', 'orr_dt']].loc[data['sby_dit_cd']==1].drop_duplicates().value_counts().groupby(['cus_id']).sum().sort_index()

# 일평균 매수 단가: 총 매수 단가 / 매수 거래일 수 
buy_pr = data[['orr_pr', 'cus_id']].loc[data['sby_dit_cd']==2].groupby(['cus_id']).sum()['orr_pr'] /   data[['cus_id', 'orr_dt']].loc[data['sby_dit_cd']==2].drop_duplicates().value_counts().groupby(['cus_id']).sum().sort_index()


# In[ ]:


# 수익: 총 매도 수량 * 평균단가 - 총 매수 수량 * 평균단가
whole_sell_qty = data[['cns_qty', 'cus_id']].loc[data['sby_dit_cd']==1].groupby(['cus_id']).sum()['cns_qty'] 
whole_buy_qty = data[['cns_qty', 'cus_id']].loc[data['sby_dit_cd']==2].groupby(['cus_id']).sum()['cns_qty'] 
whole_sell_pr = data[['orr_pr', 'cus_id']].loc[data['sby_dit_cd']==1].groupby(['cus_id']).mean()['orr_pr'] 
whole_buy_pr = data[['orr_pr', 'cus_id']].loc[data['sby_dit_cd']==2].groupby(['cus_id']).mean()['orr_pr']

temp = pd.read_csv(basic_path+'/data/2_cus_info.csv')
temp = temp.merge(whole_sell_qty.to_frame('sell_qty'),how = 'left',left_on='cus_id',right_index=True)
temp = temp.merge(whole_buy_qty.to_frame('buy_qty'),how = 'left',left_on='cus_id',right_index=True)
temp = temp.merge(whole_sell_pr.to_frame('sell_pr'),how = 'left',left_on='cus_id',right_index=True)
temp = temp.merge(whole_buy_pr.to_frame('buy_pr'),how = 'left',left_on='cus_id',right_index=True)
temp = temp.fillna(0)
earn = temp.sell_qty * temp.sell_pr - temp.buy_qty * temp.buy_pr 
earn.index = temp['cus_id']


# In[ ]:


# 투자 주식 섹터 개수 (결측값 제외)
temp = data[['sec_cd', 'cus_id']].loc[data['sec_cd'].isnull()==False].groupby(['cus_id']).count()
temp.columns = ['sec_num']
temp2 = data[['cus_id', 'sec_cd']].loc[data['sec_cd'].isnull()==False].value_counts()
temp_gr = temp2.groupby(['cus_id'])

for key, group in temp_gr:
  temp['sec_num'].loc[temp.index==key,] = len(group)

sec_num = temp


# In[ ]:


# 최빈 거래 섹터 (결측값 제외)
temp = data[['cus_id', 'sec_cd']].loc[data['sec_cd'].isnull()==False].value_counts()
temp = temp.unstack().fillna(0)
temp.columns = 'sec_cd_' + temp.columns.values 
temp1 = temp.idxmax(1).to_frame('sec_mode')

#  one-hot encoding (결측값을 기준으로)
for i in range(len(temp.columns.values)):
  temp1['sec_mode_'+temp.columns.values[i][-5:]] = temp1['sec_mode'].apply(lambda x: 1 if x==temp.columns.values[i] else 0)

sec_mode = temp1.drop(['sec_mode'], axis=1)


# In[ ]:


# 투자 종목 개수 
temp = data[['iem_cd', 'cus_id']]
temp['iem_cd'] = temp['iem_cd'].str.extract('([A-Z]*)').iloc[:,0]
item_num = temp.groupby(['cus_id']).count()
item_num.columns = ['item_num']


# In[ ]:


# 주 이용기기
temp = data[['cus_id', 'orr_mdi_dit_cd']].value_counts().to_frame('count')
temp = temp.unstack().fillna(0)
temp.columns = ['orr_mdi_dit_01', 'orr_mdi_dit_02', 'orr_mdi_dit_03', 'orr_mdi_dit_04']
temp = temp.idxmax(1).to_frame('orr_mdi_dit_cd')
# print(temp.head(5))

# one-hot encoding
# print(temp.value_counts()) # 3이 가장 보편적이므로 3을 기준으로
temp['orr_mdi_dit_01'] = temp.orr_mdi_dit_cd.apply(lambda x: 1 if x=='orr_mdi_dit_01' else 0)
temp['orr_mdi_dit_02'] = temp.orr_mdi_dit_cd.apply(lambda x: 1 if x=='orr_mdi_dit_02' else 0)
temp['orr_mdi_dit_04'] = temp.orr_mdi_dit_cd.apply(lambda x: 1 if x=='orr_mdi_dit_04' else 0)
orr_mdi_dit = temp.drop(['orr_mdi_dit_cd'], axis=1)


# * 최종 데이터 셋 생성
# 
# 앞서 만든 투자 성향을 타나내는 파생 변수와 원 데이터의 고객 특성 변수를 병합하여 최종 데이터를 생성한 후, 변수별 단위를 맞추기 위해 연속형 변수에 대해 min-max scaling을 실시하였다. 

# In[ ]:


# 최종 데이터 셋
data_fin = data.loc[:,'cus_id':'ad_act_opn_ym'].groupby(['cus_id']).mean()
data_fin = data_fin.merge(sell_num.to_frame('sell_num'), how='left',left_on='cus_id', right_index=True).fillna(0) # 일평균 매도 횟수, 매도 기록이 없는 경우 0 
data_fin = data_fin.merge(buy_num.to_frame('buy_num'), how='left',left_on='cus_id', right_index=True).fillna(0) # 일평균 매수 횟수, 매수 기록이 없는 경우 0 
data_fin = data_fin.merge(sell_qty.to_frame('sell_qty'), how='left',left_on='cus_id', right_index=True).fillna(0) # 일평균 매도 수량, 매도 수량 없는 경우 0 
data_fin = data_fin.merge(buy_qty.to_frame('buy_qty'), how='left',left_on='cus_id', right_index=True).fillna(0) # 일평균 매수 수량, 매수 수량 없는 경우 0 
data_fin = data_fin.merge(sell_pr.to_frame('sell_pr'), how='left',left_on='cus_id', right_index=True).fillna(0) # 일평균 매도 단가, 매도 기록이 없는 경우 0 
data_fin = data_fin.merge(buy_pr.to_frame('buy_pr'), how='left',left_on='cus_id', right_index=True).fillna(0) # 일평균 매수 단가, 매수 기록이 없는 경우 0 
data_fin = data_fin.merge(earn.to_frame('earn'), how='left',left_on='cus_id', right_index=True) # 총 수익 

data_fin = data_fin.merge(sec_num, how='left',left_on='cus_id', right_index=True)
data_fin = data_fin.merge(sec_mode, how='left',left_on='cus_id', right_index=True).fillna(0)
data_fin = data_fin.merge(item_num, how='left',left_on='cus_id', right_index=True)
data_fin = data_fin.merge(orr_mdi_dit, how='left',left_on='cus_id', right_index=True)

# 결측값 처리 
mode = data_fin['sec_num'].loc[(data_fin.sec_num.isnull()==False) & (data_fin.item_num>2)].mode()
data_fin['sec_num'] = data_fin['sec_num'].fillna(mode)


# In[ ]:


# 표준화 
data_fin[['tco_cus_grd_cd', 'ad_num_acc', 'ad_act_opn_ym', 'sell_num', 'buy_num','sell_qty', 'buy_qty', 'sell_pr', 'buy_pr', 'earn', 'sec_num', 'item_num']] =   data_fin[['tco_cus_grd_cd', 'ad_num_acc', 'ad_act_opn_ym', 'sell_num', 'buy_num','sell_qty', 'buy_qty', 'sell_pr', 'buy_pr', 'earn', 'sec_num', 'item_num']].apply(lambda x: (x-x.min())/(x.max()-x.min()), axis=0)


# ### 3. 모형 설계
# 
# **bigen**을 반응 변수로 하는 예측 모형을 만들어 예측력이 가장 뛰어난 모형으로 신세대와 기성세대의 차이를 분석한다. 
# 
# 모형은 결과에 대한 설명을 할 수 있어야 한다는 점과 반응 변수와 예측 변수가 가질 수 있는 관계 (선형, 비선형) 를 모두 고려해 로지스틱 회귀 모형과 의사결정나무 모형을 선정하였다. 
# 
# 먼저 전체 데이터를 훈련 데이터와 검정 데이터로 나눈 뒤 cross-validation을 통해 모형의 파라미터를 조정하고, cross-validation score가 가장 높은 모형을 각각의 최종 모형으로 결정한다. 이 때 반응 변수가 이항형이므로 cross-validation score는 AUC로 설정하였다. 

# * 로지스틱 회귀 모형

# In[ ]:


from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

x_dat = data_fin.loc[:,data_fin.columns != 'bigen']
y_dat = data_fin.loc[:,data_fin.columns == 'bigen']

X_train, X_test, y_train, y_test = train_test_split(x_dat, y_dat, test_size=0.3, stratify = y_dat, random_state=2020)


# In[ ]:


pipe_lr = Pipeline([('classifier', LogisticRegression())])
param_grid_lr = [{'classifier': [LogisticRegression(solver='liblinear')],
               'classifier__C': np.arange(0.5,1,0.001), 
               'classifier__penalty': ['l1']
               }]
cv_lr = RandomizedSearchCV(pipe_lr, param_grid_lr, scoring='roc_auc', cv=5, verbose=1, n_iter=100)
cv_lr.fit(X_train, y_train)


# In[ ]:


cv_lr_result = pd.DataFrame(cv_lr.cv_results_)
print('로지스틱 회귀 최적 파라미터 조합:',cv_lr.best_params_)
print('로지스틱 회귀 best AUC: {:.3f}'.format(cv_lr.best_score_))
cv_lr_result[['rank_test_score','params', 'mean_test_score']].sort_values('rank_test_score')[0:5]


# * 의사결정나무 모형 

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
pipe_dt = Pipeline([('classifier', DecisionTreeClassifier())])
param_grid_dt = [{'classifier': [DecisionTreeClassifier()],
                  'classifier__criterion': ['gini', 'entropy'],
                  'classifier__max_depth': np.arange(5, 26), 
                  'classifier__min_samples_leaf': np.arange(10, 110, 10),
                  'classifier__min_samples_split': np.arange(10, 110, 10)
                  }]
cv_dt = RandomizedSearchCV(pipe_dt, param_grid_dt, scoring='roc_auc', cv=5, verbose=1, n_iter=100)
cv_dt.fit(X_train, y_train)


# In[ ]:


cv_dt_result = pd.DataFrame(cv_dt.cv_results_)
print('의사결정나무 최적 파라미터 조합:',cv_dt.best_params_)
print('의사결정나무 best AUC: {:.3f}'.format(cv_dt.best_score_))
cv_dt_result[['rank_test_score','params', 'mean_test_score']].sort_values('rank_test_score')[0:5]


# In[ ]:


best_lr = list(cv_lr.best_params_.values())[2]
best_lr.fit(X_train, y_train)

pred_lr = best_lr.predict(X_test)
acc_lr = accuracy_score(y_test, pred_lr)
auc_lr= roc_auc_score(y_test, pred_lr)
print('로지스틱 회귀 모형 정확도: {:.3f}'.format(acc_lr), '\n'
      '로지스틱 회귀 모형 AUC: {:.3f}'.format(auc_lr))


# In[ ]:


best_dt = list(cv_dt.best_params_.values())[4]
best_dt.fit(X_train, y_train)

pred_dt = best_dt.predict(X_test)
acc_dt = accuracy_score(y_test, pred_dt)
auc_dt= roc_auc_score(y_test, pred_dt)
print('의사결정나무 모형 정확도: {:.3f}'.format(acc_dt), '\n'
      '의사결정나무 AUC: {:.3f}'.format(auc_dt))


# 의사결정나무 모형의 예측 정확도와 AUC가 높으나 두 모형 모두 예측 성능이 좋은 편이다. 신세대와 기성세대 간에 눈에 띄는 차이가 있음을 암시한다. 따라서 두 모형 모두 해석에 사용한다. 여기서는 결과만 간략히 보이고, 시각화는 다음 부분에서 다룬다. 먼저 로지스틱 회귀 모형에서 L1 penalty에 의해 선택된 33개의 중요 변수는 다음과 같다. 

# In[ ]:


var_coef_lr = pd.DataFrame(np.c_[X_train.columns.values, best_lr.coef_[0]])
var_coef_lr.columns = ['var', 'coef']
imp_var_lr = var_coef_lr[var_coef_lr.coef != 0]
imp_var_lr['imp'] = imp_var_lr.coef.abs()
imp_var_lr.sort_values('imp', ascending=False)


# 의사결정나무 모형에서의 변수 중요도는 다음과 같다. 

# In[ ]:


var_imp_dt = pd.DataFrame(np.c_[X_train.columns.values,best_dt.feature_importances_])
var_imp_dt.columns = ['var', 'imp']
imp_var_dt = var_imp_dt[var_imp_dt.imp != 0]
imp_var_dt.sort_values('imp', ascending=False)


# ### 4. 해석 및 결론
# 
# 최종 데이터 셋과 모형을 종합하여 신세대와 기성세대의 차이점을 비교 분석한다. 해석 시에는 표준화하지 않은 원 단위의 데이터를 사용하였다. 
# 

# In[ ]:


# 고객 특성 
temp_cus_info = pd.read_csv(basic_path+'/data/2_cus_info.csv')
temp_cus_info['bigen'] = temp_cus_info['cus_age'].apply(lambda x: (x >= 40 and '기성세대') or ('신세대')) 
temp_cus_info['gen'] = temp_cus_info['cus_age'].apply(lambda x: (x >= 55 and '베이비붐') or 
                                            (x >= 40 and 'X세대') or 
                                            (x >= 20 and 'Y세대') or
                                            ('Z세대'))
temp_cus_info['sex_dit_cd'] = temp_cus_info['sex_dit_cd'].apply(lambda x: (x==1 and '남성') or '여성') 
temp_cus_info['zip_ctp_cd'] = temp_cus_info['zip_ctp_cd'].apply(lambda x: (x=='41' and '서울/경기') or (x=='11' and '서울/경기') or                                                                 (x=='-' and '서울/경기') or x) 
temp_cus_info['zip_ctp_cd'] = temp_cus_info['zip_ctp_cd'].apply(lambda x: (x=='26' and '광역/특별시') or (x=='27' and '광역/특별시') or                                                                 (x=='28' and '광역/특별시') or (x=='29' and '광역/특별시') or                                                                 (x=='30' and '광역/특별시') or (x=='31' and '광역/특별시') or                                                                 (x=='36' and '광역/특별시') or x) # 광역시/특별시 
temp_cus_info['zip_ctp_cd'] = temp_cus_info['zip_ctp_cd'].apply(lambda x: (x=='42' and '지방') or (x=='43' and '지방') or                                                                 (x=='44' and '지방') or (x=='45' and '지방') or                                                                 (x=='46' and '지방') or (x=='47' and '지방') or                                                                 (x=='48' and '지방') or (x=='50' and '지방') or x) # 이 외 지역
temp_cus_info['tco_cus_grd_cd'] = temp_cus_info['tco_cus_grd_cd'].apply(lambda x: 1 if x=='01' else x) 
temp_cus_info['tco_cus_grd_cd'] = temp_cus_info['tco_cus_grd_cd'].apply(lambda x: 2 if x=='02' else x) 
temp_cus_info['tco_cus_grd_cd'] = temp_cus_info['tco_cus_grd_cd'].apply(lambda x: 3 if x=='03' else x) 
temp_cus_info['tco_cus_grd_cd'] = temp_cus_info['tco_cus_grd_cd'].apply(lambda x: 4 if x=='04' else x) 
temp_cus_info['tco_cus_grd_cd'] = temp_cus_info['tco_cus_grd_cd'].apply(lambda x: 5 if x=='05' else x) 
temp_cus_info['tco_cus_grd_cd'] = temp_cus_info['tco_cus_grd_cd'].apply(lambda x: 6 if x=='09' else x) 
temp_cus_info['tco_cus_grd_cd'] = temp_cus_info['tco_cus_grd_cd'].apply(lambda x: 7 if x=='_ ' else x) 
temp_cus_info['ivs_icn_cd'] = temp_cus_info['ivs_icn_cd'].apply(lambda x: 1 if x=='01' else x) # 안정형 
temp_cus_info['ivs_icn_cd'] = temp_cus_info['ivs_icn_cd'].apply(lambda x: 2 if x=='02' else x) # 안정추구형
temp_cus_info['ivs_icn_cd'] = temp_cus_info['ivs_icn_cd'].apply(lambda x: 3 if x=='03' else x) # 위험중립형 
temp_cus_info['ivs_icn_cd'] = temp_cus_info['ivs_icn_cd'].apply(lambda x: 4 if x=='04' else x) # 적극투자형
temp_cus_info['ivs_icn_cd'] = temp_cus_info['ivs_icn_cd'].apply(lambda x: 5 if x=='05' else x) # 공격투자형
temp_cus_info['ivs_icn_cd'] = temp_cus_info['ivs_icn_cd'].apply(lambda x: 9 if x=='09' else x) # 전문투자자형   
temp_cus_info['ivs_icn_cd'] = temp_cus_info['ivs_icn_cd'].apply(lambda x: 10 if x=='-' else x) # 해당없음  
temp_cus_info['ivs_icn_cd'] = temp_cus_info['ivs_icn_cd'].apply(lambda x: 10 if x=='00' else x) # 정보제공 미동의     


# In[ ]:


f, ax = plt.subplots(2,2,figsize=(16, 12))

sns.countplot('bigen', data=temp_cus_info, hue='sex_dit_cd', palette='magma', ax=ax[0,0])
ax[0,0].set_title('세대별 성별 비중')
ax[0,0].set_xlabel('')
ax[0,0].set_ylabel('고객 수')
ax[0,0].legend(title='구분', bbox_to_anchor=(1,1))
sns.countplot('bigen', data=temp_cus_info, hue='zip_ctp_cd', palette='magma', ax=ax[0,1])
ax[0,1].set_title('세대별 거주지역 비중')
ax[0,1].set_xlabel('')
ax[0,1].set_ylabel('고객 수')
ax[0,1].legend(title='구분', bbox_to_anchor=(1,1))
sns.countplot('bigen', data=temp_cus_info, hue='tco_cus_grd_cd', palette='magma', ax=ax[1,0])
ax[1,0].set_title('세대별 고객등급 비중')
ax[1,0].set_xlabel('')
ax[1,0].set_ylabel('고객 수')
legend_labels, _= ax[1,0].get_legend_handles_labels()
ax[1,0].legend(legend_labels, ['탑클래스','골드','로얄', '그린', '블루', '등급없음', '기타'], bbox_to_anchor=(1,1))
sns.countplot('bigen', data=temp_cus_info, hue='ivs_icn_cd', palette='magma', ax=ax[1,1])
ax[1,1].set_title('세대별 투자 성향 비중')
ax[1,1].set_xlabel('')
ax[1,1].set_ylabel('고객 수')
legend_labels, _= ax[1,1].get_legend_handles_labels()
ax[1,1].legend(legend_labels, ['안정형','안정추구형','위험중립형', '적극투자형', '공격투자형', '전문투자자형', '정보 없음'], bbox_to_anchor=(1,1))

plt.subplots_adjust(hspace=0.3, wspace=1)
plt.show()


# 두 세대의 성별과 거주지역 분포 모두 육안으로는 큰 차이가 나지 않는다. 고객등급은 신세대가 기성세대에 비해 낮은 등급에 몰려있는 반면, 투자 성향은 신세대의 분포가 더 고르다. 

# In[ ]:


# 투자 성향 
temp_dit = data[['cus_id', 'orr_mdi_dit_cd']].value_counts().to_frame('count')
temp_dit = temp_dit.unstack().fillna(0)
temp_dit.columns = ['01','02','03','04']
temp_dit = temp_dit.idxmax(1).to_frame('orr_mdi_dit_cd')

temp_deal_num = pd.merge(sell_num.to_frame('sell_num'), buy_num.to_frame('buy_num'), left_index=True, right_index=True, how='outer').fillna(0)
temp_deal_qty = pd.merge(sell_qty.to_frame('sell_qty'), buy_qty.to_frame('buy_qty'), left_index=True, right_index=True, how='outer').fillna(0)
temp_deal_pr = pd.merge(sell_pr.to_frame('sell_pr'), buy_pr.to_frame('buy_pr'), left_index=True, right_index=True, how='outer').fillna(0)
temp_inv = pd.merge(temp_deal_num, temp_deal_qty, left_index=True, right_index=True, how='outer').fillna(0)
temp_inv = temp_inv.merge(temp_deal_pr, left_index=True, right_index=True, how='outer').fillna(0)
temp_inv = temp_inv.merge(temp_dit, left_index=True, right_index=True, how='right').fillna(0)
temp_inv = temp_inv.merge(item_num, left_index=True, right_index=True)

temp_mode = data[['cus_id', 'sec_cd']].loc[data['sec_cd'].isnull()==False].value_counts()
temp_mode = temp_mode.unstack().fillna(0)
temp_mode.columns = temp_mode.columns.values 
temp_mode = temp_mode.idxmax(1).to_frame('sec_mode')
temp_inv = temp_inv.merge(temp_mode, left_index=True, right_index=True, how='left')
temp_inv = temp_inv.merge(sec_num, how='left',left_on='cus_id', right_index=True).fillna(mode)
temp_inv = temp_inv.merge(earn.to_frame('earn'), how='left',left_index=True, right_index=True)
temp_inv = temp_inv.merge(temp_cus_info[['cus_id', 'bigen', 'gen']], left_index=True, right_on='cus_id')

temp_inv['sec_num'][temp_inv['sec_num'].isnull()==True]=mode.values[0]


# In[ ]:


# 매도/매수 성향 
f, ax = plt.subplots(4,2,figsize=(16, 20))

temp_inv[['sell_num','bigen']].groupby(['bigen']).mean().unstack().plot.bar(color=['#952EA0','#CA3C97'],ax=ax[0,0], rot=0, alpha=0.85)
ax[0,0].set_title('세대별 일평균 매도 횟수')
ax[0,0].set_xlabel('')
ax[0,0].set_ylabel('횟수')
ax[0,0].set_xticklabels(['기성세대', '신세대'])
temp_inv[['buy_num','bigen']].groupby(['bigen']).mean().unstack().plot.bar(color=['#952EA0','#CA3C97'],ax=ax[0,1], rot=0, alpha=0.85)
ax[0,1].set_title('세대별 일평균 매수 횟수')
ax[0,1].set_xlabel('')
ax[0,1].set_ylabel('횟수')
ax[0,1].set_xticklabels(['기성세대', '신세대'])

temp_inv[['sell_qty','bigen']].groupby(['bigen']).mean().unstack().plot.bar(color=['#952EA0','#CA3C97'],ax=ax[1,0], rot=0, alpha=0.85)
ax[1,0].set_title('세대별 일평균 매도 수량')
ax[1,0].set_xlabel('')
ax[1,0].set_ylabel('수량')
ax[1,0].set_xticklabels(['기성세대', '신세대'])
temp_inv[['buy_qty','bigen']].groupby(['bigen']).mean().unstack().plot.bar(color=['#952EA0','#CA3C97'],ax=ax[1,1], rot=0, alpha=0.85)
ax[1,1].set_title('세대별 일평균 매수 수량')
ax[1,1].set_xlabel('')
ax[1,1].set_ylabel('수량')
ax[1,1].set_xticklabels(['기성세대', '신세대'])

temp_inv[['sell_pr','bigen']].groupby(['bigen']).mean().unstack().plot.bar(color=['#952EA0','#CA3C97'],ax=ax[2,0], rot=0, alpha=0.85)
ax[2,0].set_title('세대별 일평균 매도 단가')
ax[2,0].set_xlabel('')
ax[2,0].set_ylabel('단가')
ax[2,0].set_xticklabels(['기성세대', '신세대'])
temp_inv[['buy_pr','bigen']].groupby(['bigen']).mean().unstack().plot.bar(color=['#952EA0','#CA3C97'],ax=ax[2,1], rot=0, alpha=0.85)
ax[2,1].set_title('세대별 일평균 매수 단가')
ax[2,1].set_xlabel('')
ax[2,1].set_ylabel('단가')
ax[2,1].set_xticklabels(['기성세대', '신세대'])

temp_inv[['sec_num','bigen']].groupby(['bigen']).mean().unstack().plot.bar(color=['#952EA0','#CA3C97'],ax=ax[3,0], rot=0, alpha=0.85)
ax[3,0].set_title('세대별 투자 섹터 수 평균')
ax[3,0].set_xlabel('')
ax[3,0].set_ylabel('개수')
ax[3,0].set_xticklabels(['기성세대', '신세대'])
temp_inv[['item_num','bigen']].groupby(['bigen']).mean().unstack().plot.bar(color=['#952EA0','#CA3C97'],ax=ax[3,1], rot=0, alpha=0.85)
ax[3,1].set_title('세대별 투자 종목 수 평균')
ax[3,1].set_xlabel('')
ax[3,1].set_ylabel('개수')
ax[3,1].set_xticklabels(['기성세대', '신세대'])

plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.show()


# 일평균 매도/매수 횟수에는 세대 간 차이가 거의 없으나, 수량은 기성세대가 많고 단가는 신세대가 높다. 
# 
# 신세대가 기성세대에 비해 높은 단가의 종목을 더 적게 거래한다고 할 수 있다. 
# 
# 또 투자 섹터 수는 두 세대가 유사하나, 투자 종목의 종류는 기성세대가 더 다양하다. 

# In[ ]:


# 총 수익
f, ax = plt.subplots(1,1,figsize=(6,6))

temp_inv[['earn','bigen']].groupby(['bigen']).mean().unstack().plot.bar(color=['#952EA0','#CA3C97'], rot=0, alpha=0.85)
ax.set_title('세대별 총수익 평균')
ax.set_xlabel('')
ax.set_ylabel('총수익')
ax.set_xticklabels(['기성세대', '신세대'])

plt.subplots_adjust(wspace=0.3, hspace=0.3)
plt.show()


# 데이터의 거래기간이 2019년 1월 ~ 2020년 6월으로, 코로나의 영향으로 두 세대 모두 손실이 크며, 신세대가 기성세대에 비해 그 정도가 심하다. 

# In[ ]:


# 거래단말
f, ax = plt.subplots(1,1,figsize=(6,6))

sns.countplot('bigen', data=temp_inv, hue='orr_mdi_dit_cd', palette='magma')
ax.set_title('세대별 거래단말 분포')
ax.set_xlabel('')
ax.set_ylabel('고객 수')
legend_labels, _= ax.get_legend_handles_labels()
ax.legend(legend_labels, ['MTS','HTS','영업점단말', '무선단말'], title='구분', bbox_to_anchor=(1,1))

plt.show()


# 신세대가 기성세대에 비해 무선단말, 즉 모바일 거래 위주로 사용한다는 사실을 알 수 있다. 기성세대는 비교적 다양한 거래단말을 사용한다고 할 수 있다. 

# In[ ]:


# 주 투자 종목 
f, ax = plt.subplots(1,1,figsize=(20,6))

sns.countplot('bigen', data=temp_inv[temp_inv.sec_mode.isnull()==False], hue='sec_mode', palette='magma')
ax.set_title('세대별 주 투자 종목 분포')
ax.set_xlabel('')
ax.set_ylabel('고객 수')
legend_labels, _= ax.get_legend_handles_labels()
ax.legend(legend_labels, ['건강관리', '호텔/레저', '화장품/의류', '반도체', '건설', 'IT하드웨어', '미디어/교육', '디스플레이', '소프트웨어', 
                          '철강', '은행', '필수소비재', '기계', '유틸리티', '조선', '운송', '에너지', '증권', '자동차', 'IT가전', '전기통신서비스', 
                          '비철금속', '상사/자본재', '화학', '보험', '소매(유통)'],title='구분', bbox_to_anchor=(1,1.3))
plt.show()


# 대부분의 항목이 비슷한 추이를 보이나, 건강관리와 IT하드웨어 분야는 기성세대가 신세대에 비해 분포가 몰려있다. 
# 
# 기성세대와 신세대 수를 고려해 표현한 투자 섹터별 세대 비중은 다음과 같다. 

# In[ ]:


temp_inv[['sec_mode','bigen']].groupby(['bigen']).count()


# In[ ]:


# 주 투자 섹터별 세대 비중: 중공업/운송
sec26_cd_dict_inv = dict([(value, key) for key, value in sec26_cd_dict.items()]) 

f, ax = plt.subplots(3, 3, figsize=(18,18))
(temp_inv[['sec_mode','bigen']][temp_inv['sec_mode']=='WI100'].value_counts()/[100*6239/(6239+3531), 100*3531/(6239+3531)]).sort_index().plot.pie(explode=[0.01,0.01],                                                                                                   autopct='%1.1f%%', colors=['#FF4500','#FFA500'],                                                                                                   wedgeprops={'width': 0.7, 'edgecolor': 'w', 'linewidth': 0.1},                                                                                                    labels = ['기성세대', '신세대'], ax=ax[0,0])
ax[0,0].set_title(sec26_cd_dict_inv['WI100']+' 산업 세대별 비중')
ax[0,0].set_ylabel('')

(temp_inv[['sec_mode','bigen']][temp_inv['sec_mode']=='WI110'].value_counts()/[100*6239/(6239+3531), 100*3531/(6239+3531)]).sort_index().plot.pie(explode=[0.01,0.01],                                                                                                   autopct='%1.1f%%', colors=['#FF4500','#FFA500'],                                                                                                   wedgeprops={'width': 0.7, 'edgecolor': 'w', 'linewidth': 0.1},                                                                                                    labels = ['기성세대', '신세대'], ax=ax[0,1])
ax[0,1].set_title(sec26_cd_dict_inv['WI110']+' 산업 세대별 비중')
ax[0,1].set_ylabel('')

(temp_inv[['sec_mode','bigen']][temp_inv['sec_mode']=='WI200'].value_counts()/[100*6239/(6239+3531), 100*3531/(6239+3531)]).sort_index().plot.pie(explode=[0.01,0.01],                                                                                                   autopct='%1.1f%%', colors=['#FF4500','#FFA500'],                                                                                                   wedgeprops={'width': 0.7, 'edgecolor': 'w', 'linewidth': 0.1},                                                                                                    labels = ['기성세대', '신세대'], ax=ax[0,2])
ax[0,2].set_title(sec26_cd_dict_inv['WI200']+' 산업 세대별 비중')
ax[0,2].set_ylabel('')

(temp_inv[['sec_mode','bigen']][temp_inv['sec_mode']=='WI210'].value_counts()/[100*6239/(6239+3531), 100*3531/(6239+3531)]).sort_index().plot.pie(explode=[0.01,0.01],                                                                                                   autopct='%1.1f%%', colors=['#FF4500','#FFA500'],                                                                                                   wedgeprops={'width': 0.7, 'edgecolor': 'w', 'linewidth': 0.1},                                                                                                    labels = ['기성세대', '신세대'], ax=ax[1,0])
ax[1,0].set_title(sec26_cd_dict_inv['WI210']+' 산업 세대별 비중')
ax[1,0].set_ylabel('')

(temp_inv[['sec_mode','bigen']][temp_inv['sec_mode']=='WI220'].value_counts()/[100*6239/(6239+3531), 100*3531/(6239+3531)]).sort_index().plot.pie(explode=[0.01,0.01],                                                                                                   autopct='%1.1f%%', colors=['#FF4500','#FFA500'],                                                                                                   wedgeprops={'width': 0.7, 'edgecolor': 'w', 'linewidth': 0.1},                                                                                                    labels = ['기성세대', '신세대'], ax=ax[1,1])
ax[1,1].set_title(sec26_cd_dict_inv['WI220']+' 산업 세대별 비중')
ax[1,1].set_ylabel('')

(temp_inv[['sec_mode','bigen']][temp_inv['sec_mode']=='WI230'].value_counts()/[100*6239/(6239+3531), 100*3531/(6239+3531)]).sort_index().plot.pie(explode=[0.01,0.01],                                                                                                   autopct='%1.1f%%', colors=['#FF4500','#FFA500'],                                                                                                   wedgeprops={'width': 0.7, 'edgecolor': 'w', 'linewidth': 0.1},                                                                                                    labels = ['기성세대', '신세대'], ax=ax[1,2])
ax[1,2].set_title(sec26_cd_dict_inv['WI230']+' 산업 세대별 비중')
ax[1,2].set_ylabel('')

(temp_inv[['sec_mode','bigen']][temp_inv['sec_mode']=='WI240'].value_counts()/[100*6239/(6239+3531), 100*3531/(6239+3531)]).sort_index().plot.pie(explode=[0.01,0.01],                                                                                                   autopct='%1.1f%%', colors=['#FF4500','#FFA500'],                                                                                                   wedgeprops={'width': 0.7, 'edgecolor': 'w', 'linewidth': 0.1},                                                                                                    labels = ['기성세대', '신세대'], ax=ax[2,0])
ax[2,0].set_title(sec26_cd_dict_inv['WI240']+' 산업 세대별 비중')
ax[2,0].set_ylabel('')

(temp_inv[['sec_mode','bigen']][temp_inv['sec_mode']=='WI250'].value_counts()/[100*6239/(6239+3531), 100*3531/(6239+3531)]).sort_index().plot.pie(explode=[0.01,0.01],                                                                                                   autopct='%1.1f%%', colors=['#FF4500','#FFA500'],                                                                                                   wedgeprops={'width': 0.7, 'edgecolor': 'w', 'linewidth': 0.1},                                                                                                    labels = ['기성세대', '신세대'], ax=ax[2,1])
ax[2,1].set_title(sec26_cd_dict_inv['WI250']+' 산업 세대별 비중')
ax[2,1].set_ylabel('')

(temp_inv[['sec_mode','bigen']][temp_inv['sec_mode']=='WI260'].value_counts()/[100*6239/(6239+3531), 100*3531/(6239+3531)]).sort_index().plot.pie(explode=[0.01,0.01],                                                                                                   autopct='%1.1f%%', colors=['#FF4500','#FFA500'],                                                                                                   wedgeprops={'width': 0.7, 'edgecolor': 'w', 'linewidth': 0.1},                                                                                                    labels = ['기성세대', '신세대'], ax=ax[2,2])
ax[2,2].set_title(sec26_cd_dict_inv['WI260']+' 산업 세대별 비중')
ax[2,2].set_ylabel('')

plt.subplots_adjust(wspace=0, hspace=0.2)
plt.show()


# 화학 및 철강에서의 차이가 두드러진다.

# In[ ]:


# 주 투자 섹터별 세대 비중: 소비재/금융
f, ax = plt.subplots(3, 3, figsize=(18,18))
(temp_inv[['sec_mode','bigen']][temp_inv['sec_mode']=='WI310'].value_counts()/[100*6239/(6239+3531), 100*3531/(6239+3531)]).sort_index().plot.pie(explode=[0.01,0.01],                                                                                                   autopct='%1.1f%%', colors=['#FF4500','#FFA500'],                                                                                                   wedgeprops={'width': 0.7, 'edgecolor': 'w', 'linewidth': 0.1},                                                                                                    labels = ['기성세대', '신세대'], ax=ax[0,0])
ax[0,0].set_title(sec26_cd_dict_inv['WI310']+' 산업 세대별 비중')
ax[0,0].set_ylabel('')

(temp_inv[['sec_mode','bigen']][temp_inv['sec_mode']=='WI320'].value_counts()/[100*6239/(6239+3531), 100*3531/(6239+3531)]).sort_index().plot.pie(explode=[0.01,0.01],                                                                                                   autopct='%1.1f%%', colors=['#FF4500','#FFA500'],                                                                                                   wedgeprops={'width': 0.7, 'edgecolor': 'w', 'linewidth': 0.1},                                                                                                    labels = ['기성세대', '신세대'], ax=ax[0,1])
ax[0,1].set_title(sec26_cd_dict_inv['WI320']+' 산업 세대별 비중')
ax[0,1].set_ylabel('')

(temp_inv[['sec_mode','bigen']][temp_inv['sec_mode']=='WI330'].value_counts()/[100*3531/(6239+3531), 100*6239/(6239+3531)]).sort_index().plot.pie(explode=[0.01,0.01],                                                                                                   autopct='%1.1f%%', colors=['#FF4500','#FFA500'],                                                                                                   wedgeprops={'width': 0.7, 'edgecolor': 'w', 'linewidth': 0.1},                                                                                                    labels = ['기성세대', '신세대'], ax=ax[0,2])
ax[0,2].set_title(sec26_cd_dict_inv['WI330']+' 산업 세대별 비중')
ax[0,2].set_ylabel('')

(temp_inv[['sec_mode','bigen']][temp_inv['sec_mode']=='WI340'].value_counts()/[100*6239/(6239+3531), 100*3531/(6239+3531)]).sort_index().plot.pie(explode=[0.01,0.01],                                                                                                   autopct='%1.1f%%', colors=['#FF4500','#FFA500'],                                                                                                   wedgeprops={'width': 0.7, 'edgecolor': 'w', 'linewidth': 0.1},                                                                                                    labels = ['기성세대', '신세대'], ax=ax[1,0])
ax[1,0].set_title(sec26_cd_dict_inv['WI340']+' 산업 세대별 비중')
ax[1,0].set_ylabel('')

(temp_inv[['sec_mode','bigen']][temp_inv['sec_mode']=='WI400'].value_counts()/[100*6239/(6239+3531), 100*3531/(6239+3531)]).sort_index().plot.pie(explode=[0.01,0.01],                                                                                                   autopct='%1.1f%%', colors=['#FF4500','#FFA500'],                                                                                                   wedgeprops={'width': 0.7, 'edgecolor': 'w', 'linewidth': 0.1},                                                                                                    labels = ['기성세대', '신세대'], ax=ax[1,1])
ax[1,1].set_title(sec26_cd_dict_inv['WI400']+' 산업 세대별 비중')
ax[1,1].set_ylabel('')

(temp_inv[['sec_mode','bigen']][temp_inv['sec_mode']=='WI410'].value_counts()/[100*6239/(6239+3531), 100*3531/(6239+3531)]).sort_index().plot.pie(explode=[0.01,0.01],                                                                                                   autopct='%1.1f%%', colors=['#FF4500','#FFA500'],                                                                                                   wedgeprops={'width': 0.7, 'edgecolor': 'w', 'linewidth': 0.1},                                                                                                    labels = ['기성세대', '신세대'], ax=ax[1,2])
ax[1,2].set_title(sec26_cd_dict_inv['WI410']+' 산업 세대별 비중')
ax[1,2].set_ylabel('')

(temp_inv[['sec_mode','bigen']][temp_inv['sec_mode']=='WI500'].value_counts()/[100*6239/(6239+3531), 100*3531/(6239+3531)]).sort_index().plot.pie(explode=[0.01,0.01],                                                                                                   autopct='%1.1f%%', colors=['#FF4500','#FFA500'],                                                                                                   wedgeprops={'width': 0.7, 'edgecolor': 'w', 'linewidth': 0.1},                                                                                                    labels = ['기성세대', '신세대'], ax=ax[2,0])
ax[2,0].set_title(sec26_cd_dict_inv['WI500']+' 산업 세대별 비중')
ax[2,0].set_ylabel('')

(temp_inv[['sec_mode','bigen']][temp_inv['sec_mode']=='WI510'].value_counts()/[100*6239/(6239+3531), 100*3531/(6239+3531)]).sort_index().plot.pie(explode=[0.01,0.01],                                                                                                   autopct='%1.1f%%', colors=['#FF4500','#FFA500'],                                                                                                   wedgeprops={'width': 0.7, 'edgecolor': 'w', 'linewidth': 0.1},                                                                                                    labels = ['기성세대', '신세대'], ax=ax[2,1])
ax[2,1].set_title(sec26_cd_dict_inv['WI510']+' 산업 세대별 비중')
ax[2,1].set_ylabel('')

(temp_inv[['sec_mode','bigen']][temp_inv['sec_mode']=='WI520'].value_counts()/[10*6239/(6239+3531), 10*3531/(6239+3531)]).sort_index().plot.pie(explode=[0.01,0.01],                                                                                                   autopct='%1.1f%%', colors=['#FF4500','#FFA500'],                                                                                                   wedgeprops={'width': 0.7, 'edgecolor': 'w', 'linewidth': 0.1},                                                                                                    labels = ['기성세대', '신세대'], ax=ax[2,2])
ax[2,2].set_title(sec26_cd_dict_inv['WI520']+' 산업 세대별 비중')
ax[2,2].set_ylabel('')

plt.subplots_adjust(wspace=0, hspace=0.2)
plt.show()


# 미디어/교육에서의 차이가 두드러진다.

# In[ ]:


# 주 투자 섹터별 세대 비중: 자동차/IT/하드웨어
f, ax = plt.subplots(4, 2, figsize=(12,20))
(temp_inv[['sec_mode','bigen']][temp_inv['sec_mode']=='WI300'].value_counts()/[100*6239/(6239+3531), 100*3531/(6239+3531)]).sort_index().plot.pie(explode=[0.01,0.01],                                                                                                   autopct='%1.1f%%', colors=['#FF4500','#FFA500'],                                                                                                   wedgeprops={'width': 0.7, 'edgecolor': 'w', 'linewidth': 0.1},                                                                                                    labels = ['기성세대', '신세대'], ax=ax[0,0])
ax[0,0].set_title(sec26_cd_dict_inv['WI300']+' 산업 세대별 비중')
ax[0,0].set_ylabel('')

(temp_inv[['sec_mode','bigen']][temp_inv['sec_mode']=='WI600'].value_counts()/[100*6239/(6239+3531), 100*3531/(6239+3531)]).sort_index().plot.pie(explode=[0.01,0.01],                                                                                                   autopct='%1.1f%%', colors=['#FF4500','#FFA500'],                                                                                                   wedgeprops={'width': 0.7, 'edgecolor': 'w', 'linewidth': 0.1},                                                                                                    labels = ['기성세대', '신세대'], ax=ax[0,1])
ax[0,1].set_title(sec26_cd_dict_inv['WI600']+' 산업 세대별 비중')
ax[0,1].set_ylabel('')

(temp_inv[['sec_mode','bigen']][temp_inv['sec_mode']=='WI610'].value_counts()/[100*6239/(6239+3531), 100*3531/(6239+3531)]).sort_index().plot.pie(explode=[0.01,0.01],                                                                                                   autopct='%1.1f%%', colors=['#FF4500','#FFA500'],                                                                                                   wedgeprops={'width': 0.7, 'edgecolor': 'w', 'linewidth': 0.1},                                                                                                    labels = ['기성세대', '신세대'], ax=ax[1,0])
ax[1,0].set_title(sec26_cd_dict_inv['WI610']+' 산업 세대별 비중')
ax[1,0].set_ylabel('')

(temp_inv[['sec_mode','bigen']][temp_inv['sec_mode']=='WI620'].value_counts()/[100*6239/(6239+3531), 100*3531/(6239+3531)]).sort_index().plot.pie(explode=[0.01,0.01],                                                                                                   autopct='%1.1f%%', colors=['#FF4500','#FFA500'],                                                                                                   wedgeprops={'width': 0.7, 'edgecolor': 'w', 'linewidth': 0.1},                                                                                                    labels = ['기성세대', '신세대'], ax=ax[1,1])
ax[1,1].set_title(sec26_cd_dict_inv['WI620']+' 산업 세대별 비중')
ax[1,1].set_ylabel('')

(temp_inv[['sec_mode','bigen']][temp_inv['sec_mode']=='WI630'].value_counts()/[100*6239/(6239+3531), 100*3531/(6239+3531)]).sort_index().plot.pie(explode=[0.01,0.01],                                                                                                   autopct='%1.1f%%', colors=['#FF4500','#FFA500'],                                                                                                   wedgeprops={'width': 0.7, 'edgecolor': 'w', 'linewidth': 0.1},                                                                                                    labels = ['기성세대', '신세대'], ax=ax[2,0])
ax[2,0].set_title(sec26_cd_dict_inv['WI630']+' 산업 세대별 비중')
ax[2,0].set_ylabel('')

(temp_inv[['sec_mode','bigen']][temp_inv['sec_mode']=='WI640'].value_counts()/[100*6239/(6239+3531), 100*3531/(6239+3531)]).sort_index().plot.pie(explode=[0.01,0.01],                                                                                                   autopct='%1.1f%%', colors=['#FF4500','#FFA500'],                                                                                                   wedgeprops={'width': 0.7, 'edgecolor': 'w', 'linewidth': 0.1},                                                                                                    labels = ['기성세대', '신세대'], ax=ax[2,1])
ax[2,1].set_title(sec26_cd_dict_inv['WI640']+' 산업 세대별 비중')
ax[2,1].set_ylabel('')

(temp_inv[['sec_mode','bigen']][temp_inv['sec_mode']=='WI700'].value_counts()/[10*6239/(6239+3531), 10*3531/(6239+3531)]).sort_index().plot.pie(explode=[0.01,0.01],                                                                                                   autopct='%1.1f%%', colors=['#FF4500','#FFA500'],                                                                                                   wedgeprops={'width': 0.7, 'edgecolor': 'w', 'linewidth': 0.1},                                                                                                    labels = ['기성세대', '신세대'], ax=ax[3,0])
ax[3,0].set_title(sec26_cd_dict_inv['WI700']+' 산업 세대별 비중')
ax[3,0].set_ylabel('')

(temp_inv[['sec_mode','bigen']][temp_inv['sec_mode']=='WI800'].value_counts()/[100*6239/(6239+3531), 100*3531/(6239+3531)]).sort_index().plot.pie(explode=[0.01,0.01],                                                                                                   autopct='%1.1f%%', colors=['#FF4500','#FFA500'],                                                                                                   wedgeprops={'width': 0.7, 'edgecolor': 'w', 'linewidth': 0.1},                                                                                                    labels = ['기성세대', '신세대'], ax=ax[3,1])
ax[3,1].set_title(sec26_cd_dict_inv['WI800']+' 산업 세대별 비중')
ax[3,1].set_ylabel('')


plt.subplots_adjust(wspace=0, hspace=0.2)
plt.show()


# IT하드웨어와 디스플레이에서의 차이가 두드러진다.

# In[ ]:


# 모형결과 
imp_var_lr.index = imp_var_lr['var']
imp_var_lr = imp_var_lr.drop(['var'], axis=1)

plt.figure(figsize=(18, 11))

imp_var_lr.sort_values(['imp'], ascending=True)['imp'].plot.barh(color=['#4B2991','#692A99','#672CA2','#952EA0','#A3319F','#C0369D','#CA3C97','#DF488D',                                '#ED5983','#F66D7A','#F98477','#f79C79','#F3B584','#EFCC98'], rot=0, alpha=0.85)
plt.title('로지스틱 회귀 모형 변수 중요도')
plt.xlabel('회귀 계수 절댓값')
plt.ylabel('')
plt.yticks(range(33), ['계좌 보유 기간 대비 계좌 수','고객등급','나이 대비 계좌 보유 기간','거래단말(유선단말)','거래단말(영업점단말)','투자성향(안정형)','거래단말(MTS)',
                       '투자성향(공격투자형)','투자성향(안정추구형)','일평균 매수 단가', '주거래섹터(미디어/교육)','주거래섹터(IT하드웨어)','투자성향(위험중립형)','투자성향(적극투자형)',
                       '주거래섹터(에너지)', '투자 섹터 개수', '주거래섹터(철강)','성별(남성)','주거래섹터(호텔/레저)','주거래섹터(건설)','주거래섹터(건강관리)','거주지(수도권)',
                       '주거래섹터(IT가)','주거래섹터(화학)','주거래섹터(화장품/의류)','주거래섹터(디스플레이)','주거래섹터(기계)','거주지(광역/특별시)','주거래섹터(비철금속)',
                       '주거래섹터(필수소비재)', '주거래섹터(소프트웨어)','주거래섹터(반도체)','주거래섹터(자동차)'][::-1], size='small')

for i, v in enumerate(imp_var_lr.sort_values(['imp'], ascending=True)['imp']):
    plt.text(v + 0.035, i + -.25, str(round(v,3)), color='black', size='small')

plt.show()


# In[ ]:


plt.figure(figsize=(18, 11))

imp_var_lr.sort_values(['imp'], ascending=True)['coef'].plot.barh(color=['#4B2991','#692A99','#672CA2','#952EA0','#A3319F','#C0369D','#CA3C97','#DF488D',                                '#ED5983','#F66D7A','#F98477','#f79C79','#F3B584','#EFCC98'], rot=0, alpha=0.85)
plt.title('로지스틱 회귀 계수')
plt.xlabel('회귀 계수')
plt.ylabel('')
plt.yticks(range(33), ['계좌 보유 기간 대비 계좌 수','고객등급','나이 대비 계좌 보유 기간','거래단말(유선단말)','거래단말(영업점단말)','투자성향(안정형)','거래단말(MTS)',
                       '투자성향(공격투자형)','투자성향(안정추구형)','일평균 매수 단가', '주거래섹터(미디어/교육)','주거래섹터(IT하드웨어)','투자성향(위험중립형)','투자성향(적극투자형)',
                       '주거래섹터(에너지)', '투자 섹터 개수', '주거래섹터(철강)','성별(남성)','주거래섹터(호텔/레저)','주거래섹터(건설)','주거래섹터(건강관리)','거주지(수도권)',
                       '주거래섹터(IT가)','주거래섹터(화학)','주거래섹터(화장품/의류)','주거래섹터(디스플레이)','주거래섹터(기계)','거주지(광역/특별시)','주거래섹터(비철금속)',
                       '주거래섹터(필수소비재)', '주거래섹터(소프트웨어)','주거래섹터(반도체)','주거래섹터(자동차)'][::-1], size='small')

for i, v in enumerate(imp_var_lr.sort_values(['imp'], ascending=True)['coef']):
    plt.text(v + 0, i + 0.25, str(round(v,3)), color='black', size='small')

plt.show()


# 사전에 표준화를 진행하였으므로 계수의 절댓값이 클수록 중요한 변수이며,
# 반응변수를 신세대=1, 기성세대=0으로 코딩해 계수가 양수이면 해당 변수값이 높아질수록 신세대일 확률이 높고, 음수이면 기성세대일 확률이 높다고 할 수 있다. 
# 
# 가장 주요하게 작용한 변수는 계좌 보유 시간 대비 계좌 수이다. 나이 대비 계좌 보유 기간과 함께 해석하면, **신세대는 기성 세대에 비해 투자 시작 시기는 늦으나 계좌는 더 많이 개설했다고 할 수 있다.** 
# 
# 기성세대에 비해 고객등급이 낮고, 유선단말과 영업점 단말은 상대적으로 이용 비중이 적으나 MTS, 모바일 기기를 통한 투자 비중은 높다.  
# 
# 범주화된 투자 성향의 기본값 (0으로 처리된 범주) 은 '정보 없음'이다. 즉, 투자 성향 관련 변수의 계수 부호가 전부 양수라는 것은 **신세대가 기성 세대에 비해 투자 성향이 확실**하다는 것을 의미한다. 투자 성향이 고객이 스스로 응답한 값일 경우, 실제 투자 성향과 일치하지 않을 수도 있지만, 그들 스스로가 생각하는 투자 성향이 일단은 존재한다는 뜻이다. 기성세대에 비해 일평균 매수 단가가 높은 종목을 구매하는 것을 보면 실제로도 안정적인 성향을 보인다고 할 수 있다. 
# 
# 투자 섹터의 경우, 기본값을 주식 이외의 종목으로 하였을 때, 즉 주식 외 종목에 투자하는 사람 수 대비  **미디어/교육, 에너지 등의 신산업에 더 많이 투자하며, IT하드웨어, 철강 등의 전통적 산업에는 더 적게 투자**한다. 투자 섹터의 개수가 더 많으므로 기성세대에 비해 더 다양한 섹터에 투자한다는 사실을 알 수 있다. 

# In[ ]:


imp_var_dt.index = imp_var_dt['var']
imp_var_dt = imp_var_dt.drop(['var'], axis=1)

plt.figure(figsize=(18, 6))

imp_var_dt.sort_values(['imp'], ascending=True)['imp'].plot.barh(color=['#4B2991','#672CA2','#A3319F','#CA3C97','#DF488D',                                '#ED5983','#F66D7A','#F98477','#f79C79','#F3B584','#EFCC98'], rot=0, alpha=0.85)
plt.title('의사결정나무 모형 변수 중요도')
plt.xlabel('변수 중요도')
plt.ylabel('')
plt.yticks(range(15), ['계좌 보유 기간 대비 계좌 수','나이 대비 계좌 보유 기간','고객등급','일평균 매도 단가','고객평균 총수익','일평균 매수 단가',
                       '일평균 매도 수량','투자 종목 개수','일평균 매도 횟수','일평균 매수 수량','거래단말(영업점단말)','투자성향(위험중립형)',
                       '거래단말(MTS)','일평균 매수 횟수','투자 섹터 개수'][::-1], size='small')

for i, v in enumerate(imp_var_dt.sort_values(['imp'], ascending=True)['imp']):
    plt.text(v + 0.005, i - 0.25, str(round(v,3)), color='black', size='small')

plt.show()


# In[ ]:


from sklearn.tree import export_graphviz
from subprocess import call

export_graphviz(best_dt, out_file=basic_path+'/best_dt.dot',    
                class_names=['기성세대', '신세대'],
                feature_names=['고객등급', '성별(남성)', '거주지(수도권)', '거주지(광역/특별시)', '투자성향(안정형)',
       '투자성향(안정추구형)', '투자성향(위헝중립형)', '투자성향(적극투자형)', '투자성향(공격투자형)',
       '투자성향(전문투자자형)', '계좌 보유기간 대비 계좌 수', '나이 대비 계좌 보유 기간', '일평균 매도 횟수', '일평균 매수 횟수',
       '일평균 매도 수량', '일평균 매수 수량', '일평균 매도 단가', '일평균 매수 단가', '총수익', '투자 섹터 개수',
       '주거래섹터(에너지)', '주거래섹터(화학)', '주거래섹터(비철금속)',
       '주거래섹터(철강)', '주거래섹터(건설)', '주거래섹터(기계)',
       '주거래섹터(조선)', '주거래섹터(상사/자본재)', '주거래섹터(운송)',
       '주거래섹터(자동차)', '주거래섹터(화장품/의류)', '주거래섹터(호텔/레저)',
       '주거래섹터(미디어/교육)', '주거래섹터(소매(유통))', '주거래섹터(필수소비재)',
       '주거래섹터(건강관리)', '주거래섹터(은헹)', '주거래섹터(증권)',
       '주거래섹터(보험)', '주거래섹터(소프트웨어)', '주거래섹터(IT하드웨어)',
       '주거래섹터(반도체)', '주거래섹터(IT가전)', '주거래섹터(디스플레이)',
       '주거래섹터(전기통신서비스)', '주거래섹터(유틸리티)', '투자 종목 개수', '거래단말(영업점단말)',
       '거래단말(유선단말)', '거래단말(MTS)'])
call(['dot', '-Tpng', basic_path+'/best_dt.dot', '-o', basic_path+'/best_dt.png', '-Gdpi=600'])


# In[ ]:


from IPython.display import Image
Image(filename = basic_path+'/best_dt.png')


# 의사결정나무 모형에서는 로지스틱 회귀 모형과 마찬가지로 계좌 보유 기간 대비 계좌 수가 가장 중요하게 작용하며, 다음으로는 나이 대비 계좌 보유 기간과 고객등급의 영향이 크다.
# 
# 사용 단말, 투자 섹터 개수, 위험중립형 투자 성향 또한 로지스틱 회귀 모형과 동일하게 중요하게 산출되었다.
# 
# 총수익과 일평균 매도/매수 단가, 수량, 횟수 그리고 투자 종목의 종류는 의사결정나무 모형에서만 발견되었다.  
# 
# 나무 그림에서는 각 가지별 대소관계가 공통적인 변수에 대해 내용을 해석하였다.
# 
# 계좌 보유 기간 대비 계좌 수와 나이 대비 계좌 보유 기간, 고객 등급에 대한 해석은 로지스틱 회귀 모형과 동일하다. 투자를 늦게 시작한 데 비해 계좌 수가 많으며 고객 등급은 낮은 편이다.
# 
# **신세대의 매도/매수 횟수가 더 많으므로 거래를 더 빈번하게 한다**고 할 수 있고, 매수 단가는 로지스틱 회귀 모형과 동일하다. 매도/매수 횟수 평균을 그림으로 나타냈을 때에는 유사해 보였으나 의사결정나무 모형을 통해 두 세대 간 차이가 있음을 알 수 있다. 

# ----------------------------------------
# 
# ### 두 모형을 종합해 내린 결론은 다음과 같다. 
# 
# ### 신세대 (Y & Z) 는 기성세대 (X & 베이비붐)에 비해 투자 시작 연령이 늦지만 계좌 수와 거래 빈도가 더 많기에 **투자를 더 활발하게 한다**고 할 수 있다. **접근성이 좋은 모바일 단말의 높은 이용 비중**과도 관련되어 있을 것이다.
# 
# ### **스스로의 투자 성향을 자각**하고 있는 비중이 상대적으로 높으며, 그 중 안정추구형과 위험중립형 비중은 높 적극투자형 비중은 낮은 사실을 통해 **기성세대에 비해 안정을 추구**한다는 사실을 알 수 있다. 일평균 매수 종목의 단가가 더 높다는 사실이 그를 뒷받침한다. 
# 
# ### **보다 다양한 섹터에 투자한다는 점을 종합하면, 신세대의 투자 시작 연령은 늦을지언정 투자에 대한 관심이 높고 현재 활발하게 투자** 중이라는 사실을 알 수 있다. 
# 
# ### 고객등급은 기성세대에 비해 낮지만, 신세대의 투자 경력이 짧다는 사실을 감안하면 점차 높은 고객 등급에서 신세대가 차지하는 비중이 높아질 수 있을 것이다.
# 
# ### 투자 성향과 행태 모두 새로운 신세대가 앞으로 어떻게 변화해갈지 궁금해지는 결과이다. 
