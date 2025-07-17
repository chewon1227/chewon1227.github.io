---
layout: post
title: "서울시 버스 노선 정류장 승하차 분석 - 01"
permalink: /eda/bus-eda-01/
subtitle: 공공데이터 포털에 있는 서울시 버스노선별 정류장별 승하차 인원 정보 데이터를 이용해서 EDA하기 위한 기초 작업을 해보자.  
parent: data-analyze
---


[서울시 버스노선별 정류장별 승하차 인원 정보](https://data.seoul.go.kr/dataList/OA-12912/S/1/datasetView.do)

우선 여기서 csv 파일을 다운받았다. 
json 파일로 하고 싶었는데 ,, 파일을 불러올 때 계속 오류가 났다. 

구글 코랩에서 진행한다. (EDA 결과를 바로바로 볼 수 있기 때문 .. 런타임 오류가 뜨는게 좀 킹받긴 한데) 

```python
from google.colab import files
import pandas as pd
import json
```

코랩에서 파일을 불러온다. 구글 드라이브에 올려놓아야 접근이 가능하다. 그리고 인코딩은 cp949로 해야한다. utf-8로는 인식이 안됨. 

```python
file_path = '/content/drive/MyDrive/eda_data/서울시 버스노선별 정류장별 승하차 인원 정보.csv'
data = pd.read_csv(file_path, encoding='cp949')
```

우선 2023년, 2024년 자료만 보기 위해 자르고 승/하차 일자는 필요 없기 때문에 연도별로 싹 묶어 역명 순으로 정렬한다. 

```python
data['사용일자'] = pd.to_datetime(data['사용일자'], format='%Y%m%d')
data = data[(data['사용일자'].dt.year >= 2023) & (data['사용일자'].dt.year <= 2024)]
data = data[~data['노선번호'].str.contains('N', na=False)]
data = data.sort_values(by='역명')
data['사용연도'] = data['사용일자'].dt.year
```

사용 연도와 노선번호, 정류장ID가 모두 같을 경우 승차 총 승객수와 하차 총 승객수를 각각 합치는 작업을 수행한다. 2024년과 2023년만 떼오긴 했으나 뭐 .. 비슷할 것이기 때문에 합친다. 대신 마구잡이로 합칠 순 없으니 노선 번호와 정류장ID가 아예 동일할 때만 합친다. 이 두개가 거의 고유번호인 것 같았다. 

```python

clean_data = data.groupby(
    ['사용연도', '노선번호', '노선명', '표준버스정류장ID', '버스정류장ARS번호','역명']
).agg({
    '승차총승객수': 'sum',
    '하차총승객수': 'sum'
}).reset_index()

columns= ['사용연도', '노선번호', '노선명', '표준버스정류장ID', '버스정류장ARS번호', '역명', '승차총승객수', '하차총승객수']
clean_data = pd.merge(clean_data, clean_data.drop(columns=['승차총승객수', '하차총승객수']), 
                       on=['사용연도', '노선번호', '노선명', '표준버스정류장ID', '버스정류장ARS번호', '역명'], 
       how='left').drop_duplicates()
```

승차 총 승객수와 하차 총 승객수 기준으로 나열해본다. 

```python
clean_data = clean_data.sort_values(by='승차총승객수', ascending=False)
clean_data = clean_data.sort_values(by='하차총승객수', ascending=False)
```

승차 총 승객수가 많은 곳은 회기역, 당산역, 신촌전철역, 가산디지털단지역, 미아사거리전철역이 top 6 이었다. 

하차 총 승객수가 많은 곳은 구로디지털단지역, 낙선대역, 가산디저털단지역, 당산역, 홍대입구역이 top6 이었다. 

그럼 여기서 떠오르는 질문만 해도 여러가지다. 

1. 승하차 총 승객수가 많은 곳은 무조건 지하철역일까? 
2. 승차보다 하차 승객수가 많은 곳은 어떤 지역일까? 어떤 특징을 가졌을까? 반대인 지역은 어떨까? 
3. 같은 버스정류장을 지나는 버스는 몇 대일까? 
4. 주말과 주중은 어떻게 패턴이 다를까? (주중에는 산업단지 위주, 주말은 노는 곳 위주일테니 ..)

앞으로 이 질문들에 대해서 확장시켜 나갈 계획이다. 