from RegressorKit import RegressorKit

rk = RegressorKit() # 클래스 초기화

rk.read('./datasets/stock_market.csv') # 데이터셋 불러오기
'''
Dataset Source: https://bit.ly/2Q4RlMF
'''
rk.unique_record() # 데이터셋의 필드 및 고유 값 확인

rk.drop(['Symbol', 'Name', 'SEC Filings']) # 불필요한 필드 제거
rk.show(10) # 불필요한 필드가 제거된 데이터셋 확인

rk.heatmap()
'''
상관관계가 높은 필드 분석:
    Price & 52 Week Low : 1
    Price & 52 Week High : 0.98
    52 Week Low & 52 Week High : 0.98
    
    Earnings/Share & Price : 0.59
    Earnings/Share & 52 Week Low : 0.59
    Earnings/Share & 52 Week High : 0.6

    EBITDA & Market Cap : 0.77
'''

rk.pairplots(['Price', '52 Week Low', '52 Week High', 'Earnings/Share'], 'Sector')
rk.pairplots(['EBITDA', 'Market Cap'], 'Sector')
'''
열지도에서 얻은 상관관계가 높은 필드만 따로 골라 2차원 그래프 위에 나타내기:
    1. Price, 52 Week Low, 52 Week High : 세 필드는 서로 매우 높은 상관관계를 보임. 해당 필드로 기계를 학습할 경우 Bias가 낮지만, Variance가 매우 높을 위험이 있음.
    2. Earnings/Share에 대한 Price, 52 Week Low, 52 Week High: 이 네가지 상관관계는 비교적 낮은 상관관계를 보이고, 아웃라이어가 다수 존재함. Bias는 높지만, Variance가 낮을 것으로 예상됨.
    3. EBITDA와 Market Cap: 1과 2 중간의 적당한 상관관계.
'''

rk.plot3d('52 Week Low', '52 Week High', 'Price', 'Market Cap', 'Sector', 'EBITDA')
'''
앞서 골라낸 모든 필드를 3차원 그래프로 나타내기:
    x축: 52 Week Low
    y축: 52 Week High
    z축: Price
    버블 크기: Maket Cap
    버블 색: Sector
    추가 정보: EBITDA
'''

rk.boxplot('Sector', 'Price')
'''
분야 별 주식거래가 분석:
    11개의 분야 별로 주식거래가가 모두 비슷하고, 매우 넓은 범위에 걸쳐 아웃라이어가 분포해 학습에 쓰일만한 자료로 보이지 않음.
'''

# 따라서, 앞서 얻은 다섯가지 필드 ['52 Week Low', '52 Week High', 'Earnings/Share', 'EBITDA', 'Market Cap']을 이용해 'Price'를 예측
# 11개의 회귀 알고리즘을 이용해 기계를 학습시키고, 그 중 가장 높은 성능을 내는 알고리즘을 예측 모델로 채택

rk.execute(['52 Week Low', '52 Week High', 'Earnings/Share', 'EBITDA', 'Market Cap'], 'Price')
