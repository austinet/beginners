from RegressorKit import RegressorKit

rk = RegressorKit() # 클래스 초기화

rk.read('./datasets/house_pricing.csv')
rk.unique_record() # 데이터셋의 필드 및 고유 값 확인

rk.heatmap()
