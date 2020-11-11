"""
Description
---
    머신 러닝과 관련된 기능을 제공하는 라이브러리입니다.

Classes
---
    datafit
    graph

Author
---
    Austin, 2019
    제주대학교 컴퓨터공학전공 인공지능 수업
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class datafit:
    """
    Introduction
    ---
        csv 데이터셋 준비와 관련된 기능을 제공합니다.
        datafit()으로 초기화 하는 것을 권장합니다.

    Methods
    ---
        read(): csv 데이터셋 불러오기
        show(): 데이터셋의 필드와 레코드 확인
        drop(): 데이터셋의 특정 필드 제거
    """


    def read(self, csv_file):
        """
        csv 데이터셋을 불러옵니다.

        Parameters
        ---
            csv_file: csv 파일명. "example.csv" 형태로 입력
        """
        dataset = pd.read_csv(csv_file)
        return dataset


    def show(self, df, num):
        """
        데이터셋의 필드와 레코드를 확인합니다.

        Parameters
        ---
            df: read()로부터 불러온 csv 파일이 저장된 객체
            num: 확인하고 싶은 상위 레코드 수. 0이 입력되면 필드만 표시
        """
        # 데이터셋의 필드 수, 레코드 수 출력
        print("[ Fields:", df.shape[1], "Records:", df.shape[0], "]")

        # 데이터셋의 필드 및 레코드 출력
        if num == 0:
            print(list(df.head(0))) # 필드 값만 리스트로 출력
        else:
            print(df.head(num))


    def drop(self, df, field):
        """
        데이터셋에서 불필요한 필드를 제거합니다.

        Parameters
        ---
            df: read()로부터 불러온 csv 파일이 저장된 객체
            field: 제거할 필드 이름. ["필드 1", "필드 2", ...]와 같이 list 입력
        """
        df.drop(field, axis = 1, inplace = True)


class graph:
    """
    Introduction
    ---
        데이터 시각화 기능을 제공합니다.
        graph()로 초기화하는 것을 권장합니다.

    Methods
    ---
        plot(): 점 그래프 생성
        violin_plot(): 바이올린 분포 그래프 생성
        histogram(): 도수 분포 그래프 생성
        heatmap(): 히트맵 생성
    """

    
    def plot(self, df, field_x, field_y, cluster):
        """
        점 그래프를 생성합니다. 10가지 점 색상을 지원합니다.

        Parameters
        ---
            df: read()로부터 불러온 csv 파일이 저장된 객체
            field_x: 그래프 x축으로 지정될 필드. "필드 이름" 형태로 입력
            field_y: 그래프 y축으로 지정될 필드. "필드 이름" 형태로 입력
            cluster: 점 그래프 위에 표시될 필드. "필드 이름" 형태로 입력
        """
        # 클러스터 생성에 필요한 list
        fields = df[cluster].unique() # 데이터셋에서 필드만 가져와서 list로 저장
        cluster_color = ['red', 'blue', 'green', 'orange', 'purple', 'yellow', 'skyblue', 'brown', 'black', 'grey']

        # 첫번째 필드의 클러스터 생성
        fig = \
            df[df[cluster] == fields[0]].plot.scatter(x=field_x, y=field_y, color=cluster_color[0], label=cluster)
        # 2 ~ n번째 필드의 클러스터 자동 생성
        for i in range(len(fields)-1):
            df[df[cluster] == fields[i+1]].plot.scatter(x=field_x, y=field_y, color=cluster_color[i+1], label=cluster, ax=fig)
        
        fig.set_xlabel(field_x)
        fig.set_ylabel(field_y)
        fig.set_title(field_x + " vs. " + field_y)
        fig = plt.gcf()
        fig.set_size_inches(10, 6)
        
        plt.show()


    def violin_plot(self, df, field_x, field_y):
        """
        바이올린 분포 그래프를 생성합니다.

        Parameters
        ---
            df: read()로부터 불러온 csv 파일이 저장된 객체
            field_x: 그래프 x축으로 지정될 필드. "필드 이름" 형태로 입력
            field_y: 그래프 y축으로 지정될 필드. "필드 이름" 형태로 입력
        """
        plt.figure(figsize=(5, 4))
        plt.subplot(1, 1, 1)
        sns.violinplot(x=field_x, y=field_y, data=df)
        plt.show()

    
    def histogram(self, df):
        """
        도수 분포 그래프를 생성합니다. 모든 필드 간 도수 분포 관계를 확인할 수 있습니다.

        Parameters
        ---
            df: read()로부터 불러온 csv 파일이 저장된 객체
        """
        df.hist(edgecolor='black', linewidth=1.2)
        fig = plt.gcf()
        fig.set_size_inches(6,5)
        plt.show()


    def heatmap(self, df):
        """
        열지도를 생성합니다. 모든 필드 간 상관 관계를 확인할 수 있습니다.

        Parameters
        ---
            df: read()로부터 불러온 csv 파일이 저장된 객체
        """
        plt.figure(figsize=(10,6))
        sns.heatmap(df.corr(),annot=True,cmap='cubehelix_r')
        plt.show()


