import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="서울시 공원 데이터 분석",
    page_icon="🌳",
    layout="wide"
)

st.title("🌳 서울시 주요 공원 데이터 분석")
st.write("서울시 주요 공원 데이터를 살펴보는 프로젝트입니다.")

df = pd.read_csv(
    "서울시 주요 공원현황.csv",
    encoding="cp949"
)
st.success("데이터 불러오기 성공!")

st.write("데이터 크기:", df.shape)

st.subheader("데이터 미리보기")
st.dataframe(df)
# 결측치 분석
st.subheader("🔍 결측치 분석")

missing_count = df.isnull().sum()
missing_ratio = (missing_count / len(df)) * 100

missing_df = pd.DataFrame({
    "항목": missing_count.index,
    "결측치 개수": missing_count.values,
    "결측치 비율(%)": missing_ratio.values.round(2)
})

missing_df = missing_df[
    missing_df["결측치 개수"] > 0
].sort_values(
    by="결측치 개수",
    ascending=False
)

if missing_df.empty:
    st.success("결측치가 없습니다.")
else:
    st.write("항목별 결측치 현황")
    st.dataframe(missing_df, use_container_width=True)

    st.write("항목별 결측치 개수")
    st.bar_chart(
        missing_df.set_index("항목")["결측치 개수"]
    )

    st.write("항목별 결측치 비율")
    st.bar_chart(
        missing_df.set_index("항목")["결측치 비율(%)"]
    )
# 이상치 분석
st.subheader("📊 공원 면적 이상치 분석")

# 면적에서 첫 번째 숫자 추출
area_number = df["면적"].astype(str).str.extract(
    r"([\d,]+(?:\.\d+)?)"
)[0]

# 쉼표 제거 후 숫자로 변환
area_number = pd.to_numeric(
    area_number.str.replace(",", "", regex=False),
    errors="coerce"
)

# 결측치가 아닌 면적 데이터만 사용
valid_area = area_number.dropna()

# IQR 계산
Q1 = valid_area.quantile(0.25)
Q3 = valid_area.quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

# 이상치 추출
outliers = df.loc[
    area_number.notna() &
    ((area_number < lower) | (area_number > upper))
].copy()

outliers["면적_숫자"] = area_number.loc[outliers.index]

# 분석 결과 표시
st.write("**IQR 분석 기준**")
st.write(f"Q1: {Q1:,.2f}")
st.write(f"Q3: {Q3:,.2f}")
st.write(f"IQR: {IQR:,.2f}")
st.write(f"이상치 상한: {upper:,.2f}")

st.write("**이상치로 탐지된 공원**")
st.write(f"전체 분석 대상: {len(valid_area)}개")
st.write(f"탐지된 이상치: {len(outliers)}개")

st.dataframe(
    outliers[["공원명", "면적", "지역", "면적_숫자"]],
    use_container_width=True
)

st.write("**이상치 공원 면적 비교 (로그 변환)**")

if not outliers.empty:

    # 면적을 로그10으로 변환
    import numpy as np

    outliers["로그 변환 면적"] = np.log10(
        outliers["면적_숫자"].clip(lower=1)
    )

    st.bar_chart(
        outliers.set_index("공원명")["로그 변환 면적"]
    )

    st.caption(
        "로그 변환을 적용한 그래프입니다. "
        "실제 면적값은 위 표에서 확인할 수 있습니다."
    )

else:
    st.success("이상치가 발견되지 않았습니다.")
# 지역별 공원 분포 분석
st.subheader("🗺️ 지역별 공원 분포")

region_data = df["지역"].dropna().astype(str).str.strip()
region_data = region_data[region_data != ""]

region_count = region_data.value_counts()

region_table = pd.DataFrame({
    "지역": region_count.index,
    "공원 수": region_count.values
})

st.write("자치구별 공원 수")

st.dataframe(
    region_table,
    use_container_width=True
)

st.bar_chart(
    region_table.set_index("지역")["공원 수"]
)

st.write("지역 정보가 있는 데이터:", len(region_data), "개")
# 공원 면적 TOP 10
st.subheader("🏆 면적이 넓은 공원 TOP 10")

top10_data = df[["공원명", "면적"]].copy()

top10_data["면적_숫자"] = pd.to_numeric(
    top10_data["면적"]
    .astype(str)
    .str.extract(r"([\d,]+(?:\.\d+)?)")[0]
    .str.replace(",", "", regex=False),
    errors="coerce"
)

top10_data = top10_data.dropna(
    subset=["면적_숫자"]
)

top10_data = top10_data.sort_values(
    by="면적_숫자",
    ascending=False
).head(10)

st.dataframe(
    top10_data[["공원명", "면적"]],
    use_container_width=True
)

import numpy as np

st.write("면적 TOP 10 (로그 변환 그래프)")

top10_data["로그 면적"] = np.log10(
    top10_data["면적_숫자"].clip(lower=1)
)

st.bar_chart(
    top10_data.set_index("공원명")["로그 면적"]
)

st.caption(
    "로그 변환 그래프입니다. 실제 면적은 위 표에서 확인하세요."
)
# K-means 군집 분석
st.subheader("🌳 K-means 공원 유형 분석")

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np

# 분석에 사용할 데이터 복사
cluster_df = df[["공원명", "면적", "주요시설"]].copy()

# 면적에서 숫자 추출
cluster_df["면적_숫자"] = pd.to_numeric(
    cluster_df["면적"]
    .astype(str)
    .str.extract(r"([\d,]+(?:\.\d+)?)")[0]
    .str.replace(",", "", regex=False),
    errors="coerce"
)

# 단위가 ㎡ 또는 m²인 데이터만 사용
area_text = cluster_df["면적"].astype(str)
valid_unit = area_text.str.contains(
    r"㎡|m²",
    regex=True,
    na=False
)

# 북한산국립공원은 면적값 검증 전까지 학습에서 제외
valid_rows = (
    valid_unit
    & cluster_df["면적_숫자"].notna()
    & (cluster_df["공원명"] != "북한산국립공원")
)

cluster_df = cluster_df.loc[valid_rows].copy()

# 면적 로그 변환
cluster_df["면적_로그"] = np.log10(
    cluster_df["면적_숫자"].clip(lower=1)
)

# 주요시설 텍스트에서 시설 종류 추출
facility_text = cluster_df["주요시설"].fillna("").astype(str)

facility_keywords = {
    "운동시설": ["운동시설", "운동장", "테니스장", "농구장", "축구장"],
    "조경시설": ["조경시설", "연못", "분수", "정원"],
    "휴양시설": ["휴양시설", "산책로", "쉼터", "정자"],
    "편익시설": ["편익시설", "화장실", "주차장", "음수대"],
    "문화시설": ["문화시설", "야외무대", "전시관", "도서관"],
    "생태시설": ["생태", "식물원", "동물원", "자연학습장"],
    "놀이시설": ["놀이시설", "놀이터", "놀이동산"]
}

for category, keywords in facility_keywords.items():
    cluster_df[category] = facility_text.apply(
        lambda x: sum(keyword in x for keyword in keywords)
    )

# 군집 분석에 사용할 변수
feature_columns = ["면적_로그"] + list(facility_keywords.keys())

X = cluster_df[feature_columns]

# 변수 표준화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-means 군집 분석
st.subheader("🌳 K-means 공원 유형 분석")

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 분석 데이터 준비
cluster_df = df[["공원명", "면적", "주요시설"]].copy()

# 면적 숫자 추출
area_text = cluster_df["면적"].astype(str)

cluster_df["면적_숫자"] = pd.to_numeric(
    area_text.str.extract(r"([\d,]+(?:\.\d+)?)")[0]
    .str.replace(",", "", regex=False),
    errors="coerce"
)

# 면적 단위가 ㎡ 또는 m²인 데이터만 사용
valid_unit = area_text.str.contains(
    r"㎡|m²",
    regex=True,
    na=False
)

# 면적 데이터 정리
valid_rows = (
    valid_unit
    & cluster_df["면적_숫자"].notna()
    & (cluster_df["면적_숫자"] > 0)
    & (cluster_df["공원명"] != "북한산국립공원")
)

cluster_df = cluster_df.loc[valid_rows].copy()

# 면적 로그 변환
cluster_df["면적_로그"] = np.log10(
    cluster_df["면적_숫자"]
)

# 주요시설 텍스트 정리
facility_text = (
    cluster_df["주요시설"]
    .fillna("")
    .astype(str)
)

# 시설 종류별 키워드
facility_keywords = {
    "운동시설": ["운동시설", "테니스장", "농구장", "축구장", "운동기구"],
    "조경시설": ["조경시설", "연못", "분수", "정원", "파고라"],
    "휴양시설": ["휴양시설", "산책로", "쉼터", "정자", "벤치"],
    "편익시설": ["편익시설", "화장실", "주차장", "음수대", "매점"],
    "문화시설": ["교양시설", "문화시설", "도서관", "야외무대", "전시관"],
    "생태시설": ["생태시설", "생태", "식물원", "동물원", "자연학습장"],
    "놀이시설": ["놀이시설", "놀이터", "놀이동산"],
    "체험시설": ["체험시설", "체험관", "관찰대"]
}

# 시설 종류가 언급되면 1, 아니면 0
for category, keywords in facility_keywords.items():
    cluster_df[category] = facility_text.apply(
        lambda x: int(any(keyword in x for keyword in keywords))
    )

# 군집 분석 변수
feature_columns = [
    "면적_로그",
    "운동시설",
    "조경시설",
    "휴양시설",
    "편익시설",
    "문화시설",
    "생태시설",
    "놀이시설",
    "체험시설"
]

X = cluster_df[feature_columns]

# 데이터 표준화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-means 모델
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

cluster_df["군집"] = kmeans.fit_predict(X_scaled)

# 분석 결과
st.write("분석에 사용된 공원 수:", len(cluster_df))

st.subheader("공원별 군집 결과")

st.dataframe(
    cluster_df[["공원명", "면적", "군집"]],
    use_container_width=True
)

# 군집별 공원 수
st.subheader("군집별 공원 수")

cluster_count = (
    cluster_df["군집"]
    .value_counts()
    .sort_index()
)

st.bar_chart(cluster_count)

st.write("군집별 정확한 공원 수")

st.dataframe(
    cluster_count.rename("공원 수").rename_axis("군집").reset_index(),
    use_container_width=True
)

# 군집별 평균 특성
st.subheader("군집별 평균 특성")

cluster_mean = (
    cluster_df
    .groupby("군집")[feature_columns]
    .mean()
    .round(2)
)

st.dataframe(
    cluster_mean,
    use_container_width=True
)

st.caption(
    "시설 특성은 주요시설 설명에 해당 키워드가 포함되어 있는지를 "
    "기준으로 0과 1로 변환했습니다. 실제 시설 개수는 아닙니다."
)
