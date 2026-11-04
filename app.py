
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ----------------------------------
# 1. 기본 설정 및 데이터 불러오기
# ----------------------------------

st.set_page_config(
    page_title="서울시 공원 데이터 분석",
    page_icon="🌳",
    layout="wide"
)

st.title("🌳 서울시 주요 공원 데이터 분석")
st.write(
    "서울시 공원의 면적과 주요 시설을 분석하여 "
    "공원 유형을 알아보는 프로젝트입니다."
)

@st.cache_data
def load_data():
    encodings = ["cp949", "utf-8-sig", "euc-kr", "utf-8"]

    for encoding in encodings:
        try:
            return pd.read_csv(
                "서울시 주요 공원현황.csv",
                encoding=encoding
            )
        except UnicodeDecodeError:
            continue

    raise ValueError("CSV 파일의 인코딩을 확인해 주세요.")

try:
    df = load_data()
    st.success("데이터 불러오기 성공!")
except Exception as e:
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()


# ----------------------------------
# 2. 데이터 개요
# ----------------------------------

st.header("1. 데이터 개요")

col1, col2 = st.columns(2)

with col1:
    st.metric("전체 행 개수", f"{df.shape[0]:,}개")

with col2:
    st.metric("전체 열 개수", f"{df.shape[1]:,}개")

st.subheader("데이터 미리보기")
st.dataframe(df, use_container_width=True)

# 빈 문자열도 결측치로 계산
missing_count = df.apply(
    lambda col: (
        col.isna()
        | col.astype(str).str.strip().isin(["", "nan", "None"])
    ).sum()
)

overview = pd.DataFrame({
    "자료형": df.dtypes.astype(str),
    "빈 값 개수": missing_count,
    "빈 값 비율(%)": (
        missing_count / len(df) * 100
    ).round(2)
})

st.subheader("열별 자료형 및 결측치 현황")
st.dataframe(overview, use_container_width=True)


# ----------------------------------
# 3. 결측치 분석
# ----------------------------------

st.header("2. 결측치 분석")

missing_df = overview[
    overview["빈 값 개수"] > 0
].sort_values(
    by="빈 값 개수",
    ascending=False
)

if missing_df.empty:
    st.success("결측치가 없습니다.")
else:
    st.write("결측치가 있는 항목")
    st.dataframe(missing_df, use_container_width=True)

    st.subheader("항목별 결측치 개수")

    missing_chart = missing_df[
        ["빈 값 개수"]
    ].reset_index(names="항목")

    st.bar_chart(
        missing_chart,
        x="항목",
        y="빈 값 개수",
        x_label="데이터 항목",
        y_label="결측치 개수 (개)"
    )

    most_missing = missing_df["빈 값 개수"].idxmax()
    most_missing_count = missing_df.loc[
        most_missing, "빈 값 개수"
    ]

    st.write(
        f"**발견한 점:** {most_missing} 항목에 "
        "가장 많은 빈 값이 나타났다."
    )
    st.write(
        f"**근거 수치:** {most_missing} 항목의 빈 값은 "
        f"{most_missing_count}개이다."
    )

st.info(
    "결측치 처리 방법: 면적이 없거나 단위를 확인할 수 없는 공원은 "
    "군집 분석에서 제외하고, 주요 시설이 비어 있는 경우에는 "
    "시설 키워드가 없는 것으로 처리했습니다."
)


# ----------------------------------
# 4. 면적 숫자 변환 및 이상치 분석
# ----------------------------------

st.header("3. 이상치 분석")

area_number = pd.to_numeric(
    df["면적"]
    .astype(str)
    .str.extract(r"([\d,]+(?:\.\d+)?)")[0]
    .str.replace(",", "", regex=False),
    errors="coerce"
)

valid_area = area_number.dropna()
valid_area = valid_area[valid_area > 0]

if len(valid_area) >= 4:

    Q1 = valid_area.quantile(0.25)
    Q3 = valid_area.quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outlier_mask = (
        area_number.notna()
        & ((area_number < lower) | (area_number > upper))
    )

    outliers = df.loc[outlier_mask].copy()
    outliers["면적_숫자"] = area_number.loc[outlier_mask]

    st.write(f"제1사분위수(Q1): {Q1:,.2f}")
    st.write(f"제3사분위수(Q3): {Q3:,.2f}")
    st.write(f"IQR: {IQR:,.2f}")
    st.write(f"이상치 판단 하한: {lower:,.2f}")
    st.write(f"이상치 판단 상한: {upper:,.2f}")

    st.metric("이상치 탐지 개수", f"{len(outliers)}개")

    if not outliers.empty:
        st.subheader("이상치로 탐지된 공원")

        st.dataframe(
            outliers[["공원명", "면적", "지역", "면적_숫자"]],
            use_container_width=True
        )

        outlier_chart = outliers[
            ["공원명", "면적_숫자"]
        ].copy()

        outlier_chart["로그 면적"] = np.log10(
            outlier_chart["면적_숫자"].clip(lower=1)
        )

        st.subheader("이상치 공원 면적 비교")

        st.bar_chart(
            outlier_chart,
            x="공원명",
            y="로그 면적",
            x_label="공원명",
            y_label="로그 변환 면적"
        )

        largest_outlier = outliers.loc[
            outliers["면적_숫자"].idxmax()
        ]

        st.write(
            "**발견한 점:** IQR 기준을 벗어나는 면적을 가진 "
            "공원이 탐지되었다."
        )
        st.write(
            f"**근거 수치:** 탐지된 공원은 {len(outliers)}개이며, "
            f"가장 큰 값은 {largest_outlier['공원명']}의 "
            f"{largest_outlier['면적_숫자']:,.1f}㎡이다."
        )
    else:
        st.success("IQR 기준으로 탐지된 이상치가 없습니다.")

    st.info(
        "이상치는 IQR 방식으로 탐지했습니다. "
        "탐지된 모든 값을 자동 삭제하지 않고 원본 값과 단위를 확인하도록 했습니다."
    )

else:
    st.warning("이상치 분석에 사용할 유효한 면적 데이터가 부족합니다.")


# ----------------------------------
# 5. 지역별 공원 분포
# ----------------------------------

st.header("4. 지역별 공원 분포")

region_data = (
    df["지역"]
    .dropna()
    .astype(str)
    .str.strip()
)

region_data = region_data[
    (region_data != "")
    & (region_data != "nan")
]

region_count = region_data.value_counts()

region_table = region_count.rename(
    "공원 수"
).rename_axis(
    "지역"
).reset_index()

st.subheader("지역별 공원 수")

st.bar_chart(
    region_table,
    x="지역",
    y="공원 수",
    x_label="지역",
    y_label="공원 수 (개)"
)

st.dataframe(region_table, use_container_width=True)

if not region_table.empty:
    top_region = region_table.iloc[0]

    st.write(
        f"**발견한 점:** {top_region['지역']} 지역에 "
        "가장 많은 공원이 분포한다."
    )
    st.write(
        f"**근거 수치:** {top_region['지역']} 지역의 공원은 "
        f"{top_region['공원 수']}개이다."
    )


# ----------------------------------
# 6. 면적 TOP 10
# ----------------------------------

st.header("5. 면적이 넓은 공원 TOP 10")

top10_data = df[["공원명", "면적"]].copy()

top10_data["면적_숫자"] = area_number

top10_data = top10_data.dropna(
    subset=["면적_숫자"]
)

top10_data = top10_data[
    top10_data["면적_숫자"] > 0
]

top10_data = top10_data.sort_values(
    by="면적_숫자",
    ascending=False
).head(10)

st.dataframe(
    top10_data[["공원명", "면적"]],
    use_container_width=True
)

if not top10_data.empty:

    top10_chart = top10_data.copy()

    top10_chart["로그 면적"] = np.log10(
        top10_chart["면적_숫자"]
    )

    st.bar_chart(
        top10_chart,
        x="공원명",
        y="로그 면적",
        x_label="공원명",
        y_label="로그 변환 면적"
    )

    largest_park = top10_data.iloc[0]

    st.write(
        f"**발견한 점:** {largest_park['공원명']}의 면적이 "
        "상위 10개 공원 중 가장 넓다."
    )
    st.write(
        f"**근거 수치:** {largest_park['면적_숫자']:,.1f}㎡"
    )

st.caption(
    "TOP 10 그래프는 큰 면적값의 차이를 보기 쉽게 로그 변환했습니다. "
    "실제 면적은 위 표에서 확인할 수 있습니다."
)


# ----------------------------------
# 7. K-means 군집 분석을 위한 전처리
# ----------------------------------

st.header("6. K-means 공원 유형 분석")

cluster_df = df[
    ["공원명", "면적", "주요시설"]
].copy()

before_count = len(cluster_df)

# 면적 숫자 추출
area_text = cluster_df["면적"].astype(str)

cluster_df["면적_숫자"] = pd.to_numeric(
    area_text.str.extract(
        r"([\d,]+(?:\.\d+)?)"
    )[0].str.replace(",", "", regex=False),
    errors="coerce"
)

# 면적 단위 확인
valid_unit = area_text.str.contains(
    r"㎡|m²|m2|제곱미터",
    regex=True,
    case=False,
    na=False
)

# 제외 이유 기록
cluster_df["제외 사유"] = ""

cluster_df.loc[
    cluster_df["면적_숫자"].isna(),
    "제외 사유"
] = "면적값 없음 또는 숫자 변환 불가"

cluster_df.loc[
    cluster_df["제외 사유"].eq("")
    & ~valid_unit,
    "제외 사유"
] = "면적 단위 확인 불가"

cluster_df.loc[
    cluster_df["제외 사유"].eq("")
    & (cluster_df["면적_숫자"] <= 0),
    "제외 사유"
] = "면적값이 0 이하"

cluster_df.loc[
    cluster_df["제외 사유"].eq("")
    & (cluster_df["공원명"] == "북한산국립공원"),
    "제외 사유"
] = "면적값 검증 필요"

excluded_df = cluster_df[
    cluster_df["제외 사유"] != ""
].copy()

cluster_df = cluster_df[
    cluster_df["제외 사유"] == ""
].copy()

after_count = len(cluster_df)

st.subheader("데이터 전처리 결과")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("처리 전 공원 수", f"{before_count}개")

with col2:
    st.metric("처리 후 공원 수", f"{after_count}개")

with col3:
    st.metric(
        "제외된 공원 수",
        f"{before_count - after_count}개"
    )

st.write("**전처리 방법**")
st.write(
    "① 면적값이 없는 데이터 제외\n\n"
    "② 면적 단위가 ㎡인지 확인할 수 없는 데이터 제외\n\n"
    "③ 면적이 0 이하인 데이터 제외\n\n"
    "④ 면적값 검증이 필요한 북한산국립공원 제외"
)

st.subheader("분석에서 제외된 공원")

if excluded_df.empty:
    st.success("제외된 공원이 없습니다.")
else:
    st.dataframe(
        excluded_df[["공원명", "면적", "제외 사유"]],
        use_container_width=True
    )

st.info(
    "IQR 이상치 탐지와 군집 분석의 데이터 제외는 별개입니다. "
    "IQR에서 이상치로 탐지되었다는 이유만으로 자동 제외하지 않았습니다."
)


# ----------------------------------
# 8. 시설 키워드 추출 및 K-means
# ----------------------------------

if len(cluster_df) < 3:
    st.error("군집 분석을 진행하기에 데이터가 부족합니다.")
    st.stop()

# 면적 로그 변환
cluster_df["면적_로그"] = np.log10(
    cluster_df["면적_숫자"]
)

# 시설 텍스트
facility_text = (
    cluster_df["주요시설"]
    .fillna("")
    .astype(str)
)

facility_keywords = {
    "운동시설": [
        "운동시설", "테니스장", "농구장",
        "축구장", "운동기구"
    ],
    "조경시설": [
        "조경시설", "연못", "분수",
        "정원", "파고라"
    ],
    "휴양시설": [
        "휴양시설", "산책로", "쉼터",
        "정자", "벤치"
    ],
    "편익시설": [
        "편익시설", "화장실", "주차장",
        "음수대", "매점"
    ],
    "문화시설": [
        "교양시설", "문화시설", "도서관",
        "야외무대", "전시관"
    ],
    "생태시설": [
        "생태시설", "생태", "식물원",
        "동물원", "자연학습장"
    ],
    "놀이시설": [
        "놀이시설", "놀이터", "놀이동산"
    ],
    "체험시설": [
        "체험시설", "체험관", "관찰대"
    ]
}

for category, keywords in facility_keywords.items():
    cluster_df[category] = facility_text.apply(
        lambda x: int(any(keyword in x for keyword in keywords))
    )

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

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

cluster_df["군집"] = kmeans.fit_predict(X_scaled)


# ----------------------------------
# 9. 군집별 공원 수 그래프
# ----------------------------------

st.subheader("그래프 1. 군집별 공원 수 비교")

cluster_count = (
    cluster_df["군집"]
    .value_counts()
    .sort_index()
)

cluster_count_df = cluster_count.rename(
    "공원 수"
).rename_axis(
    "군집"
).reset_index()

st.bar_chart(
    cluster_count_df,
    x="군집",
    y="공원 수",
    x_label="공원 군집",
    y_label="공원 수 (개)"
)

st.dataframe(
    cluster_count_df,
    use_container_width=True
)

largest_cluster = cluster_count.idxmax()
largest_count = cluster_count.max()
largest_ratio = largest_count / len(cluster_df) * 100

st.write(
    f"**발견한 점:** {largest_cluster}번 군집에 "
    "가장 많은 공원이 포함되어 있다."
)

st.write(
    f"**근거 수치:** {largest_cluster}번 군집은 "
    f"{largest_count}개로 전체 공원의 "
    f"{largest_ratio:.1f}%를 차지한다."
)


# ----------------------------------
# 10. 군집별 시설 특성 그래프
# ----------------------------------

cluster_mean = (
    cluster_df
    .groupby("군집")[feature_columns]
    .mean()
    .round(2)
)

st.subheader("그래프 2. 군집별 주요 시설 특성 비교")

facility_columns = list(facility_keywords.keys())

facility_mean = cluster_mean[facility_columns]

facility_chart = facility_mean.reset_index()

st.bar_chart(
    facility_chart,
    x="군집",
    y=facility_columns,
    x_label="공원 군집",
    y_label="시설 키워드 포함 비율",
    stack=False
)

st.dataframe(
    facility_mean,
    use_container_width=True
)

st.write(
    "**발견한 점:** 군집마다 주요 시설의 특징이 다르게 나타난다."
)

# 각 군집에서 평균이 가장 높은 시설 찾기
for cluster_num in facility_mean.index:
    highest_facility = facility_mean.loc[
        cluster_num
    ].idxmax()

    highest_value = facility_mean.loc[
        cluster_num
    ].max()

    st.write(
        f"- {cluster_num}번 군집: "
        f"{highest_facility} 평균 {highest_value:.2f}"
    )

st.caption(
    "시설 평균은 실제 시설 개수가 아니라 해당 시설 키워드가 "
    "포함된 공원의 비율입니다. 예를 들어 0.75는 약 75%의 "
    "공원에서 해당 키워드가 발견되었다는 뜻입니다."
)


# ----------------------------------
# 11. 군집별 평균 면적 그래프
# ----------------------------------

st.subheader("그래프 3. 군집별 평균 공원 면적 비교")

area_mean = (
    cluster_df
    .groupby("군집")["면적_숫자"]
    .mean()
    .round(1)
)

area_mean_df = area_mean.rename(
    "평균 면적"
).rename_axis(
    "군집"
).reset_index()

st.bar_chart(
    area_mean_df,
    x="군집",
    y="평균 면적",
    x_label="공원 군집",
    y_label="평균 면적 (㎡)"
)

st.dataframe(
    area_mean_df,
    use_container_width=True
)

largest_area_cluster = area_mean.idxmax()
largest_area = area_mean.max()

st.write(
    f"**발견한 점:** {largest_area_cluster}번 군집의 "
    "평균 공원 면적이 가장 넓다."
)

st.write(
    f"**근거 수치:** {largest_area_cluster}번 군집의 "
    f"평균 면적은 {largest_area:,.1f}㎡이다."
)


# ----------------------------------
# 12. 군집별 대표 공원
# ----------------------------------

st.header("7. 군집별 대표 공원")

distances = kmeans.transform(X_scaled)

cluster_df["중심과 거리"] = [
    distances[i, cluster]
    for i, cluster in enumerate(cluster_df["군집"])
]

for cluster_num in sorted(cluster_df["군집"].unique()):

    st.subheader(f"{cluster_num}번 군집")

    group = cluster_df[
        cluster_df["군집"] == cluster_num
    ].sort_values("중심과 거리")

    st.write(f"공원 수: {len(group)}개")

    st.write("대표 공원 TOP 3")

    st.dataframe(
        group[["공원명", "면적"]].head(3),
        use_container_width=True
    )

    with st.expander("군집에 포함된 전체 공원 보기"):
        st.write(
            ", ".join(group["공원명"].astype(str))
        )


# ----------------------------------
# 13. 최종 결과
# ----------------------------------

st.header("8. 최종 분석 결과")

st.write(
    f"최종 군집 분석에 사용된 공원은 총 {len(cluster_df)}개이며, "
    "면적과 주요 시설의 특징을 바탕으로 3개 유형으로 분류했습니다."
)

st.subheader("공원별 군집 결과")

st.dataframe(
    cluster_df[["공원명", "면적", "군집"]],
    use_container_width=True
)

st.subheader("군집별 평균 특성")

st.dataframe(
    cluster_mean,
    use_container_width=True
)

st.caption(
    "본 분석의 군집은 K-means 알고리즘이 면적과 시설 키워드의 "
    "유사성을 바탕으로 분류한 결과이며, 공원의 공식적인 행정 분류는 아닙니다."
)
