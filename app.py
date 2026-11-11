import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
# 1. 기본 설정
st.set_page_config(
    page_title="서울시 공원 데이터 분석",
    page_icon="🌳",
    layout="wide"
)

st.title("🌳 서울시 주요 공원 데이터 분석")
st.write("서울시 공원의 면적과 주요 시설을 분석하여 공원 유형을 알아보는 프로젝트입니다.")

# 2. 데이터 불러오기
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

# 3. 데이터 개요
st.header("1. 데이터 개요")

col1, col2 = st.columns(2)

with col1:
    st.metric("전체 행 개수", f"{df.shape[0]:,}개")

with col2:
    st.metric("전체 열 개수", f"{df.shape[1]:,}개")

st.subheader("데이터 미리보기")
st.dataframe(df, use_container_width=True)

missing_count = df.apply(
    lambda col: (
        col.isna()
        | col.astype(str).str.strip().isin(["", "nan", "None"])
    ).sum()
)

overview = pd.DataFrame({
    "자료형": df.dtypes.astype(str),
    "빈 값 개수": missing_count,
    "빈 값 비율(%)": (missing_count / len(df) * 100).round(2)
})

st.subheader("열별 자료형 및 결측치 현황")
st.dataframe(overview, use_container_width=True)

# 4. 결측치 분석
st.header("2. 결측치 분석")

missing_df = overview[
    overview["빈 값 개수"] > 0
].sort_values("빈 값 개수", ascending=False)

if missing_df.empty:
    st.success("결측치가 없습니다.")
else:
    st.dataframe(missing_df, use_container_width=True)

    total_cells = df.shape[0] * df.shape[1]
    missing_cells = int(missing_count.sum())
    present_cells = total_cells - missing_cells

    missing_pie = pd.DataFrame({
        "상태": ["정상 데이터", "결측 데이터"],
        "개수": [present_cells, missing_cells]
    })

    missing_chart = alt.Chart(missing_pie).mark_arc(
        innerRadius=65
    ).encode(
        theta=alt.Theta("개수:Q"),
        color=alt.Color("상태:N", title="데이터 상태"),
        tooltip=["상태:N", "개수:Q"]
    ).properties(
        title="전체 데이터의 결측치 비율",
        height=300
    )

    st.altair_chart(missing_chart, use_container_width=True)

    most_missing = missing_df["빈 값 개수"].idxmax()
    most_missing_count = missing_df.loc[most_missing, "빈 값 개수"]

    st.write(f"**발견한 점:** {most_missing} 항목에 가장 많은 빈 값이 나타났다.")
    st.write(f"**근거 수치:** {most_missing} 항목의 빈 값은 {most_missing_count}개이다.")

st.info(
    "면적이 없거나 단위를 확인할 수 없는 공원은 군집 분석에서 제외하고, "
    "주요 시설이 비어 있는 경우에는 시설 키워드가 없는 것으로 처리했습니다."
)

# 5. 면적 변환 및 이상치 분석
st.header("3. 이상치 분석")

area_number = pd.to_numeric(
    df["면적"].astype(str)
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

    box_data = pd.DataFrame({"면적(㎡)": valid_area.values})

    box_chart = alt.Chart(box_data).mark_boxplot(
        extent="min-max"
    ).encode(
        y=alt.Y(
            "면적(㎡):Q",
            scale=alt.Scale(type="log"),
            title="면적 (㎡, 로그 눈금)"
        ),
        tooltip=[alt.Tooltip("면적(㎡):Q", format=",.0f")]
    ).properties(
        title="공원 면적 분포와 이상치",
        height=300
    )

    st.altair_chart(box_chart, use_container_width=True)

    st.caption(
        "상자수염 그래프는 중앙값과 사분위 범위, 전체 분포를 보여줍니다. "
        "면적 차이가 커서 세로축은 로그 눈금으로 표시했습니다."
    )

    if not outliers.empty:
        largest_outlier = outliers.loc[
            outliers["면적_숫자"].idxmax()
        ]

        st.write(
            f"**발견한 점:** IQR 기준을 벗어나는 공원이 "
            f"{len(outliers)}개 탐지되었다."
        )
        st.write(
            f"**근거 수치:** 가장 큰 값은 {largest_outlier['공원명']}의 "
            f"{largest_outlier['면적_숫자']:,.1f}㎡이다."
        )

    st.info(
        "IQR 방식으로 이상치를 탐지했습니다. "
        "탐지된 값을 자동 삭제하지 않고 원본 값과 단위를 확인하도록 했습니다."
    )
else:
    st.warning("이상치 분석에 사용할 유효한 면적 데이터가 부족합니다.")

# 6. 지역별 공원 분포
st.header("4. 지역별 공원 분포")

region_data = (
    df["지역"].dropna().astype(str).str.strip()
)

region_data = region_data[
    (region_data != "") & (region_data != "nan")
]

region_count = region_data.value_counts()

region_table = region_count.rename(
    "공원 수"
).rename_axis("지역").reset_index()

region_chart = alt.Chart(region_table).mark_circle(
    size=110
).encode(
    x=alt.X("공원 수:Q", title="공원 수 (개)"),
    y=alt.Y("지역:N", sort="-x", title="지역"),
    tooltip=["지역:N", "공원 수:Q"]
).properties(
    title="지역별 공원 분포",
    height=600
)

st.altair_chart(region_chart, use_container_width=True)
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

# 7. 면적 TOP 10
st.header("5. 면적이 넓은 공원 TOP 10")

top10_data = df[["공원명", "면적"]].copy()
top10_data["면적_숫자"] = area_number

top10_data = top10_data.dropna(subset=["면적_숫자"])
top10_data = top10_data[top10_data["면적_숫자"] > 0]

top10_data = top10_data.sort_values(
    by="면적_숫자",
    ascending=False
).head(10)

st.dataframe(
    top10_data[["공원명", "면적"]],
    use_container_width=True
)

if not top10_data.empty:
    top10_data["로그 면적"] = np.log10(top10_data["면적_숫자"])

    top10_chart = alt.Chart(top10_data).mark_line(
        point=alt.OverlayMarkDef(size=90)
    ).encode(
        x=alt.X("공원명:N", sort="-y", title="공원명"),
        y=alt.Y("로그 면적:Q", title="로그 변환 면적"),
        tooltip=[
            "공원명:N",
            alt.Tooltip("면적_숫자:Q", title="실제 면적(㎡)", format=",.1f")
        ]
    ).properties(
        title="면적 상위 10개 공원 비교",
        height=350
    )

    st.altair_chart(top10_chart, use_container_width=True)

    largest_park = top10_data.iloc[0]

    st.write(
        f"**발견한 점:** {largest_park['공원명']}의 면적이 "
        "상위 10개 공원 중 가장 넓다."
    )
    st.write(f"**근거 수치:** {largest_park['면적_숫자']:,.1f}㎡")

st.caption(
    "TOP 10 그래프는 큰 면적값의 차이를 보기 쉽게 로그 변환했습니다. "
    "실제 면적은 위 표에서 확인할 수 있습니다."
)

# 8. K-means 군집 분석 전처리
st.header("6. K-means 공원 유형 분석")

cluster_df = df[["공원명", "면적", "주요시설"]].copy()
before_count = len(cluster_df)

area_text = cluster_df["면적"].astype(str)

cluster_df["면적_숫자"] = pd.to_numeric(
    area_text.str.extract(r"([\d,]+(?:\.\d+)?)")[0]
    .str.replace(",", "", regex=False),
    errors="coerce"
)

valid_unit = area_text.str.contains(
    r"㎡|m²|m2|제곱미터",
    regex=True,
    case=False,
    na=False
)

cluster_df["제외 사유"] = ""

cluster_df.loc[
    cluster_df["면적_숫자"].isna(),
    "제외 사유"
] = "면적값 없음 또는 숫자 변환 불가"

cluster_df.loc[
    cluster_df["제외 사유"].eq("") & ~valid_unit,
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

excluded_df = cluster_df[cluster_df["제외 사유"] != ""].copy()
cluster_df = cluster_df[cluster_df["제외 사유"] == ""].copy()

after_count = len(cluster_df)

st.subheader("데이터 전처리 결과")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("처리 전 공원 수", f"{before_count}개")

with col2:
    st.metric("처리 후 공원 수", f"{after_count}개")

with col3:
    st.metric("제외된 공원 수", f"{before_count - after_count}개")

st.write(
    "① 면적값이 없는 데이터 제외\n\n"
    "② 면적 단위를 확인할 수 없는 데이터 제외\n\n"
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

# 9. 시설 키워드 추출 및 K-means
if len(cluster_df) < 3:
    st.error("군집 분석을 진행하기에 데이터가 부족합니다.")
    st.stop()

cluster_df["면적_로그"] = np.log10(cluster_df["면적_숫자"])

facility_text = (
    cluster_df["주요시설"].fillna("").astype(str)
)

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

facility_columns = list(facility_keywords.keys())

# 머신러닝 학습 데이터 준비
X = cluster_df[feature_columns]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 군집 개수별 모델 평가
evaluation_rows = []

max_k = min(6, len(cluster_df) - 1)

for k in range(2, max_k + 1):
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X_scaled)

    score = silhouette_score(X_scaled, labels)

    evaluation_rows.append({
        "군집 수": k,
        "Inertia": model.inertia_,
        "실루엣 계수": score
    })

evaluation_df = pd.DataFrame(evaluation_rows)

# 모델 평가 결과 표시
st.subheader("🤖 머신러닝 모델 평가")

st.write("군집 개수에 따라 모델의 성능이 어떻게 달라지는지 비교합니다.")

st.dataframe(
    evaluation_df.round(3),
    use_container_width=True
)

# 엘보우 방법 그래프
st.subheader("1. 엘보우 방법")

inertia_chart = alt.Chart(evaluation_df).mark_line(
    point=True
).encode(
    x=alt.X("군집 수:Q", axis=alt.Axis(tickMinStep=1)),
    y=alt.Y("Inertia:Q", title="Inertia"),
    tooltip=["군집 수", "Inertia"]
).properties(
    title="군집 수에 따른 Inertia 변화",
    height=300
)

st.altair_chart(inertia_chart, use_container_width=True)

st.caption(
    "Inertia는 군집 내부 데이터가 얼마나 모여 있는지를 나타냅니다. "
    "값이 감소하다가 감소 폭이 작아지는 지점을 살펴봅니다."
)

# 실루엣 계수 그래프
st.subheader("2. 실루엣 계수")

silhouette_chart = alt.Chart(evaluation_df).mark_line(
    point=True
).encode(
    x=alt.X("군집 수:Q", axis=alt.Axis(tickMinStep=1)),
    y=alt.Y("실루엣 계수:Q", scale=alt.Scale(domain=[-1, 1])),
    tooltip=[
        "군집 수",
        alt.Tooltip("실루엣 계수:Q", format=".3f")
    ]
).properties(
    title="군집 수에 따른 실루엣 계수 변화",
    height=300
)

st.altair_chart(silhouette_chart, use_container_width=True)

st.caption(
    "실루엣 계수는 군집이 얼마나 잘 구분되는지 나타냅니다. "
    "1에 가까울수록 군집이 잘 분리된 것입니다."
)

# 최종 모델 학습
final_k = 4

kmeans = KMeans(
    n_clusters=final_k,
    random_state=42,
    n_init=10
)

cluster_df["군집"] = kmeans.fit_predict(X_scaled)

# 군집별 평균 면적을 기준으로 번호 정렬
cluster_order = (
    cluster_df.groupby("군집")["면적_숫자"]
    .mean()
    .sort_values()
    .index
    .tolist()
)

cluster_mapping = {
    old: new for new, old in enumerate(cluster_order)
}

cluster_df["군집"] = cluster_df["군집"].map(cluster_mapping)

# 군집별 분석 결과
st.subheader("🌳 군집별 공원 유형 분석")

facility_columns = [
    "운동시설", "조경시설", "휴양시설", "편익시설",
    "문화시설", "생태시설", "놀이시설", "체험시설"
]

cluster_summary = cluster_df.groupby("군집").agg(
    공원수=("공원명", "count"),
    평균면적=("면적_숫자", "mean")
).reset_index()

facility_summary = cluster_df.groupby("군집")[facility_columns].mean() * 100

# 군집별 유형 이름과 특징 만들기
for _, row in cluster_summary.iterrows():
    cluster_num = int(row["군집"])
    facilities = facility_summary.loc[cluster_num]

    top_facilities = facilities.sort_values(ascending=False).head(3)

    facility_names = [
        name for name, value in top_facilities.items()
        if value > 0
    ]

    if cluster_num == 0:
        size_name = "소규모"
    elif cluster_num == 1:
        size_name = "중규모"
    else:
        size_name = "대규모"

    st.markdown(f"### 군집 {cluster_num} - {size_name} 공원 유형")

    col1, col2 = st.columns(2)
    col1.metric("공원 수", f"{int(row['공원수'])}개")
    col2.metric("평균 면적", f"{row['평균면적']:,.0f}㎡")

    st.write(
        "시설 특징: " +
        (", ".join(facility_names) if facility_names else "주요 시설 특징 없음")
    )

    st.write("시설별 포함 비율(%)")

    facility_chart_data = (
        facility_summary.loc[[cluster_num]]
        .T.reset_index()
    )
    facility_chart_data.columns = ["시설 종류", "포함 비율"]

    facility_chart = alt.Chart(facility_chart_data).mark_bar().encode(
        x=alt.X("포함 비율:Q", title="공원 중 해당 시설이 포함된 비율(%)"),
        y=alt.Y("시설 종류:N", sort="-x", title="시설 종류"),
        tooltip=["시설 종류", alt.Tooltip("포함 비율:Q", format=".1f")]
    ).properties(height=250)

    st.altair_chart(facility_chart, use_container_width=True)

    st.divider()

# 최종 모델 결과
final_score = evaluation_df.loc[
    evaluation_df["군집 수"] == final_k,
    "실루엣 계수"
].iloc[0]

st.subheader("3. 최종 모델")

col1, col2 = st.columns(2)

col1.metric("최종 군집 개수", f"{final_k}개")
col2.metric("실루엣 계수", f"{final_score:.3f}")

st.success("K-means 군집화 모델 학습이 완료되었습니다.")
# 10. 군집별 공원 수 - 도넛 그래프
st.subheader("그래프 1. 군집별 공원 수 비교")

cluster_count = cluster_df["군집"].value_counts().sort_index()

cluster_count_df = cluster_count.rename(
    "공원 수"
).rename_axis("군집").reset_index()

cluster_pie = alt.Chart(cluster_count_df).mark_arc(
    innerRadius=55
).encode(
    theta=alt.Theta("공원 수:Q"),
    color=alt.Color("군집:N", title="군집"),
    tooltip=["군집:N", "공원 수:Q"]
).properties(
    title="군집별 공원 구성 비율",
    height=330
)

st.altair_chart(cluster_pie, use_container_width=True)
st.dataframe(cluster_count_df, use_container_width=True)

largest_cluster = cluster_count.idxmax()
largest_count = cluster_count.max()
largest_ratio = largest_count / len(cluster_df) * 100

st.write(
    f"**발견한 점:** {largest_cluster}번 군집에 "
    "가장 많은 공원이 포함되어 있다."
)
st.write(
    f"**근거 수치:** {largest_cluster}번 군집은 "
    f"{largest_count}개로 전체 공원의 {largest_ratio:.1f}%를 차지한다."
)

# 11. 군집별 시설 특성 - 히트맵
cluster_mean = (
    cluster_df.groupby("군집")[feature_columns].mean().round(2)
)

st.subheader("그래프 2. 군집별 주요 시설 특성 비교")

facility_mean = cluster_mean[facility_columns]

facility_long = facility_mean.reset_index().melt(
    id_vars="군집",
    var_name="시설 종류",
    value_name="포함 비율"
)

facility_heatmap = alt.Chart(facility_long).mark_rect().encode(
    x=alt.X(
        "시설 종류:N",
        title="주요 시설",
        axis=alt.Axis(labelAngle=-35)
    ),
    y=alt.Y("군집:N", title="공원 군집"),
    color=alt.Color(
        "포함 비율:Q",
        scale=alt.Scale(scheme="blues", domain=[0, 1]),
        title="포함 비율"
    ),
    tooltip=[
        "군집:N",
        "시설 종류:N",
        alt.Tooltip("포함 비율:Q", format=".0%")
    ]
).properties(
    title="군집별 시설 키워드 포함 비율 히트맵",
    height=260
)

st.altair_chart(facility_heatmap, use_container_width=True)
st.dataframe(facility_mean, use_container_width=True)

st.write("**발견한 점:** 군집마다 주요 시설의 특징이 다르게 나타난다.")

for cluster_num in facility_mean.index:
    highest_facility = facility_mean.loc[cluster_num].idxmax()
    highest_value = facility_mean.loc[cluster_num].max()

    st.write(
        f"- {cluster_num}번 군집: "
        f"{highest_facility} 평균 {highest_value:.2f}"
    )

st.caption(
    "시설 평균은 실제 시설 개수가 아니라 해당 시설 키워드가 "
    "포함된 공원의 비율입니다. 예를 들어 0.75는 약 75%의 "
    "공원에서 해당 키워드가 발견되었다는 뜻입니다."
)

# 12. 군집별 면적 분포 - 상자수염 그래프
st.subheader("그래프 3. 군집별 평균 공원 면적 비교")

area_mean = (
    cluster_df.groupby("군집")["면적_숫자"].mean().round(1)
)

area_mean_df = area_mean.rename(
    "평균 면적"
).rename_axis("군집").reset_index()

area_box_data = cluster_df[["군집", "면적_숫자"]].rename(
    columns={"면적_숫자": "면적(㎡)"}
)

area_box_chart = alt.Chart(area_box_data).mark_boxplot(
    extent="min-max"
).encode(
    x=alt.X("군집:N", title="공원 군집"),
    y=alt.Y(
        "면적(㎡):Q",
        scale=alt.Scale(type="log"),
        title="공원 면적 (㎡, 로그 눈금)"
    ),
    color=alt.Color("군집:N", legend=None),
    tooltip=[
        "군집:N",
        alt.Tooltip("면적(㎡):Q", format=",.0f")
    ]
).properties(
    title="군집별 공원 면적 분포",
    height=350
)

st.altair_chart(area_box_chart, use_container_width=True)
st.caption(
    "평균값뿐 아니라 각 군집에 속한 공원들의 면적 분포와 중앙값을 함께 비교합니다."
)

st.dataframe(area_mean_df, use_container_width=True)

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

# 13. 면적과 시설 종류 수 - 산점도
st.subheader("그래프 4. 면적과 시설 수에 따른 공원 군집 산점도")

cluster_df["시설 종류 수"] = cluster_df[facility_columns].sum(axis=1)

scatter_chart = alt.Chart(cluster_df).mark_circle(
    size=90,
    opacity=0.75
).encode(
    x=alt.X("면적_로그:Q", title="공원 면적 (로그 변환)"),
    y=alt.Y(
        "시설 종류 수:Q",
        title="포함된 시설 종류 수",
        scale=alt.Scale(zero=True)
    ),
    color=alt.Color("군집:N", title="K-means 군집"),
    tooltip=[
        "공원명:N",
        "군집:N",
        alt.Tooltip("면적_숫자:Q", title="실제 면적(㎡)", format=",.1f"),
        "시설 종류 수:Q"
    ]
).properties(
    title="공원별 면적과 시설 종류 수의 관계",
    height=450
).interactive()

st.altair_chart(scatter_chart, use_container_width=True)

st.write(
    "**그래프 읽는 법:** 점 하나가 공원 하나를 의미합니다. "
    "색은 K-means 군집을 나타내며, 오른쪽일수록 면적이 넓고 "
    "위쪽일수록 포함된 시설 종류가 많습니다."
)

# 14. 군집별 대표 공원
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
        st.write(", ".join(group["공원명"].astype(str)))

# 15. 최종 결과
st.header("8. 최종 분석 결과")

st.write(
    f"최종 군집 분석에 사용된 공원은 총 {len(cluster_df)}개이며, "
    "면적과 주요 시설의 특징을 바탕으로 3개 군집으로 나누었습니다."
)

st.subheader("공원별 군집 결과")

st.dataframe(
    cluster_df[["공원명", "면적", "군집", "시설 종류 수"]],
    use_container_width=True
)

st.subheader("군집별 평균 특성")
st.dataframe(cluster_mean, use_container_width=True)

st.caption(
    "본 분석의 군집은 K-means 알고리즘이 면적과 시설 키워드의 "
    "유사성을 바탕으로 나눈 결과이며, 공원의 공식적인 행정 분류는 아닙니다."
)
