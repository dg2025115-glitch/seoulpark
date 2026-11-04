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

st.write("**이상치 공원 면적 비교**")

if not outliers.empty:
    st.bar_chart(
        outliers.set_index("공원명")["면적_숫자"]
    )
else:
    st.success("이상치가 발견되지 않았습니다.")
