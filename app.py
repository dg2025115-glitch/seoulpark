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
