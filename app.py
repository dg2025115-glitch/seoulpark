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
    encoding="utf-8-sig"
)

st.success("데이터 불러오기 성공!")

st.write("데이터 크기:", df.shape)

st.subheader("데이터 미리보기")
st.dataframe(df)
