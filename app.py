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
# 12. K-means 군집 산점도
# ----------------------------------

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


# ----------------------------------
# 13. 군집별 대표 공원
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
# 14. 최종 결과
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
