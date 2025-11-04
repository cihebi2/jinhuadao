import streamlit as st
import pandas as pd
from datetime import datetime
from app.logic import normalize_and_filter, compute_cohort_monthly, compute_bca_monthly

st.set_page_config(page_title="续费率仪表盘", layout="wide")

st.title("续费率仪表盘（上传 CSV/XLSX）")

uploaded = st.file_uploader("上传文件（CSV/XLSX）", type=["csv","xlsx"])

# 默认统计截止到上一个完整自然月
now = pd.Timestamp.today()
default_end = pd.Timestamp(now.year, now.month, 1) - pd.offsets.MonthBegin(1)
end_month = st.text_input("统计截至月份（YYYY-MM）", default_end.strftime("%Y-%m"))

if uploaded:
    try:
        if uploaded.name.lower().endswith(".xlsx"):
            df_raw = pd.read_excel(uploaded)
        else:
            # 尝试多编码读取
            last_err = None
            for enc in ("utf-8-sig","utf-8","gb18030"):
                try:
                    df_raw = pd.read_csv(uploaded, encoding=enc)
                    break
                except Exception as e:
                    last_err = e
            else:
                raise last_err
    except Exception as e:
        st.error(f"读取失败: {e}")
        st.stop()

    st.subheader("字段映射")
    cols = list(df_raw.columns)
    if len(cols) < 9:
        st.warning("列数较少，可能需要手动选择字段。")
    col_perm = st.selectbox("权限状态 列", cols, index=min(5, len(cols)-1))
    col_method = st.selectbox("首次加入方式 列", cols, index=min(6, len(cols)-1))
    col_first = st.selectbox("首次加入时间 列", cols, index=min(7, len(cols)-1))
    col_expire = st.selectbox("到期时间 列", cols, index=min(8, len(cols)-1))

    logic_select = st.multiselect(
        "计算口径",
        ["同月Cohort-严格12", "同月Cohort-12~14容差", "B/C/A-严格12", "B/C/A-12~14容差"],
        default=["同月Cohort-严格12","同月Cohort-12~14容差"]
    )

    if st.button("开始计算"):
        try:
            end_m = pd.Timestamp(f"{end_month}-01")
        except Exception:
            st.error("结束月份格式应为 YYYY-MM")
            st.stop()

        df = normalize_and_filter(df_raw, col_perm, col_method, col_first, col_expire)
        if df.empty:
            st.warning("过滤后无数据。请检查过滤口径与字段映射。")
            st.stop()

        tabs = st.tabs(["图表", "明细导出"])

        with tabs[0]:
            charts = []
            if "同月Cohort-严格12" in logic_select:
                cohort_strict = compute_cohort_monthly(df, end_m, tolerance_max_mod=0)
                st.subheader("同月 Cohort（严格12）")
                st.dataframe(cohort_strict)
                st.line_chart(cohort_strict.set_index("Month")["RatePct"], height=260)
                charts.append(("cohort_strict.csv", cohort_strict))
            if "同月Cohort-12~14容差" in logic_select:
                cohort_tol = compute_cohort_monthly(df, end_m, tolerance_max_mod=2)
                st.subheader("同月 Cohort（12~14 容差）")
                st.dataframe(cohort_tol)
                st.line_chart(cohort_tol.set_index("Month")["RatePct"], height=260)
                charts.append(("cohort_tol.csv", cohort_tol))
            if "B/C/A-严格12" in logic_select:
                bca_strict = compute_bca_monthly(df, end_m, tolerance_max_mod=0)
                st.subheader("B/C/A（严格12）")
                st.dataframe(bca_strict)
                st.line_chart(bca_strict.set_index("Month")["RatePct"], height=260)
                charts.append(("bca_strict.csv", bca_strict))
            if "B/C/A-12~14容差" in logic_select:
                bca_tol = compute_bca_monthly(df, end_m, tolerance_max_mod=2)
                st.subheader("B/C/A（12~14 容差）")
                st.dataframe(bca_tol)
                st.line_chart(bca_tol.set_index("Month")["RatePct"], height=260)
                charts.append(("bca_tol.csv", bca_tol))

        with tabs[1]:
            st.subheader("下载明细")
            for name, d in charts:
                st.write(name, d.head(12))
                st.download_button(
                    "下载 " + name,
                    d.to_csv(index=False).encode("utf-8-sig"),
                    file_name=name,
                    mime="text/csv",
                )
