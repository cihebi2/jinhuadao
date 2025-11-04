import os, sys
import streamlit as st
import pandas as pd
import altair as alt
from datetime import datetime

# 兼容 Streamlit 工作目录差异，确保可导入 app.logic
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from app.logic import normalize_and_filter, compute_cohort_monthly, compute_bca_monthly

st.set_page_config(page_title="续费率仪表盘", layout="wide")

st.title("续费率仪表盘（上传 CSV/XLSX）")

# 辅助：将透视表稳健还原为长表（自动识别索引列名）
def _pivot_to_long(pivot_df: pd.DataFrame, category_col: str, value_col: str) -> pd.DataFrame:
    df_reset = pivot_df.reset_index()
    # 索引列名：若原索引有名称，则为该名称；若无名称，reset_index 后默认为 'index'
    idx_col = df_reset.columns[0]
    df_long = df_reset.melt(id_vars=[idx_col], var_name=category_col, value_name=value_col)
    df_long = df_long.rename(columns={idx_col: "月份"})
    # 转为时间类型便于 Altair 使用时间轴
    df_long["月份"] = pd.to_datetime(df_long["月份"], format="%Y-%m", errors="coerce")
    return df_long

with st.sidebar:
    st.header("设置")
    uploaded = st.file_uploader("上传数据文件（CSV/XLSX）", type=["csv","xlsx"])
    # 默认统计截至到上一个完整自然月
    now = pd.Timestamp.today()
    default_end = pd.Timestamp(now.year, now.month, 1) - pd.offsets.MonthBegin(1)
    end_month = st.text_input("统计截至月份（YYYY-MM）", default_end.strftime("%Y-%m"))

if not uploaded:
    st.info("请在左侧上传文件并设置截至月份")
    st.stop()

# 读取数据
try:
    if uploaded.name.lower().endswith(".xlsx"):
        df_raw = pd.read_excel(uploaded)
    else:
        last_err = None
        for enc in ("utf-8-sig", "utf-8", "gb18030"):
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
col_perm = st.selectbox("权限状态 列", cols, index=min(5, len(cols)-1))
col_method = st.selectbox("首次加入方式 列", cols, index=min(6, len(cols)-1))
col_first = st.selectbox("首次加入时间 列", cols, index=min(7, len(cols)-1))
col_expire = st.selectbox("到期时间 列", cols, index=min(8, len(cols)-1))

logic_select = st.multiselect(
    "选择计算口径",
    ["同月Cohort-严格12", "同月Cohort-12~14容差", "B/C/A-严格12", "B/C/A-12~14容差"],
    default=["同月Cohort-严格12", "同月Cohort-12~14容差"],
)

# 执行计算
try:
    end_m = pd.Timestamp(f"{end_month}-01")
except Exception:
    st.error("结束月份格式应为 YYYY-MM")
    st.stop()

# 预处理 + 过滤
DF = normalize_and_filter(df_raw, col_perm, col_method, col_first, col_expire)
if DF.empty:
    st.warning("过滤后无数据。请检查过滤口径与字段映射。")
    st.stop()

st.divider()
st.subheader("基础看板")
col_a, col_b, col_c, col_d = st.columns(4)
with col_a:
    st.metric("过滤后样本量", f"{len(DF):,}")
with col_b:
    st.metric("最早首入月", DF["FJM"].min().strftime("%Y-%m"))
with col_c:
    st.metric("最晚到期月", DF["EXM"].max().strftime("%Y-%m"))
with col_d:
    st.metric("统计截至月", end_m.strftime("%Y-%m"))

st.caption("说明：样本已排除‘被移除’，仅保留‘付费加入/课程购买’，且需首入/到期时间有效。")

# ============ 月度分布：权限状态 / 加入方式 ============
st.divider()
st.subheader("月度分布（权限状态 / 加入方式）")
col_sel1, col_sel2 = st.columns([1,1])
with col_sel1:
    dim_perm = st.radio("权限状态分布按：", ["按首次加入月份", "按到期月份"], horizontal=True, index=1)

# 权限状态分布
month_col_for_perm = "EXM" if dim_perm == "按到期月份" else "FJM"
perm_df = (
    DF.groupby([month_col_for_perm, col_perm]).size().reset_index(name="人数")
)
if not perm_df.empty:
    pivot_perm = perm_df.pivot(index=month_col_for_perm, columns=col_perm, values="人数").fillna(0).sort_index()
    pivot_perm.index = pivot_perm.index.strftime("%Y-%m")
    st.write("权限状态分布（", dim_perm, ")")
    st.dataframe(pivot_perm)
    # Altair 堆叠面积图
    perm_long = _pivot_to_long(pivot_perm, category_col="权限状态", value_col="人数")
    chart_perm = alt.Chart(perm_long).mark_area().encode(
        x=alt.X("月份:T", title="月份"),
        y=alt.Y("人数:Q", title="人数"),
        color=alt.Color("权限状态:N", title="权限状态"),
        tooltip=["月份","权限状态","人数"]
    ).properties(height=240)
    st.altair_chart(chart_perm, use_container_width=True)
else:
    st.info("无可计算的权限状态分布")

# 加入方式分布（按首入月）
method_df = (
    DF.groupby(["FJM", col_method]).size().reset_index(name="人数")
)
if not method_df.empty:
    pivot_method = method_df.pivot(index="FJM", columns=col_method, values="人数").fillna(0).sort_index()
    pivot_method.index = pivot_method.index.strftime("%Y-%m")
    st.write("加入方式分布（按首次加入月份）")
    st.dataframe(pivot_method)
    method_long = _pivot_to_long(pivot_method, category_col="加入方式", value_col="人数")
    chart_method = alt.Chart(method_long).mark_bar().encode(
        x=alt.X("月份:T", title="月份"),
        y=alt.Y("人数:Q", title="人数"),
        color=alt.Color("加入方式:N", title="加入方式"),
        tooltip=["月份","加入方式","人数"]
    ).properties(height=240)
    st.altair_chart(chart_method, use_container_width=True)
else:
    st.info("无可计算的加入方式分布")

# ============ 续费率计算与展示 ============
st.divider()
st.subheader("续费率（同月 Cohort & B/C/A 可选）")

charts = []
if "同月Cohort-严格12" in logic_select:
    cohort_strict = compute_cohort_monthly(DF, end_m, tolerance_max_mod=0)
    st.markdown("**同月 Cohort（严格12）**：cohort=历年同月首入且首入<当月；对齐 offset%12==0；分子 offset≥12；分母 offset≥0")
    st.dataframe(cohort_strict)
    st.line_chart(cohort_strict.set_index("Month")["RatePct"], height=260)
    charts.append(("cohort_strict.csv", cohort_strict))

if "同月Cohort-12~14容差" in logic_select:
    cohort_tol = compute_cohort_monthly(DF, end_m, tolerance_max_mod=2)
    st.markdown("**同月 Cohort（12~14 容差）**：对齐 (offset%12)∈{0,1,2}；分子 offset≥1；分母 offset≥0")
    st.dataframe(cohort_tol)
    st.line_chart(cohort_tol.set_index("Month")["RatePct"], height=260)
    charts.append(("cohort_tol.csv", cohort_tol))

if "B/C/A-严格12" in logic_select:
    bca_strict = compute_bca_monthly(DF, end_m, tolerance_max_mod=0)
    st.markdown("**B/C/A（严格12）**：A=当月首入；B=当月到期；C=次年同月到期；续= C−A_in_C；底= B+续")
    st.dataframe(bca_strict)
    st.line_chart(bca_strict.set_index("Month")["RatePct"], height=260)
    charts.append(("bca_strict.csv", bca_strict))

if "B/C/A-12~14容差" in logic_select:
    bca_tol = compute_bca_monthly(DF, end_m, tolerance_max_mod=2)
    st.markdown("**B/C/A（12~14 容差）**：C/A_in_C 使用 12~14 月窗口")
    st.dataframe(bca_tol)
    st.line_chart(bca_tol.set_index("Month")["RatePct"], height=260)
    charts.append(("bca_tol.csv", bca_tol))

# ============ 续费率明细（单月拆解） ============
st.divider()
st.subheader("续费率明细（选择月份查看更多数值）")

month_list = None
if "cohort_strict" in locals():
    month_list = list(cohort_strict["Month"]) 
elif "cohort_tol" in locals():
    month_list = list(cohort_tol["Month"]) 
else:
    month_list = sorted(list({d.strftime("%Y-%m") for d in DF["FJM"].unique()}))

sel = st.selectbox("选择月份", month_list, index=len(month_list)-1 if month_list else 0)
if sel:
    m = pd.to_datetime(sel + "-01")
    m_idx = m.year*12 + m.month
    mon = m.month
    cohort = DF[(DF["FJM"].dt.month == mon) & (DF["FIdx"] < m_idx)]
    cohort_size = cohort.shape[0]
    earlier = cohort[cohort["EIdx"] < m_idx].shape[0]
    aligned_strict = cohort[(cohort["EIdx"] >= m_idx) & ((cohort["EIdx"] - m_idx) % 12 == 0)]
    aligned_tol = cohort[(cohort["EIdx"] >= m_idx) & ((cohort["EIdx"] - m_idx) % 12 <= 2)]
    den_strict = aligned_strict.shape[0]
    num_strict = aligned_strict[(aligned_strict["EIdx"] - m_idx) >= 12].shape[0]
    due_now = aligned_strict[(aligned_strict["EIdx"] - m_idx) == 0].shape[0]
    den_tol = aligned_tol.shape[0]
    num_tol = aligned_tol[(aligned_tol["EIdx"] - m_idx) >= 1].shape[0]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("cohort规模", f"{cohort_size:,}")
        st.metric("早于当月到期", f"{earlier:,}")
    with c2:
        st.metric("严格12 分母", f"{den_strict:,}")
        st.metric("严格12 分子", f"{num_strict:,}")
    with c3:
        st.metric("当月应续(严格)", f"{due_now:,}")
        st.metric("严格12 续费率", f"{(round(num_strict/den_strict*100,2) if den_strict>0 else None)}%")
    with c4:
        st.metric("12~14分母", f"{den_tol:,}")
        st.metric("12~14分子", f"{num_tol:,}")
        st.metric("12~14续费率", f"{(round(num_tol/den_tol*100,2) if den_tol>0 else None)}%")

st.caption("说明：同月 Cohort 以‘历年同月首入且首入<当月’为样本。严格12仅统计周年同月，12~14容差包含周年同月至+2月；分母含当月应续，分子仅包含当月之后窗口。")

st.divider()
st.subheader("下载明细")
for name, d in charts:
    st.write(name, d.head(12))
    st.download_button(
        "下载 " + name,
        d.to_csv(index=False).encode("utf-8-sig"),
        file_name=name,
        mime="text/csv",
    )
