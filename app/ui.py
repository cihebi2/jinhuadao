import os, sys
import streamlit as st
import pandas as pd
import altair as alt
from datetime import datetime
import io

# 兼容 Streamlit 工作目录差异，确保可导入 app.logic
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from app.logic import normalize_and_filter, compute_cohort_monthly, compute_bca_monthly

st.set_page_config(page_title="续费率仪表盘", layout="wide")

# 左侧功能导航与音频工具渲染
with st.sidebar:
    st.header("功能导航")
    _nav = st.radio("选择功能", ["续费率仪表盘", "MP3 音频音量放大"], index=0)

def _render_audio_tool():
    st.divider()
    st.subheader("音频工具：MP3 音量放大")
    st.caption("说明：拖拽或点击上传 MP3，服务端放大后导出为 MP3。需要 ffmpeg。")
    up = st.file_uploader("拖拽或点击上传 MP3 文件", type=["mp3"], key="mp3_uploader_nav")
    c1, c2 = st.columns([2,1])
    with c1:
        gain_db = st.slider("增益 (dB)", min_value=-20, max_value=20, value=6, step=1)
    with c2:
        avoid_clip = st.checkbox("避免削波(自动降至 -1dBFS)", value=True)
    if up is not None:
        try:
            from pydub import AudioSegment
            seg = AudioSegment.from_file(up, format="mp3")
            out = seg.apply_gain(gain_db)
            try:
                peak = out.max_dBFS
            except Exception:
                peak = None
            if avoid_clip and peak is not None and peak > -1.0:
                out = out.apply_gain(-1.0 - peak)
            buf = io.BytesIO()
            out.export(buf, format="mp3", bitrate="192k")
            buf.seek(0)
            base = os.path.splitext(up.name)[0]
            st.download_button("下载放大后的 MP3", data=buf.getvalue(), file_name=f"{base}_gain{gain_db:+d}dB.mp3", mime="audio/mpeg")
        except Exception as e:
            st.error(f"处理失败：{e}")

if _nav == "MP3 音频音量放大":
    _render_audio_tool()
    st.stop()


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

# 辅助：将续费结果重命名为中文列，并生成折线图所需的长表
def _rename_cohort_for_display(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Den" in out.columns:
        out = out.rename(columns={"Den":"分母","Num":"分子","RatePct":"续费率"})
    return out

def _cohort_long_pair(df_cn: pd.DataFrame) -> pd.DataFrame:
    d = df_cn.copy()
    d["月份"] = pd.to_datetime(d["Month"], format="%Y-%m", errors="coerce")
    keep = ["月份","分子","分母"]
    exist = [c for c in keep if c in d.columns]
    return d[exist].melt(id_vars=["月份"], var_name="指标", value_name="人数")

def _cohort_long_rate(df_cn: pd.DataFrame) -> pd.DataFrame:
    d = df_cn.copy()
    d["月份"] = pd.to_datetime(d["Month"], format="%Y-%m", errors="coerce")
    if "续费率" in d.columns:
        return d[["月份","续费率"]].rename(columns={"续费率":"数值"})
    else:
        return d[["月份"]].assign(数值=None)

# 辅助：输出统一的列顺序（先重命名 Month->月份，再按顺序挑列）
def _select_month_cn(df: pd.DataFrame, cols_order=None) -> pd.DataFrame:
    if cols_order is None:
        cols_order = ["Month","分子","分母","续费率"]
    df2 = df.copy()
    # 先重命名 Month → 月份
    df2 = df2.rename(columns={"Month":"月份"})
    # 将期望顺序中的 Month 替换为 月份
    cols_cn = [("月份" if c=="Month" else c) for c in cols_order if c in (list(df.columns)+["月份"])]
    # 去重保持顺序
    seen = set(); ordered = []
    for c in cols_cn:
        if c not in seen and c in df2.columns:
            seen.add(c); ordered.append(c)
    return df2[ordered]

# 组合折线图：分子/分母（左轴）+ 续费率（右轴），时间轴对齐
def _layer_counts_and_rate(counts_long: pd.DataFrame, rate_long: pd.DataFrame, height: int = 260):
    base_x = alt.X("月份:T", axis=alt.Axis(format="%Y-%m", title="月份"))
    left_y = alt.Y("人数:Q", title="人数")
    right_y = alt.Y("数值:Q", axis=alt.Axis(title="续费率(%)", orient="right"))
    layer_counts = alt.Chart(counts_long).mark_line(point=True).encode(
        x=base_x,
        y=left_y,
        color=alt.Color("指标:N", title="指标(分子/分母)"),
        tooltip=["月份","指标","人数"]
    )
    layer_rate = alt.Chart(rate_long).mark_line(point=True, color="#d62728").encode(
        x=base_x,
        y=right_y,
        tooltip=["月份","数值"]
    )
    return alt.layer(layer_counts, layer_rate).resolve_scale(y='independent').properties(height=height).interactive()

# 年度续费率（事件口径）：从 B/C/A 月度结果按年汇总
def _annual_from_bca(bca_monthly: pd.DataFrame) -> pd.DataFrame:
    if bca_monthly is None or bca_monthly.empty:
        return pd.DataFrame(columns=["Year","A_sum","B_sum","C_sum","A_in_C_sum","分子","分母","续费率"])
    d = bca_monthly.copy()
    d["Year"] = d["Month"].str.slice(0, 4)
    g = d.groupby("Year", as_index=False).agg({
        "A":"sum","B":"sum","C":"sum","A_in_C":"sum","Renew":"sum","Base":"sum"
    })
    g["RatePct"] = (g["Renew"] / g["Base"] * 100).round(2)
    g = g.rename(columns={
        "A":"A_sum","B":"B_sum","C":"C_sum","A_in_C":"A_in_C_sum",
        "Renew":"分子","Base":"分母","RatePct":"续费率"
    })
    return g

def _annual_plot_frames(df_year_cn: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df_year_cn is None or df_year_cn.empty:
        return pd.DataFrame(columns=["月份","指标","人数"]), pd.DataFrame(columns=["月份","数值"]) 
    d = df_year_cn.copy()
    d["月份"] = pd.to_datetime(d["Year"] + "-01-01", errors="coerce")
    counts_long = d.melt(id_vars=["月份"], value_vars=["分子","分母"], var_name="指标", value_name="人数")
    rate_long = d[["月份","续费率"]].rename(columns={"续费率":"数值"})
    return counts_long, rate_long

# 预计算：同月 Cohort 明细（避免每次选择都重新计算）
def _compute_cohort_details(df: pd.DataFrame, end_month: pd.Timestamp, tol_max_mod: int) -> pd.DataFrame:
    import pandas as pd
    from app.logic import months_range
    rows = []
    min_month = df["FJM"].min()
    for m in months_range(min_month, end_month):
        m_idx = m.year*12 + m.month
        mon = m.month
        cohort = df[(df["FJM"].dt.month == mon) & (df["FIdx"] < m_idx)]
        cohort_size = cohort.shape[0]
        earlier = cohort[cohort["EIdx"] < m_idx].shape[0]
        if tol_max_mod == 0:
            aligned = cohort[(cohort["EIdx"] >= m_idx) & ((cohort["EIdx"] - m_idx) % 12 == 0)]
            den = aligned.shape[0]
            num = aligned[(aligned["EIdx"] - m_idx) >= 12].shape[0]
            due_now = aligned[(aligned["EIdx"] - m_idx) == 0].shape[0]
        else:
            aligned = cohort[(cohort["EIdx"] >= m_idx) & ((cohort["EIdx"] - m_idx) % 12 <= tol_max_mod)]
            den = aligned.shape[0]
            num = aligned[(aligned["EIdx"] - m_idx) >= 1].shape[0]
            due_now = aligned[(aligned["EIdx"] - m_idx) == 0].shape[0]
        rate = round(num/den*100,2) if den>0 else None
        rows.append({
            "Month": m.strftime("%Y-%m"),
            "cohort规模": cohort_size,
            "早于当月到期": earlier,
            "分母": den,
            "分子": num,
            "续费率": rate,
            "当月应续(严格)": due_now if tol_max_mod==0 else None,
            "口径": "严格12" if tol_max_mod==0 else "12-14容差",
        })
    return pd.DataFrame(rows)

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
    cstrict_cn = _rename_cohort_for_display(cohort_strict)
    # 表格列顺序：月份、分子、分母、续费率
    cols_show = [c for c in ["Month","分子","分母","续费率"] if c in (list(cstrict_cn.columns)+["Month"])]
    st.dataframe(_select_month_cn(cstrict_cn, cols_show))
    dl_pair = _cohort_long_pair(cstrict_cn)
    dl_rate = _cohort_long_rate(cstrict_cn)
    st.altair_chart(_layer_counts_and_rate(dl_pair, dl_rate), use_container_width=True)
    charts.append(("cohort_strict.csv", _select_month_cn(cstrict_cn, cols_show)))

if "同月Cohort-12~14容差" in logic_select:
    cohort_tol = compute_cohort_monthly(DF, end_m, tolerance_max_mod=2)
    st.markdown("**同月 Cohort（12~14 容差）**：对齐 (offset%12)∈{0,1,2}；分子 offset≥1；分母 offset≥0")
    ctol_cn = _rename_cohort_for_display(cohort_tol)
    cols_show2 = [c for c in ["Month","分子","分母","续费率"] if c in (list(ctol_cn.columns)+["Month"])]
    st.dataframe(_select_month_cn(ctol_cn, cols_show2))
    dl_pair2 = _cohort_long_pair(ctol_cn)
    dl_rate2 = _cohort_long_rate(ctol_cn)
    st.altair_chart(_layer_counts_and_rate(dl_pair2, dl_rate2), use_container_width=True)
    charts.append(("cohort_tol.csv", _select_month_cn(ctol_cn, cols_show2)))

if "B/C/A-严格12" in logic_select:
    bca_strict = compute_bca_monthly(DF, end_m, tolerance_max_mod=0)
    st.markdown("**B/C/A（严格12）**：A=当月首入；B=当月到期；C=次年同月到期；分子(续)= C−A_in_C；分母(底)= B+分子")
    bcn = bca_strict.rename(columns={"Renew":"分子","Base":"分母","RatePct":"续费率"})
    cols_b1 = [c for c in ["Month","分子","分母","续费率"] if c in (list(bcn.columns)+["Month"])]
    st.dataframe(_select_month_cn(bcn, cols_b1))
    dl_b_pair = _cohort_long_pair(bcn)
    dl_b_rate = _cohort_long_rate(bcn)
    st.altair_chart(_layer_counts_and_rate(dl_b_pair, dl_b_rate), use_container_width=True)
    charts.append(("bca_strict.csv", _select_month_cn(bcn, cols_b1)))

if "B/C/A-12~14容差" in logic_select:
    bca_tol = compute_bca_monthly(DF, end_m, tolerance_max_mod=2)
    st.markdown("**B/C/A（12~14 容差）**：C/A_in_C 使用 12~14 月窗口；分子= C(12~14) − A_in_C(12~14)；分母= B+分子")
    btn = bca_tol.rename(columns={"Renew":"分子","Base":"分母","RatePct":"续费率"})
    cols_b2 = [c for c in ["Month","分子","分母","续费率"] if c in (list(btn.columns)+["Month"])]
    st.dataframe(_select_month_cn(btn, cols_b2))
    dl_btn_pair = _cohort_long_pair(btn)
    dl_btn_rate = _cohort_long_rate(btn)
    st.altair_chart(_layer_counts_and_rate(dl_btn_pair, dl_btn_rate), use_container_width=True)
    charts.append(("bca_tol.csv", _select_month_cn(btn, cols_b2)))

# ===== 年度续费率（事件口径） =====
st.divider()
st.subheader("年度续费率（事件口径）")
try:
    bca_m_strict = compute_bca_monthly(DF, end_m, tolerance_max_mod=0)
    bca_m_tol = compute_bca_monthly(DF, end_m, tolerance_max_mod=2)
    ann_strict = _annual_from_bca(bca_m_strict)
    ann_tol = _annual_from_bca(bca_m_tol)
    # 严格12
    if not ann_strict.empty:
        st.write("年度（严格12）")
        st.dataframe(ann_strict[["Year","分子","分母","续费率","A_sum","B_sum","C_sum","A_in_C_sum"]])
        cnt_long, rate_long = _annual_plot_frames(ann_strict)
        st.altair_chart(_layer_counts_and_rate(cnt_long, rate_long), use_container_width=True)
    # 12~14 容差
    if not ann_tol.empty:
        st.write("年度（12~14 容差）")
        st.dataframe(ann_tol[["Year","分子","分母","续费率","A_sum","B_sum","C_sum","A_in_C_sum"]])
        cnt_long2, rate_long2 = _annual_plot_frames(ann_tol)
        st.altair_chart(_layer_counts_and_rate(cnt_long2, rate_long2), use_container_width=True)
except Exception:
    pass

# ============ 续费率明细（单月拆解） ============
st.divider()
st.subheader("续费率明细（全部月份，已预计算）")

# 预计算两版明细，直接展示（并提供折叠）
detail_strict = _compute_cohort_details(DF, end_m, tol_max_mod=0)
detail_tol = _compute_cohort_details(DF, end_m, tol_max_mod=2)

with st.expander("展开/收起 明细 - 同月Cohort 严格12"):
    if not detail_strict.empty:
        cols = [c for c in ["Month","cohort规模","早于当月到期","分子","分母","续费率","当月应续(严格)","口径"] if c in detail_strict.columns]
        st.dataframe(detail_strict[cols].rename(columns={"Month":"月份"}))
    else:
        st.info("无明细数据")

with st.expander("展开/收起 明细 - 同月Cohort 12~14 容差"):
    if not detail_tol.empty:
        cols2 = [c for c in ["Month","cohort规模","早于当月到期","分子","分母","续费率","口径"] if c in detail_tol.columns]
        st.dataframe(detail_tol[cols2].rename(columns={"Month":"月份"}))
    else:
        st.info("无明细数据")

st.caption("说明：同月 Cohort 以‘历年同月首入且首入<当月’为样本。严格12仅统计周年同月，12~14容差包含周年同月至+2月；分母含当月应续，分子仅包含当月之后窗口。")

st.divider()
st.subheader("下载明细")
for name, d in charts:
    try:
        st.write(name, d.head(12))
        st.download_button(
            "下载 " + name,
            d.to_csv(index=False).encode("utf-8-sig"),
            file_name=name,
            mime="text/csv",
        )
    except Exception:
        pass

# 导出：分析工作簿（Excel）汇总
try:
    import pandas as _pd
    import io as _io
    _buf = _io.BytesIO()
    with _pd.ExcelWriter(_buf, engine="openpyxl") as _writer:
        try:
            c1 = cstrict_cn if 'cstrict_cn' in locals() else None
            if c1 is not None:
                (c1.rename(columns={"Month":"月份"})).to_excel(_writer, index=False, sheet_name="同月Cohort_严格12")
        except Exception:
            pass
        try:
            c2 = ctol_cn if 'ctol_cn' in locals() else None
            if c2 is not None:
                (c2.rename(columns={"Month":"月份"})).to_excel(_writer, index=False, sheet_name="同月Cohort_12-14")
        except Exception:
            pass
        try:
            if 'bcn' in locals():
                (bcn.rename(columns={"Month":"月份"})).to_excel(_writer, index=False, sheet_name="BCA_严格12")
        except Exception:
            pass
        try:
            if 'btn' in locals():
                (btn.rename(columns={"Month":"月份"})).to_excel(_writer, index=False, sheet_name="BCA_12-14")
        except Exception:
            pass
        try:
            if 'pivot_perm' in locals():
                pivot_perm.to_excel(_writer, sheet_name="权限状态分布", index=True)
        except Exception:
            pass
        try:
            if 'pivot_method' in locals():
                pivot_method.to_excel(_writer, sheet_name="加入方式分布", index=True)
        except Exception:
            pass
        try:
            if 'detail_strict' in locals():
                detail_strict.to_excel(_writer, index=False, sheet_name="明细_严格12")
        except Exception:
            pass
        try:
            if 'detail_tol' in locals():
                detail_tol.to_excel(_writer, index=False, sheet_name="明细_12-14")
        except Exception:
            pass
    _buf.seek(0)
        # annual summaries (事件口径)
        try:
            if 'ann_strict' in locals():
                ann_strict.to_excel(_writer, index=False, sheet_name="annual_12")
            else:
                _b1 = compute_bca_monthly(DF, end_m, tolerance_max_mod=0)
                _a1 = _annual_from_bca(_b1)
                if _a1 is not None and not _a1.empty:
                    _a1.to_excel(_writer, index=False, sheet_name="annual_12")
        except Exception:
            pass
        try:
            if 'ann_tol' in locals():
                ann_tol.to_excel(_writer, index=False, sheet_name="annual_12_14")
            else:
                _b2 = compute_bca_monthly(DF, end_m, tolerance_max_mod=2)
                _a2 = _annual_from_bca(_b2)
                if _a2 is not None and not _a2.empty:
                    _a2.to_excel(_writer, index=False, sheet_name="annual_12_14")
        except Exception:
            pass


# ===== Annual renewal (4-year cohort) =====
st.divider()
st.subheader("Annual Renewal (4-year cohort)")
st.caption("For year Y: Numerator=A where join_year in {Y-1..Y-4} and expire_year=Y+1; Denominator=A + B where B has expire_year=Y.")

def _annual_4yr_cohort(df: pd.DataFrame, end_month: pd.Timestamp) -> pd.DataFrame:
    d = df.copy()
    d["JoinYear"] = d["FJM"].dt.year
    d["ExpireYear"] = d["EXM"].dt.year
    years = sorted(d["ExpireYear"].unique())
    rows = []
    for Y in years:
        join_set = {Y-1, Y-2, Y-3, Y-4}
        a = d[(d["JoinYear"].isin(join_set)) & (d["ExpireYear"] == Y+1)].shape[0]
        b = d[(d["JoinYear"].isin(join_set)) & (d["ExpireYear"] == Y)].shape[0]
        base = a + b
        rate = round(a/base*100, 2) if base > 0 else None
        rows.append({"Year": str(Y), "分子": a, "分母": base, "续费率": rate})
    return pd.DataFrame(rows)

_ann = _annual_4yr_cohort(DF, end_m)
if not _ann.empty:
    st.dataframe(_ann[["Year","分子","分母","续费率"]])
    _ann_plot = _ann.copy(); _ann_plot["月份"] = pd.to_datetime(_ann_plot["Year"]+"-01-01")
    _counts = _ann_plot.melt(id_vars=["月份"], value_vars=["分子","分母"], var_name="指标", value_name="人数")
    _rate = _ann_plot[["月份","续费率"]].rename(columns={"续费率":"数值"})
    st.altair_chart(_layer_counts_and_rate(_counts, _rate), use_container_width=True)
