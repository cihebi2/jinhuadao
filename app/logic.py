import pandas as pd
import re
from datetime import datetime
from typing import Optional

ALLOWED_METHODS = {"付费加入", "课程购买"}
REMOVED_STATUS = "被移除"


def _coerce_dt(x: Optional[str]) -> Optional[pd.Timestamp]:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    s = str(x)
    s = re.sub(r'^[\s"\t]+', "", s)
    s = re.sub(r'[\s"\t]+$', "", s)
    if not s:
        return None
    try:
        return pd.to_datetime(s, errors="coerce")
    except Exception:
        return None


def _month_floor(ts: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(ts.year, ts.month, 1)


def normalize_and_filter(df: pd.DataFrame,
                         col_perm: str, col_method: str,
                         col_first_join: str, col_expire: str) -> pd.DataFrame:
    out = df.copy()
    out[col_perm] = out[col_perm].astype(str).str.strip()
    out[col_method] = out[col_method].astype(str).str.strip()
    out["FirstJoin"] = out[col_first_join].map(_coerce_dt)
    out["Expire"] = out[col_expire].map(_coerce_dt)
    out = out[(out["FirstJoin"].notna()) & (out["Expire"].notna())]
    out = out[out[col_perm] != REMOVED_STATUS]
    out = out[out[col_method].isin(ALLOWED_METHODS)]
    out["FJM"] = out["FirstJoin"].map(_month_floor)
    out["EXM"] = out["Expire"].map(_month_floor)
    out["FIdx"] = out["FJM"].dt.year * 12 + out["FJM"].dt.month
    out["EIdx"] = out["EXM"].dt.year * 12 + out["EXM"].dt.month
    return out


def months_range(min_month: pd.Timestamp, end_month: pd.Timestamp):
    cur = pd.Timestamp(min_month.year, min_month.month, 1)
    end = pd.Timestamp(end_month.year, end_month.month, 1)
    while cur <= end:
        yield cur
        cur = (cur + pd.offsets.MonthBegin(1))


def compute_cohort_monthly(df: pd.DataFrame, end_month: pd.Timestamp,
                           tolerance_max_mod: int = 0) -> pd.DataFrame:
    """
    同月 Cohort 续费率：
    - cohort = 首次加入月份 == 目标月的月份，且 FIdx < MIdx（排除当月新客）
    - 严格同月：对齐 offset%12==0，分子 offset>=12；分母 offset>=0
    - 12-14 容差：对齐 offset%12 in {0,1,2}，分子 offset>=1；分母 offset>=0
    返回：Month, Den, Num, RatePct
    """
    min_month = df["FJM"].min()
    rows = []
    for m in months_range(min_month, end_month):
        m_idx = m.year * 12 + m.month
        mon = m.month
        cohort = df[(df["FJM"].dt.month == mon) & (df["FIdx"] < m_idx)]
        if tolerance_max_mod == 0:
            aligned = cohort[(cohort["EIdx"] >= m_idx) & ((cohort["EIdx"] - m_idx) % 12 == 0)]
            num = aligned[(aligned["EIdx"] - m_idx) >= 12].shape[0]
            den = aligned.shape[0]
        else:
            aligned = cohort[(cohort["EIdx"] >= m_idx) & ((cohort["EIdx"] - m_idx) % 12 <= tolerance_max_mod)]
            num = aligned[(aligned["EIdx"] - m_idx) >= 1].shape[0]
            den = aligned.shape[0]
        rate = round(num / den * 100, 2) if den > 0 else None
        rows.append({"Month": m.strftime("%Y-%m"), "Den": den, "Num": num, "RatePct": rate})
    return pd.DataFrame(rows)


def compute_bca_monthly(df: pd.DataFrame, end_month: pd.Timestamp,
                        tolerance_max_mod: int = 0) -> pd.DataFrame:
    """
    B/C/A 快照重构：
    - A = 当月首入； B = 当月到期； C = 次年同月到期（容差版为 12~14 月）
    - 续费人数 = max(0, C - A_in_C)； 底数 = B + 续费人数
    返回：Month, A,B,C,A_in_C, Renew, Base, RatePct
    """
    min_month = df["FJM"].min()
    rows = []
    for m in months_range(min_month, end_month):
        m_next = m + pd.offsets.MonthBegin(1)
        a = df[(df["FirstJoin"] >= m) & (df["FirstJoin"] < m_next)].shape[0]
        b = df[(df["Expire"] >= m) & (df["Expire"] < m_next)].shape[0]
        if tolerance_max_mod == 0:
            c_start = m + pd.offsets.DateOffset(years=1)
            c_end = c_start + pd.offsets.MonthBegin(1)
            c = df[(df["Expire"] >= c_start) & (df["Expire"] < c_end)].shape[0]
            a_c = df[(df["FirstJoin"] >= m) & (df["FirstJoin"] < m_next) &
                     (df["Expire"] >= c_start) & (df["Expire"] < c_end)].shape[0]
        else:
            c = 0; a_c = 0
            for i in range(0, tolerance_max_mod + 1):
                c_s = m + pd.offsets.DateOffset(years=1, months=i)
                c_e = c_s + pd.offsets.MonthBegin(1)
                c += df[(df["Expire"] >= c_s) & (df["Expire"] < c_e)].shape[0]
                a_c += df[(df["FirstJoin"] >= m) & (df["FirstJoin"] < m_next) &
                          (df["Expire"] >= c_s) & (df["Expire"] < c_e)].shape[0]
        renew = max(0, c - a_c)
        base = b + renew
        rate = round(renew / base * 100, 2) if base > 0 else None
        rows.append({
            "Month": m.strftime("%Y-%m"),
            "A": a, "B": b, "C": c, "A_in_C": a_c,
            "Renew": renew, "Base": base, "RatePct": rate,
        })
    return pd.DataFrame(rows)
