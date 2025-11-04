# 续费率仪表盘（Streamlit）

一个可在 Windows 与 Linux 部署的简单应用：上传 CSV/XLSX，计算并可视化每月续费率。

## 功能
- 上传数据，字段映射
- 过滤口径：排除“被移除”，仅保留“付费加入/课程购买”
- 两种续费计算：
  - 同月 Cohort（严格12、12~14 容差）
  - B/C/A 快照重构（严格12、12~14 容差，可选）
- 图表（折线）与明细下载（CSV）

## 本地运行（Windows 11）
```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/ui.py
```
访问：http://localhost:8501

## Docker 部署（Linux）
```bash
docker build -t renew-dashboard:latest .
docker run -d --name renew --restart unless-stopped -p 8501:8501 renew-dashboard:latest
```

## 数据要求
- 必填字段：权限状态、首次加入方式、首次加入时间、到期时间
- 编码：优先 UTF-8（含 BOM），也支持 GB18030
- 时间窗口：按自然月，左闭右开

## 计算口径（简述）
- 同月 Cohort：
  - cohort=历年同月首入且首次加入<当月
  - 严格同月：对齐 offset%12==0；分子 offset>=12；分母 offset>=0
  - 12~14 容差：对齐 offset%12∈{0,1,2}；分子 offset>=1；分母 offset>=0
- B/C/A：A=当月首入；B=当月到期；C=次年同月（容差版 12~14）

## 许可证
内部项目示例，后续按需要补充。
