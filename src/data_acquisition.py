# -*- coding: utf-8 -*-
"""
==============================================================================
数据获取脚本 — 区域健康风险预测与智能诊疗决策
功能: 从多个公开数据源批量下载医疗数据集
==============================================================================
"""

import os, sys, json, shutil, time, hashlib, zipfile, io
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import requests
from tqdm import tqdm

# ========================= 全局配置 =========================
DATA_RAW   = Path("./data/raw")
DATA_RAW.mkdir(parents=True, exist_ok=True)

LOG_FILE  = DATA_RAW / "download_log.json"
REPORT    = {}
SUMMARY   = []      # 汇总信息
HEADERS   = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
TIMEOUT   = 60

# ========================= 工具函数 =========================
def log(msg, level="INFO"):
    print(f"[{datetime.now().strftime('%H:%M:%S')}][{level}] {msg}")

def safe_request(url, stream=False, timeout=TIMEOUT):
    """安全的 HTTP 请求，带重试"""
    for attempt in range(3):
        try:
            resp = requests.get(url, headers=HEADERS, stream=stream, timeout=timeout)
            resp.raise_for_status()
            return resp
        except Exception as e:
            log(f"请求失败 (尝试 {attempt+1}/3): {url[:80]} — {e}", "WARN")
            if attempt < 2:
                time.sleep(3)
    return None

def save_csv(df, name, subdir=None):
    """保存 DataFrame 到 raw 目录"""
    dest = DATA_RAW if subdir is None else DATA_RAW / subdir
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / f"{name}.csv"
    df.to_csv(path, index=False)
    size = os.path.getsize(path)
    log(f"已保存: {path} ({size/1024:.1f} KB, {df.shape[0]}行×{df.shape[1]}列)")
    return str(path), size

def record(dataset, source, url, path, size, rows, cols, status="成功"):
    """记录下载结果"""
    SUMMARY.append({
        "dataset": dataset, "source": source, "url": url,
        "file": path, "size_kb": round(size/1024, 1),
        "rows": rows, "cols": cols, "status": status
    })

# ======================================================================
# 步骤 1: Kaggle 数据集下载 (kagglehub)
# ======================================================================
def download_kaggle_datasets():
    """使用 kagglehub 下载 Kaggle 公开数据集"""
    log("=" * 60)
    log("开始下载 Kaggle 公开医疗数据集")
    log("=" * 60)

    import kagglehub

    KAGGLE_LIST = [
        # (kaggle路径, 简称, 预期文件)
        ("uciml/pima-indians-diabetes-database", "pima_diabetes", "diabetes.csv"),
        ("johnsmith88/heart-disease-dataset", "heart_disease", "heart.csv"),
        ("sulianova/cardiovascular-disease-dataset", "cardio_disease", "cardio_train.csv"),
        ("uciml/breast-cancer-wisconsin-data", "breast_cancer", "data.csv"),
        ("fedesoriano/stroke-prediction-dataset", "stroke_prediction", "healthcare-dataset-stroke-data.csv"),
    ]

    for kg_path, name, csv_file in KAGGLE_LIST:
        try:
            log(f"下载: {kg_path}")
            dl_path = kagglehub.dataset_download(kg_path)
            log(f"  缓存路径: {dl_path}")

            # 查找 CSV 文件
            src = Path(dl_path)
            found = list(src.rglob(csv_file))
            if not found:
                # 尝试找任意 CSV
                found = list(src.rglob("*.csv"))
            if not found:
                log(f"  未找到 CSV 文件, 跳过", "WARN")
                record(name, "Kaggle", kg_path, "", 0, 0, 0, "未找到CSV")
                continue

            # 复制到目标目录
            dest = DATA_RAW / name
            dest.mkdir(exist_ok=True)
            for f in found:
                shutil.copy2(f, dest / f.name)

            # 读取并记录
            main_csv = dest / csv_file
            if main_csv.exists():
                df = pd.read_csv(main_csv)
                record(name, "Kaggle", kg_path, str(main_csv),
                       os.path.getsize(main_csv), df.shape[0], df.shape[1])
            else:
                total_size = sum(f.stat().st_size for f in dest.iterdir())
                record(name, "Kaggle", kg_path, str(dest), total_size, 0, 0)

        except Exception as e:
            log(f"下载失败: {kg_path} — {e}", "ERROR")
            record(name, "Kaggle", kg_path, "", 0, 0, 0, f"失败:{e}")

# ======================================================================
# 步骤 2: UCI ML Repository 数据集
# ======================================================================
def download_uci_datasets():
    """从 UCI Machine Learning Repository 下载数据集"""
    log("=" * 60)
    log("开始下载 UCI 公开数据集")
    log("=" * 60)

    UCI_LIST = [
        # (URL, 名称, 描述)
        ("https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
         "heart_cleveland", "克利夫兰心脏病数据"),
        ("https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data",
         "heart_hungarian", "匈牙利心脏病数据"),
        ("https://archive.ics.uci.edu/ml/machine-learning-databases/00383/risk_factors_cervical_cancer.csv",
         "cervical_cancer", "宫颈癌风险因素"),
        ("https://archive.ics.uci.edu/ml/machine-learning-databases/00296/dataset_diabetes.zip",
         "diabetes_130us", "糖尿病130家医院再入院"),
    ]

    for url, name, desc in UCI_LIST:
        try:
            log(f"下载: {desc} ({name})")
            resp = safe_request(url)
            if resp is None:
                record(name, "UCI", url, "", 0, 0, 0, "请求失败")
                continue

            dest = DATA_RAW / name
            dest.mkdir(exist_ok=True)

            if url.endswith(".zip"):
                # ZIP 文件
                zf = zipfile.ZipFile(io.BytesIO(resp.content))
                zf.extractall(dest)
                csvs = list(dest.rglob("*.csv"))
                if csvs:
                    df = pd.read_csv(csvs[0])
                    record(name, "UCI", url, str(csvs[0]),
                           os.path.getsize(csvs[0]), df.shape[0], df.shape[1])
                else:
                    total = sum(f.stat().st_size for f in dest.iterdir())
                    record(name, "UCI", url, str(dest), total, 0, 0)
            else:
                fname = url.split("/")[-1]
                fpath = dest / fname
                fpath.write_bytes(resp.content)

                # 尝试读取
                try:
                    df = pd.read_csv(fpath, header=None)
                    # 为 heart disease 数据添加列名
                    if "heart" in name:
                        df.columns = ["age","sex","cp","trestbps","chol","fbs","restecg",
                                      "thalach","exang","oldpeak","slope","ca","thal","target"]
                    record(name, "UCI", url, str(fpath),
                           os.path.getsize(fpath), df.shape[0], df.shape[1])
                except Exception:
                    record(name, "UCI", url, str(fpath), os.path.getsize(fpath), 0, 0)

        except Exception as e:
            log(f"下载失败: {name} — {e}", "ERROR")
            record(name, "UCI", url, "", 0, 0, 0, f"失败:{e}")

# ======================================================================
# 步骤 3: CDC NHANES 数据
# ======================================================================
def download_nhanes_data():
    """下载 CDC NHANES 代表性数据集"""
    log("=" * 60)
    log("开始下载 CDC NHANES 数据")
    log("=" * 60)

    dest = DATA_RAW / "nhanes"
    dest.mkdir(exist_ok=True)

    # NHANES 2017-2018 代表性数据文件 (XPT 格式)
    NHANES_FILES = [
        # (文件名, 描述, 周期)
        ("DEMO_J.XPT", "人口统计学", "2017-2018"),
        ("BMX_J.XPT",  "身体测量(BMI)", "2017-2018"),
        ("BPX_J.XPT",  "血压测量", "2017-2018"),
        ("BMX_H.XPT",  "身体测量", "2013-2014"),
        ("BPX_H.XPT",  "血压测量", "2013-2014"),
        ("DEMO_H.XPT", "人口统计学", "2013-2014"),
    ]
    BASE_URL = "https://wwwn.cdc.gov/Nchs/Nhanes"

    for fname, desc, cycle in NHANES_FILES:
        try:
            letter = fname.split("_")[1][0]
            yr = "2017-2018" if letter == "J" else "2013-2014"
            url = f"{BASE_URL}/{yr}/{fname}"
            fpath = dest / fname

            if fpath.exists():
                log(f"已存在，跳过: {fname}")
                try:
                    df = pd.read_sas(fpath)
                    record(f"nhanes_{fname.split('.')[0]}", "CDC NHANES", url,
                           str(fpath), os.path.getsize(fpath), df.shape[0], df.shape[1])
                except:
                    record(f"nhanes_{fname.split('.')[0]}", "CDC NHANES", url,
                           str(fpath), os.path.getsize(fpath), 0, 0, "无法解析")
                continue

            log(f"下载: {desc} ({fname})")
            resp = safe_request(url)
            if resp is None:
                record(f"nhanes_{fname.split('.')[0]}", "CDC NHANES", url, "", 0,0,0, "请求失败")
                continue

            fpath.write_bytes(resp.content)

            # 尝试用 pandas 读取 SAS XPT 格式
            try:
                df = pd.read_sas(fpath)
                record(f"nhanes_{fname.split('.')[0]}", "CDC NHANES", url,
                       str(fpath), os.path.getsize(fpath), df.shape[0], df.shape[1])
            except Exception as e:
                log(f"XPT 解析警告: {e}", "WARN")
                record(f"nhanes_{fname.split('.')[0]}", "CDC NHANES", url,
                       str(fpath), os.path.getsize(fpath), 0, 0, "需SAS解析")

        except Exception as e:
            log(f"下载失败: {fname} — {e}", "ERROR")
            record(f"nhanes_{fname.split('.')[0]}", "CDC NHANES", url, "", 0,0,0, f"失败:{e}")

    # 也下载 BRFSS 年度汇总数据 (行为风险因素)
    download_brfss_data(dest.parent / "brfss")

def download_brfss_data(dest_dir):
    """下载 BRFSS 行为风险因素监测数据"""
    dest_dir.mkdir(exist_ok=True)
    log("下载 CDC BRFSS 行为风险因素数据...")

    # BRFSS 2022 全年 SAS 数据 (较大的文件)
    brfss_url = "https://www.cdc.gov/brfss/annual_data/2022/files/LLCP2022XPT.zip"
    try:
        resp = safe_request(brfss_url)
        if resp:
            fpath = dest_dir / "LLCP2022XPT.zip"
            fpath.write_bytes(resp.content)
            try:
                zf = zipfile.ZipFile(io.BytesIO(resp.content))
                zf.extractall(dest_dir)
                xpt_files = list(dest_dir.rglob("*.XPT"))
                if xpt_files:
                    try:
                        df = pd.read_sas(xpt_files[0])
                        record("brfss_2022", "CDC BRFSS", brfss_url,
                               str(xpt_files[0]), os.path.getsize(xpt_files[0]),
                               df.shape[0], df.shape[1])
                    except:
                        record("brfss_2022", "CDC BRFSS", brfss_url,
                               str(xpt_files[0]), os.path.getsize(xpt_files[0]),
                               0, 0, "需SAS解析")
            except zipfile.BadZipFile:
                record("brfss_2022", "CDC BRFSS", brfss_url, str(fpath),
                       os.path.getsize(fpath), 0, 0, "ZIP下载完成")
        else:
            log("BRFSS 大文件下载失败，尝试旧年份...", "WARN")
            download_brfss_alt(dest_dir)
    except Exception as e:
        log(f"BRFSS 下载失败: {e}", "ERROR")
        download_brfss_alt(dest_dir)

def download_brfss_alt(dest_dir):
    """BRFSS 备选: 下载较小的 CDI 数据集"""
    cdi_url = "https://data.cdc.gov/api/views/5h56-3jrr/rows.csv?accessType=DOWNLOAD"
    try:
        resp = safe_request(cdi_url)
        if resp:
            fpath = dest_dir / "cdi_disease_indicators.csv"
            fpath.write_bytes(resp.content)
            df = pd.read_csv(fpath)
            record("cdi_indicators", "CDC", cdi_url, str(fpath),
                   os.path.getsize(fpath), df.shape[0], df.shape[1])
    except Exception as e:
        log(f"CDI 也失败: {e}", "ERROR")

# ======================================================================
# 步骤 4: WHO 全球卫生数据
# ======================================================================
def download_who_data():
    """下载 WHO 全球卫生统计数据"""
    log("=" * 60)
    log("开始下载 WHO 全球卫生统计数据")
    log("=" * 60)

    dest = DATA_RAW / "who"
    dest.mkdir(exist_ok=True)

    # WHO GHO API — 获取几个关键指标
    WHO_INDICATORS = {
        "NCD_BMI_MEAN": "BMI均值-年龄标准化",
        "NCD_GLUC_MEAN": "空腹血糖均值",
        "NCD_BP_MEAN": "血压均值",
        "NCD_CHOL_MEAN": "胆固醇均值",
        "LIFE_0000000035": "预期寿命",
        "NCDMORT3070": "30-70岁NCD死亡率",
    }

    for code, desc in WHO_INDICATORS.items():
        try:
            url = (f"https://ghoapi.azureedge.net/api/{code}"
                   f"?$format=csv")
            log(f"下载: {desc} ({code})")
            fpath = dest / f"{code}.csv"
            resp = safe_request(url)
            if resp is None:
                record(f"who_{code}", "WHO GHO", url, "", 0, 0, 0, "请求失败")
                continue
            fpath.write_bytes(resp.content)
            # 替换BOM
            content = resp.content.decode("utf-8-sig")
            fpath.write_text(content, encoding="utf-8")
            df = pd.read_csv(fpath)
            record(f"who_{code}", "WHO GHO", url, str(fpath),
                   os.path.getsize(fpath), df.shape[0], df.shape[1])
        except Exception as e:
            log(f"WHO下载失败: {code} — {e}", "ERROR")
            record(f"who_{code}", "WHO GHO", url, "", 0, 0, 0, f"失败:{e}")

# ======================================================================
# 步骤 5: COVID-19 / 公共卫生时序数据
# ======================================================================
def download_covid_data():
    """下载 COVID-19 和全球健康时序数据"""
    log("=" * 60)
    log("开始下载公共卫生时序数据 (COVID-19 等)")
    log("=" * 60)

    dest = DATA_RAW / "covid_public_health"
    dest.mkdir(exist_ok=True)

    # 5.1 JHU COVID-19 全球数据 (已归档)
    COVID_URLS = [
        ("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv",
         "covid19_confirmed_global"),
        ("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv",
         "covid19_deaths_global"),
    ]

    for url, name in COVID_URLS:
        try:
            log(f"下载: {name}")
            fpath = dest / f"{name}.csv"
            resp = safe_request(url)
            if resp is None:
                record(name, "JHU COVID-19", url, "", 0, 0, 0, "请求失败")
                continue
            fpath.write_bytes(resp.content)
            df = pd.read_csv(fpath)
            record(name, "JHU COVID-19", url, str(fpath),
                   os.path.getsize(fpath), df.shape[0], df.shape[1])
        except Exception as e:
            log(f"失败: {name} — {e}", "WARN")
            record(name, "JHU COVID-19", url, "", 0, 0, 0, f"失败:{e}")

    # 5.2 OWID 全球健康数据 (含多指标)
    try:
        owid_url = "https://catalog.ourworldindata.org/garden/health/latest/global_health_indicators/global_health_indicators.csv"
        log("下载: Our World in Data 全球健康指标")
        fpath = dest / "owid_global_health.csv"
        resp = safe_request(owid_url)
        if resp:
            fpath.write_bytes(resp.content)
            df = pd.read_csv(fpath)
            record("owid_global_health", "OWID", owid_url, str(fpath),
                   os.path.getsize(fpath), df.shape[0], df.shape[1])
    except Exception as e:
        log(f"OWID 下载失败: {e}", "WARN")

# ======================================================================
# 步骤 6: 生成数据清单报告
# ======================================================================
def generate_report():
    """生成数据获取报告"""
    log("=" * 60)
    log("生成数据获取报告")
    log("=" * 60)

    # 统计数据总量
    total_size = sum(s["size_kb"] for s in SUMMARY if s["status"] == "成功")
    total_files = len(SUMMARY)
    success = sum(1 for s in SUMMARY if s["status"] == "成功")
    failed  = sum(1 for s in SUMMARY if s["status"] != "成功")

    report_text = f"""# 数据获取报告

**生成时间:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 汇总统计

| 指标 | 数值 |
|------|------|
| 总计数据源 | {total_files} |
| 成功获取 | {success} |
| 失败/部分 | {failed} |
| 总数据量 | {total_size/1024:.1f} MB |

## 详细清单

| 数据集 | 来源 | 行数 | 列数 | 大小(KB) | 状态 |
|--------|------|------|------|----------|------|
"""
    for s in SUMMARY:
        report_text += f"| {s['dataset']} | {s['source']} | {s['rows']} | {s['cols']} | {s['size_kb']} | {s['status']} |\n"

    report_text += f"""
## 文件目录结构

```
data/raw/
"""
    for root, dirs, files in os.walk(DATA_RAW):
        level = root.replace(str(DATA_RAW), "").count(os.sep)
        indent = "  " * (level + 1)
        folder = os.path.basename(root)
        if folder != "raw":
            report_text += f"{indent}{folder}/\n"
        for f in sorted(files)[:5]:  # 每目录最多显示5个文件
            size = os.path.getsize(os.path.join(root, f))
            report_text += f"{indent}  {f} ({size/1024:.1f} KB)\n"
        if len(files) > 5:
            report_text += f"{indent}  ... 及其他 {len(files)-5} 个文件\n"

    report_text += "```\n\n"
    report_text += "## 下一步操作\n\n"
    report_text += "1. 运行 `data_cleaning.py` 进行数据清洗\n"
    report_text += "2. 运行 `data_fusion.py` 进行多模态融合\n"
    report_text += "3. 运行模型训练脚本\n"

    report_path = DATA_RAW.parent / "data_acquisition_report.md"
    report_path.write_text(report_text, encoding="utf-8")
    log(f"报告已保存: {report_path}")

    # 同时保存 JSON
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(SUMMARY, f, ensure_ascii=False, indent=2)
    log(f"JSON日志: {LOG_FILE}")

    return total_size, success, failed

# ======================================================================
# 主流程
# ======================================================================
def main():
    log("╔══════════════════════════════════════════════════════════╗")
    log("║   区域健康风险预测 — 多源医疗数据批量获取              ║")
    log("╚══════════════════════════════════════════════════════════╝")

    os.chdir(Path(__file__).parent.parent)  # 回到项目根目录
    log(f"工作目录: {os.getcwd()}")
    log(f"数据目录: {DATA_RAW.resolve()}")

    # ---- 依次下载 ----
    try:
        download_kaggle_datasets()      # Kaggle (kagglehub)
    except Exception as e:
        log(f"Kaggle 阶段异常: {e}", "ERROR")

    try:
        download_uci_datasets()         # UCI ML Repository
    except Exception as e:
        log(f"UCI 阶段异常: {e}", "ERROR")

    try:
        download_nhanes_data()          # CDC NHANES + BRFSS
    except Exception as e:
        log(f"CDC 阶段异常: {e}", "ERROR")

    try:
        download_who_data()             # WHO GHO
    except Exception as e:
        log(f"WHO 阶段异常: {e}", "ERROR")

    try:
        download_covid_data()           # COVID-19 / OWID
    except Exception as e:
        log(f"COVID 阶段异常: {e}", "ERROR")

    # ---- 生成报告 ----
    total_size, success, failed = generate_report()

    log("=" * 60)
    log(f"数据获取完成! 成功: {success}, 失败: {failed}, "
        f"总数据量: {total_size/1024:.1f} MB")
    log(f"所有数据位于: {DATA_RAW.resolve()}")
    log(f"查看报告: data/data_acquisition_report.md")
    log("=" * 60)

if __name__ == "__main__":
    main()
