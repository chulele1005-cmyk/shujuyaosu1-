# -*- coding: utf-8 -*-
"""
==============================================================================
数据清洗与标准化脚本 — 区域健康风险预测与智能诊疗决策
功能: 对原始医疗数据执行缺失值处理、异常值检测、特征标准化
输入: data/raw/ 目录下的 CSV/XPT/JSON 文件
输出: data/processed/ 目录下的 .parquet 清洗后文件
==============================================================================
"""

import os, sys, json, warnings
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OrdinalEncoder
from scipy import stats

warnings.filterwarnings("ignore")

# ========================= 全局配置 =========================
DATA_RAW      = Path("./data/raw")
DATA_PROCESSED = Path("./data/processed")
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

MISSING_THRESHOLD = 0.4       # 缺失率 > 40% 的列直接删除
OUTLIER_METHOD    = "iqr"     # 异常检测方法: iqr / zscore / isolation_forest
IMPUTE_METHOD     = "mice"    # 插补方法: median / knn / mice
SCALING_METHOD    = "standard" # 标准化: standard / minmax

RANDOM_STATE = 42
CLEANING_LOG = []             # 清洗日志
PROCESSED_SUMMARY = []        # 处理汇总


def log(msg, level="INFO"):
    """打印带时间戳的日志"""
    ts = datetime.now().strftime('%H:%M:%S')
    print(f"[{ts}][{level}] {msg}")


def auto_read_file(filepath):
    """
    智能读取文件，自动检测分隔符、编码和格式。
    支持 CSV (含多分隔符)、XPT (SAS)、JSON、Parquet、.data 文件。
    """
    ext = filepath.suffix.lower()
    fname = filepath.name.lower()

    # XPT (SAS Transport) 文件
    if ext == ".xpt":
        try:
            df = pd.read_sas(filepath)
            log(f"  [SAS XPT] 读取成功: {df.shape}")
            return df
        except Exception as e:
            log(f"  [SAS XPT] 读取失败: {e}", "WARN")
            return None

    # JSON 文件
    if ext == ".json":
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            # 尝试展开为 DataFrame
            if "value" in data:
                df = pd.json_normalize(data["value"])
            elif isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                df = pd.json_normalize(data)
            log(f"  [JSON] 读取成功: {df.shape}")
            return df
        except Exception as e:
            log(f"  [JSON] 读取失败: {e}", "WARN")
            return None

    # .data 文件 (UCI 格式)
    if ext == ".data":
        try:
            df = pd.read_csv(filepath, header=None)
            log(f"  [.data] 读取成功: {df.shape}")
            return df
        except Exception as e:
            log(f"  [.data] 读取失败: {e}", "WARN")
            return None

    # ZIP 文件 — 跳过
    if ext == ".zip":
        log(f"  [ZIP] 跳过 (不解压)", "INFO")
        return None

    # CSV — 尝试多种配置
    # 先读第一行探测分隔符
    try:
        with open(filepath, "r", encoding="utf-8-sig") as f:
            first_line = f.readline()
        # 检测分隔符
        semicolons = first_line.count(";")
        commas     = first_line.count(",")
        tabs       = first_line.count("\t")
        if semicolons > commas and semicolons > tabs:
            sep = ";"
        elif tabs > commas:
            sep = "\t"
        else:
            sep = ","
    except UnicodeDecodeError:
        sep = ","  # 回退
        try:
            with open(filepath, "r", encoding="latin-1") as f:
                first_line = f.readline()
            semicolons = first_line.count(";")
            commas     = first_line.count(",")
            if semicolons > commas:
                sep = ";"
        except:
            sep = ","

    # 读取 CSV
    encodings = ["utf-8", "utf-8-sig", "latin-1", "iso-8859-1", "cp1252"]
    for enc in encodings:
        try:
            df = pd.read_csv(filepath, sep=sep, encoding=enc, low_memory=False)
            log(f"  [CSV] 分隔符='{sep}', 编码={enc}, shape={df.shape}")
            return df
        except UnicodeDecodeError:
            continue
        except Exception as e:
            if enc == encodings[-1]:
                log(f"  [CSV] 全部编码尝试失败: {e}", "WARN")
                return None
            continue

    return None


# ======================================================================
# 步骤 1: 缺失值处理
# ======================================================================
def handle_missing(df, threshold=MISSING_THRESHOLD, method=IMPUTE_METHOD):
    """
    缺失值处理策略:
    1. 缺失率 > threshold 的列 → 删除
    2. 其余数值列 → MICE/KNN/中位数插补
    3. 分类列 → 众数填充
    """
    n_before = len(df.columns)
    missing_ratio = df.isnull().mean()

    # 删除高缺失率列
    cols_to_drop = missing_ratio[missing_ratio > threshold].index.tolist()
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        log(f"  删除高缺失列 ({len(cols_to_drop)}): {cols_to_drop}")

    # 删除全空的行
    df = df.dropna(how="all")

    # 分离数值列和分类列
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols     = df.select_dtypes(exclude=[np.number]).columns.tolist()

    # 数值列插补
    if numeric_cols and df[numeric_cols].isnull().sum().sum() > 0:
        n_missing = df[numeric_cols].isnull().sum().sum()
        log(f"  数值列缺失值: {n_missing} 个")

        if method == "median":
            imputer = SimpleImputer(strategy="median")
            df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
        elif method == "knn":
            # KNN 插补 — 对于小数据集
            if len(df) < 50000:
                imputer = KNNImputer(n_neighbors=5)
                df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
            else:
                # 大数据集用中位数回退
                imputer = SimpleImputer(strategy="median")
                df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
                log(f"  大数据集(>{len(df)}行)，使用中位数插补替代KNN")
        elif method == "mice":
            # MICE (多重插补) — 对中等大小数据集
            if len(df) < 10000:
                imputer = IterativeImputer(max_iter=10, random_state=RANDOM_STATE)
                df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
            else:
                imputer = SimpleImputer(strategy="median")
                df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
                log(f"  大数据集(>{len(df)}行)，使用中位数插补替代MICE")

    # 分类列 — 众数填充
    if cat_cols:
        for col in cat_cols:
            if df[col].isnull().any():
                mode_val = df[col].mode()
                if len(mode_val) > 0:
                    df[col] = df[col].fillna(mode_val[0])
                else:
                    df[col] = df[col].fillna("Unknown")

    # 最终确认无缺失
    remaining = df.isnull().sum().sum()
    log(f"  缺失值处理完成: {n_before}→{len(df.columns)}列, 剩余缺失: {remaining}")

    return df


# ======================================================================
# 步骤 2: 异常值处理
# ======================================================================
def handle_outliers(df, method=OUTLIER_METHOD):
    """异常值处理 — Winsorize 截尾法"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    outliers_count = 0

    for col in numeric_cols:
        col_data = df[col].dropna()
        if len(col_data) < 10:
            continue

        if method == "iqr":
            Q1, Q3 = col_data.quantile(0.25), col_data.quantile(0.75)
            IQR = Q3 - Q1
            if IQR == 0:
                continue
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            n_out = ((df[col] < lower) | (df[col] > upper)).sum()
            if n_out > 0:
                df[col] = df[col].clip(lower, upper)
                outliers_count += n_out

        elif method == "zscore":
            mean, std = col_data.mean(), col_data.std()
            if std == 0:
                continue
            lower = mean - 3 * std
            upper = mean + 3 * std
            n_out = ((df[col] < lower) | (df[col] > upper)).sum()
            if n_out > 0:
                df[col] = df[col].clip(lower, upper)
                outliers_count += n_out

    log(f"  异常值处理: {outliers_count} 个值被截尾")
    return df


# ======================================================================
# 步骤 3: 分类变量编码
# ======================================================================
def encode_categorical(df):
    """
    将分类变量转换为数值:
    - 二分类 → LabelEncoder (0/1)
    - 多分类低基数 (≤10) → One-Hot
    - 多分类高基数 (>10) → OrdinalEncoder
    """
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if not cat_cols:
        return df

    for col in cat_cols:
        n_unique = df[col].nunique()
        if n_unique <= 1:
            df = df.drop(columns=[col])
        elif n_unique == 2:
            # 二分类 — Label Encoding
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))
        elif n_unique <= 10:
            # 低基数 — One-Hot (带 drop_first 避免共线性)
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
        else:
            # 高基数 — Ordinal Encoding
            df[col] = OrdinalEncoder(handle_unknown="use_encoded_value",
                                      unknown_value=-1).fit_transform(
                df[[col]].astype(str))

    log(f"  分类编码完成: 当前形状 {df.shape}")
    return df


# ======================================================================
# 步骤 4: 特征标准化
# ======================================================================
def normalize_features(df, method=SCALING_METHOD):
    """数值特征标准化"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        return df, None

    if method == "standard":
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()

    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    log(f"  {method} 标准化完成: {len(numeric_cols)} 个数值列")

    return df, scaler


# ======================================================================
# 步骤 5: 特殊数据集处理
# ======================================================================
def fix_special_datasets(name, df):
    """
    针对特定数据集的修复:
    - cardio_disease: 分号分隔的 CSV，需重新解析
    - breast_cancer: 删除无用 ID 列
    - stroke_prediction: 删除 id 列
    - heart_cleveland/hungarian: 补充列名
    - nhanes: 合并多文件
    """
    if df is None:
        return None

    # 统一列名小写
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    # 删除无用列
    cols_to_drop = []
    for col in df.columns:
        if col in ["id", "unnamed:_0", "unnamed: 0", "unnamed:_32"]:
            cols_to_drop.append(col)
        # 删除全为 NaN 的列
        if df[col].isnull().all():
            cols_to_drop.append(col)
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    # 删除重复行
    n_dup = df.duplicated().sum()
    if n_dup > 0:
        df = df.drop_duplicates()
        log(f"  删除重复行: {n_dup}")

    return df


# ======================================================================
# 步骤 6: NHANES 多文件合并
# ======================================================================
def merge_nhanes_files(nhanes_dir):
    """将 NHANES 的 DEMO/BMX/BPX 按 SEQN 合并"""
    nhanes_dir = Path(nhanes_dir)
    xpt_files = sorted(nhanes_dir.glob("*.XPT"))
    if len(xpt_files) < 2:
        return None

    merged = None
    for fpath in xpt_files:
        try:
            df = pd.read_sas(fpath)
            # 统一 SEQN 列名
            seqn_col = None
            for c in df.columns:
                if c.upper() == "SEQN":
                    seqn_col = c
                    break
            if seqn_col is None:
                log(f"  NHANES {fpath.name}: 无 SEQN 列, 跳过", "WARN")
                continue

            suffix = fpath.stem.split("_")[0]
            df = df.add_suffix(f"_{suffix}").rename(
                columns={f"{seqn_col}_{suffix}": "seqn"})

            if merged is None:
                merged = df
            else:
                merged = merged.merge(df, on="seqn", how="outer")
        except Exception as e:
            log(f"  NHANES 读取失败 {fpath.name}: {e}", "WARN")

    if merged is not None:
        log(f"  NHANES 合并完成: {merged.shape}")

    return merged


# ======================================================================
# 主清洗流水线
# ======================================================================
def clean_single_file(filepath, output_dir):
    """清洗单个文件并保存"""
    fname = filepath.stem
    parent = filepath.parent.name
    safe_name = f"{parent}__{fname}" if parent != "raw" else fname
    out_path = output_dir / f"{safe_name}_cleaned.parquet"

    log(f"\n{'='*50}")
    log(f"清洗: {filepath.relative_to(filepath.parents[2])}")
    log(f"{'='*50}")

    # 读取
    df = auto_read_file(filepath)
    if df is None or len(df) == 0:
        CLEANING_LOG.append({"file": str(filepath), "status": "跳过", "reason": "空文件/不可读"})
        return None

    # 特殊修复
    df = fix_special_datasets(fname, df)
    if df is None or len(df.columns) == 0:
        CLEANING_LOG.append({"file": str(filepath), "status": "跳过", "reason": "无有效列"})
        return None

    original_shape = df.shape

    # 清洗步骤
    df = handle_missing(df)
    df = handle_outliers(df)
    df = encode_categorical(df)
    df, scaler = normalize_features(df)

    # 保存
    df.to_parquet(out_path, index=False)
    size_kb = os.path.getsize(out_path) / 1024

    # 验证
    missing_rate = df.isnull().mean().max() if len(df.columns) > 0 else 0
    n_dup = df.duplicated().sum()

    log(f"  保存: {out_path.name} ({size_kb:.1f} KB)")
    log(f"  结果: {original_shape} → {df.shape}, 缺失率={missing_rate:.4f}, 重复={n_dup}")

    CLEANING_LOG.append({
        "file": str(filepath),
        "status": "成功",
        "original_shape": str(original_shape),
        "final_shape": str(df.shape),
        "missing_rate": round(missing_rate, 4),
        "duplicates": n_dup,
        "size_kb": round(size_kb, 1),
    })

    PROCESSED_SUMMARY.append({
        "name": safe_name,
        "rows": df.shape[0],
        "cols": df.shape[1],
        "missing_rate": missing_rate,
        "size_kb": size_kb,
    })

    return df


def clean_all():
    """遍历 data/raw/ 下所有文件并清洗"""
    log("╔══════════════════════════════════════════════════════════╗")
    log("║   医疗数据清洗与标准化流水线                           ║")
    log("╚══════════════════════════════════════════════════════════╝")
    log(f"配置: 缺失阈值={MISSING_THRESHOLD}, 异常检测={OUTLIER_METHOD}, "
        f"插补={IMPUTE_METHOD}, 标准化={SCALING_METHOD}")

    # 待处理的文件列表
    files_to_process = []
    for root, dirs, files in os.walk(DATA_RAW):
        root_path = Path(root)
        for f in files:
            fpath = root_path / f
            # 跳过特殊文件
            if f.endswith((".zip", ".jpeg", ".png", ".jpg", ".log", ".json")):
                continue
            if f == "download_log.json":
                continue
            # 跳过已有子目录的同名文件
            if "__MACOSX" in str(fpath):
                continue
            files_to_process.append(fpath)

    log(f"\n找到 {len(files_to_process)} 个待处理文件\n")

    # 逐个清洗
    for fpath in sorted(files_to_process):
        try:
            clean_single_file(fpath, DATA_PROCESSED)
        except Exception as e:
            log(f"清洗失败 {fpath}: {e}", "ERROR")
            CLEANING_LOG.append({"file": str(fpath), "status": "失败", "reason": str(e)})

    # NHANES 特殊合并处理
    nhanes_dir = DATA_RAW / "nhanes"
    if nhanes_dir.exists() and list(nhanes_dir.glob("*.XPT")):
        try:
            log(f"\n{'='*50}")
            log("NHANES 多文件合并处理")
            log(f"{'='*50}")
            merged_nhanes = merge_nhanes_files(nhanes_dir)
            if merged_nhanes is not None:
                merged_nhanes = fix_special_datasets("nhanes_merged", merged_nhanes)
                merged_nhanes = handle_missing(merged_nhanes)
                merged_nhanes = handle_outliers(merged_nhanes)
                merged_nhanes = encode_categorical(merged_nhanes)
                merged_nhanes, _ = normalize_features(merged_nhanes)
                out_path = DATA_PROCESSED / "nhanes__merged_cleaned.parquet"
                merged_nhanes.to_parquet(out_path, index=False)
                log(f"NHANES 合并结果保存: {out_path} ({merged_nhanes.shape})")
                CLEANING_LOG.append({
                    "file": "nhanes_merged", "status": "成功",
                    "original_shape": "N/A",
                    "final_shape": str(merged_nhanes.shape),
                    "missing_rate": round(merged_nhanes.isnull().mean().max(), 4),
                    "duplicates": merged_nhanes.duplicated().sum(),
                    "size_kb": round(os.path.getsize(out_path)/1024, 1),
                })
        except Exception as e:
            log(f"NHANES 合并失败: {e}", "ERROR")

    return generate_report()


# ======================================================================
# 生成清洗报告
# ======================================================================
def generate_report():
    """生成数据清洗报告"""
    log("\n" + "=" * 60)
    log("生成清洗报告")
    log("=" * 60)

    success = sum(1 for l in CLEANING_LOG if l["status"] == "成功")
    failed  = sum(1 for l in CLEANING_LOG if l["status"] == "失败")
    skipped = sum(1 for l in CLEANING_LOG if l["status"] == "跳过")

    total_rows = sum(s.get("rows", 0) for s in PROCESSED_SUMMARY)
    total_cols = sum(s.get("cols", 0) for s in PROCESSED_SUMMARY)

    report = f"""# 数据清洗与标准化报告

**生成时间:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**处理配置:** 缺失阈值={MISSING_THRESHOLD}, 异常检测={OUTLIER_METHOD}, 插补={IMPUTE_METHOD}, 标准化={SCALING_METHOD}

## 汇总统计

| 指标 | 数值 |
|------|------|
| 处理文件总数 | {len(CLEANING_LOG)} |
| 成功 | {success} |
| 失败 | {failed} |
| 跳过 | {skipped} |
| 累计数据行数 | {total_rows:,} |
| 累计特征数 | {total_cols} |

## 详细处理日志

| 文件 | 状态 | 原始形状 | 最终形状 | 缺失率 | 重复数 | 大小(KB) |
|------|------|----------|----------|--------|--------|----------|
"""
    for entry in CLEANING_LOG:
        report += (f"| {Path(entry['file']).name} | {entry['status']} | "
                   f"{entry.get('original_shape','-')} | {entry.get('final_shape','-')} | "
                   f"{entry.get('missing_rate','-')} | {entry.get('duplicates','-')} | "
                   f"{entry.get('size_kb','-')} |\n")

    report += f"""

## 清洗后数据集速览

| 数据集 | 行数 | 列数 | 大小(KB) |
|--------|------|------|----------|
"""
    for s in sorted(PROCESSED_SUMMARY, key=lambda x: x["rows"], reverse=True):
        report += f"| {s['name']} | {s['rows']:,} | {s['cols']} | {s['size_kb']} |\n"

    report += """
## 清洗质量验证

| 检验项 | 目标 | 实际 |
|--------|------|------|
"""
    avg_missing = np.mean([s["missing_rate"] for s in PROCESSED_SUMMARY]) if PROCESSED_SUMMARY else 0
    report += f"| 平均缺失率 | < 5% | {avg_missing:.2%} |\n"

    report += """
## 下一步操作

清洗后的数据已保存至 `data/processed/`，继续执行:
1. `data_fusion.py` — 多模态特征融合
2. `privacy_protection.py` — 隐私保护处理
"""

    report_path = DATA_PROCESSED.parent / "data_cleaning_report.md"
    report_path.write_text(report, encoding="utf-8")
    log(f"报告已保存: {report_path}")

    # JSON 日志
    json_path = DATA_PROCESSED / "cleaning_log.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(CLEANING_LOG, f, ensure_ascii=False, indent=2)

    return success, failed, skipped


# ======================================================================
# 主入口
# ======================================================================
def main():
    os.chdir(Path(__file__).parent.parent)  # 回到项目根目录
    log(f"工作目录: {os.getcwd()}")
    log(f"原始数据: {DATA_RAW.resolve()}")
    log(f"输出目录: {DATA_PROCESSED.resolve()}")

    success, failed, skipped = clean_all()

    log("\n" + "=" * 60)
    log(f"数据清洗完成! 成功: {success}, 失败: {failed}, 跳过: {skipped}")
    log(f"清洗后数据位于: {DATA_PROCESSED.resolve()}")
    log(f"查看报告: data/data_cleaning_report.md")
    log("=" * 60)


if __name__ == "__main__":
    main()
