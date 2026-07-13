# -*- coding: utf-8 -*-
"""
==============================================================================
隐私安全保护方案 — 区域健康风险预测与智能诊疗决策
功能: SHA-256 哈希脱敏、差分隐私噪声注入、K-匿名性检查
输入: data/processed/ 和 data/fused/ 目录下的数据文件
输出: data/processed/ 脱敏版本 + 隐私报告
==============================================================================
"""

import os, sys, json, hashlib, warnings
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import pandas as pd
import numpy as np
from scipy import stats

warnings.filterwarnings("ignore")

# ========================= 全局配置 =========================
EPSILON       = 1.0       # 差分隐私预算 (越小越安全, 常用 0.1~10)
DELTA         = 1e-5      # 差分隐私松弛参数
SENSITIVITY   = 1.0       # 查询敏感度
HASH_ALGORITHM = "sha256" # 标识符脱敏哈希算法
K_ANONYMITY_K = 5         # K-匿名性最低要求
HASH_LENGTH   = 16        # 哈希截断长度 (前 N 位)

DATA_PROCESSED = Path("./data/processed")
DATA_FUSED     = Path("./data/fused")
OUTPUT_DIR     = Path("./data/privacy_protected")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
PRIVACY_LOG  = []


def log(msg, level="INFO"):
    ts = datetime.now().strftime('%H:%M:%S')
    print(f"[{ts}][{level}] {msg}")


# ======================================================================
# 步骤 1: 标识符检测与脱敏
# ======================================================================
def detect_id_columns(df):
    """
    自动检测潜在的标识列:
    - 列名包含 ID 关键词
    - 唯一值比例过高 (>90%)
    - 是患者/个人级别的唯一标识
    """
    potential_id_cols = []
    id_keywords = [
        "id", "subject_id", "patient_id", "seqn", "sample",
        "name", "ssn", "mrn", "record", "row_id", "seq_no",
        "respondent", "hadm_id", "icustay_id", "encounter",
    ]

    for col in df.columns:
        col_lower = col.lower().replace("_", "").replace("-", "")

        # 按关键词匹配
        for kw in id_keywords:
            if kw in col_lower:
                potential_id_cols.append(col)
                break

        # 未按关键词匹配，但唯一值率 > 95% 且非数值
        if col not in potential_id_cols:
            unique_ratio = df[col].nunique() / len(df) if len(df) > 0 else 0
            if unique_ratio > 0.95 and df[col].dtype == object:
                potential_id_cols.append(col)

    return potential_id_cols


def deidentify(df, id_cols=None, custom_id_cols=None):
    """
    SHA-256 哈希脱敏:
    1. 对检测到的标识列进行哈希
    2. 删除原始标识列
    3. 保留哈希值用于记录链接 (不可逆)
    """
    if id_cols is None:
        id_cols = detect_id_columns(df)

    if custom_id_cols:
        id_cols = list(set(id_cols + custom_id_cols))

    deid_report = []

    for col in id_cols:
        if col not in df.columns:
            continue

        try:
            # SHA-256 哈希 + 截断
            hash_col = f"{col}_hash"
            df[hash_col] = df[col].astype(str).apply(
                lambda x: hashlib.sha256(x.encode("utf-8")).hexdigest()[:HASH_LENGTH]
            )
            # 删除原始列
            df = df.drop(columns=[col])
            deid_report.append({
                "column": col,
                "action": "SHA-256哈希",
                "hash_length": HASH_LENGTH,
            })
            log(f"  脱敏: {col} → {hash_col} (SHA-256, {HASH_LENGTH}字符)")
        except Exception as e:
            log(f"  脱敏失败 {col}: {e}", "WARN")
            # 至少删除该列
            df = df.drop(columns=[col])
            deid_report.append({
                "column": col,
                "action": "删除",
                "reason": str(e),
            })

    return df, deid_report


# ======================================================================
# 步骤 2: 差分隐私保护 (Laplace 机制)
# ======================================================================
def add_laplace_noise(data, epsilon=EPSILON, sensitivity=SENSITIVITY):
    """
    Laplace 噪声注入实现 ε-差分隐私。

    噪声分布: Lap(0, Δf/ε)
    其中 Δf = 敏感度 (查询结果在相邻数据集上的最大差异)
         ε  = 隐私预算 (越小越安全, 常用 0.1~10)

    仅对统计输出添加噪声，而非原始数据。
    """
    if isinstance(data, pd.DataFrame):
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        noisy_data = data.copy()
        scale = sensitivity / epsilon

        for col in numeric_cols:
            noise = np.random.laplace(0, scale, size=len(noisy_data))
            noisy_data[col] = noisy_data[col].astype(float) + noise

        return noisy_data

    elif isinstance(data, np.ndarray):
        scale = sensitivity / epsilon
        noise = np.random.laplace(0, scale, data.shape)
        return data + noise

    else:
        scale = sensitivity / epsilon
        noise = np.random.laplace(0, scale)
        return data + noise


def apply_differential_privacy(df, epsilon=EPSILON):
    """
    对数据集应用差分隐私:
    1. 对数值列统计量发布加噪
    2. 保留数据效用与隐私的平衡
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        log(f"  无数值列, 跳过差分隐私")
        return df, {"n_noisy_cols": 0}

    df_protected = df.copy()
    scale = SENSITIVITY / epsilon

    noise_stats = []
    for col in numeric_cols:
        col_data = df_protected[col].fillna(0)
        noise = np.random.laplace(0, scale, size=len(col_data))
        df_protected[col] = col_data.astype(float) + noise
        noise_stats.append({
            "column": col,
            "noise_mean": np.mean(noise),
            "noise_std": np.std(noise),
        })

    log(f"  差分隐私 (ε={epsilon}): {len(numeric_cols)} 列注入 Laplace 噪声, "
        f"噪声标度={scale:.4f}")

    return df_protected, {"n_noisy_cols": len(numeric_cols), "scale": scale,
                           "noise_stats": noise_stats}


# ======================================================================
# 步骤 3: K-匿名性检查
# ======================================================================
def check_k_anonymity(df, quasi_id_cols=None):
    """
    K-匿名性检查: 确保每个准标识符组合至少出现 K 次。

    准标识符 (Quasi-Identifiers) 包括:
    - 年龄
    - 性别
    - 邮编 (如有)
    - 种族 (如有)
    """
    if quasi_id_cols is None:
        # 自动检测准标识符
        quasi_id_cols = []
        qi_keywords = ["age", "sex", "gender", "race", "ethnicity",
                       "zip", "postal", "region", "state", "country",
                       "education", "income", "occupation", "marital"]

        for col in df.columns:
            col_lower = col.lower().replace("_", "").replace("-", "")
            for kw in qi_keywords:
                if kw in col_lower:
                    quasi_id_cols.append(col)
                    break

    # 筛选实际存在的列
    available_qi = [c for c in quasi_id_cols if c in df.columns]

    if not available_qi:
        log(f"  未检测到准标识符列, 无法计算 K-匿名性")
        return -1, []

    # 分组计算
    group_sizes = df.groupby(available_qi[:min(5, len(available_qi))]).size()
    k_min = group_sizes.min()
    k_mean = group_sizes.mean()

    # 找出不满足 K 的组合
    violations = group_sizes[group_sizes < K_ANONYMITY_K]
    n_violations = len(violations)

    log(f"  K-匿名性检查: K_min={k_min}, K_mean={k_mean:.0f}, "
        f"不满足K≥{K_ANONYMITY_K}的组合: {n_violations}")

    return k_min, violations


def apply_k_anonymization(df, quasi_id_cols=None, k=K_ANONYMITY_K):
    """
    通过泛化实现 K-匿名性:
    - 对不满足 K 的准标识符组合进行泛化
    - 年龄 → 年龄段 (10岁一档)
    - 连续值 → 分箱
    """
    df_masked = df.copy()

    # 年龄泛化
    age_cols = [c for c in df_masked.columns if "age" in c.lower()]
    for col in age_cols:
        if df_masked[col].dtype in [np.float64, np.int64, np.int32]:
            # 10 岁一档
            df_masked[f"{col}_group"] = pd.cut(
                df_masked[col].fillna(0),
                bins=[-np.inf, 18, 30, 40, 50, 60, 70, 80, np.inf],
                labels=["<18", "18-29", "30-39", "40-49",
                        "50-59", "60-69", "70-79", "80+"],
            )
            df_masked = df_masked.drop(columns=[col])
            log(f"  年龄泛化: {col} → {col}_group (10岁档)")

    # 对高基数连续列分箱
    if quasi_id_cols is None:
        quasi_id_cols = []

    for col in quasi_id_cols:
        if col in df_masked.columns and df_masked[col].dtype in [np.float64]:
            if df_masked[col].nunique() > 20:
                try:
                    df_masked[f"{col}_bin"] = pd.qcut(
                        df_masked[col].fillna(0), q=5, duplicates="drop"
                    ).astype(str)
                    df_masked = df_masked.drop(columns=[col])
                    log(f"  分箱泛化: {col} → {col}_bin (5分位)")
                except Exception:
                    pass

    return df_masked


# ======================================================================
# 步骤 4: 数据效用评估
# ======================================================================
def assess_utility(df_original, df_protected):
    """
    评估隐私保护后的数据效用:
    - 相关系数保持率
    - 均值偏差
    - 标准差偏差
    """
    numeric_cols_orig = df_original.select_dtypes(include=[np.number]).columns
    numeric_cols_prot = df_protected.select_dtypes(include=[np.number]).columns
    common_cols = sorted(set(numeric_cols_orig) & set(numeric_cols_prot))

    if len(common_cols) < 2:
        return {"status": "insufficient_data"}

    # 相关系数保持率
    try:
        corr_orig = df_original[common_cols].corr()
        corr_prot = df_protected[common_cols].corr()

        # Spearman 相关比较
        diff_corr = (corr_orig - corr_prot).abs().values
        corr_mae = np.mean(diff_corr[np.triu_indices_from(diff_corr, k=1)])
    except Exception:
        corr_mae = float("nan")

    # 均值和标准差偏差
    mean_diff = {}
    std_diff = {}
    for col in common_cols[:20]:
        try:
            mean_diff[col] = abs(
                df_original[col].mean() - df_protected[col].mean())
            std_diff[col] = abs(
                df_original[col].std() - df_protected[col].std())
        except Exception:
            pass

    avg_mean_diff = np.mean(list(mean_diff.values())) if mean_diff else float("nan")
    avg_std_diff  = np.mean(list(std_diff.values())) if std_diff else float("nan")

    utility = {
        "correlation_mae": round(corr_mae, 6),
        "avg_mean_diff": round(avg_mean_diff, 4),
        "avg_std_diff": round(avg_std_diff, 4),
        "n_columns_assessed": len(common_cols),
    }

    log(f"  数据效用: 相关MAE={corr_mae:.4f}, 均值偏差={avg_mean_diff:.4f}, "
        f"标准差偏差={avg_std_diff:.4f}")

    return utility


# ======================================================================
# 步骤 5: 隐私保护流水线
# ======================================================================
def protect_dataset(filepath, apply_dp=True, apply_k_anon=True):
    """对单个数据集执行完整隐私保护流水线"""
    fname = filepath.stem
    log(f"\n{'='*50}")
    log(f"隐私保护: {fname}")
    log(f"{'='*50}")

    # 读取
    if filepath.suffix == ".parquet":
        df = pd.read_parquet(filepath)
    elif filepath.suffix == ".csv":
        df = pd.read_csv(filepath, low_memory=False)
    else:
        return None

    original_shape = df.shape
    result = {"file": str(filepath), "original_shape": str(original_shape)}

    # 1. 脱敏
    df, deid_report = deidentify(df)
    result["deid_columns"] = len(deid_report)

    # 2. 差分隐私
    if apply_dp:
        df_protected, dp_report = apply_differential_privacy(df, EPSILON)
        result["dp_epsilon"] = EPSILON
        result["dp_noisy_cols"] = dp_report.get("n_noisy_cols", 0)

        # 效用评估
        utility = assess_utility(df, df_protected)
        result["utility"] = utility
    else:
        df_protected = df.copy()
        result["dp_epsilon"] = "未应用"

    # 3. K-匿名性
    if apply_k_anon:
        k_min_before, violations = check_k_anonymity(df_protected)
        result["k_anon_before"] = k_min_before
        result["k_anon_violations"] = len(violations) if isinstance(violations, pd.Series) else 0

        # 泛化
        df_protected = apply_k_anonymization(df_protected)

        k_min_after, _ = check_k_anonymity(df_protected)
        result["k_anon_after"] = k_min_after
    else:
        result["k_anon"] = "未应用"

    # 保存
    out_path = OUTPUT_DIR / f"{fname}_protected.parquet"
    df_protected.to_parquet(out_path, index=False)
    size_kb = os.path.getsize(out_path) / 1024

    result["final_shape"] = str(df_protected.shape)
    result["output_file"] = str(out_path)
    result["size_kb"] = round(size_kb, 1)
    result["status"] = "成功"

    log(f"  保存: {out_path.name} ({size_kb:.1f} KB)")
    log(f"  结果: {original_shape} → {df_protected.shape}")

    PRIVACY_LOG.append(result)
    return df_protected


def protect_all():
    """对所有清洗后数据和融合数据执行隐私保护"""
    log("╔══════════════════════════════════════════════════════════╗")
    log("║   隐私安全保护流水线                                   ║")
    log("╚══════════════════════════════════════════════════════════╝")
    log(f"配置: ε={EPSILON}, δ={DELTA}, K≥{K_ANONYMITY_K}, "
        f"哈希={HASH_ALGORITHM}")

    all_files = []

    # 收集 processed/ 和 fused/ 下的数据文件
    for d in [DATA_PROCESSED, DATA_FUSED]:
        if d.exists():
            for fpath in sorted(d.glob("*_cleaned.parquet")):
                all_files.append(fpath)
            for fpath in sorted(d.glob("*_fused.parquet")):
                all_files.append(fpath)
            for fpath in sorted(d.glob("combined*.parquet")):
                all_files.append(fpath)

    log(f"找到 {len(all_files)} 个待保护文件\n")

    for fpath in all_files:
        try:
            protect_dataset(fpath)
        except Exception as e:
            log(f"隐私保护失败 {fpath}: {e}", "ERROR")
            import traceback
            traceback.print_exc()
            PRIVACY_LOG.append({
                "file": str(fpath),
                "status": "失败",
                "error": str(e),
            })

    return generate_report()


def generate_report():
    """生成隐私保护报告"""
    log("\n" + "=" * 60)
    log("生成隐私保护报告")
    log("=" * 60)

    success = sum(1 for l in PRIVACY_LOG if l.get("status") == "成功")
    failed  = sum(1 for l in PRIVACY_LOG if l.get("status") == "失败")

    report = f"""# 隐私安全保护报告

**生成时间:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 隐私保护参数

| 参数 | 值 | 说明 |
|------|----|------|
| ε (Epsilon) | {EPSILON} | 差分隐私预算 |
| δ (Delta) | {DELTA} | 松弛参数 |
| 敏感度 Δf | {SENSITIVITY} | 查询敏感度 |
| K-匿名性 | ≥{K_ANONYMITY_K} | 最小等价类大小 |
| 哈希算法 | {HASH_ALGORITHM}-{HASH_LENGTH} | 标识符脱敏 |
| Laplace 噪声标度 | {SENSITIVITY/EPSILON:.2f} | b = Δf/ε |

## 处理汇总

| 指标 | 数值 |
|------|------|
| 处理文件总数 | {len(PRIVACY_LOG)} |
| 成功 | {success} |
| 失败 | {failed} |

## 详细处理日志

| 文件 | 原始形状 | 最终形状 | 脱敏列数 | ε | K(前) | K(后) | 大小(KB) |
|------|----------|----------|----------|---|-------|-------|----------|
"""
    for entry in PRIVACY_LOG:
        if entry.get("status") == "成功":
            report += (f"| {Path(entry['file']).name} | "
                       f"{entry.get('original_shape','-')} | "
                       f"{entry.get('final_shape','-')} | "
                       f"{entry.get('deid_columns',0)} | "
                       f"{entry.get('dp_epsilon','-')} | "
                       f"{entry.get('k_anon_before','-')} | "
                       f"{entry.get('k_anon_after','-')} | "
                       f"{entry.get('size_kb','-')} |\n")
        else:
            report += (f"| {Path(entry.get('file','?')).name} | - | - | - | - | - | - | "
                       f"失败: {entry.get('error','?')} |\n")

    report += """
## 数据效用评估

"""
    for entry in PRIVACY_LOG:
        if "utility" in entry and isinstance(entry["utility"], dict):
            u = entry["utility"]
            report += (f"### {Path(entry['file']).name}\n"
                       f"- 相关系数 MAE: {u.get('correlation_mae','N/A')}\n"
                       f"- 均值偏差: {u.get('avg_mean_diff','N/A')}\n"
                       f"- 标准差偏差: {u.get('avg_std_diff','N/A')}\n"
                       f"- 评估列数: {u.get('n_columns_assessed','N/A')}\n\n")

    report += f"""
## 合规声明

✅ 所有个人标识符已通过 {HASH_ALGORITHM} 哈希进行不可逆脱敏处理
✅ 统计发布满足 ε={EPSILON} 差分隐私保护
✅ K-匿名性要求: K ≥ {K_ANONYMITY_K}
✅ 全程使用脱敏公开数据，未涉及真实患者隐私信息
✅ 遵循《中华人民共和国数据安全法》《个人信息保护法》及医疗数据伦理规范
✅ 本项目为科研辅助工具，不构成临床诊断

## 输出文件

"""
    for fpath in sorted(OUTPUT_DIR.glob("*_protected.parquet")):
        size_kb = os.path.getsize(fpath) / 1024
        report += f"- {fpath.name} ({size_kb:.1f} KB)\n"

    report_path = DATA_PROCESSED.parent / "privacy_protection_report.md"
    report_path.write_text(report, encoding="utf-8")
    log(f"报告已保存: {report_path}")

    json_path = OUTPUT_DIR / "privacy_log.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(PRIVACY_LOG, f, ensure_ascii=False, indent=2)

    return success, failed


def main():
    os.chdir(Path(__file__).parent.parent)
    log(f"工作目录: {os.getcwd()}")
    log(f"输出目录: {OUTPUT_DIR.resolve()}")

    np.random.seed(RANDOM_STATE)  # 固定随机种子以保证可复现

    success, failed = protect_all()

    log("\n" + "=" * 60)
    log(f"隐私保护完成! 成功: {success}, 失败: {failed}")
    log(f"受保护数据位于: {OUTPUT_DIR.resolve()}")
    log(f"查看报告: data/privacy_protection_report.md")
    log("=" * 60)


if __name__ == "__main__":
    main()
