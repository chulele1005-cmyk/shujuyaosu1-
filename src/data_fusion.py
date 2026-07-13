# -*- coding: utf-8 -*-
"""
==============================================================================
多模态数据融合与特征工程 — 区域健康风险预测与智能诊疗决策
功能: 将清洗后的多源数据表进行融合、特征工程和降维
输入: data/processed/ 目录下的 .parquet 文件
输出: data/fused/ 目录下的融合特征矩阵
==============================================================================
"""

import os, sys, json, warnings
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

warnings.filterwarnings("ignore")

# ========================= 全局配置 =========================
DATA_PROCESSED = Path("./data/processed")
DATA_FUSED     = Path("./data/fused")
DATA_FUSED.mkdir(parents=True, exist_ok=True)

# 特征工程参数
FEATURE_SELECTION_K  = 50      # 互信息保留的 Top-K 特征数
PCA_COMPONENTS       = 30      # PCA 降维目标维度
USE_POLY_FEATURES    = True    # 是否构造多项式交互特征
POLY_DEGREE          = 2       # 多项式阶数
MAX_INTERACTIONS     = 100     # 最大交互特征数
VARIANCE_THRESHOLD   = 0.01    # 低方差过滤阈值

RANDOM_STATE = 42
FUSION_LOG = []


def log(msg, level="INFO"):
    ts = datetime.now().strftime('%H:%M:%S')
    print(f"[{ts}][{level}] {msg}")


# ======================================================================
# 步骤 1: 加载所有清洗后数据
# ======================================================================
def load_processed_datasets():
    """加载 data/processed/ 下所有清洗后的 parquet 文件"""
    datasets = {}
    for fpath in sorted(DATA_PROCESSED.glob("*_cleaned.parquet")):
        name = fpath.stem.replace("_cleaned", "")
        try:
            df = pd.read_parquet(fpath)
            datasets[name] = df
            log(f"加载: {fpath.name} ({df.shape[0]}行 × {df.shape[1]}列)")
        except Exception as e:
            log(f"加载失败 {fpath.name}: {e}", "ERROR")

    log(f"共加载 {len(datasets)} 个清洗后数据集")
    return datasets


# ======================================================================
# 步骤 2: 主题域分组 — 将相关数据集归并
# ======================================================================
def group_by_domain(datasets):
    """
    按分析主题将数据集分组:
    - chronic_disease: 慢性病相关 (BRFSS糖尿病/心脏病, 心血管, 中风, Pima)
    - clinical_risk: 临床风险因素 (心脏病, 乳腺癌, 宫颈癌)
    - population_health: 人群健康 (WHO, NHANES, BRFSS)
    - healthcare_ops: 医疗运营 (Healthcare Ops, Medical Insurance)
    - covid_timeseries: 疫情时序 (COVID India, OWID)
    """
    domains = {
        "chronic_disease": [
            "brfss_diabetes__diabetes_binary_health_indicators_BRFSS2015",
            "brfss_heart__heart_disease_health_indicators_BRFSS2015",
            "cardio_disease__cardio_train",
            "stroke_prediction__healthcare-dataset-stroke-data",
            "pima_diabetes__diabetes",
            "chronic_disease__chronic_disease_dataset",
        ],
        "clinical_risk": [
            "heart_disease__heart",
            "heart_cleveland__processed.cleveland",
            "heart_hungarian__processed.hungarian",
            "breast_cancer__data",
            "cervical_cancer__risk_factors_cervical_cancer",
        ],
        "population_health": [
            "who__NCD_BMI_MEAN", "who__LIFE_0000000035",
            "who__NCDMORT3070", "who__SA_0000001688",
            "nhanes__merged",
        ],
        "healthcare_ops": [
            "healthcare_ops__healthcare_dataset",
            "medical_insurance__insurance",
            "diabetes_130us__dataset_diabetes__diabetic_data",
        ],
        "covid_timeseries": [
            "covid_india__complete", "covid_india__patients_data",
            "covid_india__nation_level_daily", "covid_india__state_level_daily",
        ],
    }

    grouped = {}
    all_matched = set()
    for domain, keys in domains.items():
        matched = {k: datasets[k] for k in keys if k in datasets}
        if matched:
            grouped[domain] = matched
            all_matched.update(matched.keys())
            log(f"  [{domain}] 匹配 {len(matched)} 个数据集: {list(matched.keys())}")

    # 未匹配的数据集
    unmatched = [k for k in datasets if k not in all_matched]
    if unmatched:
        log(f"  未分组数据集 ({len(unmatched)}): {unmatched}")

    return grouped


# ======================================================================
# 步骤 3: 纵向合并 — 同主题域数据集
# ======================================================================
def vertical_merge(datasets_dict, domain_name):
    """
    纵向合并同一主题域的数据集。
    策略: 找出共同特征列，对齐后 concatenate。
    如果共同列太少 (< 3)，则分别保留各数据集的特征。
    """
    if len(datasets_dict) == 1:
        name, df = list(datasets_dict.items())[0]
        log(f"  [{domain_name}] 单数据集, 直接使用: {name} ({df.shape})")
        return df, {"method": "single", "source": name}

    df_list = list(datasets_dict.values())
    n_samples = sum(len(df) for df in df_list)
    log(f"  [{domain_name}] 纵向合并 {len(df_list)} 个数据集 (共 {n_samples} 行)")

    # 找所有数据集都有的共同列
    common_cols = set(df_list[0].columns)
    for df in df_list[1:]:
        common_cols &= set(df.columns)

    common_cols = sorted(common_cols)
    log(f"  [{domain_name}] 共同列: {len(common_cols)}")

    if len(common_cols) >= 3:
        # 对齐并合并
        aligned = []
        for df in df_list:
            sub = df[common_cols].copy()
            aligned.append(sub)
        merged = pd.concat(aligned, ignore_index=True)
        log(f"  [{domain_name}] 按共同列合并: {merged.shape}")
        return merged, {"method": "common_cols", "n_cols": len(common_cols)}
    else:
        # 共同列太少 — 尝试填充缺失值为 0 再合并
        all_cols = set()
        for df in df_list:
            all_cols |= set(df.columns)
        all_cols = sorted(all_cols)
        log(f"  [{domain_name}] 全部列: {len(all_cols)}, 共同列不足, 全列合并")
        aligned = []
        for df in df_list:
            sub = df.reindex(columns=all_cols, fill_value=0)
            aligned.append(sub)
        merged = pd.concat(aligned, ignore_index=True)
        log(f"  [{domain_name}] 全列合并: {merged.shape}")
        return merged, {"method": "all_cols", "n_cols": len(all_cols)}


# ======================================================================
# 步骤 4: 特征工程
# ======================================================================
def engineer_features(df, max_interactions=MAX_INTERACTIONS):
    """
    高级特征工程:
    1. 对数变换 (正偏态特征)
    2. 统计交互特征 (Top-N 重要特征的乘积)
    3. 多项式特征组合 (可选)
    4. 比率特征 (如 BMI 相关)
    """
    n_cols_before = len(df.columns)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    log(f"  特征工程: 输入 {df.shape}, 数值列 {len(numeric_cols)}")

    new_features = {}

    # 1. 对数变换 — 对偏斜严重的特征
    skewed_cols = []
    for col in numeric_cols[:50]:  # 限前50列以防止过度膨胀
        col_data = df[col].dropna()
        if len(col_data) < 10:
            continue
        skewness = col_data.skew()
        if abs(skewness) > 1.5 and col_data.min() >= 0:
            new_features[f"{col}_log"] = np.log1p(df[col].clip(lower=0))
            skewed_cols.append(col)
    log(f"    对数变换: {len(skewed_cols)} 个特征")

    # 2. 比率特征 — BMI / 血压 / 血糖 相关
    bmi_cols = [c for c in numeric_cols if "bmi" in c.lower()]
    bp_cols  = [c for c in numeric_cols if "bp" in c.lower() or "blood" in c.lower()]
    gluc_cols = [c for c in numeric_cols if "gluc" in c.lower() or "glucose" in c.lower()]
    age_cols = [c for c in numeric_cols if "age" in c.lower()]

    # BMI * 年龄交互
    for bmi_c in bmi_cols[:3]:
        for age_c in age_cols[:3]:
            new_features[f"{bmi_c}_x_{age_c}"] = df[bmi_c] * df[age_c]

    # 3. Top 特征交互 — 选择方差最大的 Top-15 特征进行交互
    if len(numeric_cols) > 5:
        var_rank = df[numeric_cols].var().sort_values(ascending=False)
        top_features = var_rank.head(15).index.tolist()

        interaction_count = 0
        for i, c1 in enumerate(top_features):
            for c2 in top_features[i+1:]:
                if interaction_count >= max_interactions:
                    break
                new_features[f"{c1}_x_{c2}"] = df[c1] * df[c2]
                interaction_count += 1
            if interaction_count >= max_interactions:
                break
        log(f"    交互特征: {interaction_count} 个")

    # 添加新特征
    for k, v in new_features.items():
        df[k] = v

    # 4. 多项式特征 (可选)
    if USE_POLY_FEATURES and len(numeric_cols) <= 30:
        try:
            top_n = min(10, len(numeric_cols))
            top_cols = df[numeric_cols].var().sort_values(ascending=False).head(top_n).index.tolist()
            poly = PolynomialFeatures(degree=POLY_DEGREE, interaction_only=True,
                                       include_bias=False)
            poly_features = poly.fit_transform(df[top_cols].fillna(0))
            poly_names = poly.get_feature_names_out(top_cols)
            # 只保留新生成的交互项 (非原始项)
            for j, name in enumerate(poly_names):
                if "^" in name or " " in name:  # 交互项
                    df[f"poly_{name.replace(' ', '_x_')}"] = poly_features[:, j]
            log(f"    多项式特征: {len(poly_names) - top_n} 个交互项")
        except Exception as e:
            log(f"    多项式特征异常 (跳过): {e}", "WARN")

    log(f"  特征工程完成: {n_cols_before} → {len(df.columns)} 列")
    return df


# ======================================================================
# 步骤 5: 特征选择 + 降维
# ======================================================================
def reduce_dimensions(df, k=FEATURE_SELECTION_K, pca_n=PCA_COMPONENTS,
                      target_cols=None):
    """
    特征降维:
    1. 低方差过滤
    2. SelectKBest (如有标签) 或 方差选择
    3. PCA 降维
    """
    n_before = len(df.columns)

    # 分离标签列
    y = None
    label_col = None
    potential_targets = target_cols or [
        "outcome", "target", "cardio", "stroke", "diagnosis_m",
        "diabetes_binary", "heartdiseaseorattack", "readmitted",
        "diabetes_012", "highbp",
    ]

    X = df.copy()
    for col in potential_targets:
        if col in X.columns:
            # 检查是否适合作为标签 (二分类, 非ID)
            if X[col].nunique() <= 10 and X[col].nunique() >= 2:
                y = X[col].copy()
                label_col = col
                X = X.drop(columns=[col])
                log(f"  检测到标签列: '{col}' ({y.nunique()} 类)")
                break

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        return df, None, None, None

    X_num = X[numeric_cols].fillna(0)

    # 1. 低方差过滤
    variances = X_num.var()
    low_var = variances[variances < VARIANCE_THRESHOLD].index.tolist()
    if low_var:
        X_num = X_num.drop(columns=low_var)
        log(f"  低方差过滤: 删除 {len(low_var)} 列")

    # 2. SelectKBest
    selector = None
    if y is not None and len(X_num) > 50:
        k_actual = min(k, X_num.shape[1])
        try:
            selector = SelectKBest(mutual_info_classif, k=k_actual)
            X_sel = selector.fit_transform(X_num, y)
            selected_idx = selector.get_support(indices=True)
            selected_cols = [X_num.columns[i] for i in selected_idx]
            log(f"  SelectKBest: {X_num.shape[1]} → {k_actual} 列 (MI)")
            X_num = pd.DataFrame(X_sel, columns=selected_cols)
        except Exception as e:
            log(f"  SelectKBest 失败 ({e}), 使用方差选择", "WARN")
            k_actual = min(k, X_num.shape[1])
            top_var = variances.nlargest(k_actual).index.tolist()
            X_num = X_num[top_var]
    else:
        # 无标签 — 方差选择
        if X_num.shape[1] > k:
            top_var = variances.nlargest(min(k, X_num.shape[1])).index.tolist()
            X_num = X_num[top_var]
            log(f"  方差选择: {len(top_var)} 列")

    # 3. PCA 降维
    pca = None
    pca_actual = min(pca_n, X_num.shape[0], X_num.shape[1])
    if pca_actual > 2:
        try:
            pca = PCA(n_components=pca_actual, random_state=RANDOM_STATE)
            X_pca = pca.fit_transform(StandardScaler().fit_transform(X_num))
            explained = pca.explained_variance_ratio_.sum()
            pca_cols = [f"PC{i+1}" for i in range(pca_actual)]
            df_pca = pd.DataFrame(X_pca, columns=pca_cols)
            log(f"  PCA: {X_num.shape[1]} → {pca_actual} 维, "
                f"解释方差={explained:.3f}")
        except Exception as e:
            log(f"  PCA 失败 ({e})", "WARN")
            df_pca = X_num.copy()
            pca = None
    else:
        df_pca = X_num.copy()

    # 保留原始标签
    if label_col and y is not None:
        df_pca[label_col] = y.values

    log(f"  降维完成: {n_before} → {len(df_pca.columns)} 列")
    return df_pca, selector, pca, label_col


# ======================================================================
# 步骤 6: 构建全局特征字典 (跨域特征映射)
# ======================================================================
def build_unified_feature_map(all_datasets):
    """
    构建统一的特征字典，记录所有出现过的重要特征，
    用于跨数据集的语义对齐。
    """
    feature_map = defaultdict(list)

    key_patterns = {
        "age": ["age", "edad", "alter", "âge"],
        "sex": ["sex", "gender", "sexo", "geschlecht"],
        "bmi": ["bmi", "body_mass", "imc", "body_mass_index"],
        "blood_pressure": ["bp", "blood_pressure", "trestbps", "ap_hi", "ap_lo",
                           "systolic", "diastolic", "hypertension"],
        "cholesterol": ["chol", "cholesterol", "lipid", "ldl", "hdl"],
        "glucose": ["glucose", "gluc", "blood_sugar", "hba1c", "glycated"],
        "smoking": ["smoke", "smoker", "tobacco", "nicotine", "cigarette"],
        "alcohol": ["alco", "alcohol", "drink", "wine", "beer"],
        "physical_activity": ["active", "activity", "exercise", "phys", "fit"],
        "weight": ["weight", "kg", "kilo", "peso"],
        "height": ["height", "cm", "stature", "altura"],
        "diabetes": ["diabetes", "diabetic", "glucose", "insulin"],
        "heart_disease": ["heart", "cardio", "cardiac", "coronary", "mi", "infarct"],
        "cancer": ["cancer", "tumor", "malignant", "neoplasm"],
        "kidney": ["kidney", "renal", "creatinine", "gfr"],
        "liver": ["liver", "hepatic", "alt", "ast", "bilirubin"],
    }

    for dataset_name, df in all_datasets.items():
        for col in df.columns:
            col_lower = col.lower().replace("_", "").replace("-", "")
            for category, patterns in key_patterns.items():
                for pat in patterns:
                    if pat in col_lower:
                        feature_map[category].append({
                            "dataset": dataset_name,
                            "column": col,
                        })
                        break

    return dict(feature_map)


# ======================================================================
# 主融合流水线
# ======================================================================
def run_fusion_pipeline():
    """完整的数据融合流水线"""
    log("╔══════════════════════════════════════════════════════════╗")
    log("║   多模态数据融合与特征工程                             ║")
    log("╚══════════════════════════════════════════════════════════╝")
    log(f"配置: SelectKBest={FEATURE_SELECTION_K}, PCA={PCA_COMPONENTS}, "
        f"多项式={USE_POLY_FEATURES}")

    # 1. 加载
    datasets = load_processed_datasets()
    if not datasets:
        log("无可用数据集, 退出", "ERROR")
        return

    # 2. 分组
    log("\n--- 主题域分组 ---")
    domains = group_by_domain(datasets)

    # 3. 构建全局特征映射
    log("\n--- 构建全局特征映射 ---")
    feature_map = build_unified_feature_map(datasets)
    log(f"特征类别: {list(feature_map.keys())}")
    for cat, cols in feature_map.items():
        log(f"  {cat}: {len(cols)} 个列 — {[c['column'] for c in cols[:5]]}")

    # 4. 逐主题域融合
    log("\n--- 逐主题域融合 ---")
    fused_results = {}

    for domain_name, ds_dict in domains.items():
        log(f"\n{'='*50}")
        log(f"融合主题域: {domain_name}")
        log(f"{'='*50}")

        try:
            # 4a. 纵向合并
            merged, merge_info = vertical_merge(ds_dict, domain_name)

            # 4b. 特征工程
            merged = engineer_features(merged)

            # 4c. 特征选择 + 降维
            fused, selector, pca, label = reduce_dimensions(merged)

            # 保存
            out_path = DATA_FUSED / f"{domain_name}_fused.parquet"
            fused.to_parquet(out_path, index=False)
            size_kb = os.path.getsize(out_path) / 1024
            log(f"  保存: {out_path.name} ({size_kb:.1f} KB, {fused.shape})")

            # 保存 PCA 模型
            if pca is not None:
                import joblib
                joblib.dump(pca, DATA_FUSED / f"{domain_name}_pca.pkl")

            FUSION_LOG.append({
                "domain": domain_name,
                "n_datasets": len(ds_dict),
                "merge_method": merge_info.get("method", "unknown"),
                "original_cols": merge_info.get("n_cols", 0),
                "final_rows": fused.shape[0],
                "final_cols": fused.shape[1],
                "label_col": label,
                "size_kb": round(size_kb, 1),
            })

            fused_results[domain_name] = {
                "data": fused,
                "label": label,
                "selector": selector,
                "pca": pca,
            }

        except Exception as e:
            log(f"融合失败 [{domain_name}]: {e}", "ERROR")
            import traceback
            traceback.print_exc()
            FUSION_LOG.append({
                "domain": domain_name,
                "status": "失败",
                "error": str(e),
            })

    # 5. 创建综合特征集 (跨域拼接)
    log(f"\n{'='*50}")
    log("构建跨域综合特征集")
    log(f"{'='*50}")

    try:
        combined = create_combined_features(fused_results)
        if combined is not None:
            out_path = DATA_FUSED / "combined_features.parquet"
            combined.to_parquet(out_path, index=False)
            log(f"综合特征集: {combined.shape}, 已保存至 {out_path}")
    except Exception as e:
        log(f"综合特征集构建失败: {e}", "ERROR")

    # 6. 报告
    generate_fusion_report(feature_map)

    return fused_results


def create_combined_features(fused_results):
    """跨主题域拼接特征，构建综合特征矩阵"""
    # 选择行数适中的数据集进行拼接
    dfs = []
    labels = []
    for domain, info in fused_results.items():
        df = info["data"]
        # 采样以避免某些数据集过大
        if len(df) > 50000:
            df = df.sample(50000, random_state=RANDOM_STATE)
        dfs.append(df)

    if not dfs:
        return None

    # 寻找共同样本量
    min_rows = min(len(df) for df in dfs)
    log(f"综合特征: {len(dfs)} 个域, 统一采样 {min_rows} 行")

    aligned = []
    for df in dfs:
        sampled = df.sample(min_rows, random_state=RANDOM_STATE).reset_index(drop=True)
        # 重命名列以避免冲突
        sampled = sampled.add_prefix(f"{df.attrs.get('domain', 'x')}_")
        aligned.append(sampled)

    combined = pd.concat(aligned, axis=1)
    return combined


def generate_fusion_report(feature_map):
    """生成融合报告"""
    report = f"""# 多模态数据融合与特征工程报告

**生成时间:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**配置:** 特征选择K={FEATURE_SELECTION_K}, PCA={PCA_COMPONENTS}

## 融合结果汇总

| 主题域 | 数据集数 | 合并方法 | 最终行数 | 最终列数 | 标签列 | 大小(KB) |
|--------|----------|----------|----------|----------|--------|----------|
"""
    for entry in FUSION_LOG:
        if "final_rows" in entry:
            report += (f"| {entry['domain']} | {entry['n_datasets']} | "
                       f"{entry['merge_method']} | {entry['final_rows']:,} | "
                       f"{entry['final_cols']} | {entry.get('label_col','-')} | "
                       f"{entry['size_kb']} |\n")
        else:
            report += (f"| {entry['domain']} | - | - | - | - | - | "
                       f"{entry.get('error','失败')} |\n")

    report += "\n## 全局特征映射\n\n"
    for cat, cols in sorted(feature_map.items()):
        report += f"### {cat} ({len(cols)} 个列)\n"
        for c in cols[:8]:
            report += f"- `{c['column']}` (来自 {c['dataset']})\n"
        if len(cols) > 8:
            report += f"- ... 及其他 {len(cols)-8} 个列\n"
        report += "\n"

    report += """## 输出文件

| 文件 | 用途 |
|------|------|
"""
    for fpath in sorted(DATA_FUSED.glob("*_fused.parquet")):
        report += f"| {fpath.name} | 主题域融合特征 |\n"
    for fpath in sorted(DATA_FUSED.glob("*.parquet")):
        if "_fused" not in fpath.name:
            report += f"| {fpath.name} | 综合特征集 |\n"

    report += """
## 下一步操作

融合特征已保存至 `data/fused/`，继续执行:
1. `privacy_protection.py` — 隐私保护处理
2. `chronic_disease_model.py` — 模型训练
"""

    report_path = DATA_PROCESSED.parent / "data_fusion_report.md"
    report_path.write_text(report, encoding="utf-8")
    log(f"报告已保存: {report_path}")

    json_path = DATA_FUSED / "fusion_log.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(FUSION_LOG, f, ensure_ascii=False, indent=2)


def main():
    os.chdir(Path(__file__).parent.parent)
    log(f"工作目录: {os.getcwd()}")
    log(f"输入目录: {DATA_PROCESSED.resolve()}")
    log(f"输出目录: {DATA_FUSED.resolve()}")

    run_fusion_pipeline()

    log("\n" + "=" * 60)
    log(f"数据融合完成! 输出位于: {DATA_FUSED.resolve()}")
    log(f"查看报告: data/data_fusion_report.md")
    log("=" * 60)


if __name__ == "__main__":
    main()
