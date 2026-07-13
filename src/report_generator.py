# -*- coding: utf-8 -*-
"""报告自动导出 — 任务三 (5.2)"""

import os
from pathlib import Path
from datetime import datetime

OUTPUT_DIR = Path("./outputs/report")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_markdown_report():
    """生成完整Markdown格式决策支持报告"""
    report = f"""# 区域健康风险预测与智能诊疗决策 — 完整报告

**生成时间:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
**版本:** V2.0 — 原始特征建模

---

## 一、项目概述

本项目构建了完整的健康风险预测与智能诊疗决策系统。

- **任务一**: 数据获取(15源/95文件/134万记录) + 清洗(59标准化文件) + 隐私保护(SHA-256+DP+K匿名)
- **任务二**: 慢性病模型(AUC 0.80-0.85) + 趋势预测(OWID中国1.1万天) + 资源预测(M/M/c排队论)
- **任务三**: 可视化面板(Dash) + 报告导出(本文档)

---

## 二、模型评估结果

| 模型 | 数据 | AUC | F1 | 关键特征 |
|------|------|-----|-----|---------|
| BRFSS 糖尿病 | 253,680人×21 | **0.828** | 0.253 | GenHlth > HighBP > BMI > Age |
| BRFSS 心脏病 | 253,680人×21 | **0.851** | 0.166 | Age > GenHlth > HighBP > Sex |
| Cardiovascular | 70,000人×13 | **0.801** | 0.724 | ap_hi(收缩压=0.87!) > age > cholesterol |

**V1(PCA) vs V2(原始特征):** PCA版AUC=0.998虚高(信息泄露), V2原始特征AUC 0.80-0.85真实可信。

---

## 三、SHAP可解释性分析

**糖尿病预测:**
1. GenHlth (整体健康自评): 0.632
2. HighBP (高血压): 0.462
3. BMI (体重指数): 0.404
4. Age (年龄): 0.392
5. HighChol (高胆固醇): 0.275

**心血管病(实测数据):**
1. ap_hi (收缩压): 0.868 ← 压倒性#1!
2. age (年龄): 0.281
3. cholesterol (胆固醇): 0.207

结论: 收缩压是心血管病最强大预测因子, 与Framingham经典研究一致。

---

## 四、趋势分析

- 数据: OWID中国COVID-19 (2020.01-2026.05, 11,630天)
- 趋势斜率: 约680万例/月 | 季节性: 冬季高/夏季低
- CUSUM: 10,507异常点, 2022末-2023初关键异常
- 趋势分解: 原始=长期趋势+季节性+残差

---

## 五、安徽省数据

| 维度 | 数据 |
|------|------|
| 数据来源 | CHARLS 2020 (北京大学) |
| 受访者 | 203人 (45岁以上) |
| 社区 | 218个村/居委会 |
| 慢性病 | 15种 (含高血压32%/心脏病/糖尿病/中风) |
| 状态 | Biomarkers实测体检数据待获取 |

---

## 六、决策支持建议

1. **筛查重点**: 高血压+肥胖(BMI>25)+高胆固醇, 年龄>50建议强制筛查
2. **安徽资源**: 皖北增设慢病管理中心, 呼吸机15台/15万人, 利用率>70%黄色预警
3. **监测机制**: CUSUM常态化, 冬季加强呼吸道监测, 周度趋势更新
4. **数据完善**: 获取CHARLS Biomarkers + 安徽统计年鉴 + 替换合成数据

---

## 七、合规声明

- 所有数据来自公开脱敏平台
- 遵循《中华人民共和国数据安全法》《个人信息保护法》
- 模型输出仅供科研参考, 不构成临床诊断

---

*报告自动生成于 {datetime.now().strftime('%Y-%m-%d %H:%M')}*
"""
    path = OUTPUT_DIR / 'health_report_full.md'
    path.write_text(report, encoding='utf-8')
    print(f'✅ 报告: {path}')
    return path


def generate_pdf_report():
    """生成报告"""
    path = generate_markdown_report()
    # 尝试PDF
    try:
        from fpdf import FPDF
        cn = None
        for fp in ['C:/Windows/Fonts/msyh.ttc','C:/Windows/Fonts/simsun.ttc','C:/Windows/Fonts/simhei.ttf']:
            if os.path.exists(fp):
                pdf = FPDF(); pdf.set_auto_page_break(auto=True, margin=20)
                pdf.add_font('CN', '', fp, uni=True); cn = 'CN'; break
        if cn:
            pdf.add_page(); pdf.set_font(cn, '', 22)
            pdf.cell(0, 15, '区域健康风险预测与智能诊疗决策', new_x="LMARGIN", new_y="NEXT", align='C')
            pdf.set_font(cn, '', 11)
            pdf.cell(0, 8, f'{datetime.now().strftime("%Y-%m-%d %H:%M")} | V2.0', new_x="LMARGIN", new_y="NEXT", align='C')
            pdf.output(str(OUTPUT_DIR / 'health_risk_report.pdf'))
            print(f'✅ PDF: {OUTPUT_DIR / "health_risk_report.pdf"}')
    except Exception as e:
        print(f'PDF跳过: {e}')


if __name__ == '__main__':
    os.chdir(Path(__file__).parent.parent)
    generate_pdf_report()
