"""Write reports from audited stage outputs; engineering runs never become scientific evidence."""
from collections import defaultdict
from pathlib import Path
from statistics import mean

from scripts.openi.precheck_science import read_rows
from scripts.openi.state import write_json


def points(value):
    return f"{100*value:+.2f} 个百分点"


def work1(root):
    clean, mixed = read_rows(root / "work1/clean_mixed_comparison.csv")
    _, high, random = read_rows(root / "work1/loss_random_comparison.csv")
    rows = read_rows(root / "work1/paired_interventions.csv")
    last = max(int(r["quantile"]) for r in rows if r["signal"] == "loss_i")
    treated = [r for r in rows if r["phase"] == "analysis" and r["signal"] == "loss_i" and
               int(r["quantile"]) == last and r["arm"] == "downweight"]
    harmful = sum(float(r["gain_robust"]) < 0 for r in treated)
    targeted = mean(float(r["gain_robust"]) for r in treated)
    lines = [
        "# 研究一必要性预检",
        "",
        "数据为四客户端、单种子、CIFAR-10 训练来源的独立诊断验证。",
        f"- Mixed CE 相对 Clean CE 的干净及鲁棒差分别为 {points(float(mixed['clean_acc'])-float(clean['clean_acc']))}"
        f"和 {points(float(mixed['corrupt_acc'])-float(clean['corrupt_acc']))}。",
        f"- 匹配降权访问量下，高 Loss 相对随机降权的鲁棒差为 "
        f"{points(float(high['corrupt_acc'])-float(random['corrupt_acc']))}。",
        f"- Analysis checkpoint 的最高 Loss 分位软降权相对正常训练平均 {points(targeted)}。"
        f"{len(treated)} 个配对中 {harmful} 个出现鲁棒下降。",
        "- 下表汇总软降权 Spearman 的可定义客户端项。按架构和阶段列出，原始逐客户端值在 CSV。",
        "",
        "| 阶段 | 架构 | 信号 | 有定义客户端项 | ρ 均值 |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    grouped = defaultdict(list)
    for r in read_rows(root / "work1/signal_value_correlations.csv"):
        if r["arm"] == "downweight":
            values = grouped[(r["phase"], r["architecture"], r["signal"])]
            if r["spearman"]:
                values.append(float(r["spearman"]))
    for (phase, arch, signal), values in sorted(grouped.items()):
        estimate = f"{mean(values):+.3f}" if values else "未定义"
        lines.append(f"| {phase} | {arch} | {signal} | {len(values)} | {estimate} |")
    lines.extend(["", "损坏检测 AUROC 另见 corruption_detection.csv。训练样本行和分位配对共享模型、数据与种子。"
                  "这些是特定训练状态的探索性效应。是否推进差异化优化，需看跨架构方向与多种子复现。"])
    return lines, ("CONDITIONAL" if treated and (harmful > 0 or abs(targeted) > 0) else "尚不可判定",
                   "单种子高损失配对效应及匹配随机基线，需复验方向与信号稳定性。")


def work2(root, stages):
    lines = ["# 研究二必要性预检", ""]
    evidence = False
    if "work2_heterogeneity" in stages:
        rows = read_rows(root / "work2_heterogeneity/kd_collaboration_gains.csv")
        grouped = defaultdict(list)
        for row in rows:
            grouped[row["scenario"]].append(float(row["kd_gain"]))
        means = {name: mean(gains) for name, gains in grouped.items()}
        difference = means["heterogeneous"] - means["uniform_mixture"]
        hom = (means["gaussian"]+means["blur"])/2
        lines.extend(f"- {name} 的客户端平均 KD 增益为 {points(value)}。" for name, value in means.items())
        lines.append(f"- 跨端异质相对端内均匀混合为 {points(difference)}。"
                     f"相对 Gaussian 与 Blur 同质场景平均为 {points(means['heterogeneous']-hom)}。")
        lines.append(f"- 共 {sum(float(r['kd_gain'])<0 for r in rows)} 个客户端×场景出现负协作增益。")
        for arch in sorted({r["architecture"] for r in rows}):
            subset = [r for r in rows if r["architecture"] == arch]
            means_arch = {s: mean(float(r["kd_gain"]) for r in subset if r["scenario"] == s)
                          for s in ("heterogeneous", "uniform_mixture")}
            lines.append(f"- {arch} 的异质减均匀混合增益为 "
                         f"{points(means_arch['heterogeneous']-means_arch['uniform_mixture'])}。")
        evidence = difference != 0
    if "work2_frequency" in stages:
        rows = read_rows(root / "work2_frequency/delta_r_matrix.csv")
        lines.extend(["", "## 频率短期迁移", "",
                      "| 代理域 | 频带 | ΔR 均值 | 正 / 零 / 负 |",
                      "| --- | --- | ---: | ---: |"])
        for domain in dict.fromkeys(r["slice"] for r in rows):
            vals = {}
            for band in ("full", "low", "high"):
                gains = [float(r["delta_r"]) for r in rows if (r["slice"], r["band"]) == (domain, band)]
                signs = [sum((g > 0) - (g < 0) == sign for g in gains) for sign in (1, 0, -1)]
                vals[band] = mean(gains)
                lines.append(f"| {domain} | {band} | {points(vals[band])} | {' / '.join(map(str, signs))} |")
            evidence |= len(set(vals.values())) > 1
        lines.extend(["", "| 域 | 无标签评分 | Spearman ρ | AUROC |", "| --- | --- | ---: | ---: |"])
        for row in read_rows(root / "work2_frequency/score_delta_r_correlations.csv"):
            if row["student"] == row["band"] == "all":
                rho = f"{float(row['spearman']):+.3f}" if row["spearman"] else "未定义"
                auc = f"{float(row['auroc']):.3f}" if row["auroc"] else "未定义"
                lines.append(f"| {row['domain']} | {row['signal']} | {rho} | {auc} |")
        lines.append("同一域可直接比较学生依赖 JS、教师置信度和 MI 的表中指标。"
                     "分学生与分频带的结果在原始 CSV，跨域差异仅适用于本次代理样本。")
    lines.extend(["", "Gaussian 和 Defocus 同一等级的有效信息损失可能不同。短期 ΔR 受当前学生及优化器状态影响。"
                  "配对分组和公共批次共享单种子。"])
    return lines, ("CONDITIONAL" if evidence else "尚不可判定",
                   "优先复验异质减均匀混合及同学生内频带排序，确认收益后再设计选择性蒸馏。")


def run_reports(root, metadata):
    out = root / "reports"
    out.mkdir(exist_ok=True)
    stages = {name for name, record in metadata["stages"].items() if record["status"] == "complete"}
    research = metadata["purpose"] == "research"
    names = {}
    decisions = {}
    if research and "work1" in stages:
        names["WORK1_PRECHECK_REPORT.md"], decisions["work1"] = work1(root)
    if research and {"work2_heterogeneity", "work2_frequency"} & stages:
        names["WORK2_PRECHECK_REPORT.md"], decisions["work2"] = work2(root, stages)
    for key, name in (("work1", "WORK1_PRECHECK_REPORT.md"), ("work2", "WORK2_PRECHECK_REPORT.md")):
        if name not in names:
            names[name] = ["# 实验状态", "", "尚未有可用于科学判断的正式预实验结果。"]
            decisions[key] = ("尚不可判定", "等待通过质量门槛的正式预实验")
    for name, lines in names.items():
        (out / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    report = ["# 导师汇报结果附录", "", f"运行 {metadata['run_id']}，种子 {metadata['seed']}。", ""]
    for key, (verdict, reason) in decisions.items():
        report.append(f"- {key} 当前判断 **{verdict}**。{reason}")
    report.extend(["", *names["WORK1_PRECHECK_REPORT.md"], "", *names["WORK2_PRECHECK_REPORT.md"]])
    (out / "ADVISOR_PRELIMINARY_RESULTS.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    write_json(out / "scientific_status.json", {"purpose": metadata["purpose"], "stages": sorted(stages),
               "scientific_evidence": bool(research and stages & {"work1", "work2_heterogeneity", "work2_frequency"}),
               "decisions": {k: {"status": status, "basis": note} for k, (status, note) in decisions.items()}})
