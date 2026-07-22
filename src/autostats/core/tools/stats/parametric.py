from typing import Literal

import numpy as np
from pydantic import BaseModel
from scipy import stats as scipy_stats

from autostats.core.schemas.stat_result import StatResult
from autostats.core.tools.base import BaseTool, ToolContext, ToolInput
from autostats.core.tools.registry import REGISTRY
from autostats.core.tools.stats.assumptions import (
    check_homogeneity_of_variance,
    check_independence_note,
    check_normality,
    check_sample_size,
)

Alternative = Literal["two-sided", "less", "greater"]


def _pooled_std(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    return float(np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2)))


class OneSampleTTestInput(ToolInput):
    column: str
    popmean: float
    alpha: float = 0.05
    alternative: Alternative = "two-sided"


@REGISTRY.register
class OneSampleTTestTool(BaseTool):
    name = "one_sample_t_test"
    description = "Test whether a column's mean differs from a hypothesized population mean."
    category = "hypothesis_test"
    input_model = OneSampleTTestInput

    def run(self, ctx: ToolContext, params: OneSampleTTestInput) -> BaseModel:
        df = ctx.data_manager.load(params.dataset_id)
        x = df[params.column].dropna().to_numpy()
        res = scipy_stats.ttest_1samp(x, params.popmean, alternative=params.alternative)
        norm_check = check_normality(df[params.column], params.column)
        size_check = check_sample_size(len(x), 30, params.column)
        d = float((x.mean() - params.popmean) / x.std(ddof=1))
        ci = scipy_stats.t.interval(1 - params.alpha, len(x) - 1, loc=x.mean(), scale=scipy_stats.sem(x))
        assumptions = [norm_check, size_check, check_independence_note()]
        assumptions_met = norm_check.passed or len(x) >= 30
        sig = "significant" if res.pvalue < params.alpha else "not significant"
        return StatResult(
            tool_name=self.name,
            test_name="one_sample_t_test",
            statistic=float(res.statistic),
            p_value=float(res.pvalue),
            degrees_of_freedom=float(len(x) - 1),
            effect_size={"cohens_d": d},
            confidence_interval=(float(ci[0]), float(ci[1])),
            confidence_level=1 - params.alpha,
            sample_sizes={params.column: len(x)},
            assumptions=assumptions,
            assumptions_met=assumptions_met,
            recommended_alternative=None if assumptions_met else "wilcoxon_signed_rank_test",
            interpretation=(
                f"One-sample t-test: mean of '{params.column}' ({x.mean():.3g}) vs. hypothesized "
                f"mean {params.popmean:.3g} is {sig} (t={res.statistic:.3f}, p={res.pvalue:.4f}, "
                f"Cohen's d={d:.3f})."
            ),
            warnings=[] if assumptions_met else ["Normality assumption may be violated; consider a nonparametric alternative."],
        )


class TwoSampleTTestInput(ToolInput):
    group_col: str
    value_col: str
    equal_var: bool = False
    alpha: float = 0.05
    alternative: Alternative = "two-sided"


@REGISTRY.register
class TwoSampleTTestTool(BaseTool):
    name = "two_sample_t_test"
    description = "Compare means of two independent groups (Welch's t-test by default)."
    category = "hypothesis_test"
    input_model = TwoSampleTTestInput

    def run(self, ctx: ToolContext, params: TwoSampleTTestInput) -> BaseModel:
        df = ctx.data_manager.load(params.dataset_id)
        groups = df[params.group_col].dropna().unique()
        if len(groups) != 2:
            raise ValueError(f"'{params.group_col}' must have exactly 2 groups, found {len(groups)}")
        a = df.loc[df[params.group_col] == groups[0], params.value_col].dropna()
        b = df.loc[df[params.group_col] == groups[1], params.value_col].dropna()

        res = scipy_stats.ttest_ind(a, b, equal_var=params.equal_var, alternative=params.alternative)
        norm_a = check_normality(a, str(groups[0]))
        norm_b = check_normality(b, str(groups[1]))
        var_check = check_homogeneity_of_variance(a, b, labels=[str(groups[0]), str(groups[1])])
        size_a = check_sample_size(len(a), 30, str(groups[0]))
        size_b = check_sample_size(len(b), 30, str(groups[1]))

        d = float((a.mean() - b.mean()) / _pooled_std(a.to_numpy(), b.to_numpy()))
        assumptions = [norm_a, norm_b, var_check, size_a, size_b, check_independence_note()]
        assumptions_met = (norm_a.passed and norm_b.passed) or (len(a) >= 30 and len(b) >= 30)
        sig = "significant" if res.pvalue < params.alpha else "not significant"
        method = "Welch's" if not params.equal_var else "Student's"
        return StatResult(
            tool_name=self.name,
            test_name="two_sample_t_test",
            statistic=float(res.statistic),
            p_value=float(res.pvalue),
            degrees_of_freedom=float(getattr(res, "df", len(a) + len(b) - 2)),
            effect_size={"cohens_d": d},
            confidence_level=1 - params.alpha,
            sample_sizes={str(groups[0]): len(a), str(groups[1]): len(b)},
            assumptions=assumptions,
            assumptions_met=assumptions_met,
            recommended_alternative=None if assumptions_met else "mann_whitney_u_test",
            interpretation=(
                f"{method} two-sample t-test comparing '{params.value_col}' between "
                f"{groups[0]!s} (n={len(a)}, mean={a.mean():.3g}) and {groups[1]!s} "
                f"(n={len(b)}, mean={b.mean():.3g}) is {sig} (t={res.statistic:.3f}, "
                f"p={res.pvalue:.4f}, Cohen's d={d:.3f})."
            ),
            warnings=[] if assumptions_met else ["Normality assumption may be violated in one or both groups; consider Mann-Whitney U."],
        )


class PairedTTestInput(ToolInput):
    column_a: str
    column_b: str
    alpha: float = 0.05
    alternative: Alternative = "two-sided"


@REGISTRY.register
class PairedTTestTool(BaseTool):
    name = "paired_t_test"
    description = "Compare means of two paired/repeated measurements on the same subjects."
    category = "hypothesis_test"
    input_model = PairedTTestInput

    def run(self, ctx: ToolContext, params: PairedTTestInput) -> BaseModel:
        df = ctx.data_manager.load(params.dataset_id)
        paired = df[[params.column_a, params.column_b]].dropna()
        a, b = paired[params.column_a], paired[params.column_b]
        diff = a - b
        res = scipy_stats.ttest_rel(a, b, alternative=params.alternative)
        norm_check = check_normality(diff, "difference")
        size_check = check_sample_size(len(diff), 30, "pairs")
        d = float(diff.mean() / diff.std(ddof=1))
        assumptions = [norm_check, size_check, check_independence_note()]
        assumptions_met = norm_check.passed or len(diff) >= 30
        sig = "significant" if res.pvalue < params.alpha else "not significant"
        return StatResult(
            tool_name=self.name,
            test_name="paired_t_test",
            statistic=float(res.statistic),
            p_value=float(res.pvalue),
            degrees_of_freedom=float(len(diff) - 1),
            effect_size={"cohens_d": d},
            confidence_level=1 - params.alpha,
            sample_sizes={"pairs": len(diff)},
            assumptions=assumptions,
            assumptions_met=assumptions_met,
            recommended_alternative=None if assumptions_met else "wilcoxon_signed_rank_test",
            interpretation=(
                f"Paired t-test between '{params.column_a}' and '{params.column_b}' "
                f"(mean difference={diff.mean():.3g}) is {sig} (t={res.statistic:.3f}, "
                f"p={res.pvalue:.4f}, Cohen's d={d:.3f})."
            ),
            warnings=[] if assumptions_met else ["Normality of differences may be violated; consider Wilcoxon signed-rank test."],
        )


class OneWayAnovaInput(ToolInput):
    group_col: str
    value_col: str
    alpha: float = 0.05


@REGISTRY.register
class OneWayAnovaTool(BaseTool):
    name = "one_way_anova"
    description = "Compare means across 3+ independent groups using one-way ANOVA."
    category = "hypothesis_test"
    input_model = OneWayAnovaInput

    def run(self, ctx: ToolContext, params: OneWayAnovaInput) -> BaseModel:
        df = ctx.data_manager.load(params.dataset_id)
        groups_df = df[[params.group_col, params.value_col]].dropna()
        labels = sorted(groups_df[params.group_col].unique().tolist())
        samples = [groups_df.loc[groups_df[params.group_col] == g, params.value_col] for g in labels]

        res = scipy_stats.f_oneway(*samples)
        norm_checks = [check_normality(s, str(g)) for g, s in zip(labels, samples)]
        var_check = check_homogeneity_of_variance(*samples, labels=[str(g) for g in labels])
        size_checks = [check_sample_size(len(s), 30, str(g)) for g, s in zip(labels, samples)]

        grand_mean = groups_df[params.value_col].mean()
        ss_between = sum(len(s) * (s.mean() - grand_mean) ** 2 for s in samples)
        ss_total = ((groups_df[params.value_col] - grand_mean) ** 2).sum()
        eta_sq = float(ss_between / ss_total) if ss_total else 0.0

        assumptions = [*norm_checks, var_check, *size_checks, check_independence_note()]
        assumptions_met = all(c.passed for c in norm_checks) and var_check.passed
        sig = "significant" if res.pvalue < params.alpha else "not significant"
        return StatResult(
            tool_name=self.name,
            test_name="one_way_anova",
            statistic=float(res.statistic),
            p_value=float(res.pvalue),
            degrees_of_freedom=(float(len(labels) - 1), float(len(groups_df) - len(labels))),
            effect_size={"eta_squared": eta_sq},
            sample_sizes={str(g): int(len(s)) for g, s in zip(labels, samples)},
            assumptions=assumptions,
            assumptions_met=assumptions_met,
            recommended_alternative=None if assumptions_met else "kruskal_wallis_test",
            interpretation=(
                f"One-way ANOVA on '{params.value_col}' across groups {labels} is {sig} "
                f"(F={res.statistic:.3f}, p={res.pvalue:.4f}, eta-squared={eta_sq:.3f}). "
                + ("Consider a Tukey post-hoc test to identify which groups differ." if res.pvalue < params.alpha else "")
            ),
            warnings=[] if assumptions_met else ["Normality and/or equal-variance assumptions may be violated; consider Kruskal-Wallis."],
        )


class PairwiseTukeyInput(ToolInput):
    group_col: str
    value_col: str
    alpha: float = 0.05


@REGISTRY.register
class PairwiseTukeyPosthocTool(BaseTool):
    name = "pairwise_tukey_posthoc"
    description = "Run Tukey's HSD post-hoc test to find which group pairs differ after a significant ANOVA."
    category = "hypothesis_test"
    input_model = PairwiseTukeyInput

    def run(self, ctx: ToolContext, params: PairwiseTukeyInput) -> BaseModel:
        from statsmodels.stats.multicomp import pairwise_tukeyhsd

        df = ctx.data_manager.load(params.dataset_id)
        clean = df[[params.group_col, params.value_col]].dropna()
        res = pairwise_tukeyhsd(clean[params.value_col], clean[params.group_col], alpha=params.alpha)
        from autostats.core.schemas.stat_result import TableArtifact

        rows = [list(r) for r in res._results_table.data[1:]]
        table = TableArtifact(
            table_id=f"tukey_{params.dataset_id}",
            title="Tukey HSD pairwise comparisons",
            columns=list(res._results_table.data[0]),
            rows=rows,
        )
        n_sig = sum(1 for r in rows if r[-1])
        return StatResult(
            tool_name=self.name,
            test_name="pairwise_tukey_posthoc",
            tables=[table],
            interpretation=f"Tukey HSD post-hoc found {n_sig} of {len(rows)} pairwise comparisons significant at alpha={params.alpha}.",
        )


class PearsonCorrelationTestInput(ToolInput):
    column_a: str
    column_b: str
    alpha: float = 0.05


@REGISTRY.register
class PearsonCorrelationTestTool(BaseTool):
    name = "pearson_correlation_test"
    description = "Test the significance of the Pearson linear correlation between two numeric columns."
    category = "hypothesis_test"
    input_model = PearsonCorrelationTestInput

    def run(self, ctx: ToolContext, params: PearsonCorrelationTestInput) -> BaseModel:
        df = ctx.data_manager.load(params.dataset_id)
        paired = df[[params.column_a, params.column_b]].dropna()
        a, b = paired[params.column_a], paired[params.column_b]
        r, p = scipy_stats.pearsonr(a, b)
        norm_a = check_normality(a, params.column_a)
        norm_b = check_normality(b, params.column_b)
        assumptions = [norm_a, norm_b, check_sample_size(len(paired), 30, "pairs"), check_independence_note()]
        assumptions_met = norm_a.passed and norm_b.passed
        sig = "significant" if p < params.alpha else "not significant"
        return StatResult(
            tool_name=self.name,
            test_name="pearson_correlation_test",
            statistic=float(r),
            p_value=float(p),
            degrees_of_freedom=float(len(paired) - 2),
            sample_sizes={"pairs": len(paired)},
            assumptions=assumptions,
            assumptions_met=assumptions_met,
            recommended_alternative=None if assumptions_met else "spearman_correlation_test",
            interpretation=(
                f"Pearson correlation between '{params.column_a}' and '{params.column_b}' is {sig} "
                f"(r={r:.3f}, p={p:.4f}, n={len(paired)})."
            ),
            warnings=[] if assumptions_met else ["Normality assumption may be violated; consider Spearman's correlation."],
        )
