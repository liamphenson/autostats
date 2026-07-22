import numpy as np
from pydantic import BaseModel
from scipy import stats as scipy_stats

from autostats.core.schemas.stat_result import StatResult
from autostats.core.tools.base import BaseTool, ToolContext, ToolInput
from autostats.core.tools.registry import REGISTRY
from autostats.core.tools.stats.assumptions import check_independence_note, check_sample_size


class MannWhitneyUInput(ToolInput):
    group_col: str
    value_col: str
    alpha: float = 0.05
    alternative: str = "two-sided"


@REGISTRY.register
class MannWhitneyUTool(BaseTool):
    name = "mann_whitney_u_test"
    description = "Nonparametric alternative to the two-sample t-test; compares distributions of two independent groups."
    category = "hypothesis_test"
    input_model = MannWhitneyUInput

    def run(self, ctx: ToolContext, params: MannWhitneyUInput) -> BaseModel:
        df = ctx.data_manager.load(params.dataset_id)
        groups = df[params.group_col].dropna().unique()
        if len(groups) != 2:
            raise ValueError(f"'{params.group_col}' must have exactly 2 groups, found {len(groups)}")
        a = df.loc[df[params.group_col] == groups[0], params.value_col].dropna()
        b = df.loc[df[params.group_col] == groups[1], params.value_col].dropna()
        res = scipy_stats.mannwhitneyu(a, b, alternative=params.alternative)
        sig = "significant" if res.pvalue < params.alpha else "not significant"
        return StatResult(
            tool_name=self.name,
            test_name="mann_whitney_u_test",
            statistic=float(res.statistic),
            p_value=float(res.pvalue),
            sample_sizes={str(groups[0]): len(a), str(groups[1]): len(b)},
            assumptions=[check_sample_size(len(a), 20, str(groups[0])), check_sample_size(len(b), 20, str(groups[1])), check_independence_note()],
            interpretation=(
                f"Mann-Whitney U test comparing '{params.value_col}' between {groups[0]!s} and {groups[1]!s} "
                f"is {sig} (U={res.statistic:.3f}, p={res.pvalue:.4f})."
            ),
        )


class WilcoxonSignedRankInput(ToolInput):
    column_a: str
    column_b: str
    alpha: float = 0.05


@REGISTRY.register
class WilcoxonSignedRankTool(BaseTool):
    name = "wilcoxon_signed_rank_test"
    description = "Nonparametric alternative to the paired t-test for two related samples."
    category = "hypothesis_test"
    input_model = WilcoxonSignedRankInput

    def run(self, ctx: ToolContext, params: WilcoxonSignedRankInput) -> BaseModel:
        df = ctx.data_manager.load(params.dataset_id)
        paired = df[[params.column_a, params.column_b]].dropna()
        res = scipy_stats.wilcoxon(paired[params.column_a], paired[params.column_b])
        sig = "significant" if res.pvalue < params.alpha else "not significant"
        return StatResult(
            tool_name=self.name,
            test_name="wilcoxon_signed_rank_test",
            statistic=float(res.statistic),
            p_value=float(res.pvalue),
            sample_sizes={"pairs": len(paired)},
            assumptions=[check_sample_size(len(paired), 20, "pairs"), check_independence_note()],
            interpretation=(
                f"Wilcoxon signed-rank test between '{params.column_a}' and '{params.column_b}' "
                f"is {sig} (W={res.statistic:.3f}, p={res.pvalue:.4f})."
            ),
        )


class KruskalWallisInput(ToolInput):
    group_col: str
    value_col: str
    alpha: float = 0.05


@REGISTRY.register
class KruskalWallisTool(BaseTool):
    name = "kruskal_wallis_test"
    description = "Nonparametric alternative to one-way ANOVA for 3+ independent groups."
    category = "hypothesis_test"
    input_model = KruskalWallisInput

    def run(self, ctx: ToolContext, params: KruskalWallisInput) -> BaseModel:
        df = ctx.data_manager.load(params.dataset_id)
        clean = df[[params.group_col, params.value_col]].dropna()
        labels = sorted(clean[params.group_col].unique().tolist())
        samples = [clean.loc[clean[params.group_col] == g, params.value_col] for g in labels]
        res = scipy_stats.kruskal(*samples)
        sig = "significant" if res.pvalue < params.alpha else "not significant"
        return StatResult(
            tool_name=self.name,
            test_name="kruskal_wallis_test",
            statistic=float(res.statistic),
            p_value=float(res.pvalue),
            degrees_of_freedom=float(len(labels) - 1),
            sample_sizes={str(g): int(len(s)) for g, s in zip(labels, samples)},
            assumptions=[check_independence_note()],
            interpretation=(
                f"Kruskal-Wallis test on '{params.value_col}' across groups {labels} is {sig} "
                f"(H={res.statistic:.3f}, p={res.pvalue:.4f})."
            ),
        )


class SpearmanCorrelationTestInput(ToolInput):
    column_a: str
    column_b: str
    alpha: float = 0.05


@REGISTRY.register
class SpearmanCorrelationTestTool(BaseTool):
    name = "spearman_correlation_test"
    description = "Nonparametric (rank-based) correlation test between two columns."
    category = "hypothesis_test"
    input_model = SpearmanCorrelationTestInput

    def run(self, ctx: ToolContext, params: SpearmanCorrelationTestInput) -> BaseModel:
        df = ctx.data_manager.load(params.dataset_id)
        paired = df[[params.column_a, params.column_b]].dropna()
        rho, p = scipy_stats.spearmanr(paired[params.column_a], paired[params.column_b])
        sig = "significant" if p < params.alpha else "not significant"
        return StatResult(
            tool_name=self.name,
            test_name="spearman_correlation_test",
            statistic=float(rho),
            p_value=float(p),
            sample_sizes={"pairs": len(paired)},
            assumptions=[check_independence_note()],
            interpretation=(
                f"Spearman correlation between '{params.column_a}' and '{params.column_b}' is {sig} "
                f"(rho={rho:.3f}, p={p:.4f}, n={len(paired)})."
            ),
        )


class ChiSquareIndependenceInput(ToolInput):
    column_a: str
    column_b: str
    alpha: float = 0.05


@REGISTRY.register
class ChiSquareIndependenceTool(BaseTool):
    name = "chi_square_test_independence"
    description = "Test independence between two categorical columns using a chi-square contingency test."
    category = "hypothesis_test"
    input_model = ChiSquareIndependenceInput

    def run(self, ctx: ToolContext, params: ChiSquareIndependenceInput) -> BaseModel:
        df = ctx.data_manager.load(params.dataset_id)
        contingency = df[[params.column_a, params.column_b]].dropna()
        table = contingency.groupby([params.column_a, params.column_b]).size().unstack(fill_value=0)
        chi2, p, dof, expected = scipy_stats.chi2_contingency(table)
        low_expected_pct = float((expected < 5).mean())
        assumption = check_sample_size(len(contingency), 20, "observations")
        assumption_low_cells = (
            assumption.model_copy(update={"name": "expected_cell_counts", "passed": low_expected_pct <= 0.2,
                                           "detail": f"{low_expected_pct*100:.0f}% of expected cell counts are below 5."})
        )
        sig = "significant" if p < params.alpha else "not significant"
        n = len(table.to_numpy().flatten())
        cramers_v = float(np.sqrt(chi2 / (contingency.shape[0] * (min(table.shape) - 1)))) if min(table.shape) > 1 else 0.0
        return StatResult(
            tool_name=self.name,
            test_name="chi_square_test_independence",
            statistic=float(chi2),
            p_value=float(p),
            degrees_of_freedom=float(dof),
            effect_size={"cramers_v": cramers_v},
            sample_sizes={"observations": len(contingency)},
            assumptions=[assumption, assumption_low_cells],
            assumptions_met=assumption_low_cells.passed,
            interpretation=(
                f"Chi-square test of independence between '{params.column_a}' and '{params.column_b}' "
                f"is {sig} (chi2={chi2:.3f}, df={dof}, p={p:.4f}, Cramer's V={cramers_v:.3f})."
            ),
            warnings=[] if assumption_low_cells.passed else ["More than 20% of expected cell counts are below 5; chi-square approximation may be unreliable."],
        )


class ChiSquareGoodnessOfFitInput(ToolInput):
    column: str
    expected: list[float] | None = None
    alpha: float = 0.05


@REGISTRY.register
class ChiSquareGoodnessOfFitTool(BaseTool):
    name = "chi_square_goodness_of_fit"
    description = "Test whether a categorical column's observed frequencies match an expected distribution (uniform by default)."
    category = "hypothesis_test"
    input_model = ChiSquareGoodnessOfFitInput

    def run(self, ctx: ToolContext, params: ChiSquareGoodnessOfFitInput) -> BaseModel:
        df = ctx.data_manager.load(params.dataset_id)
        observed = df[params.column].dropna().value_counts().sort_index()
        expected = params.expected
        if expected is not None:
            expected_counts = np.array(expected) / sum(expected) * observed.sum()
        else:
            expected_counts = None
        res = scipy_stats.chisquare(observed.to_numpy(), f_exp=expected_counts)
        sig = "significant" if res.pvalue < params.alpha else "not significant"
        return StatResult(
            tool_name=self.name,
            test_name="chi_square_goodness_of_fit",
            statistic=float(res.statistic),
            p_value=float(res.pvalue),
            degrees_of_freedom=float(len(observed) - 1),
            sample_sizes={"observations": int(observed.sum())},
            assumptions=[check_sample_size(int(observed.sum()), 20, "observations")],
            interpretation=(
                f"Chi-square goodness-of-fit test on '{params.column}' is {sig} "
                f"(chi2={res.statistic:.3f}, p={res.pvalue:.4f})."
            ),
        )
