import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

class StatsmodelsHandler:
    """
    A wrapper for summarizing and visualizing fitted `statsmodels` regression models.

    This class provides clean LaTeX summaries and coefficient plots with 95% confidence intervals,
    optionally applying custom cleaning to variable names and supporting categorical encodings.

    Attributes:
        model (statsmodels model): A fitted statsmodels model (e.g., from OLS, Logit, MixedLM).

    Example:
        >>> import statsmodels.formula.api as smf
        >>> from statsmodels_handler import StatsmodelsHandler, clean_var_name_from_formula
        >>> mod = smf.ols("y ~ C(condition, Treatment(reference='control'))", data=df).fit()
        >>> handler = StatsmodelsHandler(mod)
        >>> handler.to_latex()
        >>> handler.plot(clean_var_name=clean_var_name_from_formula)
        >>> plt.xlabel("Coefficient")
        >>> plt.ylabel("Condition")
        >>> plt.title("Estimated Effects with 95% CIs")
        >>> plt.show()
    """

    def __init__(self, model):
        """
        Initialize with a fitted statsmodels model (OLS, GLM, or MixedLM).
        """
        self.model = model

        # MixedLMResults have fe_params for fixed effects
        if hasattr(model, "fe_params"):
            self.params = model.fe_params
            self.errors = model.bse_fe
            self.conf_int = model.conf_int().loc[model.fe_params.index]
        else:  # OLS/GLM Results
            self.params = model.params
            self.errors = model.bse
            self.conf_int = model.conf_int()

        # Keep only p-values for the params we selected
        self.pvalues = model.pvalues.loc[self.params.index]

    def to_latex(self, beta_digits=2, se_digits=2, p_digits=3, ci_digits=2, print_sci_not=False):
        """
        Print LaTeX-formatted summaries of all parameters in the model.
        """
        for param_name, beta in self.params.items():
            safe_param_name = param_name.replace('_', '\\_')
            se = self.errors[param_name]
            p = self.pvalues[param_name]
            ci_lower, ci_upper = self.conf_int.loc[param_name]

            if p < 0.001:
                p_formatted = f"= {p:.2e}" if print_sci_not else "<0.001"
            else:
                p_formatted = f"= {p:.{p_digits}f}"

            summary = (f"{safe_param_name}: $\\beta = {beta:.{beta_digits}f}$, "
                       f"$SE = {se:.{se_digits}f}$, $p {p_formatted}$, "
                       f"$95\\% CI = [{ci_lower:.{ci_digits}f}, {ci_upper:.{ci_digits}f}]$")
            print(summary)

    def plot(self, exp=False, figsize=(10, 6), color_dict=None, drop_intercept=True, clean_var_name=None):
        """
        Plot model coefficients with confidence intervals and optional color coding.
        Works with OLS/GLM and MixedLM results.
        """
        color_dict = color_dict or {'ns': 'gray', 'neg': '#D41876', 'pos': '#89DAFF'}

        # Build summary DataFrame directly from stored attributes
        sum_res = pd.DataFrame({
            "var": self.params.index,
            "coef": self.params.values,
            "se": self.errors.values,
            "pval": self.pvalues.values,
            "ci_lower": self.conf_int.iloc[:, 0].values,
            "ci_upper": self.conf_int.iloc[:, 1].values,
        })

        if drop_intercept:
            sum_res = sum_res[sum_res["var"] != "Intercept"]

        if exp:
            threshold = 1
            sum_res["coef"] = np.exp(sum_res["coef"])
            sum_res["ci_lower"] = np.exp(sum_res["ci_lower"])
            sum_res["ci_upper"] = np.exp(sum_res["ci_upper"])
        else:
            threshold = 0

        if clean_var_name:
            sum_res["var"] = sum_res["var"].apply(clean_var_name)

        # Error bars
        sum_res["error_lower"] = sum_res["coef"] - sum_res["ci_lower"]
        sum_res["error_upper"] = sum_res["ci_upper"] - sum_res["coef"]

        # Assign colors by sign & significance
        sum_res["color"] = sum_res.apply(
            lambda row: (
                color_dict["pos"] if (row["pval"] < 0.05 and row["coef"] > threshold)
                else color_dict["neg"] if (row["pval"] < 0.05 and row["coef"] < threshold)
                else color_dict["ns"]
            ),
            axis=1,
        )

        # Sort by coefficient for nicer plotting
        sum_res = sum_res.sort_values(by="coef", ascending=True)

        # Plot
        plt.figure(figsize=figsize)
        for _, row in sum_res.iterrows():
            plt.errorbar(
                row["coef"], row["var"],
                xerr=[[row["error_lower"]], [row["error_upper"]]],
                fmt="o", capsize=5, color=row["color"]
            )
        plt.axvline(x=threshold, color="gray", linestyle="dashed")

        return plt

    def clean_var_name_from_formula(term_str):
        import re
        """
        Converts a statsmodels formula term like:
        'C(var_name, Treatment(reference="baseline"))[T.level]'
        or the short form 'C(var_name)[T.level]'
        into a clean label for plotting.

        Also cleans z-scored variable names like 'z_source_reliability' → 'source reliability'.

        Args:
            term_str (str): Variable name from the model output.

        Returns:
            str: Readable label for plotting.
        """
        term_str = term_str.strip()
        clean = lambda x: x.replace('_', ' ')

        # Pattern 1: C(var, Treatment(reference="ref"))[T.level]
        pattern_full = r'C\(([^,]+),\s*Treatment\(reference="([^"]+)"\)\)\[([^\]]+)\]'
        match = re.match(pattern_full, term_str)
        if match:
            question, reference, coef = match.groups()
            if coef.startswith("T."):
                coef = coef[2:]
            return f"{clean(question)} = {clean(coef)}\n(reference = {clean(reference)})"

        # Pattern 2: C(var)[T.level]  (short form, no explicit reference)
        pattern_short = r'C\(([^)]+)\)\[T\.([^\]]+)\]'
        match = re.match(pattern_short, term_str)
        if match:
            question, coef = match.groups()
            return f"{clean(question)} = {clean(coef)}"

        # Pattern 3: z_var_name → strip z_ prefix and clean underscores
        if term_str.startswith('z_'):
            return clean(term_str[2:])

        # Pattern 4: interaction terms like 'z_a:z_b' or 'C(x)[T.y]:z_a'
        if ':' in term_str:
            parts = term_str.split(':')
            cleaned_parts = [StatsmodelsHandler.clean_var_name_from_formula(p) for p in parts]
            return ' × '.join(cleaned_parts)

        return clean(term_str)
