# Returns Analytics

## Research question
How should a price series be converted into return, volatility, and drawdown summaries without confusing arithmetic and log returns?

## Methodology and formulas
For positive prices P_t, simple return is r_t=P_t/P_(t-1)-1, while log return is ell_t=log(P_t/P_(t-1)). The annualized arithmetic mean is A times mean(r); annualized sample volatility is sqrt(A) times s_r, where A is the configured periods per year. Maximum drawdown is the minimum of W_t / max(W_0...W_t) - 1, with W_t the compounded return path.

## Assumptions and limitations
- The default annualization factor is 252 and assumes daily observations; it does not adjust for holidays or irregular sampling.
- Volatility uses sample standard deviation (ddof=1); the arithmetic mean is not a compounded CAGR.
- The included 252-return path is generated with NumPy seed 2025. It is synthetic and makes no claim about a traded strategy or real investment performance.
- No benchmark, risk-free rate, fees, taxes, or survivorship-adjusted asset universe is modeled.

## Reproducibility
With Python 3.11+, run python -m pip install -e ".[test]", then python -m pytest and returns-demo --seed 2025. The command writes reports/synthetic_price_path.png and reports/summary.json. There are no data downloads or credentials.

