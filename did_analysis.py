"""청주·청원 통합 전후의 재정자립도: 이중차분 계산 실습.

옛 청주시와 통합 청주시를 연결한 자료를 사용한다. 청원군을 합산한
고정 경계 패널이 아니므로 계산값을 통합의 인과효과로 해석하지 않는다.
KOSIS DT_1YL20921, 세입과목 개편 전 기준, 보관된 2001-2025 CSV.
필요 패키지: pandas, matplotlib, statsmodels.
"""

from pathlib import Path
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
import statsmodels.formula.api as smf

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "fiscal_autonomy.csv"
PRE_YEARS = [2009, 2010, 2011]
POST_YEARS = [2015, 2016, 2017, 2018, 2019]
CONTROL_POOL = ["전주시", "원주시", "포항시", "천안시"]
CITY_PROVINCES = {
    "청주시": "충청북도", "충북 청주시": "충청북도",
    "전주시": "전북특별자치도", "원주시": "강원특별자치도",
    "포항시": "경상북도", "천안시": "충청남도",
}


def load_kosis(path):
    """보관된 CP949·2단 헤더 CSV에서 개편 전 기준을 읽는다."""
    raw = pd.read_csv(path, encoding="cp949", header=[0, 1])
    raw.columns = [
        "sido" if i == 0 else "sgg" if i == 1 else f"{yr}_{kind}"
        for i, (yr, kind) in enumerate(raw.columns)
    ]
    cols = [c for c in raw.columns if "세입과목개편전" in c]
    if not cols:
        raise ValueError("세입과목 개편 전 기준의 열이 없습니다.")
    df = raw[["sido", "sgg"] + cols].melt(
        id_vars=["sido", "sgg"], var_name="col", value_name="fa")
    df["year"] = df["col"].str.extract(r"(\d{4})").astype(int)
    df["fa"] = pd.to_numeric(df["fa"], errors="coerce")
    return df.dropna(subset=["fa"])[["sido", "sgg", "year", "fa"]]


def unify_cheongju(df):
    """실습용 이름 연결. 통합 전후의 공간적 단위를 일치시키지 않는다."""
    df = df.copy()
    df["sido"] = df["sido"].replace({
        "전라북도": "전북특별자치도", "강원도": "강원특별자치도"})
    keep = df["sgg"].isin(CITY_PROVINCES)
    keep &= df["sido"].eq(df["sgg"].map(CITY_PROVINCES))
    df = df.loc[keep].copy()
    df["source_sgg"] = df["sgg"]
    df["sgg"] = df["sgg"].replace({"충북 청주시": "청주시"})
    if df.duplicated(["sgg", "year"]).any():
        raise ValueError("도시·연도 중복값의 원자료를 확인해야 합니다.")
    return df.sort_values(["sgg", "year"]).reset_index(drop=True)


def comparison_sample(df, treated, control, pre, post):
    """지정한 모든 도시·연도가 있는지 확인하고 비교 표본을 구성한다."""
    pre, post = list(pre), list(post)
    if not pre or not post or max(pre) >= min(post):
        raise ValueError("사전 기간은 사후 기간보다 앞서야 합니다.")
    if len(set(pre + post)) != len(pre + post) or treated == control:
        raise ValueError("연도와 비교집단의 중복을 확인하십시오.")
    cities, years = [treated, control], pre + post
    sub = df[df["sgg"].isin(cities) & df["year"].isin(years)].copy()
    expected = pd.MultiIndex.from_product([cities, years])
    actual = pd.MultiIndex.from_frame(sub[["sgg", "year"]])
    if actual.has_duplicates or len(expected.difference(actual)):
        raise ValueError("필요한 도시·연도 관측값이 없거나 중복됩니다.")
    if sub["fa"].isna().any():
        raise ValueError("결과변수에 결측값이 있습니다.")
    sub["post"] = sub["year"].isin(post).astype(int)
    sub["treat"] = sub["sgg"].eq(treated).astype(int)
    return sub


def did_2x2(df, treated, control, pre=PRE_YEARS, post=POST_YEARS):
    sub = comparison_sample(df, treated, control, pre, post)
    cells = sub.groupby(["treat", "post"])["fa"].mean().unstack()
    did = ((cells.loc[1, 1] - cells.loc[1, 0])
           - (cells.loc[0, 1] - cells.loc[0, 0]))
    return {
        "treated_pre": float(cells.loc[1, 0]),
        "treated_post": float(cells.loc[1, 1]),
        "control_pre": float(cells.loc[0, 0]),
        "control_post": float(cells.loc[0, 1]),
        "did": float(did), "n_obs": len(sub),
    }


def did_regression(df, treated, control, pre=PRE_YEARS, post=POST_YEARS):
    """계수의 일치를 확인하는 점추정만 반환한다. 검정에는 사용하지 않는다."""
    sub = comparison_sample(df, treated, control, pre, post)
    fit = smf.ols("fa ~ treat * post", data=sub).fit()
    return float(fit.params["treat:post"])


def placebo_pretrend(df, treated, control):
    return did_2x2(df, treated, control,
                   pre=[2005, 2006, 2007, 2008],
                   post=[2009, 2010, 2011])


def plot_trends(df, treated, controls, out):
    available = {f.name for f in font_manager.fontManager.ttflist}
    for font in ["AppleGothic", "Malgun Gothic", "NanumGothic",
                 "Noto Sans CJK KR"]:
        if font in available:
            mpl.rcParams["font.family"] = font
            break
    mpl.rcParams["axes.unicode_minus"] = False
    keep = df[df["sgg"].isin([treated] + controls)
              & df["year"].between(2005, 2022)]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    colors = ["#657785", "#8a7059", "#738268", "#92869c"]
    for city, color in zip(controls, colors):
        sub = keep[keep["sgg"] == city]
        ax.plot(sub["year"], sub["fa"], "--", color=color,
                linewidth=1.2, label=city)
    for original, label in [("충북 청주시", "옛 청주시"),
                             ("청주시", "통합 청주시")]:
        sub = keep[(keep["sgg"] == treated)
                   & (keep["source_sgg"] == original)]
        ax.plot(sub["year"], sub["fa"], "o-", color="#234b6c",
                markersize=3.5, linewidth=2, label=label)
    ax.axvline(2014.5, color="#4c5661", linestyle=":")
    ax.axvspan(2012, 2014.5, color="#dae0e5", alpha=0.5)
    ax.text(2014.7, ax.get_ylim()[1] - 1, "2014년 7월 통합",
            fontsize=9, va="top", color="#4c5661")
    ax.set_xlabel("연도")
    ax.set_xticks([2005, 2008, 2011, 2014, 2017, 2020, 2022])
    ax.set_ylabel("재정자립도 (%, 세입과목 개편 전 기준)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17),
              ncol=3, fontsize=9, frameon=False)
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    df = unify_cheongju(load_kosis(DATA))
    base = did_2x2(df, "청주시", "전주시")
    print("기본 비교:", base)
    print(f"회귀계수: {did_regression(df, '청주시', '전주시'):.3f}")
    print("비교도시별 계산 (단위: %p)")
    for city in CONTROL_POOL:
        r = did_2x2(df, "청주시", city)
        print(f"{city}: DiD = {r['did']:+.2f}")
    print("사전 기간 위약 비교 (단위: %p)")
    for city in CONTROL_POOL:
        r = placebo_pretrend(df, "청주시", city)
        print(f"{city}: placebo DiD = {r['did']:+.3f}")
    plot_trends(df, "청주시", CONTROL_POOL, HERE / "figures" / "trends.png")
