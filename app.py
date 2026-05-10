

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import seaborn as sns
import streamlit as st
import warnings
from matplotlib import ticker
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

T = {
    "bg_app": "#0f1116",
    "bg_sidebar": "#12151d",
    "bg_card": "#181c26",
    "border": "#2a3142",
    "text_main": "#e8e6e3",
    "text_muted": "#8f8c85",
    "accent_gold": "#d4af6a",
    "accent_sage": "#6b9074",
    "accent_copper": "#b8735a",
    "plot_face": "#1a1f2a",
    "plot_grid": "#2d3548",
}

PLOTLY_SEQUENCE = ["#d4af6a", "#6b9074", "#7eb8c9", "#b8735a", "#9a92b4", "#c49a6c"]

FEATURES = ["year", "sale_year", "month", "originalmillionusd"]
TARGET = "adjustedmillionusd"

ROOT = Path(__file__).resolve().parent.parent
CSV_CANDIDATES = [
    ROOT / "paintings_cleaned.csv",
    ROOT / "lab3-4" / "paintings_cleaned.csv",
    ROOT / "lab6" / "paintings_cleaned.csv",
]

st.set_page_config(
    page_title="Atelier — Paintings Pipeline",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,500;0,600;1,500&family=Source+Sans+3:wght@400;500;600&display=swap');
    html, body, [class*="css"] {{
        font-family: 'Source Sans 3', 'Segoe UI', sans-serif;
    }}
    .stApp {{
        background-color: {T["bg_app"]};
        background-image:
            radial-gradient(ellipse 120% 80% at 20% -20%, rgba(212, 175, 106, 0.07), transparent),
            radial-gradient(ellipse 80% 60% at 100% 0%, rgba(107, 144, 116, 0.06), transparent);
        color: {T["text_main"]};
    }}
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {T["bg_sidebar"]} 0%, {T["bg_app"]} 100%) !important;
        border-right: 1px solid {T["border"]};
    }}
    [data-testid="stSidebar"] * {{ color: {T["text_main"]} !important; }}
    h1 {{
        font-family: 'Cormorant Garamond', Georgia, serif !important;
        font-weight: 600 !important;
        letter-spacing: 0.03em !important;
        color: {T["text_main"]} !important;
    }}
    h2, h3, .stMarkdown h2, .stMarkdown h3 {{
        font-family: 'Cormorant Garamond', Georgia, serif !important;
        color: {T["accent_gold"]} !important;
    }}
    p, span, label {{ color: {T["text_main"]}; }}
    [data-testid="stMetric"] {{
        background: {T["bg_card"]} !important;
        border: 1px solid {T["border"]};
        border-radius: 14px !important;
        box-shadow: 0 10px 40px rgba(0,0,0,0.25);
        padding: 1rem 1.1rem !important;
    }}
    [data-testid="stMetricValue"] {{
        color: {T["accent_gold"]} !important;
        font-weight: 600 !important;
    }}
    [data-testid="stMetricLabel"] {{ color: {T["text_muted"]} !important; }}
    .hero-title {{
        font-family: 'Cormorant Garamond', Georgia, serif;
        font-size: clamp(2.1rem, 4vw, 2.85rem);
        font-weight: 600;
        color: {T["text_main"]};
        letter-spacing: 0.04em;
        margin-bottom: 0.35rem;
    }}
    .hero-sub {{
        color: {T["text_muted"]};
        font-size: 1rem;
        max-width: 42rem;
        margin: 0 auto;
        line-height: 1.55;
    }}
    .section-title {{
        font-family: 'Cormorant Garamond', Georgia, serif;
        font-size: 1.55rem;
        font-weight: 600;
        color: {T["accent_gold"]};
        border-bottom: 1px solid {T["border"]};
        padding-bottom: 0.4rem;
        margin: 1.25rem 0 1rem 0;
    }}
    .insight-box {{
        background: {T["bg_card"]};
        border: 1px solid {T["border"]};
        border-radius: 12px;
        padding: 14px 18px;
        margin: 10px 0;
        border-left: 3px solid {T["accent_sage"]};
        color: {T["text_main"]};
        line-height: 1.55;
    }}
    .pipeline-badge {{
        display: inline-block;
        font-size: 0.72rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        padding: 3px 10px;
        border-radius: 4px;
        background: rgba(212, 175, 106, 0.15);
        color: {T["accent_gold"]};
        margin-right: 10px;
        font-weight: 600;
        border: 1px solid rgba(212, 175, 106, 0.35);
    }}
    hr {{ border-color: {T["border"]} !important; opacity: 0.85; }}
    div[data-testid="stDecoration"] {{
        background: rgba(212,175,106,0.12);
    }}
    .stTabs [data-baseweb="tab-list"] {{
        background: {T["bg_card"]} !important;
        border-radius: 10px;
        gap: 4px;
        padding: 4px;
        border: 1px solid {T["border"]};
    }}
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(135deg, rgba(212,175,106,0.25), rgba(107,144,116,0.18)) !important;
        border-radius: 8px !important;
        color: {T["text_main"]} !important;
        font-weight: 600 !important;
    }}
</style>
""",
    unsafe_allow_html=True,
)

plotly_theme = dict(
    template="plotly_dark",
    paper_bgcolor=T["bg_app"],
    plot_bgcolor=T["plot_face"],
    font=dict(family="Source Sans Pro, sans-serif", color=T["text_main"], size=13),
    colorway=PLOTLY_SEQUENCE,
    xaxis=dict(gridcolor=T["plot_grid"], linecolor=T["border"], zeroline=False),
    yaxis=dict(gridcolor=T["plot_grid"], linecolor=T["border"], zeroline=False),
)


def mpl_setup():
    plt.rcParams.update(
        {
            "figure.facecolor": T["bg_app"],
            "axes.facecolor": T["plot_face"],
            "axes.edgecolor": T["border"],
            "axes.labelcolor": T["text_main"],
            "xtick.color": T["text_muted"],
            "ytick.color": T["text_muted"],
            "text.color": T["text_main"],
            "grid.color": T["plot_grid"],
            "grid.alpha": 0.45,
            "axes.grid": True,
            "axes.titlesize": 12,
            "axes.titleweight": "600",
            "font.family": "sans-serif",
        }
    )


mpl_setup()


def resolve_csv() -> Path:
    for p in CSV_CANDIDATES:
        if p.is_file():
            return p
    raise FileNotFoundError(
        "paintings_cleaned.csv not found. Place at repo root, lab3-4/, or lab6/."
    )


@st.cache_data
def load_paintings() -> pd.DataFrame:
    path = resolve_csv()
    df = pd.read_csv(path)
    df["date_of_sale"] = pd.to_datetime(df["date_of_sale"], errors="coerce")
    df["month"] = df["date_of_sale"].dt.month

    def classify_house(h: object) -> str:
        h = str(h)
        if "Christie" in h:
            return "Christie's"
        if "Sotheby" in h:
            return "Sotheby's"
        if "Private" in h:
            return "Private Sale"
        return "Other"

    df["house_category"] = df["auction_house"].apply(classify_house)
    return df


def model_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df[FEATURES + [TARGET]].dropna().reset_index(drop=True)


def mets(y_t, y_p):
    return {
        "R²": r2_score(y_t, y_p),
        "RMSE": mean_squared_error(y_t, y_p) ** 0.5,
        "MAE": mean_absolute_error(y_t, y_p),
    }


@st.cache_data
def regression_bundle(test_size: float, random_state: int) -> dict:
    df_full = load_paintings()
    df_m = model_frame(df_full)
    X = df_m[FEATURES]
    y = df_m[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=int(random_state)
    )
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    ols = LinearRegression().fit(X_train_sc, y_train)
    ridge = Ridge(alpha=1.0).fit(X_train_sc, y_train)
    lasso = Lasso(alpha=0.1, max_iter=10000).fit(X_train_sc, y_train)

    pred_tr = ols.predict(X_train_sc)
    pred_te = ols.predict(X_test_sc)

    kf = KFold(n_splits=5, shuffle=True, random_state=int(random_state))
    pipe_ols = Pipeline([("sc", StandardScaler()), ("m", LinearRegression())])
    cv_r2 = cross_val_score(pipe_ols, X, y, cv=kf, scoring="r2")

    coef_df = pd.DataFrame(
        {
            "Feature": FEATURES,
            "OLS": ols.coef_,
            "Ridge": ridge.coef_,
            "Lasso": lasso.coef_,
        }
    )

    resid_test = np.asarray(y_test) - pred_te

    return {
        "df_model": df_m,
        "ols": ols,
        "pred_train": pred_tr,
        "pred_test": pred_te,
        "y_train": y_train,
        "y_test": y_test,
        "metrics_train": mets(y_train, pred_tr),
        "metrics_test": mets(y_test, pred_te),
        "cv_r2_mean": float(cv_r2.mean()),
        "coef_df": coef_df,
        "intercept": float(ols.intercept_),
        "scaler": scaler,
        "resid_test": resid_test,
        "fitted_test": pred_te,
    }


# ─── Load ───
try:
    df_raw = load_paintings()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

df_analysis = df_raw.dropna(subset=[TARGET]).copy()
MODEL_N = len(model_frame(df_raw))


with st.sidebar:
    st.markdown(
        f'<p style="color:{T["accent_gold"]}; font-family:Georgia,serif;font-size:1.35rem;margin:0;">Atelier dashboard</p>',
        unsafe_allow_html=True,
    )
    st.caption("Most expensive paintings · full pipeline")
    st.divider()
    page = st.radio(
        "Navigate",
        [
            "Overview",
            "Collection",
            "Preprocessing",
            "Exploration",
            "Visual narrative",
            "Regression",
        ],
    )
    st.divider()
    test_frac = st.slider("Hold-out fraction", 0.15, 0.35, 0.20, 0.05)
    seed = st.number_input("Random state", value=42, step=1)
    st.divider()
    st.caption(f"Source: `{resolve_csv().name}`")

bund = regression_bundle(test_size=test_frac, random_state=int(seed))

# ═══ Overview ═══
if page == "Overview":
    st.markdown(
        f'<p class="hero-title" style="text-align:center;">Paintings pipeline</p>'
        f'<p class="hero-sub" style="text-align:center;">From Wikipedia extract to exploratory graphics and linear regression on adjusted prices '
        "(millions&nbsp;USD)."
        '</p>',
        unsafe_allow_html=True,
    )

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Records", len(df_raw))
    r2.metric("Regression set (complete)", MODEL_N)
    r3.metric("Mean adjusted · M USD", f"{df_analysis[TARGET].mean():.2f}")
    r4.metric("OLS test R²", f"{bund['metrics_test']['R²']:.3f}")

    st.markdown('<div class="section-title">Analytical pathway</div>', unsafe_allow_html=True)
    for lab, tit, dsc in [
        ("1 · Source", "Acquisition", "`requests` · `BeautifulSoup` → spreadsheet of marquee sales."),
        ("2 · Data", "Cleaning", "Notebook workflow → **`paintings_cleaned.csv`**."),
        ("3 · EDA", "Exploration", "Distributions · correlations · relationships."),
        ("4 · App", "Visualization", "Streamlit — timelines, strata, price patterns."),
        ("5 · Model", "Regression", "OLS / Ridge / Lasso on **`adjustedmillionusd`**."),
    ]:
        st.markdown(
            f'<div class="insight-box"><span class="pipeline-badge">{lab}</span><strong>{tit}</strong> — {dsc}</div>',
            unsafe_allow_html=True,
        )

    ys = df_analysis.groupby("sale_year").agg(count=(TARGET, "size"), mean_px=(TARGET, "mean")).reset_index()
    fig_o = px.line(
        ys,
        x="sale_year",
        y="mean_px",
        markers=True,
        title="Mean inflation-adjusted price by sale year",
        labels={"mean_px": "Mean adj. · M USD", "sale_year": "Sale year"},
    )
    fig_o.update_traces(line_color=T["accent_gold"], marker=dict(size=7))
    fig_o.update_layout(height=340, margin=dict(l=40, r=20, t=52, b=36), **plotly_theme)
    st.plotly_chart(fig_o, use_container_width=True)

    st.markdown('<div class="section-title">Highest adjusted prices — table</div>', unsafe_allow_html=True)
    tops = df_raw.nlargest(10, TARGET)[["name", "artist", "sale_year", TARGET, "originalmillionusd"]]
    tops.index = range(1, len(tops) + 1)
    st.dataframe(
        tops.style.format({TARGET: "{:.2f}", "originalmillionusd": "{:.2f}"}).background_gradient(
            subset=[TARGET], cmap="copper_r", axis=None, vmin=tops[TARGET].min(), vmax=tops[TARGET].max()
        ),
        use_container_width=True,
    )

# ═══ Collection ═══
elif page == "Collection":
    st.markdown('<div class="section-title">Data collection</div>', unsafe_allow_html=True)
    st.markdown(
        """
        Wikipedia tabular article **« List of most expensive paintings »** was harvested with **`requests`**
        and **`BeautifulSoup`** (see scraping notebook under `lab3-4/`). The cleaned export feeds every downstream step.
        """
    )
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Paintings presently in frame", len(df_raw))
        st.metric("Artists tracked", df_raw["artist"].nunique())
    with c2:
        yrs = pd.to_numeric(df_raw["sale_year"], errors="coerce")
        st.metric(
            "Span of realised sales · years",
            f"{int(np.nanmin(yrs.values)):.0f} — {int(np.nanmax(yrs.values)):.0f}",
        )
    st.markdown("Preview")
    st.dataframe(df_raw.head(12), use_container_width=True, height=440)

# ═══ Preprocessing ═══
elif page == "Preprocessing":
    st.markdown('<div class="section-title">Preprocessing</div>', unsafe_allow_html=True)
    tabs = st.tabs(["Narrative", "Missing inventory", "Engineered cues"])
    with tabs[0]:
        st.markdown(
            """
            **Column hygiene** clears bracketed references; monetary fields become numeric **M USD**;
            **`date_of_sale`** becomes datetime powering **`month`** (used in regression) and reporting charts.
            **Duplicate sales** suppressed per notebook rules.
            """
        )
        st.info("Operational table in repository: **`paintings_cleaned.csv`**.")
    with tabs[1]:
        miss = df_raw.isnull().sum()
        miss = miss[miss > 0].sort_values(ascending=False)
        if len(miss) == 0:
            st.success("No unresolved nulls.")
        else:
            fig_m = px.bar(
                x=miss.values,
                y=miss.index.astype(str),
                orientation="h",
                labels={"x": "Count", "y": ""},
                title="Residual missingness",
            )
            fig_m.update_traces(marker_color=T["accent_gold"])
            fig_m.update_layout(height=max(260, len(miss) * 28), margin=dict(l=20, r=40, t=54, b=36), **plotly_theme)
            st.plotly_chart(fig_m, use_container_width=True)
        show_cols = ["year", "sale_year", "month", TARGET, "originalmillionusd"]
        show_cols = [c for c in show_cols if c in df_raw.columns]
        st.caption(".dtypes excerpt")
        st.dataframe(df_raw[show_cols].dtypes.astype(str).to_frame("dtype"))
    with tabs[2]:
        st.dataframe(df_raw[["auction_house", "house_category", "date_of_sale", "month"]].head(22), height=460)

# ═══ Exploration ═══
elif page == "Exploration":
    st.markdown('<div class="section-title">Exploratory analysis</div>', unsafe_allow_html=True)
    dm = bund["df_model"]

    cc1, cc2 = st.columns(2)
    with cc1:
        fig_d, axd = plt.subplots(figsize=(6.8, 3.9))
        sns.histplot(dm[TARGET], kde=True, ax=axd, color=T["accent_gold"], edgecolor=T["border"], line_kws=dict(linewidth=1.8))
        axd.set_title("Adjusted price density")
        axd.set_xlabel("M USD")
        st.pyplot(fig_d)
    with cc2:
        fig_bp, axb = plt.subplots(figsize=(6.8, 3.9))
        order_h = df_analysis["house_category"].value_counts().index.tolist()
        sns.boxplot(data=df_analysis, x="house_category", y=TARGET, order=order_h, ax=axb, palette=PLOTLY_SEQUENCE)
        plt.setp(axb.xaxis.get_majorticklabels(), rotation=22, ha="right")
        axb.set_title("Price by disposition channel · C→Q glance")
        axb.set_xlabel("")
        axb.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0f}"))
        st.pyplot(fig_bp)

    corr = dm[FEATURES + [TARGET]].corr()
    fig_hm, axh = plt.subplots(figsize=(7, 5.8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="Greys_r", center=0,
        square=True, linewidths=1.16, ax=axh, linecolor=T["border"],
        cbar_kws={"label": "ρ"}, vmin=-1, vmax=1,
    )
    axh.set_title("Feature–target correlations")
    st.pyplot(fig_hm)

    st.markdown("*Pairwise scatter vs **adjusted USD** · complete-case regression rows.*")
    fig_sc, axes = plt.subplots(2, 2, figsize=(10.8, 8.8))
    for ax, ft in zip(axes.ravel(), FEATURES):
        ax.scatter(dm[ft], dm[TARGET], alpha=0.55, s=56, color=T["accent_sage"], edgecolors="none", zorder=2)
        m, b = np.polyfit(dm[ft], dm[TARGET], 1)
        xs = np.linspace(dm[ft].min(), dm[ft].max(), 120)
        ax.plot(xs, m * xs + b, "--", color=T["accent_gold"], linewidth=2, alpha=0.85, zorder=3)
        ax.set_xlabel(ft)
        ax.set_ylabel(TARGET)
    plt.tight_layout()
    st.pyplot(fig_sc)

    assoc = corr[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)
    sdf = assoc.iloc[::-1].reset_index()
    sdf.columns = ["feature", "rho"]
    sdf["abs_rho"] = np.abs(sdf["rho"].astype(float))
    strong_fig = px.bar(
        sdf,
        x="rho",
        y="feature",
        orientation="h",
        color="abs_rho",
        color_continuous_scale=[[0, T["accent_copper"]], [0.5, T["text_muted"]], [1, T["accent_sage"]]],
        labels={"color": "|ρ|", "rho": "ρ"},
        title="Strength of Pearson association with adjusted price",
    )
    strong_fig.update_layout(height=340, margin=dict(l=20, r=44, t=54, b=34), **plotly_theme)
    st.plotly_chart(strong_fig, use_container_width=True)

# ═══ Visual narrative ═══
elif page == "Visual narrative":
    st.markdown('<div class="section-title">Visual narrative</div>', unsafe_allow_html=True)
    df = df_analysis

    yearly = df.groupby("sale_year").size().rename("transactions").reset_index()
    fy = px.bar(
        yearly,
        x="sale_year",
        y="transactions",
        title="Number of marquee sales accounted for · by year",
        labels={"transactions": "Count", "sale_year": ""},
        text_auto=True,
    )
    fy.update_traces(marker=dict(color=PLOTLY_SEQUENCE[1], line_width=0))
    fy.update_layout(height=400, bargap=0.22, margin=dict(l=42, r=34, t=56, b=52), **plotly_theme)
    st.plotly_chart(fy, use_container_width=True)

    rev_y = df.groupby("sale_year")[TARGET].sum().rename("usd_m_adj").reset_index()
    fz = px.area(
        rev_y,
        x="sale_year",
        y="usd_m_adj",
        title="Cumulative inflation-adjusted turnover · annual sum (M USD metaphor)",
        labels={"usd_m_adj": "Σ adj. · M USD"},
    )
    fz.update_traces(fill="tozeroy", line_color=T["accent_gold"], fillcolor="rgba(212,175,106,0.22)")
    fz.update_layout(height=360, margin=dict(l=40, r=32, t=52, b=52), **plotly_theme)
    st.plotly_chart(fz, use_container_width=True)

    fig_pdf, axp = plt.subplots(figsize=(9.2, 3.95))
    df[TARGET].hist(bins=min(34, len(df) // 2), ax=axp, color=T["accent_gold"], edgecolor=T["border"])
    axp.set_title("Empirical marginal — adjusted hammer prices")
    axp.set_xlabel("M USD · adjusted")
    st.pyplot(fig_pdf)

    if st.checkbox("Expose raw tableau (first eighty rows)", value=False):
        cols = ["name", "artist", "year", TARGET, "originalmillionusd", "house_category", "sale_year"]
        st.dataframe(df[cols].head(80), use_container_width=True)

    a1, a2 = st.columns(2)
    with a1:
        art = df["artist"].value_counts().head(14).iloc[::-1]
        fx = px.bar(x=art.values, y=art.index.astype(str), orientation="h", title="Frequent attribution · top strata")
        fx.update_layout(height=470, margin=dict(l=188, r=42, t=46, b=36), **plotly_theme)
        fx.update_traces(marker_color=T["accent_sage"])
        st.plotly_chart(fx, use_container_width=True)
    with a2:
        hou = df["house_category"].value_counts().iloc[::-1]
        fw = px.pie(values=hou.values, names=hou.index, title="Structural share · channel", hole=0.42)
        fw.update_layout(height=470, margin=dict(l=46, r=52, t=62, b=46), **plotly_theme)
        fw.update_traces(marker=dict(line=dict(color=T["bg_app"], width=1)), textfont=dict(color=T["text_main"]))
        st.plotly_chart(fw, use_container_width=True)

    g_data, g_labels, g_colors = [], [], []
    for label, col in zip(
        ["Christie's", "Sotheby's", "Private Sale"], [T["accent_gold"], T["accent_sage"], T["accent_copper"]]
    ):
        lst = df.loc[df["house_category"] == label, TARGET].tolist()
        if lst:
            g_data.append(lst)
            g_labels.append(label)
            g_colors.append(col)
    if len(g_data) >= 2:
        fig_dist = ff.create_distplot(
            g_data, group_labels=g_labels, bin_size=18, show_rug=False,
            curve_type="normal", histnorm="probability density",
            colors=g_colors,
        )
        fig_dist.update_layout(title="Adjusted price densities · juxtaposed strata", legend=dict(orientation="h", yanchor="bottom"), **plotly_theme)
        st.plotly_chart(fig_dist, use_container_width=True)

    dec_counts = ((df["year"] // 10) * 10).value_counts().sort_index().reset_index()
    dec_counts.columns = ["decade", "n"]
    fd = px.line(dec_counts, x="decade", y="n", markers=True, title="Creations clustered by originating decade · availability")
    fd.update_traces(line_color=PLOTLY_SEQUENCE[2], marker=dict(size=9))
    fd.update_layout(height=350, margin=dict(l=42, r=42, t=54, b=44), **plotly_theme)
    st.plotly_chart(fd, use_container_width=True)

    if st.button("Toggle · principal seller concentration"):
        sc = df["seller"].value_counts().head(18).iloc[::-1]
        st.plotly_chart(
            px.bar(x=sc.values, y=sc.index.astype(str), orientation="h", title="Top sellers cited").update_layout(height=620, margin=dict(l=268, r=54, t=62, b=54), **plotly_theme),
            use_container_width=True,
        )

    fg = px.scatter(
        df,
        x="year",
        y=TARGET,
        color="house_category",
        hover_data=["name", "artist", "sale_year", "originalmillionusd"],
        opacity=0.68,
        size_max=18,
        title="Antiquity of object vs realised adjusted price · colour ⇒ channel",
    )
    fg.update_layout(**plotly_theme, height=500, legend=dict(title="Disposition"))
    st.plotly_chart(fg, use_container_width=True)

# ═══ Regression ═══
elif page == "Regression":
    st.markdown('<div class="section-title">Linear regression</div>', unsafe_allow_html=True)
    st.markdown(
        """
        **Dependent variable:** `adjustedmillionusd` · **Predictors:**
        `year`, `sale_year`, `month` (from sale date), `originalmillionusd`.  
        **Scaling:** `StandardScaler` fitted **only** on training rows (**random state**
        synced with sidebar).
        """
    )

    tm = bund["metrics_test"]
    trm = bund["metrics_train"]

    cc1, cc2, cc3, cc4, cc5, cc6 = st.columns(6)
    cc1.metric("Train R²", f"{trm['R²']:.4f}")
    cc2.metric("Test R²", f"{tm['R²']:.4f}")
    cc3.metric("Train RMSE", f"{trm['RMSE']:.3f}")
    cc4.metric("Test RMSE", f"{tm['RMSE']:.3f}")
    cc5.metric("Test MAE", f"{tm['MAE']:.3f}")
    cc6.metric("CV R² (mean · 5-f.)", f"{bund['cv_r2_mean']:.4f}")

    gap = abs(trm["R²"] - tm["R²"])
    st.caption(
        "Train − test ΔR² = "
        f"**{gap:.4f}** — sizeable gap ⇒ revisit complexity; negligible gap ⇒ no dramatic overfitting for this affine model."
    )

    st.markdown(
        f"**OLS intercept** (β₀) at centred standardized predictors ≈ **`{bund['intercept']:.4f}` M USD."
    )

    st.dataframe(
        bund["coef_df"].style.format({"OLS": "{:.4f}", "Ridge": "{:.4f}", "Lasso": "{:.4f}"}, subset=["OLS", "Ridge", "Lasso"]),
        use_container_width=True,
    )

    co = bund["coef_df"].set_index("Feature")["OLS"].sort_values()
    fig_bc, axb = plt.subplots(figsize=(8.2, 4.2))
    cols_b = [T["accent_copper"] if v < 0 else T["accent_gold"] for v in co.values]
    axb.barh(co.index.astype(str), co.values, color=cols_b, edgecolor=T["border"], linewidth=0.7)
    axb.axvline(0, color=T["text_muted"], linestyle=":", linewidth=1.1)
    axb.set_title("OLS standardized partial slopes")
    st.pyplot(fig_bc)

    rc1, rc2 = st.columns((1, 1))
    yt = np.asarray(bund["y_test"])
    pt = np.asarray(bund["pred_test"])

    with rc1:
        fig_av, axa = plt.subplots(figsize=(5.7, 5.4))
        axa.scatter(yt, pt, alpha=0.68, s=84, facecolors=T["accent_sage"], edgecolors=T["accent_gold"], linewidths=0.45)
        lo, hi = 0.0, float(np.nanmax(np.r_[yt, pt]) * 1.06 + 10)
        axa.plot([lo, hi], [lo, hi], color=T["accent_copper"], ls="--", lw=2, label="y = ŷ diagonal")
        axa.legend(loc="upper left")
        axa.set_xlim(lo, hi)
        axa.set_ylim(lo, hi)
        axa.set_xlabel("Realised adjusted · hold-out")
        axa.set_ylabel("Model prediction")
        axa.set_title("Generalisation · parity plot")
        st.pyplot(fig_av)

    with rc2:
        rez = yt - pt
        fig_rz, axes = plt.subplots(2, 1, figsize=(5.95, 5.85), gridspec_kw={"height_ratios": [1.85, 1.0], "hspace": 0.28})
        axes[0].scatter(pt, rez, alpha=0.66, color=T["accent_gold"], s=74, edgecolors="none")
        axes[0].axhline(0.0, color=T["accent_sage"], ls=":", lw=2)
        axes[0].set_ylabel("Residual")
        axes[0].set_xlabel("")
        axes[0].set_title("Residual choreography · vs fitted · test shard")
        axes[1].hist(rez, bins=18, color=T["accent_sage"], edgecolor=T["border"])
        axes[1].set_xlabel("Residual · M USD adj.")
        axes[1].set_ylabel("Frequency")
        st.pyplot(fig_rz)

    st.success(
        "Interpretive anchor (`lab7/README.md`): **`originalmillionusd`** explains most variance together with "
        "the inflation-adjusted target — caveat if you describe **prediction** before the nominal price exists."
    )

st.markdown(
    f'<div style="text-align:center;color:{T["text_muted"]};font-size:.78rem;margin-top:2.5rem;">'
    "Laboratory artefacts · pedagogical tableau · **`dashboard/examplestreamlit.py`** ◆ **`lab7_paintings.ipynb`** · "
    "**`lab7/README.md`**</div>",
    unsafe_allow_html=True,
)
