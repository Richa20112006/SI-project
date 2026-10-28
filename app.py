import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Airbnb Listing Analysis", page_icon="🏡", layout="wide")
px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = ["#FF5A5F", "#00A699", "#FC642D", "#484848", "#767676"]

NUMERIC = ["price", "accommodates", "bedrooms", "number_of_reviews", "rating"]
FEATURES = ["accommodates", "bedrooms", "number_of_reviews", "rating", "city", "room_type"]


# ---------------------------------------------------------------- helpers
@st.cache_data
def load_data():
    if not os.path.exists("airbnb.csv"):
        raise FileNotFoundError("airbnb.csv not found. Run generate_data.py first.")
    d = pd.read_csv("airbnb.csv")
    d["rating"] = d["rating"].fillna(d["rating"].median())
    d = d[d["price"] > 0].copy()
    d["log_price"] = np.log(d["price"])
    d["price_band"] = pd.qcut(d["price"], 3, labels=["Low", "Medium", "High"]).astype(str)
    return d


@st.cache_data
def scrape_links():
    """Scrape the list of listing files from Inside Airbnb."""
    import requests
    from bs4 import BeautifulSoup

    html = requests.get("http://insideairbnb.com/get-the-data/",
                        headers={"User-Agent": "Mozilla/5.0"}, timeout=15).text
    soup = BeautifulSoup(html, "html.parser")
    rows = []
    for a in soup.find_all("a", href=True):
        link = a["href"]
        if link.endswith("listings.csv.gz"):
            parts = link.split("/")
            rows.append({"country": parts[3], "city": parts[-4], "date": parts[-3], "url": link})
    return pd.DataFrame(rows)


def plot(fig, height=380):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, width="stretch")


def conclude(p, alpha, h0, h1):
    """Show the test result in plain English."""
    if p < alpha:
        st.error(f"**p = {p:.4f} < α = {alpha}  →  Reject H0**\n\nThere is significant evidence that {h1}.")
    else:
        st.success(f"**p = {p:.4f} ≥ α = {alpha}  →  Fail to reject H0**\n\nNo significant evidence against: {h0}.")


def skew_text(s):
    if abs(s) < 0.5:
        return "roughly symmetric"
    return "right-skewed (long tail of high values)" if s > 0 else "left-skewed (long tail of low values)"


def design_matrix(d, features):
    return pd.get_dummies(d[features], drop_first=True).astype(float)


@st.cache_resource
def train_final_model(name):
    """Model used by the Price predictor page (trained on the full dataset)."""
    X = design_matrix(df, FEATURES)
    scaler = StandardScaler().fit(X)
    model = {"Ridge": Ridge(alpha=10), "Lasso": Lasso(alpha=0.01),
             "Multiple (OLS)": LinearRegression()}[name].fit(scaler.transform(X), df["log_price"])
    resid_std = np.std(df["log_price"] - model.predict(scaler.transform(X)))
    return model, scaler, X.columns, resid_std


def predict_price(name, **inputs):
    model, scaler, cols, _ = train_final_model(name)
    row = pd.get_dummies(pd.DataFrame([inputs])).reindex(columns=cols, fill_value=0).astype(float)
    return float(np.exp(model.predict(scaler.transform(row))[0]))


try:
    df = load_data()
except Exception as exc:
    st.error(str(exc))
    st.stop()

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("🏡 Airbnb Analysis")
    page = st.radio("Navigate", ["📊 Overview", "🌐 Web scraping", "🔍 EDA", "📈 Distributions",
                                 "🎲 Central Limit Theorem", "🧪 Hypothesis tests",
                                 "📉 Regression", "💰 Price predictor"], label_visibility="collapsed")
    st.divider()
    st.subheader("Filters")
    cities = sorted(df["city"].unique())
    selected_cities = st.multiselect("Cities", cities, default=cities)
    rooms = sorted(df["room_type"].unique())
    selected_rooms = st.multiselect("Room types", rooms, default=rooms)
    st.caption("Filters apply to all pages except the Price predictor.")

data = df[df["city"].isin(selected_cities) & df["room_type"].isin(selected_rooms)].copy()
if data.empty:
    st.warning("Select at least one city and one room type in the sidebar.")
    st.stop()
st.sidebar.caption(f"Showing {len(data):,} of {len(df):,} listings")

# ---------------------------------------------------------------- Overview
if page == "📊 Overview":
    st.title("📊 Overview")
    k = st.columns(5)
    k[0].metric("Listings", f"{len(data):,}", border=True)
    k[1].metric("Average price", f"₹{data['price'].mean():,.0f}", border=True)
    k[2].metric("Median price", f"₹{data['price'].median():,.0f}", border=True)
    k[3].metric("Average rating", f"{data['rating'].mean():.2f} / 5", border=True)
    k[4].metric("Avg guests", f"{data['accommodates'].mean():.1f}", border=True)

    by_city = data.groupby("city")["price"].mean().sort_values()
    by_room = data.groupby("room_type")["price"].mean().sort_values()
    st.info(f"💡 **{by_city.index[-1]}** is the most expensive city (₹{by_city.iloc[-1]:,.0f} on average) and "
            f"**{by_city.index[0]}** the cheapest (₹{by_city.iloc[0]:,.0f}). "
            f"**{by_room.index[-1]}** listings cost the most (₹{by_room.iloc[-1]:,.0f}).")

    left, right = st.columns(2)
    with left:
        plot(px.bar(by_city.reset_index(), x="city", y="price", text_auto=".0f",
                    title="Average price by city"))
    with right:
        plot(px.bar(by_room.reset_index(), x="room_type", y="price", text_auto=".0f",
                    title="Average price by room type"))

    left, right = st.columns(2)
    with left:
        pivot = data.pivot_table(index="city", columns="room_type", values="price", aggfunc="mean")
        plot(px.imshow(pivot, text_auto=".0f", color_continuous_scale="Reds",
                       title="Average price: city × room type"))
    with right:
        plot(px.histogram(data, x="price", nbins=40, color="room_type", marginal="box",
                          title="Nightly price distribution"))

    with st.expander("Browse the listings"):
        lo, hi = int(data["price"].min()), int(data["price"].max())
        rng = st.slider("Price range (₹)", lo, hi, (lo, hi))
        view = data[data["price"].between(*rng)].drop(columns=["log_price", "price_band"])
        st.write(f"**{len(view):,}** listings")
        st.dataframe(view, width="stretch")
        st.download_button("Download CSV", view.to_csv(index=False), "airbnb_filtered.csv")

# ---------------------------------------------------------------- Web scraping
elif page == "🌐 Web scraping":
    st.title("🌐 Web scraping – Inside Airbnb")
    st.write("This page uses `requests` + `BeautifulSoup` to read the *Get the data* page and "
             "collect every `listings.csv.gz` link.")
    st.caption("The rest of the app uses the synthetic airbnb.csv, not these scraped files.")
    with st.expander("How the scraper works"):
        st.code('''html = requests.get(url, headers=..., timeout=15).text
soup = BeautifulSoup(html, "html.parser")
for a in soup.find_all("a", href=True):
    if a["href"].endswith("listings.csv.gz"):
        ...  # split the URL into country / city / date''', language="python")
    if st.button("Scrape now", type="primary"):
        try:
            with st.spinner("Scraping..."):
                links = scrape_links()
            c = st.columns(3)
            c[0].metric("Files found", len(links), border=True)
            c[1].metric("Countries", links["country"].nunique(), border=True)
            c[2].metric("Cities", links["city"].nunique(), border=True)
            top = links["country"].value_counts().head(15).reset_index()
            plot(px.bar(top, x="country", y="count", title="Files per country (top 15)"))
            country = st.selectbox("Filter by country", ["All"] + sorted(links["country"].unique()))
            shown = links if country == "All" else links[links["country"] == country]
            st.dataframe(shown, width="stretch")
        except Exception as exc:
            st.error(f"Scraping failed – check your internet connection. ({exc})")

# ---------------------------------------------------------------- EDA
elif page == "🔍 EDA":
    st.title("🔍 Exploratory data analysis")
    t1, t2, t3, t4, t5 = st.tabs(["Data", "Quality", "Outliers", "Correlation", "Relationships"])

    with t1:
        st.dataframe(data.drop(columns=["log_price", "price_band"]).head(50), width="stretch")
        st.caption(f"{data.shape[0]:,} rows × {data.shape[1] - 2} columns")
    with t2:
        c1, c2 = st.columns(2)
        c1.write("**Missing values**")
        c1.dataframe(data.isna().sum().rename("missing"), width="stretch")
        c1.metric("Duplicate rows", int(data.duplicated().sum()), border=True)
        c2.write("**Summary statistics**")
        c2.dataframe(data[NUMERIC].describe().T.round(2), width="stretch")
    with t3:
        group = st.selectbox("Group by", ["city", "room_type"])
        plot(px.box(data, x=group, y="price", color=group, title="Price outliers appear as dots"))
        q1, q3 = data["price"].quantile([0.25, 0.75])
        n_out = int((data["price"] > q3 + 1.5 * (q3 - q1)).sum())
        st.info(f"IQR rule: **{n_out}** listings ({n_out / len(data):.1%}) are priced above "
                f"₹{q3 + 1.5 * (q3 - q1):,.0f}.")
    with t4:
        corr = data[NUMERIC].corr()
        plot(px.imshow(corr, text_auto=".2f", zmin=-1, zmax=1, color_continuous_scale="RdBu_r",
                       title="Correlation matrix"), height=450)
        st.caption("Close to +1 or -1 = strong relationship; close to 0 = little linear relationship.")
    with t5:
        c1, c2 = st.columns(2)
        x = c1.selectbox("X axis", [c for c in NUMERIC if c != "price"])
        color = c2.selectbox("Colour by", ["city", "room_type"])
        plot(px.scatter(data, x=x, y="price", color=color, opacity=0.5, title=f"Price vs {x}"))

# ---------------------------------------------------------------- Distributions
elif page == "📈 Distributions":
    st.title("📈 Distributions")
    c1, c2 = st.columns([2, 1])
    col = c1.selectbox("Column", NUMERIC)
    use_log = c2.checkbox("Log transform") if col == "price" else False
    values = np.log(data[col]) if use_log else data[col]
    label = f"log({col})" if use_log else col

    skew, kurt = stats.skew(values), stats.kurtosis(values)
    shapiro_p = stats.shapiro(values.sample(min(len(values), 500), random_state=1))[1]
    m = st.columns(4)
    m[0].metric("Mean", f"{values.mean():,.2f}", border=True)
    m[1].metric("Skewness", f"{skew:.2f}", border=True)
    m[2].metric("Excess kurtosis", f"{kurt:.2f}", border=True)
    m[3].metric("Shapiro-Wilk p", f"{shapiro_p:.4f}", border=True)
    st.info(f"💡 **{label}** is {skew_text(skew)}. Shapiro-Wilk says it is "
            f"**{'not normal' if shapiro_p < 0.05 else 'consistent with normal'}** (p {'<' if shapiro_p < 0.05 else '≥'} 0.05).")

    left, right = st.columns(2)
    with left:
        fig = px.histogram(values, nbins=30, histnorm="probability density", marginal="box",
                           title="Histogram with fitted normal curve")
        xs = np.linspace(values.min(), values.max(), 200)
        fig.add_trace(go.Scatter(x=xs, y=stats.norm.pdf(xs, values.mean(), values.std()),
                                 name="Normal curve", line=dict(color="#222", width=2)), row=1, col=1)
        fig.update_layout(showlegend=False, xaxis_title=label)
        plot(fig)
    with right:
        (osm, osr), (slope, intercept, _) = stats.probplot(values, dist="norm")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=osm, y=osr, mode="markers", name="Data", opacity=0.5))
        fig.add_trace(go.Scatter(x=osm, y=intercept + slope * osm, name="Normal line",
                                 line=dict(color="#222")))
        fig.update_layout(title="Q-Q plot (points on the line = normal)",
                          xaxis_title="Theoretical quantiles", yaxis_title="Sample quantiles",
                          showlegend=False)
        plot(fig)

# ---------------------------------------------------------------- CLT
elif page == "🎲 Central Limit Theorem":
    st.title("🎲 Central Limit Theorem")
    st.write("Even when the population is skewed, the **means of many random samples** form a bell curve.")
    c1, c2, c3 = st.columns(3)
    col = c1.selectbox("Population column", ["price", "number_of_reviews", "rating", "accommodates"])
    n = c2.slider("Sample size (n)", 2, 200, 30)
    k = c3.slider("Number of samples", 100, 5000, 1000, step=100)

    pop = data[col].values
    means = np.random.default_rng().choice(pop, size=(k, n)).mean(axis=1)

    left, right = st.columns(2)
    with left:
        plot(px.histogram(x=pop, nbins=30, title=f"Population of {col}"))
    with right:
        fig = px.histogram(x=means, nbins=30, histnorm="probability density",
                           title=f"{k:,} sample means (n = {n})")
        xs = np.linspace(means.min(), means.max(), 200)
        fig.add_trace(go.Scatter(x=xs, y=stats.norm.pdf(xs, pop.mean(), pop.std() / np.sqrt(n)),
                                 name="Theory", line=dict(color="#222", width=2)))
        fig.update_layout(showlegend=False)
        plot(fig)

    st.table(pd.DataFrame({
        "Population": [pop.mean(), pop.std()],
        "Sample means": [means.mean(), means.std()],
        "Theory": [pop.mean(), pop.std() / np.sqrt(n)],
    }, index=["Mean", "Std dev (for sample means = standard error)"]).round(3))
    st.info(f"💡 The population skewness is **{stats.skew(pop):.2f}**, but the sample means have skewness "
            f"**{stats.skew(means):.2f}**. Increase n to see them get closer to normal. "
            "Their spread follows σ/√n.")

# ---------------------------------------------------------------- Hypothesis tests
elif page == "🧪 Hypothesis tests":
    st.title("🧪 Hypothesis tests")
    c1, c2 = st.columns(2)
    alpha = c1.select_slider("Significance level (α)", [0.01, 0.05, 0.10], value=0.05)
    use_log = c2.checkbox("Use log(price)", help="Price is skewed; the log makes the tests more reliable.")
    target = "log_price" if use_log else "price"
    tab_z, tab_t, tab_chi, tab_anova = st.tabs(["Z-test", "T-test", "Chi-square", "ANOVA"])

    with tab_z:
        st.write("**One-sample Z-test** – does one city's mean differ from the overall mean? "
                 "(σ is treated as known: the std of the full dataset.)")
        city = st.selectbox("City sample", sorted(data["city"].unique()), key="z_city")
        sample = data.loc[data["city"] == city, target]
        mu0, sigma = df[target].mean(), df[target].std()
        z = (sample.mean() - mu0) / (sigma / np.sqrt(len(sample)))
        p = 2 * (1 - stats.norm.cdf(abs(z)))
        m = st.columns(4)
        m[0].metric("n", len(sample), border=True)
        m[1].metric("Sample mean", f"{sample.mean():,.2f}", border=True)
        m[2].metric("Population mean", f"{mu0:,.2f}", border=True)
        m[3].metric("z", f"{z:.3f}", border=True)
        conclude(p, alpha, f"{city}'s mean equals the overall mean", f"{city}'s mean differs from the overall mean")

    with tab_t:
        kind = st.radio("Type", ["One-sample", "Two-sample (Welch)"], horizontal=True)
        if kind == "One-sample":
            mu0 = st.number_input("Hypothesised mean (μ0)", value=float(round(df[target].mean(), 2)))
            t, p = stats.ttest_1samp(data[target], mu0)
            st.metric("t statistic", f"{t:.3f}", border=True)
            conclude(p, alpha, f"the mean {target} equals {mu0}", f"the mean {target} is not {mu0}")
        else:
            by = st.selectbox("Compare groups of", ["city", "room_type"])
            groups = sorted(data[by].unique())
            if len(groups) < 2:
                st.warning(f"Select at least two {by} values in the sidebar.")
            else:
                c1, c2 = st.columns(2)
                a = c1.selectbox("Group A", groups, index=0)
                b = c2.selectbox("Group B", groups, index=1)
                x1, x2 = data.loc[data[by] == a, target], data.loc[data[by] == b, target]
                t, p = stats.ttest_ind(x1, x2, equal_var=False)
                m = st.columns(3)
                m[0].metric(f"{a} mean", f"{x1.mean():,.2f}", border=True)
                m[1].metric(f"{b} mean", f"{x2.mean():,.2f}", border=True)
                m[2].metric("t statistic", f"{t:.3f}", border=True)
                conclude(p, alpha, f"{a} and {b} have the same mean {target}",
                         f"{a} and {b} have different mean {target}")

    with tab_chi:
        st.write("**Chi-square test of independence** between two categorical columns.")
        cats = ["city", "room_type", "price_band"]
        c1, c2 = st.columns(2)
        v1 = c1.selectbox("Variable 1", cats, index=0)
        v2 = c2.selectbox("Variable 2", cats, index=2)
        table = pd.crosstab(data[v1], data[v2])
        if v1 == v2:
            st.warning("Pick two different variables.")
        elif table.shape[0] < 2 or table.shape[1] < 2:
            st.warning("Need at least two categories in each variable.")
        else:
            chi2, p, dof, expected = stats.chi2_contingency(table)
            left, right = st.columns(2)
            left.write("Observed counts")
            left.dataframe(table, width="stretch")
            with right:
                plot(px.bar(table.reset_index().melt(id_vars=v1), x=v1, y="value", color=v2,
                            barmode="group", title="Counts"), height=280)
            st.write(f"χ² = **{chi2:.2f}**, degrees of freedom = **{dof}**")
            conclude(p, alpha, f"{v1} and {v2} are independent", f"{v1} and {v2} are related")

    with tab_anova:
        st.write("**One-way ANOVA** – do the group means differ?")
        by = st.selectbox("Groups", ["city", "room_type"], key="anova_by")
        samples = [g[target].values for _, g in data.groupby(by)]
        if len(samples) < 2:
            st.warning(f"Select at least two {by} values in the sidebar.")
        else:
            f, p = stats.f_oneway(*samples)
            st.metric("F statistic", f"{f:.2f}", border=True)
            conclude(p, alpha, f"all {by} groups have the same mean {target}",
                     f"at least one {by} group has a different mean {target}")
            plot(px.box(data, x=by, y=target, color=by, points=False), height=320)

# ---------------------------------------------------------------- Regression
elif page == "📉 Regression":
    st.title("📉 Regression")
    st.caption("Target is log(price). Features are standardised so Ridge and Lasso penalties are fair.")
    tab1, tab2, tab3 = st.tabs(["Simple linear", "Multiple · Ridge · Lasso", "Model check"])

    with tab1:
        feat = st.selectbox("One feature", ["accommodates", "bedrooms", "number_of_reviews", "rating"])
        slope, intercept, r, p, se = stats.linregress(data[feat], data["log_price"])
        fig = px.scatter(data, x=feat, y="log_price", opacity=0.3, title=f"log(price) vs {feat}")
        xs = np.array([data[feat].min(), data[feat].max()])
        fig.add_trace(go.Scatter(x=xs, y=intercept + slope * xs, name="Fit", line=dict(color="#222", width=3)))
        plot(fig)
        m = st.columns(3)
        m[0].metric("Slope", f"{slope:.4f}", border=True)
        m[1].metric("R²", f"{r ** 2:.3f}", border=True)
        m[2].metric("p-value", f"{p:.3g}", border=True)
        st.info(f"💡 Each extra unit of **{feat}** changes the price by about **{(np.exp(slope) - 1):+.1%}**. "
                f"It explains **{r ** 2:.1%}** of the variation in log price.")

    with tab2:
        c1, c2, c3 = st.columns([3, 1, 1])
        features = c1.multiselect("Features", FEATURES, default=FEATURES)
        alpha_r = c2.slider("Ridge alpha", 0.01, 100.0, 10.0)
        alpha_l = c3.slider("Lasso alpha", 0.001, 0.5, 0.01, format="%.3f")
        if not features:
            st.info("Select at least one feature.")
        else:
            X = design_matrix(data, features)
            X_tr, X_te, y_tr, y_te = train_test_split(X, data["log_price"], test_size=0.2, random_state=42)
            scaler = StandardScaler().fit(X_tr)
            models = {"Multiple (OLS)": LinearRegression(), "Ridge": Ridge(alpha=alpha_r),
                      "Lasso": Lasso(alpha=alpha_l)}
            rows, coefs, preds = [], {}, {}
            for name, mdl in models.items():
                mdl.fit(scaler.transform(X_tr), y_tr)
                preds[name] = mdl.predict(scaler.transform(X_te))
                rows.append({"Model": name,
                             "Train R²": r2_score(y_tr, mdl.predict(scaler.transform(X_tr))),
                             "Test R²": r2_score(y_te, preds[name]),
                             "Test RMSE": np.sqrt(mean_squared_error(y_te, preds[name]))})
                coefs[name] = mdl.coef_
            results = pd.DataFrame(rows).round(3)
            st.session_state["reg"] = (results, preds, y_te)

            st.dataframe(results, hide_index=True, width="stretch")
            best = results.loc[results["Test R²"].idxmax(), "Model"]
            st.info(f"💡 Best test R²: **{best}**. Higher R² and lower RMSE are better.")
            coef_df = pd.DataFrame(coefs, index=X.columns).reset_index(names="feature").melt(
                id_vars="feature", var_name="model", value_name="coefficient")
            plot(px.bar(coef_df, x="feature", y="coefficient", color="model", barmode="group",
                        title="Coefficients (standardised)"), height=420)
            dropped = [f for f, c in zip(X.columns, coefs["Lasso"]) if c == 0]
            st.write("**Lasso removed:** " + (", ".join(dropped) if dropped else "no features at this alpha"))

    with tab3:
        if "reg" not in st.session_state:
            st.info("Open the 'Multiple · Ridge · Lasso' tab first and choose features.")
        else:
            results, preds, y_te = st.session_state["reg"]
            name = st.radio("Model", list(preds), horizontal=True)
            pred = preds[name]
            left, right = st.columns(2)
            with left:
                fig = px.scatter(x=np.exp(y_te), y=np.exp(pred), opacity=0.4,
                                 labels={"x": "Actual price (₹)", "y": "Predicted price (₹)"},
                                 title="Actual vs predicted (test set)")
                top = float(np.exp(y_te).max())
                fig.add_trace(go.Scatter(x=[0, top], y=[0, top], name="Perfect",
                                         line=dict(color="#222", dash="dash")))
                fig.update_layout(showlegend=False)
                plot(fig)
            with right:
                plot(px.histogram(x=y_te - pred, nbins=30, title="Residuals (should be centred on 0)",
                                  labels={"x": "Residual (log price)"}))

# ---------------------------------------------------------------- Predictor
else:
    st.title("💰 Price predictor")
    st.caption("Trained on the project dataset. Predictions are estimates, not real quotes.")
    model_name = st.radio("Model", ["Ridge", "Lasso", "Multiple (OLS)"], horizontal=True)

    with st.form("predict"):
        c1, c2, c3 = st.columns(3)
        city = c1.selectbox("City", sorted(df["city"].unique()))
        room = c1.selectbox("Room type", sorted(df["room_type"].unique()))
        guests = c2.slider("Guests", 1, 8, 2)
        bedrooms = c2.slider("Bedrooms", 1, 6, 1)
        reviews = c3.number_input("Number of reviews", 0, 10000, 20)
        rating = c3.slider("Rating", 1.0, 5.0, 4.5, 0.1)
        go_btn = st.form_submit_button("Predict nightly price", type="primary")

    if go_btn:
        inputs = dict(accommodates=guests, bedrooms=bedrooms, number_of_reviews=reviews,
                      rating=rating, city=city, room_type=room)
        est = predict_price(model_name, **inputs)
        resid_std = train_final_model(model_name)[3]
        low, high = est * np.exp(-1.28 * resid_std), est * np.exp(1.28 * resid_std)
        similar = df[(df["city"] == city) & (df["room_type"] == room)]["price"].mean()

        m = st.columns(3)
        m[0].metric("Estimated price / night", f"₹{est:,.0f}", border=True)
        m[1].metric("Likely range (80%)", f"₹{low:,.0f} – ₹{high:,.0f}", border=True)
        m[2].metric(f"Avg {room} in {city}", f"₹{similar:,.0f}", f"{est / similar - 1:+.0%} vs average",
                    border=True)

        by_city = pd.DataFrame({"city": sorted(df["city"].unique())})
        by_city["price"] = [predict_price(model_name, **{**inputs, "city": c}) for c in by_city["city"]]
        plot(px.bar(by_city, x="city", y="price", text_auto=".0f",
                    title="Same listing, predicted price in each city"), height=320)