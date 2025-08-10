import pandas as pd
import numpy as np
import re
from functools import lru_cache
import plotly.express as px
import plotly.graph_objects as go
from dash import dcc, html, Input, Output
from jupyter_dash import JupyterDash

FILE_PATH = "nationaldatabaseofchildcareprices.xlsx"
SHEET = "nationaldatabaseofchildcare"

@lru_cache(maxsize=1)
def load_data():
    try:
        df = pd.read_excel(FILE_PATH, sheet_name=SHEET)
    except Exception:
        df = pd.read_excel("/mnt/data/nationaldatabaseofchildcareprices.xlsx", sheet_name=SHEET)
    df.columns = (
        df.columns.str.strip()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )
    return df

def detect_measures(df):
    patt = re.compile(r"^(M[A-Z]*?)(Infant|Toddler|Preschool)$")
    pairs = []
    for c in df.columns:
        m = patt.match(c)
        if m:
            pairs.append((m.group(1), m.group(2)))
    care_types = sorted({p[0] for p in pairs})
    ages = ["Infant", "Toddler", "Preschool"]
    care_types = [c for c in care_types if any(f"{c}{a}" in df.columns for a in ages)]
    ages = [a for a in ages if any(f"{c}{a}" in df.columns for c in care_types)]
    return care_types, ages

def map_frame(df, care_type, age, year):
    col = f"{care_type}{age}"
    d = df[df["StudyYear"] == year]
    if col not in d.columns:
        return pd.DataFrame(columns=["State_Abbreviation", "Value"])
    g = (
        d.groupby(["State_Name", "State_Abbreviation"])[col]
        .mean()
        .reset_index()
        .rename(columns={col: "Value"})
    )
    return g.dropna(subset=["State_Abbreviation", "Value"])

def timeseries(df, care_type, age, geo_level, geo_val):
    col = f"{care_type}{age}"
    d = df.dropna(subset=["StudyYear"])
    if col not in d.columns:
        return pd.DataFrame(columns=["StudyYear", "Value"])
    if geo_level == "Nation":
        g = d.groupby("StudyYear")[col].mean().reset_index().rename(columns={col: "Value"})
    elif geo_level == "State":
        g = (
            d[d["State_Name"] == geo_val]
            .groupby("StudyYear")[col]
            .mean()
            .reset_index()
            .rename(columns={col: "Value"})
        )
    else:
        if " | " in geo_val:
            st, ct = geo_val.split(" | ", 1)
            g = d[(d["State_Name"] == st) & (d["County_Name"] == ct)]
        else:
            g = d[d["County_Name"] == geo_val]
        g = g.groupby("StudyYear")[col].mean().reset_index().rename(columns={col: "Value"})
    return g

def pivot_for_compare(df, ages, care_types, year, geo_level, geo_val):
    d = df[df["StudyYear"] == year].copy()
    if geo_level == "State":
        d = d[d["State_Name"] == geo_val]
    elif geo_level == "County":
        if " | " in geo_val:
            st, ct = geo_val.split(" | ", 1)
            d = d[(d["State_Name"] == st) & (d["County_Name"] == ct)]
        else:
            d = d[d["County_Name"] == geo_val]
    records = []
    for ct in care_types:
        for age in ages:
            col = f"{ct}{age}"
            if col in d.columns:
                v = d[col].mean(skipna=True)
                if pd.notna(v):
                    records.append({"CareType": ct, "AgeGroup": age, "Value": float(v)})
    return pd.DataFrame(records)

def scatter_unemp_frame(df, care_type, age, year, metric, agg):
    col = f"{care_type}{age}"
    if metric not in df.columns or col not in df.columns:
        return pd.DataFrame(columns=[metric, "Value", "State_Name"])
    d = df[df["StudyYear"] == year][[metric, col, "State_Name"]].copy()
    d = d.rename(columns={col: "Value"})
    d = d.dropna(subset=[metric, "Value"])
    if d.empty:
        return d
    if agg == "state":
        d = d.groupby("State_Name", as_index=False)[[metric, "Value"]].mean()
    return d

def affordability_frame(df, care_type, age, year):
    if "MHI" not in df.columns:
        return pd.DataFrame(columns=["State_Abbreviation", "AffordPct"])
    col = f"{care_type}{age}"
    if col not in df.columns:
        return pd.DataFrame(columns=["State_Abbreviation", "AffordPct"])
    d = df[df["StudyYear"] == year][["State_Name", "State_Abbreviation", "MHI", col]].dropna()
    if d.empty:
        return pd.DataFrame(columns=["State_Abbreviation", "AffordPct"])
    g = d.groupby(["State_Name", "State_Abbreviation"], as_index=False)[["MHI", col]].mean()
    annual_cost = 52.0 * g[col]
    g["AffordPct"] = np.where(g["MHI"] > 0, 100.0 * annual_cost / g["MHI"], np.nan)
    g = g.dropna(subset=["AffordPct"])
    return g[["State_Name", "State_Abbreviation", "AffordPct"]]

df = load_data()
care_types, ages = detect_measures(df)
years = sorted(df["StudyYear"].dropna().unique().astype(int)) if "StudyYear" in df.columns else []
states = sorted(df["State_Name"].dropna().unique().tolist()) if "State_Name" in df.columns else []
counties = []
if "County_Name" in df.columns and "State_Name" in df.columns:
    tmp = df[["State_Name", "County_Name"]].dropna().drop_duplicates()
    counties = sorted((tmp["State_Name"] + " | " + tmp["County_Name"]).tolist())

app = JupyterDash(__name__)
app.layout = html.Div([
    html.H2("Childcare Cost Explorer (NDCP)"),
    html.Div([
        html.Div([html.Label("Geography Level"),
                  dcc.Dropdown(id="geo-level",
                               options=[{"label": "Nation", "value": "Nation"},
                                        {"label": "State", "value": "State"},
                                        {"label": "County", "value": "County"}],
                               value="Nation", clearable=False)],
                 style={"width": "18%", "display": "inline-block", "marginRight": "1%"}),
        html.Div([html.Label("State / County"),
                  dcc.Dropdown(id="geo-value")],
                 style={"width": "32%", "display": "inline-block", "marginRight": "1%"}),
        html.Div([html.Label("Year"),
                  dcc.Dropdown(id="year",
                               options=[{"label": int(y), "value": int(y)} for y in years],
                               value=(years[-1] if years else None), clearable=False)],
                 style={"width": "12%", "display": "inline-block", "marginRight": "1%"}),
        html.Div([html.Label("Age Group"),
                  dcc.Dropdown(id="age",
                               options=[{"label": a, "value": a} for a in ages],
                               value=("Infant" if "Infant" in ages else (ages[0] if ages else None)), clearable=False)],
                 style={"width": "12%", "display": "inline-block", "marginRight": "1%"}),
        html.Div([html.Label("Care Type"),
                  dcc.Dropdown(id="care",
                               options=[{"label": c, "value": c} for c in care_types],
                               value=(care_types[0] if care_types else None), clearable=False)],
                 style={"width": "12%", "display": "inline-block"})
    ], style={"marginBottom": "10px"}),
    dcc.Tabs([
        dcc.Tab(label="Map + Top States", children=[dcc.Graph(id="map"), dcc.Graph(id="top-bar")]),
        dcc.Tab(label="Trends", children=[dcc.Graph(id="trend-line")]),
        dcc.Tab(label="Care-Type Compare", children=[
            html.Div([
                dcc.Dropdown(id="compare-ages", options=[{"label": a, "value": a} for a in ages], value=ages, multi=True),
                dcc.Dropdown(id="compare-care", options=[{"label": c, "value": c} for c in care_types], value=care_types, multi=True)
            ], style={"padding": "8px"}),
            dcc.Graph(id="compare-bars")
        ]),
        dcc.Tab(label="Unemployment vs Cost", children=[
            html.Div([
                dcc.Dropdown(
                    id="unemp-metric",
                    options=[{"label": m, "value": m} for m in ["UNR_20to64", "FUNR_20to64", "MUNR_20to64"] if m in df.columns],
                    value=("UNR_20to64" if "UNR_20to64" in df.columns else ([c for c in ["FUNR_20to64", "MUNR_20to64"] if c in df.columns][0] if any(c in df.columns for c in ["FUNR_20to64", "MUNR_20to64"]) else None)),
                    clearable=False
                ),
                dcc.Dropdown(
                    id="scatter-agg",
                    options=[{"label": "County points", "value": "county"}, {"label": "State averages", "value": "state"}],
                    value="county",
                    clearable=False
                ),
            ], style={"display": "flex", "gap": "8px", "padding": "8px", "maxWidth": "520px"}),
            dcc.Graph(id="scatter")
        ]),
        dcc.Tab(label="Affordability (% of MHI)", children=[
            dcc.Graph(id="afford-map"),
            dcc.Graph(id="afford-bar")
        ])
    ])
], style={"fontFamily": "Arial, sans-serif", "padding": "10px"})

@app.callback(
    Output("geo-value","options"),
    Output("geo-value","value"),
    Input("geo-level","value")
)
def update_geo_value(level):
    if level == "Nation":
        return [], None
    if level == "State":
        opts = [{"label": s, "value": s} for s in states]
        return opts, (states[0] if states else None)
    opts = [{"label": c, "value": c} for c in counties]
    return opts, (counties[0] if counties else None)

@app.callback(
    Output("map","figure"),
    Output("top-bar","figure"),
    Input("year","value"),
    Input("age","value"),
    Input("care","value")
)
def update_map_and_top(year, age, care):
    m = map_frame(df, care, age, year) if year else pd.DataFrame(columns=["State_Abbreviation", "Value"])
    if m.empty:
        return go.Figure().update_layout(title="No data"), go.Figure().update_layout(title="No data")

    fig_map = px.choropleth(
        m,
        locations="State_Abbreviation",
        locationmode="USA-states",
        scope="usa",
        color="Value",
        color_continuous_scale="YlOrRd",
        labels={"Value": "Weekly cost"}
    )
    fig_map.update_layout(title=f"Average {care}{age} Weekly Cost by State - {year}")

    top = m.sort_values("Value", ascending=False).head(10)
    order = top["State_Abbreviation"].tolist()[::-1]

    fig_bar = px.bar(
        top[::-1],
        x="Value",
        y="State_Abbreviation",
        orientation="h",
        color="Value",
        color_continuous_scale="YlOrRd",
        labels={"Value": "Weekly cost", "State_Abbreviation": "State"}
    )
    fig_bar.update_layout(
        title="Top 10 States by Weekly Cost",
        yaxis=dict(categoryorder="array", categoryarray=order),
        coloraxis_showscale=False
    )

    return fig_map, fig_bar

@app.callback(
    Output("trend-line","figure"),
    Input("geo-level","value"),
    Input("geo-value","value"),
    Input("care","value"),
    Input("age","value")
)
def update_trend(level, val, care, age):
    ts = timeseries(df, care, age, level, val if val else "")
    if ts.empty:
        return go.Figure().update_layout(title="No data")
    fig = px.line(ts, x="StudyYear", y="Value", markers=True, labels={"Value": "Weekly cost", "StudyYear": "Year"})
    ttl = f"Trend - {care}{age} Weekly Cost ({level if level != 'County' else val})"
    fig.update_layout(title=ttl)
    return fig

@app.callback(
    Output("compare-bars","figure"),
    Input("compare-ages","value"),
    Input("compare-care","value"),
    Input("year","value"),
    Input("geo-level","value"),
    Input("geo-value","value")
)
def update_compare(sel_ages, sel_care, year, level, val):
    if not sel_ages or not sel_care or year is None:
        return go.Figure().update_layout(title="No selection")
    p = pivot_for_compare(df, sel_ages, sel_care, year, level, val if val else "")
    if p.empty:
        return go.Figure().update_layout(title="No data")
    fig = px.bar(p, x="Value", y="AgeGroup", color="CareType", barmode="group", orientation="h",
                 labels={"Value": "Weekly cost", "AgeGroup": "Age"})
    fig.update_layout(title=f"Compare Care Types Weekly Cost - {year} ({level if level != 'County' else val})")
    return fig

@app.callback(
    Output("scatter","figure"),
    Input("year","value"),
    Input("care","value"),
    Input("age","value"),
    Input("unemp-metric","value"),
    Input("scatter-agg","value")
)
def update_scatter(year, care, age, metric, agg):
    s = scatter_unemp_frame(df, care, age, year, metric, agg) if year else pd.DataFrame()
    if s.empty or metric is None:
        return go.Figure().update_layout(title="No data for current selection")
    r = s[[metric, "Value"]].corr().iloc[0, 1]
    x = s[metric].to_numpy()
    y = s["Value"].to_numpy()
    if x.size >= 2 and y.size >= 2:
        coef = np.polyfit(x, y, 1)
        line_x = np.linspace(x.min(), x.max(), 100)
        line_y = coef[0] * line_x + coef[1]
        fig = px.scatter(s, x=metric, y="Value",
                         hover_data=(["State_Name"] if "State_Name" in s.columns else None),
                         labels={metric: "Unemployment (20-64)", "Value": "Weekly cost"})
        fig.add_trace(go.Scatter(x=line_x, y=line_y, mode="lines", name="Trendline"))
    else:
        fig = px.scatter(s, x=metric, y="Value",
                         hover_data=(["State_Name"] if "State_Name" in s.columns else None),
                         labels={metric: "Unemployment (20-64)", "Value": "Weekly cost"})
    fig.update_layout(title=f"Unemployment vs Weekly Cost - {care}{age} ({year}, {agg}) | r={r:.2f}")
    return fig

@app.callback(
    Output("afford-map","figure"),
    Output("afford-bar","figure"),
    Input("year","value"),
    Input("care","value"),
    Input("age","value")
)
def update_afford(year, care, age):
    a = affordability_frame(df, care, age, year) if year else pd.DataFrame(columns=["State_Abbreviation", "AffordPct"])
    if a.empty:
        return go.Figure().update_layout(title="No data"), go.Figure().update_layout(title="No data")

    fig_map = px.choropleth(
        a,
        locations="State_Abbreviation",
        locationmode="USA-states",
        scope="usa",
        color="AffordPct",
        color_continuous_scale="Reds",
        labels={"AffordPct": "Annual cost as % of Median Household Income"}
    )
    fig_map.update_layout(title=f"Affordability: Annualized From Weekly Cost as % of MHI - {care}{age} ({year})")

    top = a.sort_values("AffordPct", ascending=False).head(10)
    order = top["State_Name"].tolist()[::-1]

    fig_bar = px.bar(
        top[::-1],
        x="AffordPct",
        y="State_Name",
        orientation="h",
        color="AffordPct",
        color_continuous_scale="Reds",
        labels={"AffordPct": "Annual cost as % of Median Household Income", "State_Name": "State"}
    )
    fig_bar.update_layout(
        title="Top 10 Most Unaffordable States (Annualized from weekly)",
        yaxis=dict(categoryorder="array", categoryarray=order),
        coloraxis_showscale=False
    )

    return fig_map, fig_bar

app.run(mode="external")
