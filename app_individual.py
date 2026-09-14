import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import zlib
import io
import os

st.set_page_config(page_title="NIFTY × SENSEX Strategy Tester", layout="wide")

DB_PATH = os.path.join(os.path.dirname(__file__), "strategy_data.db")
NIFTY_STEP = 50
SENSEX_STEP = 100

st.markdown("""
<style>
.main-title{font-size:42px;font-weight:800;letter-spacing:-1.5px;margin-bottom:2px}
.subtitle{color:#6b7280;font-size:15px;margin-bottom:18px}
.section-title{font-size:21px;font-weight:750;margin-top:10px;margin-bottom:8px}
.formula-box{padding:14px 16px;border-radius:12px;background:#f6f7fb;border:1px solid #e5e7eb;font-family:monospace;font-size:15px}
div[data-testid="stMetric"]{padding:12px 14px;border-radius:12px;border:1px solid #e5e7eb;background:#fff;box-shadow:0 2px 8px rgba(0,0,0,.04)}
div[data-testid="stDataFrame"]{border-radius:12px;overflow:hidden}
</style>
<div class="main-title">NIFTY × SENSEX Strategy Tester</div>
<div class="subtitle">Historical straddle analytics • Expiry comparison • India VIX • Backtesting</div>
""", unsafe_allow_html=True)

if not os.path.exists(DB_PATH):
    st.error("strategy_data.db is missing. Keep it in the same GitHub folder as app.py.")
    st.stop()

def db_connect():
    return sqlite3.connect(DB_PATH)

@st.cache_data(show_spinner=False)
def available_dates():
    con=db_connect()
    q="""SELECT trade_date FROM days
         WHERE nifty_blob IS NOT NULL AND length(nifty_blob)>0
           AND sensex_blob IS NOT NULL AND length(sensex_blob)>0
         ORDER BY trade_date"""
    out=pd.read_sql_query(q,con)["trade_date"].tolist()
    con.close()
    return out

@st.cache_data(show_spinner=False)
def market_values(dt):
    con=db_connect()
    row=con.execute("SELECT nifty_spot,sensex_spot,india_vix FROM days WHERE trade_date=?",(dt,)).fetchone()
    con.close()
    if row is None:
        return np.nan,np.nan,np.nan
    return tuple(float(x) if x is not None else np.nan for x in row)

@st.cache_data(show_spinner=False)
def day_options(dt, symbol):
    con=db_connect()
    col="nifty_blob" if symbol=="NIFTY" else "sensex_blob"
    row=con.execute(f"SELECT {col} FROM days WHERE trade_date=?",(dt,)).fetchone()
    con.close()
    if row is None or not row[0]:
        return pd.DataFrame(columns=["expiry","strike","option_type","close"])
    raw=zlib.decompress(row[0]).decode("utf-8")
    df=pd.read_csv(io.StringIO(raw),header=None,names=["expiry","strike","option_type","close"])
    df["strike"]=pd.to_numeric(df["strike"],errors="coerce")
    df["close"]=pd.to_numeric(df["close"],errors="coerce")
    df["option_type"]=df["option_type"].astype(str).str.upper()
    return df.dropna(subset=["expiry","strike"])

@st.cache_data(show_spinner=False)
def expiry_values(dt, symbol):
    df=day_options(dt,symbol)
    return sorted(df["expiry"].dropna().astype(str).unique().tolist())

def round_strike(value,step):
    if not np.isfinite(value): return np.nan
    return int(np.floor(value/step+0.5)*step)

def locked_straddle(df,symbol,expiry,step,spot):
    x=df[df["expiry"].eq(expiry) & df["option_type"].isin(["CE","PE"])].copy()
    empty={"spot":spot,"spot_source":"market OHLC","provisional_atm":np.nan,
           "atm_ce":np.nan,"atm_pe":np.nan,"synthetic_future":np.nan,
           "final_strike":np.nan,"final_ce":np.nan,"final_pe":np.nan,
           "straddle":np.nan,"status":"No usable data"}
    if x.empty or not np.isfinite(spot): return empty

    provisional=round_strike(spot,step)
    piv=x.pivot_table(index="strike",columns="option_type",values="close",aggfunc="first")
    if "CE" not in piv.columns or "PE" not in piv.columns:
        empty["provisional_atm"]=provisional
        empty["status"]="CE/PE missing"
        return empty
    piv=piv.dropna(subset=["CE","PE"])
    if piv.empty:
        empty["provisional_atm"]=provisional
        empty["status"]="CE/PE missing"
        return empty

    if provisional in piv.index:
        used=provisional
    else:
        used=piv.index[np.argmin(np.abs(piv.index.to_numpy(dtype=float)-provisional))]

    atm_ce=float(piv.loc[used,"CE"]); atm_pe=float(piv.loc[used,"PE"])
    synthetic=float(used+atm_ce-atm_pe)
    final_strike=round_strike(synthetic,step)

    if final_strike not in piv.index:
        return {"spot":spot,"spot_source":"market OHLC","provisional_atm":used,
                "atm_ce":atm_ce,"atm_pe":atm_pe,"synthetic_future":synthetic,
                "final_strike":final_strike,"final_ce":np.nan,"final_pe":np.nan,
                "straddle":np.nan,"status":"Final-strike CE/PE missing"}

    final_ce=float(piv.loc[final_strike,"CE"]); final_pe=float(piv.loc[final_strike,"PE"])
    return {"spot":spot,"spot_source":"market OHLC","provisional_atm":used,
            "atm_ce":atm_ce,"atm_pe":atm_pe,"synthetic_future":synthetic,
            "final_strike":final_strike,"final_ce":final_ce,"final_pe":final_pe,
            "straddle":final_ce+final_pe,"status":"OK"}

@st.cache_data(show_spinner=False)
def calculate_day(dt,n_exp,s_exp,multiplier):
    nspot,sspot,vix=market_values(dt)
    n=locked_straddle(day_options(dt,"NIFTY"),"NIFTY",n_exp,NIFTY_STEP,nspot)
    s=locked_straddle(day_options(dt,"SENSEX"),"SENSEX",s_exp,SENSEX_STEP,sspot)
    fv=s["straddle"]-(n["straddle"]*multiplier) if np.isfinite(n["straddle"]) and np.isfinite(s["straddle"]) else np.nan
    return n,s,vix,fv

@st.cache_data(show_spinner=False)
def run_backtest(dates,n_exp,s_exp,multiplier):
    rows=[]
    for dt in dates:
        try:
            n,s,vix,fv=calculate_day(dt,n_exp,s_exp,multiplier)
            if np.isfinite(fv):
                rows.append({"Date":dt,"India VIX":vix,"NIFTY Spot":n["spot"],
                             "NIFTY Straddle":n["straddle"],"Adjusted NIFTY":n["straddle"]*multiplier,
                             "SENSEX Spot":s["spot"],"SENSEX Straddle":s["straddle"],
                             "Final Value":fv,"NIFTY Final Strike":n["final_strike"],
                             "SENSEX Final Strike":s["final_strike"]})
        except Exception:
            continue
    return pd.DataFrame(rows)


# -----------------------------
# INDIVIDUAL STRADDLE TEST UI
# -----------------------------
st.markdown('<div class="main-title">Individual NIFTY × SENSEX Straddle</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Standalone test app • Original Strategy Tester is not changed</div>', unsafe_allow_html=True)

dates=available_dates()
if not dates:
    st.error("No common NIFTY + SENSEX option dates are available in the database.")
    st.stop()

st.markdown('<div class="section-title">📅 Date & Expiry Selection</div>',unsafe_allow_html=True)
c1,c2,c3=st.columns(3)
with c1: start_date=st.selectbox("Start Date",dates,index=max(0,len(dates)-20))
with c2:
    valid_end=[d for d in dates if d>=start_date]
    end_date=st.selectbox("End Date",valid_end,index=len(valid_end)-1)
with c3:
    view_dates=[d for d in dates if start_date<=d<=end_date]
    selected_date=st.selectbox("View Date",view_dates,index=len(view_dates)-1)

n_expiries=expiry_values(selected_date,"NIFTY")
s_expiries=expiry_values(selected_date,"SENSEX")
if not n_expiries or not s_expiries:
    st.error("Expiry data is not available for this date.")
    st.stop()

e1,e2=st.columns(2)
with e1: nifty_exp=st.selectbox("NIFTY Expiry",n_expiries)
with e2: sensex_exp=st.selectbox("SENSEX Expiry",s_expiries)

n,s,vix,_=calculate_day(selected_date,nifty_exp,sensex_exp,3.30)

# Key market values: Spot + Synthetic Future
st.markdown('<div class="section-title">📊 Spot & Synthetic Future</div>',unsafe_allow_html=True)
k1,k2,k3,k4=st.columns(4)
with k1:
    st.metric("NIFTY Spot",f"{n['spot']:.2f}" if np.isfinite(n['spot']) else "N/A")
with k2:
    st.metric("NIFTY Synthetic Future",f"{n['synthetic_future']:.2f}" if np.isfinite(n['synthetic_future']) else "N/A")
with k3:
    st.metric("SENSEX Spot",f"{s['spot']:.2f}" if np.isfinite(s['spot']) else "N/A")
with k4:
    st.metric("SENSEX Synthetic Future",f"{s['synthetic_future']:.2f}" if np.isfinite(s['synthetic_future']) else "N/A")

st.markdown('<div class="section-title">💰 Individual Straddle</div>',unsafe_allow_html=True)
m1,m2=st.columns(2)
with m1:
    st.metric("NIFTY Individual Straddle",f"{n['straddle']:.2f}" if np.isfinite(n['straddle']) else "N/A")
    st.caption(f"Expiry: {nifty_exp} | Final Strike: {n['final_strike']}" if np.isfinite(n['final_strike']) else f"Expiry: {nifty_exp}")
with m2:
    st.metric("SENSEX Individual Straddle",f"{s['straddle']:.2f}" if np.isfinite(s['straddle']) else "N/A")
    st.caption(f"Expiry: {sensex_exp} | Final Strike: {s['final_strike']}" if np.isfinite(s['final_strike']) else f"Expiry: {sensex_exp}")

st.divider()
st.markdown('<div class="section-title">🔍 Calculation Details</div>',unsafe_allow_html=True)
l,r=st.columns(2)
with l:
    st.subheader("NIFTY")
    st.dataframe(pd.DataFrame([{"Spot":n['spot'],"Provisional ATM":n['provisional_atm'],"ATM CE":n['atm_ce'],"ATM PE":n['atm_pe'],"Synthetic Future":n['synthetic_future'],"Final Strike":n['final_strike'],"Final CE":n['final_ce'],"Final PE":n['final_pe'],"Straddle":n['straddle'],"Status":n['status']}]),use_container_width=True,hide_index=True)
with r:
    st.subheader("SENSEX")
    st.dataframe(pd.DataFrame([{"Spot":s['spot'],"Provisional ATM":s['provisional_atm'],"ATM CE":s['atm_ce'],"ATM PE":s['atm_pe'],"Synthetic Future":s['synthetic_future'],"Final Strike":s['final_strike'],"Final CE":s['final_ce'],"Final PE":s['final_pe'],"Straddle":s['straddle'],"Status":s['status']}]),use_container_width=True,hide_index=True)

@st.cache_data(show_spinner=False)
def individual_range(dates_range,n_exp,s_exp):
    rows=[]
    for dt in dates_range:
        try:
            n,s,vix,_=calculate_day(dt,n_exp,s_exp,3.30)
            if np.isfinite(n['straddle']) and np.isfinite(s['straddle']):
                rows.append({
                    'Date':dt,
                    'India VIX':vix,
                    'NIFTY Spot':n['spot'],
                    'NIFTY Synthetic Future':n['synthetic_future'],
                    'NIFTY Final Strike':n['final_strike'],
                    'NIFTY Straddle':n['straddle'],
                    'SENSEX Spot':s['spot'],
                    'SENSEX Synthetic Future':s['synthetic_future'],
                    'SENSEX Final Strike':s['final_strike'],
                    'SENSEX Straddle':s['straddle']
                })
        except Exception: continue
    return pd.DataFrame(rows)

st.divider()
st.markdown('<div class="section-title">📈 Date-wise Individual Straddle</div>',unsafe_allow_html=True)
ind=individual_range(view_dates,nifty_exp,sensex_exp)
if ind.empty:
    st.warning("Selected range માટે બંને individual straddlesનો usable data મળ્યો નથી.")
else:
    st.line_chart(ind.set_index('Date')[['NIFTY Straddle','SENSEX Straddle']],use_container_width=True)

    # Flexible, full-width date-wise table with light visual grouping.
    def style_result_table(df):
        sty = df.style
        nifty_cols = ['NIFTY Spot','NIFTY Synthetic Future','NIFTY Final Strike','NIFTY Straddle']
        sensex_cols = ['SENSEX Spot','SENSEX Synthetic Future','SENSEX Final Strike','SENSEX Straddle']
        vix_cols = ['India VIX']

        for c in nifty_cols:
            if c in df.columns:
                sty = sty.set_properties(subset=[c], **{'background-color':'#eef6ff'})
        for c in sensex_cols:
            if c in df.columns:
                sty = sty.set_properties(subset=[c], **{'background-color':'#eefbf2'})
        for c in vix_cols:
            if c in df.columns:
                sty = sty.set_properties(subset=[c], **{'background-color':'#f7f7f7'})
        if 'Date' in df.columns:
            sty = sty.set_properties(subset=['Date'], **{'font-weight':'600'})
        return sty

    st.caption("💡 Table full-width છે અને columns ને mouse થી drag કરીને તમારી જરૂર મુજબ resize કરી શકો છો. નીચે horizontal scroll પણ મળશે.")
    display_ind = ind.copy()
    for c in ['India VIX','NIFTY Spot','NIFTY Synthetic Future','NIFTY Final Strike','NIFTY Straddle',
              'SENSEX Spot','SENSEX Synthetic Future','SENSEX Final Strike','SENSEX Straddle']:
        if c in display_ind.columns:
            display_ind[c] = pd.to_numeric(display_ind[c], errors='coerce')

    st.dataframe(
        style_result_table(display_ind),
        width="stretch",
        height=520,
        hide_index=True,
        column_config={
            "Date": st.column_config.TextColumn("Date", width="medium"),
            "India VIX": st.column_config.NumberColumn("India VIX", format="%.2f", width="small"),
            "NIFTY Spot": st.column_config.NumberColumn("NIFTY Spot", format="%.2f", width="medium"),
            "NIFTY Synthetic Future": st.column_config.NumberColumn("NIFTY Synthetic Future", format="%.2f", width="medium"),
            "NIFTY Final Strike": st.column_config.NumberColumn("NIFTY Final Strike", format="%.0f", width="medium"),
            "NIFTY Straddle": st.column_config.NumberColumn("NIFTY Straddle", format="%.2f", width="medium"),
            "SENSEX Spot": st.column_config.NumberColumn("SENSEX Spot", format="%.2f", width="medium"),
            "SENSEX Synthetic Future": st.column_config.NumberColumn("SENSEX Synthetic Future", format="%.2f", width="medium"),
            "SENSEX Final Strike": st.column_config.NumberColumn("SENSEX Final Strike", format="%.0f", width="medium"),
            "SENSEX Straddle": st.column_config.NumberColumn("SENSEX Straddle", format="%.2f", width="medium"),
        },
    )
    st.download_button('Download Individual Straddle CSV',ind.to_csv(index=False).encode('utf-8'),'individual_straddles.csv','text/csv')
