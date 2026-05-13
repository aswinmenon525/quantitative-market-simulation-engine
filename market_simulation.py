# ╔══════════════════════════════════════════════════════════════════╗
# ║        STOCK PRICE ANALYSER — Random Process Project            ║
# ║        Course    : BECE207L — Random Processes                  ║
# ║        Platform  : Thonny IDE / Python 3                        ║
# ║        Input     : Yahoo Finance CSV files                      ║
# ║        Output    : BUY / SELL / HOLD + Combined Graph + Table   ║
# ║        Author    : Aswin — VIT Vellore (24BEC0525)              ║
# ╚══════════════════════════════════════════════════════════════════╝

import pandas            as pd
import numpy             as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from   scipy.signal      import lfilter
from   scipy.stats       import norm
import warnings
warnings.filterwarnings('ignore')


STOCK_FILES = {
    'zomato' : 'zomato.csv',
    'swiggy' : 'swiggy.csv',
}

MA_WINDOW = 20

# ══════════════════════════════════════════════════════════════════
#  STEP 1 — LOAD AND CLEAN CSV DATA
# ══════════════════════════════════════════════════════════════════

def load_stock(filepath, name):
    try:
        df = pd.read_csv(filepath)
        df.columns = [c.strip() for c in df.columns]
        print(f"     Columns found : {list(df.columns)}")

        date_matches  = [c for c in df.columns if 'date' in c.lower()]
        date_col      = date_matches[0] if date_matches else df.columns[0]
        close_matches = [c for c in df.columns
                         if 'close' in c.lower() and 'adj' not in c.lower()]
        if close_matches:
            close_col = close_matches[0]
        else:
            price_matches = [c for c in df.columns if 'price' in c.lower()]
            close_col = price_matches[0] if price_matches else df.columns[4]

        print(f"     Using date col  : {date_col}")
        print(f"     Using price col : {close_col}")

        df[date_col]  = pd.to_datetime(df[date_col])
        df[close_col] = pd.to_numeric(df[close_col], errors='coerce')
        df = df.dropna(subset=[close_col]).sort_values(date_col).reset_index(drop=True)

        prices = df[close_col].values
        dates  = df[date_col].values

        print(f"  v {name:<8} loaded — {len(prices)} days of data")
        print(f"     Date range : {str(dates[0])[:10]}  to  {str(dates[-1])[:10]}")
        print(f"     Price range: Rs{prices.min():.2f}  to  Rs{prices.max():.2f}")
        return prices, dates

    except FileNotFoundError:
        print(f"  x {name}: File '{filepath}' not found.")
        return None, None
    except Exception as e:
        print(f"  x {name}: Error: {e}")
        return None, None


# ══════════════════════════════════════════════════════════════════
#  STEP 2 — RANDOM PROCESS FUNCTIONS
# ══════════════════════════════════════════════════════════════════

def compute_daily_returns(prices):
    return np.diff(prices) / prices[:-1] * 100

def apply_MA_filter(signal, W):
    return lfilter(np.ones(W) / W, [1], signal)

def compute_autocorrelation(signal, max_lag=30):
    s      = signal - signal.mean()
    result = np.correlate(s, s, mode='full')
    result = result / (result.max() + 1e-10)
    mid    = len(result) // 2
    lags   = np.arange(-max_lag, max_lag + 1)
    acf    = result[mid - max_lag : mid + max_lag + 1]
    return lags, acf

def compute_volatility(returns):
    return np.std(returns)

def generate_signal(prices, filtered):
    last_price = prices[-1]
    last_ma    = filtered[-1]
    prev_price = prices[-2]
    prev_ma    = filtered[-2]
    diff_pct   = (last_price - last_ma) / last_ma * 100

    crossed_above = (prev_price <= prev_ma) and (last_price > last_ma)
    crossed_below = (prev_price >= prev_ma) and (last_price < last_ma)

    if crossed_above or diff_pct > 1.5:
        return ('BUY',
                f'Price (Rs{last_price:.2f}) is ABOVE {MA_WINDOW}-day average '
                f'(Rs{last_ma:.2f}). Upward momentum detected.',
                '#1D9E75', 'BUY', diff_pct)
    elif crossed_below or diff_pct < -1.5:
        return ('SELL',
                f'Price (Rs{last_price:.2f}) is BELOW {MA_WINDOW}-day average '
                f'(Rs{last_ma:.2f}). Downward momentum detected.',
                '#E24B4A', 'SELL', diff_pct)
    else:
        return ('HOLD',
                f'Price (Rs{last_price:.2f}) is NEAR {MA_WINDOW}-day average '
                f'(Rs{last_ma:.2f}). No strong trend. Wait and watch.',
                '#BA7517', 'HOLD', diff_pct)

def risk_level(vol):
    if   vol < 1.0: return 'LOW RISK',    '#1D9E75'
    elif vol < 2.5: return 'MEDIUM RISK', '#BA7517'
    else:           return 'HIGH RISK',   '#E24B4A'


# ══════════════════════════════════════════════════════════════════
#  STEP 3 — LOAD ALL STOCKS
# ══════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("   STOCK PRICE ANALYSER — Random Process Project")
print("=" * 60 + "\n  Loading stock data...\n")

stocks = {}

for name, filepath in STOCK_FILES.items():
    prices, dates = load_stock(filepath, name)
    if prices is not None and len(prices) >= MA_WINDOW + 5:
        returns    = compute_daily_returns(prices)
        filtered   = apply_MA_filter(prices, MA_WINDOW)
        volatility = compute_volatility(returns)
        risk, _    = risk_level(volatility)
        signal, reason, sig_color, arrow, diff_pct = generate_signal(prices, filtered)
        stocks[name] = dict(prices=prices, dates=dates, returns=returns,
                            filtered=filtered, volatility=volatility,
                            risk=risk, signal=signal, reason=reason,
                            sig_color=sig_color, arrow=arrow, diff_pct=diff_pct)
    print()

if not stocks:
    print("  x No valid stock data loaded.")
    exit()


# ══════════════════════════════════════════════════════════════════
#  STEP 4 — CONSOLE RESULTS TABLE
# ══════════════════════════════════════════════════════════════════

print("\n" + "=" * 62)
print("   STOCK ANALYSIS RESULTS".center(62))
print("=" * 62)
for name, d in stocks.items():
    print(f"  {name:<8} Rs{d['prices'][-1]:>8.2f}  MA={d['filtered'][-1]:.2f}"
          f"  {d['signal']:<5}  {d['risk']:<13}  {d['diff_pct']:+.2f}%")
print("=" * 62)

print("\n  DETAILED REASONS:\n  " + "-" * 55)
for name, d in stocks.items():
    print(f"  {d['arrow']} {name} - {d['signal']}")
    print(f"    {d['reason']}")
    print(f"    Volatility : {d['volatility']:.2f}%  |  Risk : {d['risk']}\n")


# ══════════════════════════════════════════════════════════════════
#  STEP 5 — ALL 4 GRAPHS IN ONE COMBINED FIGURE
#
#  Layout  (GridSpec 4 rows x max(n,2) cols):
#  Row 0   Graph 1 — Price + MA Trend        (one panel per stock)
#  Row 1   Graph 2 — Gaussian Returns        (one panel per stock)
#  Row 2   Graph 3 — Autocorrelation         (one panel per stock)
#  Row 3   Graph 4 — Volatility + Risk       (2 panels, full width)
# ══════════════════════════════════════════════════════════════════

n       = len(stocks)
names   = list(stocks.keys())
PALETTE = ['#378ADD', '#1D9E75', '#BA7517', '#E24B4A', '#534AB7']
SIG_COL = {'BUY': '#1D9E75', 'SELL': '#E24B4A', 'HOLD': '#BA7517'}
SIG_MRK = {'BUY': '^',        'SELL': 'v',        'HOLD': 'D'}

fig = plt.figure(figsize=(7 * max(n, 2), 26))
fig.suptitle(
    'QUANTITATIVE MARKET SIMULATION ENGINE\n'
    'All Graphs Combined  |  Random Process Analysis — BECE207L  |  VIT Vellore',
    fontsize=14, fontweight='bold', y=1.002
)

gs = gridspec.GridSpec(4, max(n, 2), figure=fig, hspace=0.6, wspace=0.35)

# ── Row 0 : Price + MA Trend ──────────────────────────────────────
for idx, name in enumerate(names):
    d   = stocks[name]
    ax  = fig.add_subplot(gs[0, idx])
    N   = len(d['prices'])
    t   = np.arange(N)
    col = PALETTE[idx]

    ax.plot(t, d['prices'],   color=col,   lw=0.9, alpha=0.7,
            label=f'{name} Close Price')
    ax.plot(t, d['filtered'], color='#222', lw=2.0,
            label=f'{MA_WINDOW}-Day MA (LTI Filter)')
    ax.fill_between(t, d['prices'], d['filtered'],
                    where=(d['prices'] >= d['filtered']),
                    alpha=0.15, color='#1D9E75', label='Bullish zone')
    ax.fill_between(t, d['prices'], d['filtered'],
                    where=(d['prices'] <  d['filtered']),
                    alpha=0.15, color='#E24B4A', label='Bearish zone')
    ax.scatter(N-1, d['prices'][-1],
               color=SIG_COL[d['signal']], marker=SIG_MRK[d['signal']],
               s=180, zorder=5, label=f"Signal: {d['signal']}")
    ax.text(0.99, 0.97, f"  {d['arrow']}  {d['signal']}  ",
            transform=ax.transAxes, fontsize=12, fontweight='bold',
            color='white',
            bbox=dict(boxstyle='round,pad=0.4',
                      facecolor=SIG_COL[d['signal']], alpha=0.9),
            ha='right', va='top')
    ax.set_title(
        f'GRAPH 1 — {name}  |  Rs{d["prices"][-1]:.2f}'
        f'  |  MA({MA_WINDOW}d): Rs{d["filtered"][-1]:.2f}'
        f'  |  {d["risk"]}',
        fontsize=10, fontweight='bold')
    ax.set_xlabel('Trading Days')
    ax.set_ylabel('Price (Rs)')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3, linestyle='--')

# ── Row 1 : Gaussian Returns ──────────────────────────────────────
for idx, name in enumerate(names):
    d   = stocks[name]
    ax  = fig.add_subplot(gs[1, idx])
    ret = d['returns']
    col = PALETTE[idx]

    ax.hist(ret, bins=30, density=True, color=col, alpha=0.6,
            edgecolor='white', lw=0.5, label='Actual daily returns')
    x_fit = np.linspace(ret.min(), ret.max(), 300)
    ax.plot(x_fit, norm.pdf(x_fit, ret.mean(), ret.std()),
            color='black', lw=2.5,
            label=f'Gaussian N({ret.mean():.2f}, {ret.std():.2f}^2)')
    ax.axvline(ret.mean(), color='red', lw=1.5, linestyle='--',
               label=f'Mean = {ret.mean():.3f}%')
    ax.set_title(
        f'GRAPH 2 — {name} | Daily Returns Histogram\n'
        f'mu={ret.mean():.3f}%   sigma={ret.std():.2f}%',
        fontsize=10, fontweight='bold')
    ax.set_xlabel('Daily Return (%)')
    ax.set_ylabel('Probability Density')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, linestyle='--')

# ── Row 2 : Autocorrelation ───────────────────────────────────────
for idx, name in enumerate(names):
    d         = stocks[name]
    ax        = fig.add_subplot(gs[2, idx])
    col       = PALETTE[idx]
    lags, acf = compute_autocorrelation(d['returns'], max_lag=30)
    ci        = 1.96 / np.sqrt(len(d['returns']))

    ax.bar(lags[:len(acf)], acf, color=col, alpha=0.6,
           width=0.8, label='Autocorrelation')
    ax.axhline( ci, color='red', linestyle='--', lw=1,
                label=f'95% CI = +/-{ci:.3f}')
    ax.axhline(-ci, color='red', linestyle='--', lw=1)
    ax.axhline(0,   color='black', lw=0.6)
    ax.set_title(
        f'GRAPH 3 — {name} | Autocorrelation of Returns\n'
        f'Spike at lag 0 only = random process (good)',
        fontsize=10, fontweight='bold')
    ax.set_xlabel('Lag (days)')
    ax.set_ylabel('Autocorrelation')
    ax.set_ylim(-0.4, 1.1)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, linestyle='--')

# ── Row 3 : Volatility + Signal Divergence ────────────────────────
ax_vol = fig.add_subplot(gs[3, 0])
ax_sig = fig.add_subplot(gs[3, 1])

vols     = [stocks[nm]['volatility']         for nm in names]
bar_cols = [PALETTE[i]                        for i in range(len(names))]
sig_cols = [SIG_COL[stocks[nm]['signal']]     for nm in names]

bars1 = ax_vol.bar(names, vols, color=bar_cols,
                   edgecolor='white', lw=0.8, width=0.5)
ax_vol.axhline(1.0, color='#1D9E75', linestyle='--', lw=1.5,
               label='Low/Medium boundary (1%)')
ax_vol.axhline(2.5, color='#E24B4A', linestyle='--', lw=1.5,
               label='Medium/High boundary (2.5%)')
for bar, vol in zip(bars1, vols):
    ax_vol.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.05,
                f'{vol:.2f}%', ha='center', va='bottom',
                fontweight='bold', fontsize=11)
ax_vol.set_title('GRAPH 4 — Daily Volatility (sigma of returns)',
                 fontsize=10, fontweight='bold')
ax_vol.set_ylabel('Volatility (%)')
ax_vol.set_xlabel('Stock')
ax_vol.legend(fontsize=9)
ax_vol.grid(True, alpha=0.3, linestyle='--', axis='y')

bars2 = ax_sig.bar(names,
                   [abs(stocks[nm]['diff_pct']) for nm in names],
                   color=sig_cols, edgecolor='white', lw=0.8, width=0.5)
for bar, nm in zip(bars2, names):
    d = stocks[nm]
    ax_sig.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.05,
                f"{d['arrow']} {d['signal']}",
                ha='center', va='bottom',
                fontweight='bold', fontsize=12,
                color=SIG_COL[d['signal']])
ax_sig.set_title('GRAPH 4 — Price vs MA Divergence\n'
                 '(Determines BUY / SELL / HOLD signal)',
                 fontsize=10, fontweight='bold')
ax_sig.set_ylabel('|Price - MA| / MA  (%)')
ax_sig.set_xlabel('Stock')
ax_sig.grid(True, alpha=0.3, linestyle='--', axis='y')

plt.savefig('all_graphs_combined.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n  v Combined graph saved — all_graphs_combined.png")


# ══════════════════════════════════════════════════════════════════
#  STEP 6 — FINAL INVESTMENT ADVICE
# ══════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("   FINAL INVESTMENT ADVICE")
print("=" * 60 + "\n")

for name, d in stocks.items():
    print(f"  +-- {name} " + "-" * (50 - len(name)))
    print(f"  |  Signal      :  {d['arrow']}  {d['signal']}")
    print(f"  |  Last price  :  Rs{d['prices'][-1]:.2f}")
    print(f"  |  MA Trend    :  Rs{d['filtered'][-1]:.2f}")
    print(f"  |  Difference  :  {d['diff_pct']:+.2f}%")
    print(f"  |  Volatility  :  {d['volatility']:.2f}%")
    print(f"  |  Risk level  :  {d['risk']}")
    print(f"  |  Reason      :  {d['reason']}")
    print(f"  +" + "-" * 52 + "\n")

print("  RANDOM PROCESS CONCEPTS USED:")
print("  " + "-" * 50)
print("  v Gaussian Random Variable  - daily returns follow bell curve")
print("  v LTI Moving Average Filter - removes noise, reveals trend")
print("  v Autocorrelation R_xx(t)   - checks if past predicts future")
print("  v Volatility (Std Deviation)- measures randomness = risk level")
print()
print("  ! DISCLAIMER: This is a student project.")
print("    Do not make real investment decisions based on this output.")
print()
print("  v all_graphs_combined.png saved in the same folder")
print("  v Analysis complete!\n")
