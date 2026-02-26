import yfinance as yf
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import os
import numpy as np

# -----------------------------
# Step 1: Download and Align Data
# -----------------------------
manu = yf.download("MANU", start="2025-01-01")
sp500 = yf.download("^GSPC", start="2025-01-01")

# Keep only Close prices
manu = manu[['Close']].rename(columns={'Close': 'MANU_Close'})
sp500 = sp500[['Close']].rename(columns={'Close': 'SP500_Close'})

# Merge to ensure perfect date alignment
data = manu.merge(sp500, left_index=True, right_index=True, how='inner')

# -----------------------------
# Step 2: Calculate Returns
# -----------------------------
data['MANU_Returns'] = data['MANU_Close'].pct_change()
data['SP500_Returns'] = data['SP500_Close'].pct_change()

data.dropna(inplace=True)

# -----------------------------
# Step 3: Alpha & Beta (252-day Estimation Window)
# -----------------------------
event_date = pd.to_datetime("2026-01-13")

estimation_window = data.loc[:event_date].iloc[-253:-1]

y = estimation_window['MANU_Returns']
X = estimation_window['SP500_Returns']

X = sm.add_constant(X)
model = sm.OLS(y, X).fit()

alpha = model.params['const']
beta = model.params['SP500_Returns']

print(f"\nAlpha: {alpha:.6f}")
print(f"Beta:  {beta:.6f}")

# -----------------------------
# Step 4: Expected & Abnormal Returns
# -----------------------------
data['Expected'] = alpha + beta * data['SP500_Returns']
data['Abnormal'] = data['MANU_Returns'] - data['Expected']

# -----------------------------
# Step 5: Event Window (±21 Days)
# -----------------------------
if event_date not in data.index:
    raise ValueError("Event date not in trading calendar.")

event_index = data.index.get_loc(event_date)
event_window = data.iloc[event_index-21:event_index+22].copy()

# -----------------------------
# Step 6: CAR
# -----------------------------
event_window['CAR'] = event_window['Abnormal'].cumsum()

# -----------------------------
# Step 7: Significance Testing
# -----------------------------
est_std = estimation_window['MANU_Returns'].std()

event_window['t_AR'] = event_window['Abnormal'] / est_std

N = len(event_window)
CAR_std = est_std * np.sqrt(N)
t_CAR = event_window['CAR'].iloc[-1] / CAR_std

print(f"Cumulative Abnormal Return t-statistic: {t_CAR:.3f}")

# Identify significant days
sig_pos = event_window[event_window['t_AR'] > 2]
sig_neg = event_window[event_window['t_AR'] < -2]

# -----------------------------
# Step 8: Professional CAR Plot
# -----------------------------
output_folder = "manu_event_analysis"
os.makedirs(output_folder, exist_ok=True)

plt.figure(figsize=(12,6))

# CAR line
plt.plot(event_window.index,
         event_window['CAR'],
         marker='o',
         linewidth=2,
         label='CAR')

# Significant AR markers (plotted at CAR level)
if not sig_pos.empty:
    plt.scatter(sig_pos.index,
                sig_pos['CAR'],
                s=120,
                zorder=5,
                label='Significant Positive AR')

if not sig_neg.empty:
    plt.scatter(sig_neg.index,
                sig_neg['CAR'],
                s=120,
                zorder=5,
                label='Significant Negative AR')

# Market overlay (scaled cumulative)
market_overlay = event_window['SP500_Returns'].cumsum() * beta
plt.plot(event_window.index,
         market_overlay,
         linestyle='--',
         label='Market (scaled)')

# Event line
plt.axvline(event_date,
            linestyle='--',
            linewidth=1.5,
            label='Manager Change')

# Formatting
plt.title("Cumulative Abnormal Return with Significant AR Days and Market Overlay", fontsize=14)
plt.xlabel("Date", fontsize=12)
plt.ylabel("CAR / Scaled Market Returns", fontsize=12)
plt.grid(True)
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

# Save plot
plot_file = os.path.join(output_folder, "CAR_plot.png")
plt.savefig(plot_file)
plt.show()

print(f"Professional CAR plot saved to {plot_file}")

# -----------------------------
# Step 9: Export Event Window Table
# -----------------------------
table_file = os.path.join(output_folder, "event_window.csv")
event_window.to_csv(table_file)

print(f"Event window table saved to {table_file}")