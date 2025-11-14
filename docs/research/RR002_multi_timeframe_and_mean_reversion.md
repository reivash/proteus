# Research Request RR-002: Multi-Timeframe Prediction & Mean Reversion

**Date:** 2025-11-14
**Requested By:** User
**Priority:** HIGH
**Type:** Strategy Enhancement

---

## RESEARCH OBJECTIVES

### 1. Multi-Timeframe Prediction Analysis

**Question:** How does model accuracy vary across different prediction horizons?

**Timeframes to Test:**
- **Intraday:** Same-day price movement (requires minute data)
- **Next Day:** Tomorrow's direction (current focus)
- **Next Week:** 5-trading-day ahead
- **Next Month:** 20-trading-day ahead
- **Next Quarter:** 60-trading-day ahead
- **Long-term:** 6 months to 1 year

**Hypothesis:**
- Shorter timeframes = More noise, harder to predict
- Medium timeframes (week/month) = Sweet spot?
- Long-term = Easier but less actionable

---

### 2. Overcorrection Detection & Mean Reversion

**Concept:** Identify when stock has moved too far too fast, predict reversion

**User Observation:**
> "Sometimes the stock jumps down or up, but then it tends to normalize itself...
> If an action suddenly becomes cheap, in 1-2 days it tends to recover"

**Strategy:**
1. Detect abnormal moves (>2σ from recent average)
2. Check if fundamentals support the move
3. If not → predict mean reversion within 1-2 days
4. High-frequency trading opportunity

---

## PART 1: MULTI-TIMEFRAME PREDICTION

### Approach

**For each timeframe, measure:**
- Directional accuracy
- Magnitude accuracy (MAE, RMSE)
- Confidence calibration
- Profit potential (if traded)

### Implementation Plan

```python
# Create multi-horizon targets
def create_multi_timeframe_targets(df):
    """Create targets for multiple prediction horizons."""

    # Next day (current)
    df['direction_1d'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df['return_1d'] = df['Close'].pct_change(-1)

    # Next week (5 trading days)
    df['direction_5d'] = (df['Close'].shift(-5) > df['Close']).astype(int)
    df['return_5d'] = (df['Close'].shift(-5) / df['Close'] - 1) * 100

    # Next month (20 trading days)
    df['direction_20d'] = (df['Close'].shift(-20) > df['Close']).astype(int)
    df['return_20d'] = (df['Close'].shift(-20) / df['Close'] - 1) * 100

    # Next quarter (60 trading days)
    df['direction_60d'] = (df['Close'].shift(-60) > df['Close']).astype(int)
    df['return_60d'] = (df['Close'].shift(-60) / df['Close'] - 1) * 100

    return df
```

### Expected Results

| Timeframe | Hypothesis | Reasoning |
|-----------|------------|-----------|
| **Intraday** | 50-52% | High noise, random walk |
| **Next Day** | **55-60%** | Current: 58.94% ✅ |
| **Next Week** | **60-65%** | Smooths daily noise, captures trends |
| **Next Month** | **65-70%** | Clear trends emerge |
| **Next Quarter** | 60-65% | Too long, new info changes outlook |

**Sweet Spot Hypothesis:** **1 week to 1 month** predictions will likely have highest accuracy

### Trading Implications

- **Day trading:** Lower accuracy, but many opportunities
- **Swing trading (week):** Better accuracy, medium frequency
- **Position trading (month):** Best accuracy, low frequency
- **Long-term:** High accuracy but less relevant for active trading

---

## PART 2: MEAN REVERSION / OVERCORRECTION DETECTION

### Concept

**Mean Reversion:** Prices tend to revert to their average/trend after extreme moves

**Types of Moves to Detect:**

#### A. Statistical Overcorrection
```python
# Detect moves > 2 standard deviations
recent_mean = df['Close'].rolling(20).mean()
recent_std = df['Close'].rolling(20).std()

df['z_score'] = (df['Close'] - recent_mean) / recent_std
df['overcorrection'] = (df['z_score'].abs() > 2).astype(int)
```

#### B. Momentum Exhaustion
```python
# Detect when RSI is extreme
df['rsi_oversold'] = (df['rsi'] < 30).astype(int)  # Likely to bounce
df['rsi_overbought'] = (df['rsi'] > 70).astype(int)  # Likely to drop
```

#### C. Volume Spike + Price Move
```python
# Unusual volume with big price move = potential overreaction
avg_volume = df['Volume'].rolling(20).mean()
df['volume_spike'] = (df['Volume'] > avg_volume * 2).astype(int)

df['panic_sell'] = (df['volume_spike'] & (df['return_1d'] < -3)).astype(int)
df['panic_buy'] = (df['volume_spike'] & (df['return_1d'] > 3)).astype(int)
```

### Strategy: Fade the Overreaction

**Entry Signal:**
1. Stock drops >3% in one day
2. Volume is 2x normal
3. No fundamental news (earnings, etc.)
4. Technical indicators show oversold (RSI < 30)

**Expected Outcome:**
- Stock recovers 1-2% within 1-2 days
- Quick profit opportunity

**Trade Setup:**
- **Buy:** When overcorrection detected (panic sell)
- **Hold:** 1-2 days
- **Sell:** When recovered to recent average OR 2-day timeout
- **Stop Loss:** -2% (in case it's not overcorrection but real bad news)

### Implementation

```python
class MeanReversionDetector:
    """Detect overcorrections and predict mean reversion."""

    def detect_overcorrection(self, df):
        """Identify potential overcorrections."""

        # Calculate signals
        df['z_score'] = self.calculate_z_score(df)
        df['rsi'] = self.calculate_rsi(df)
        df['volume_anomaly'] = self.detect_volume_spike(df)

        # Overcorrection flags
        df['oversold'] = (
            (df['z_score'] < -2) &  # 2σ below mean
            (df['rsi'] < 30) &       # RSI oversold
            (df['volume_anomaly'] == 1) &  # Volume spike
            (df['return_1d'] < -3)   # Big drop
        ).astype(int)

        df['overbought'] = (
            (df['z_score'] > 2) &
            (df['rsi'] > 70) &
            (df['volume_anomaly'] == 1) &
            (df['return_1d'] > 3)
        ).astype(int)

        return df

    def predict_reversion(self, df):
        """Predict if/when price will revert."""

        # For oversold: predict +1-3% recovery in 1-2 days
        # For overbought: predict -1-3% drop in 1-2 days

        df['reversion_target'] = np.where(
            df['oversold'] == 1,
            df['Close'] * 1.02,  # +2% recovery
            np.where(
                df['overbought'] == 1,
                df['Close'] * 0.98,  # -2% correction
                df['Close']  # No prediction
            )
        )

        df['reversion_timeframe'] = 2  # days

        return df
```

### Backtesting Mean Reversion Strategy

```python
def backtest_mean_reversion(df, initial_capital=10000):
    """
    Backtest mean reversion strategy.

    Rules:
    - Buy when oversold detected
    - Sell after 2 days OR if recovered +2% OR stop loss -2%
    """

    capital = initial_capital
    position = None
    trades = []

    for i in range(len(df) - 2):
        # Entry signal
        if df.iloc[i]['oversold'] == 1 and position is None:
            entry_price = df.iloc[i]['Close']
            position = {
                'entry_price': entry_price,
                'entry_date': df.iloc[i]['Date'],
                'shares': capital / entry_price
            }

        # Exit signal (if in position)
        if position is not None:
            current_price = df.iloc[i]['Close']
            entry_price = position['entry_price']

            # Calculate return
            return_pct = (current_price / entry_price - 1) * 100
            days_held = (df.iloc[i]['Date'] - position['entry_date']).days

            # Exit conditions
            should_exit = (
                return_pct >= 2 or  # Hit profit target
                return_pct <= -2 or  # Stop loss
                days_held >= 2  # Timeout
            )

            if should_exit:
                # Close position
                profit = (current_price - entry_price) * position['shares']
                capital += profit

                trades.append({
                    'entry': entry_price,
                    'exit': current_price,
                    'return_pct': return_pct,
                    'profit': profit,
                    'days_held': days_held
                })

                position = None

    # Calculate metrics
    total_trades = len(trades)
    winning_trades = sum(1 for t in trades if t['profit'] > 0)
    win_rate = winning_trades / total_trades if total_trades > 0 else 0

    total_return = (capital / initial_capital - 1) * 100

    return {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'total_return': total_return,
        'final_capital': capital,
        'trades': trades
    }
```

---

## EXPECTED OUTCOMES

### Multi-Timeframe Analysis

**Best Case:**
- Identify optimal prediction horizon (likely 1-4 weeks)
- Accuracy improves to 65-70% for monthly predictions
- Clear trading strategy for each timeframe

**Insights:**
- Which timeframe suits model best
- Risk/reward at different horizons
- Optimal holding period

### Mean Reversion Strategy

**Best Case:**
- 60-70% win rate on overcorrection trades
- Average gain: +2-3% per trade
- 20-30 opportunities per year (S&P 500)
- Annualized return: 15-25% (if consistent)

**Risk Factors:**
- False signals (real bad news vs overreaction)
- "Catching falling knife" (trend continuation)
- Requires quick execution

---

## IMPLEMENTATION PRIORITY

### Phase 1: Multi-Timeframe Prediction (1-2 weeks)

**Steps:**
1. Create multi-horizon targets (1d, 5d, 20d, 60d)
2. Train XGBoost for each timeframe
3. Compare accuracies
4. Identify sweet spot
5. Document findings

**Expected Result:** Clear answer on best prediction horizon

### Phase 2: Mean Reversion Detection (2-3 weeks)

**Steps:**
1. Implement overcorrection detectors (z-score, RSI, volume)
2. Backtest on historical data (last 5-10 years)
3. Measure win rate, avg profit, max drawdown
4. Optimize entry/exit rules
5. Paper trade for 1 month

**Expected Result:** Profitable mean reversion strategy (if viable)

---

## RESEARCH QUESTIONS

### Multi-Timeframe:
1. Does accuracy improve with longer horizons?
2. What's the optimal timeframe for our feature set?
3. Do different features matter for different timeframes?
4. How does market regime affect timeframe performance?

### Mean Reversion:
1. What % of large moves are overcorrections vs real trends?
2. What's the best signal combination (z-score + RSI + volume)?
3. How long does mean reversion typically take?
4. Does it work better for indices vs individual stocks?
5. Does it work in all market regimes?

---

## SUCCESS METRICS

### Multi-Timeframe:
- ✅ Accuracy >60% for at least one non-daily timeframe
- ✅ Clear ranking of timeframes by predictability
- ✅ Actionable trading strategy for each horizon

### Mean Reversion:
- ✅ Win rate >60%
- ✅ Average gain >1.5% per trade
- ✅ Max drawdown <10%
- ✅ At least 15-20 opportunities per year
- ✅ Positive Sharpe ratio (>1.0)

---

## RISKS & CHALLENGES

### Multi-Timeframe:
- Longer timeframes = less data (60-day prediction needs 60-day forward data)
- May need 10+ years of data
- Different market regimes affect different timeframes differently

### Mean Reversion:
- "Markets can remain irrational longer than you can remain solvent"
- Distinguishing overcorrection from genuine bad news
- Requires very fast execution (minutes matter)
- Higher transaction costs for frequent trading
- False signals can rack up losses quickly

---

## RECOMMENDED NEXT EXPERIMENTS

### EXP-007: Multi-Timeframe Prediction
**Goal:** Test model across 1d, 5d, 20d, 60d horizons
**Expected:** 1-4 week predictions most accurate (65-70%)
**Timeline:** 1 week

### EXP-008: Mean Reversion Strategy
**Goal:** Detect and trade overcorrections
**Expected:** 60% win rate, +2% avg gain
**Timeline:** 2 weeks

### EXP-009: Combined Strategy
**Goal:** Use multi-timeframe + mean reversion together
**Expected:** Best overall performance
**Timeline:** 3 weeks

---

## POTENTIAL GAME-CHANGERS

1. **Weekly Predictions:**
   - If accuracy reaches 65-70% for weekly predictions
   - Lower frequency = lower transaction costs
   - More reliable signals

2. **Mean Reversion Profits:**
   - If 60%+ win rate with +2% avg gain
   - 20-30 trades/year = 30-40% annual return
   - Quick turnaround (1-2 days holding)
   - High ROI per trade

3. **Combined Strategy:**
   - Use multi-timeframe for trend direction
   - Use mean reversion for entry timing
   - Synergistic effect

---

## CONCLUSION

**User insights are highly valuable!** Both research directions:

1. **Multi-timeframe prediction** → Find optimal prediction horizon
2. **Mean reversion trading** → Exploit market overreactions

These complement our current system perfectly and could significantly improve profitability.

**Recommendation:** Prioritize both for next research cycle.

---

**Research Request:** RR-002
**Status:** APPROVED
**Priority:** HIGH
**Assigned To:** Hermes → Athena → Prometheus
**Expected Completion:** 3-4 weeks

---
