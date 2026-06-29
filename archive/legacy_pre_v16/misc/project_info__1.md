# Clarifying ros_roll, rob_roll and How to Simplify the Sweep

## What ros_roll and rob_roll actually mean (in price action terms)

### `ros_roll` — "How recently was fast WR oversold?"

```python
fast_was_os = df["wr_X"].rolling(ros_roll).min() <= os_v
```

This **takes the minimum of WR over the last `ros_roll` bars**, then checks if it was ≤ `os_v`. 

**In English:** "At some point in the last `ros_roll` bars, was the fast WR oversold?"

- `ros_roll=10` → the oversold dip must have occurred **within the last 10 bars**. Tight. The entry fires soon after the oversold condition.
- `ros_roll=20` → the dip could have been **up to 20 bars ago**. More relaxed. Catches entries where price was oversold further back and is only now warming.

### `rob_roll` — "How recently was slow WR overbought?"

```python
slow_was_ob = df["wr_Y"].rolling(rob_roll).max() >= ob_v
```

This **takes the maximum of WR over the last `rob_roll` bars**, then checks if it was ≥ `ob_v`.

**In English:** "At some point in the last `rob_roll` bars, was the slow WR overbought?"

- `rob_roll=20` → the overbought peak was within the last 20 bars.
- `rob_roll=30` → the peak was up to 30 bars ago.

### Why these exist at all

The rolling min/max is how the code implements **sequential logic** ("was X, then became Y") without explicitly tracking state. Without it, you'd only catch the exact bar where WR crosses the threshold, which is too narrow for a real entry pattern. The rolling window says: "it happened recently enough, and now conditions are right."

---

## What `slk` does (the one you want to remove)

```python
fast_slope_pos = (df["wr_15"] - df["wr_15"].shift(slk)) > 0
```

**In English:** "Has WR been rising over the last `slk` bars?" 

This is a momentum filter — confirms that not only was WR oversold, but it's now **actively recovering** (price bottoming). You want to remove it. That's fine — just delete this line from the mask creation.

---

## Simplified sweep with WR(30) and no slk

Your conditions become 4 conjunctions:

```python
fast_was_os   = df["wr_30"].rolling(ros_roll).min()  <= os_v     # Fast was oversold
fast_warmed   = df["wr_30"]                           >= warm_v   # Now recovered
slow_was_ob   = df["wr_60"].rolling(rob_roll).max()   >= ob_v     # Slow was overbought
slow_cooled   = df["wr_60"]                           <= cool_v   # Now cooled

mask = fast_was_os & fast_warmed & slow_was_ob & slow_cooled
```

### What each parameter now controls:

| Parameter | What it tests | With WR(30) meaning |
|-----------|---------------|---------------------|
| `OS_VALS` | How deep the fast dip was | -30 = moderate dip, -50 = extreme dip |
| `WARM_VALS` | How far it recovered from that dip | -20 = still slightly weak, -10 = near neutral |
| `OB_VALS` | How high the slow peak was | -5 = barely overbought, 5 = clearly overbought |
| `COOL_VALS` | How far it pulled back from that peak | -30 = moderate pullback, -50 = deep pullback |
| `ROS_ROLL` | How recently was the fast dip? (bars) | 10 = very recent, 20 = moderately recent |
| `ROB_ROLL` | How recently was the slow peak? (bars) | 20 = recent, 30 = looser |

### With WR period=30 vs WR period=15

WR(30) cycles slower than WR(15) — one full oscillation takes roughly 30 bars. That means:

- A `ros_roll=10` with WR(30) = "oversold happened within the last **third** of a WR cycle" (tight)
- A `ros_roll=20` with WR(30) = "oversold happened within the last **two-thirds** of a WR cycle" (relaxed)

The sweep is **3×2×3×3×2×2 = 216 combos**, which is manageable. After filtering to combos with ≥1000 bars, you walk-forward test the top 10.

### Recommended sweep values to start:

```python
OS_VALS = [-30, -40, -50]      # sweep strictness of oversold
WARM_VALS = [-20, -10]          # sweep how much recovery needed
OB_VALS = [-5, 0, 5]           # sweep how overbought slow was
COOL_VALS = [-30, -40, -50]    # sweep how much slow cooled
ROS_ROLL = [10, 20]            # sweep recency of fast oversold
ROB_ROLL = [20, 30]            # sweep recency of slow overbought
```

This gives you the **same 4-condition structure as the short strategy**, just mirrored for longs, with WR(30) instead of WR(15), and no slope filter.