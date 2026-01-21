# Inflation Direction Accuracy Evaluation

Evaluation of the `inflation_direction` field extracted from model predictions.

**Date:** 2025-01-20
**Dataset:** ProQuest 2010-2025 (partial - 10/45 chunks)
**Sample Size:** 210 sentences (70 per direction)

## Distribution in Dataset

| inflation_direction | Count |
|---------------------|-------|
| NaN (no narrative) | 85,799 |
| up | 44,141 |
| down | 20,888 |
| same | 13,809 |

## Accuracy Summary

| Direction | Estimated Accuracy | Notes |
|-----------|-------------------|-------|
| up | **~95%** | High accuracy |
| down | **~70-75%** | Some mislabels |
| same | **~35-40%** | Problematic - used as catch-all |

---

## Detailed Analysis

### UP (70 samples) - ~95% accurate

Almost all correct. Clear indicators include: "inflation pressure", "spiraling inflation", "double digits", "soaring", "high inflation", "inflation hit", "pushing up prices"

**Sample correct labels:**
- "Consumers are pulling back from spending amid inflation pressure"
- "inflation was cracking double digits"
- "substantial input cost inflation"
- "high U.S. inflation and weak economic growth"
- "spiraling inflation"
- "inflation soaring at one point beyond 85 percent"

**Questionable (2-3):**
- Dictionary definition of inflation (educational, not directional)
- "hasn't triggered excessive inflation" (potential future, not current)

---

### DOWN (70 samples) - ~70-75% accurate

**Clearly correct (~50):**
- "get inflation under control"
- "inflation falls to 3.5 percent"
- "inflation is easing"
- "inflation should fall back sharply"
- "bring down inflation"
- "lowering inflation"
- "cooled inflation"
- "inflation has come down"
- "inflation continued to slow"

**Questionable/Wrong (~20):**

| Sample | Text | Issue |
|--------|------|-------|
| #5 | "limits collective bargaining to wage increases not above the rate of inflation" | About wage caps, not inflation direction |
| #10 | "Trend inflation...determined largely by public expectations" | Educational, not directional |
| #20 | "eroding their real value over time" | About tax credits not indexed, not inflation falling |
| #28, #47 | Mentions "Inflation Reduction Act" | Just references the act name |
| #37 | "High inflation, slowing economic growth" | Should be **UP** |
| #40 | "struggling under inflation" | Should be **UP** |

---

### SAME (70 samples) - ~35-40% accurate

The "same" category has significant accuracy issues. The model appears to use it as a catch-all for sentences that mention inflation without an explicit trend word, but many actually convey direction through context.

**Correctly neutral/stable (~25-30):**
- "adjusts tax brackets every year to ward off 'bracket creep'" (inflation adjustment)
- "slowing the annual inflation adjustments" (policy about adjustments)
- "interest rate paid on TIPS will rise with inflation" (inflation-indexed)
- "learning about things like inflation" (educational)
- "depending on the mix of inflation and growth" (general/comparative)

**Should be UP (~25-30):**

| Sample | Text | Why it's UP |
|--------|------|-------------|
| #15 | "high interest rates and inflation" | High inflation mentioned |
| #17 | "where inflation is very high, you would recommend...tight monetary policy" | High inflation |
| #23 | "a 0.5 percent rate hike because of Israel's high inflation" | High inflation |
| #24 | "inflation remained stubbornly in the 9-10% range" | Clearly high/up |
| #34 | "high inflation, combined with the Fed's zero rates" | High inflation |
| #55 | "stagnant state funding during high inflation" | High inflation |
| #59 | "increases the Argentine inflation rate, which was over 200%" | **Clearly UP** |
| #60 | "high inflation and rampant corruption" | High inflation |
| #70 | "cost of living crisis in America" | Implies high inflation |

**Should be DOWN (~10):**

| Sample | Text | Why it's DOWN |
|--------|------|---------------|
| #7 | "inflation is now about where the Fed wants it to be, in the 2% range" | Achieved target = down |
| #13 | "even though inflation eased a little bit last month" | "eased" = down |

---
