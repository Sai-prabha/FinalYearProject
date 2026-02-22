/**
 * Client-side Excel export utility for trade history.
 * Uses ExcelJS for .xlsx generation with image embedding support.
 *
 * Sheets (in order):
 *   1. Dashboard         – executive KPI summary (first tab)
 *   2. Trade Decisions   – per-trade decision log with full audit columns
 *   3. Trade Summary     – compact per-trade rows
 *   4. Statistics         – aggregated metrics + embedded chart images
 *   5. Risk Metrics       – Sharpe, Sortino, VaR, Kelly, streaks, drawdown
 *   6. Time Analysis      – hourly, day-of-week, duration breakdowns
 *   7. Trade Quality      – by signal quality, probability, exit reason, direction
 *   8. Parameters         – model & strategy config (enhanced)
 *   9. Top Features       – all 50 features + horizontal bar chart
 *
 * Key fix (v2): "Price Chg %" is the raw ratio movement; "Portfolio Rtn %"
 * is the actual percentage impact on the account (accounts for leverage &
 * compounding balance). The Statistics sheet now uses portfolio returns
 * instead of naively summing raw price-change percentages.
 */
import type { Trade, ModelSignalData } from '../types';
import { MODEL_SERVER_REST_URL } from '../constants';
import { getAuthHeaders } from './auth';

/* ------------------------------------------------------------------ */
/*  Type alias for ExcelJS worksheet (used across helpers)             */
/* ------------------------------------------------------------------ */
type Worksheet = import('exceljs').Worksheet;
type Workbook = import('exceljs').Workbook;

/* ------------------------------------------------------------------ */
/*  ExcelJS lazy import                                                */
/* ------------------------------------------------------------------ */
let ExcelJS: typeof import('exceljs') | null = null;

const ensureExcelJS = async () => {
  if (!ExcelJS) {
    ExcelJS = await import('exceljs');
  }
  return ExcelJS;
};

/* ------------------------------------------------------------------ */
/*  Public interface                                                    */
/* ------------------------------------------------------------------ */
export interface ExportOptions {
  trades: Trade[];
  signalData?: ModelSignalData | null;
  filename?: string;
}

/* ------------------------------------------------------------------ */
/*  Enriched trade type — adds computed columns                        */
/* ------------------------------------------------------------------ */
interface EnrichedTrade extends Trade {
  index: number;
  balanceBefore: number;
  balanceAfter: number;
  portfolioRtnPct: number;   // (pnl_dollar / balanceBefore) * 100
  cumulativePnl: number;
  cumulativeRtnPct: number;  // ((balanceAfter - starting) / starting) * 100
  quality: 'WEAK' | 'MEDIUM' | 'STRONG';
  event: string;
  exitHourUTC: number;
  exitDayOfWeek: number;     // 0=Sun … 6=Sat
}

/* ------------------------------------------------------------------ */
/*  Pre-computation: enrich trades with balance tracking               */
/* ------------------------------------------------------------------ */
function enrichTrades(trades: Trade[], startingBalance: number): EnrichedTrade[] {
  let runningBalance = startingBalance;
  let cumPnl = 0;
  return trades.map((t, i) => {
    const balanceBefore = runningBalance;
    cumPnl += t.pnl_dollar;
    runningBalance += t.pnl_dollar;
    const portfolioRtnPct = balanceBefore > 0 ? (t.pnl_dollar / balanceBefore) * 100 : 0;
    const cumulativeRtnPct = startingBalance > 0
      ? ((runningBalance - startingBalance) / startingBalance) * 100
      : 0;
    const strength = t.entry_strength ?? 0;
    let quality: 'WEAK' | 'MEDIUM' | 'STRONG' = 'WEAK';
    if (strength > 0.15) quality = 'STRONG';
    else if (strength > 0.08) quality = 'MEDIUM';

    const event = t.pnl_dollar > 0 ? 'EXIT (WIN)' : t.pnl_dollar < 0 ? 'EXIT (LOSS)' : 'EXIT (BE)';
    const exitDate = new Date(t.exit_time * 1000);

    return {
      ...t,
      index: i,
      balanceBefore,
      balanceAfter: runningBalance,
      portfolioRtnPct,
      cumulativePnl: cumPnl,
      cumulativeRtnPct,
      quality,
      event,
      exitHourUTC: exitDate.getUTCHours(),
      exitDayOfWeek: exitDate.getUTCDay(),
    };
  });
}

/* ------------------------------------------------------------------ */
/*  Risk metrics computation                                           */
/* ------------------------------------------------------------------ */
interface RiskMetrics {
  sharpeRatio: number;
  sortinoRatio: number;
  var95: number;
  maxDrawdownPct: number;
  maxDrawdownDollar: number;
  maxDDPeakDate: string;
  maxDDTroughDate: string;
  recoveryFactor: number;
  kellyCriterion: number;
  expectancy: number;
  maxConsecutiveWins: number;
  maxConsecutiveLosses: number;
  currentStreak: number;         // positive = wins, negative = losses
  currentStreakType: string;
}

function computeRiskMetrics(
  enriched: EnrichedTrade[],
  startingBalance: number,
): RiskMetrics {
  const n = enriched.length;
  const empty: RiskMetrics = {
    sharpeRatio: 0, sortinoRatio: 0, var95: 0,
    maxDrawdownPct: 0, maxDrawdownDollar: 0,
    maxDDPeakDate: '-', maxDDTroughDate: '-',
    recoveryFactor: 0, kellyCriterion: 0, expectancy: 0,
    maxConsecutiveWins: 0, maxConsecutiveLosses: 0,
    currentStreak: 0, currentStreakType: '-',
  };
  if (n === 0) return empty;

  // Portfolio returns per trade (%)
  const returns = enriched.map(t => t.portfolioRtnPct);
  const mean = returns.reduce((s, r) => s + r, 0) / n;
  const variance = returns.reduce((s, r) => s + (r - mean) ** 2, 0) / n;
  const std = Math.sqrt(variance);

  // Downside deviation (only negative returns)
  const downsideVariance = returns.reduce((s, r) => {
    const d = Math.min(r, 0);
    return s + d * d;
  }, 0) / n;
  const downsideStd = Math.sqrt(downsideVariance);

  // Annualization: estimate trades per year from avg bars held
  // Each bar = 1 minute, so bars_per_year ≈ 525600
  const avgBars = enriched.reduce((s, t) => s + (t.bars_held || 25), 0) / n;
  const tradesPerYear = avgBars > 0 ? 525600 / avgBars : 1000;
  const annFactor = Math.sqrt(tradesPerYear);

  const sharpeRatio = std > 0 ? (mean / std) * annFactor : 0;
  const sortinoRatio = downsideStd > 0 ? (mean / downsideStd) * annFactor : 0;

  // VaR (95%): 5th percentile of returns
  const sorted = [...returns].sort((a, b) => a - b);
  const varIdx = Math.max(0, Math.floor(n * 0.05) - 1);
  const var95 = sorted[varIdx] ?? 0;

  // Max drawdown with dates
  let peak = startingBalance;
  let peakTime = enriched[0]?.entry_time ?? 0;
  let maxDDPct = 0, maxDDDollar = 0;
  let ddPeakTime = 0, ddTroughTime = 0;
  let bal = startingBalance;
  for (const t of enriched) {
    bal = t.balanceAfter;
    if (bal > peak) {
      peak = bal;
      peakTime = t.exit_time;
    }
    const ddPct = peak > 0 ? ((peak - bal) / peak) * 100 : 0;
    const ddDol = peak - bal;
    if (ddPct > maxDDPct) {
      maxDDPct = ddPct;
      maxDDDollar = ddDol;
      ddPeakTime = peakTime;
      ddTroughTime = t.exit_time;
    }
  }

  const totalPnL = enriched[enriched.length - 1].cumulativePnl;
  const recoveryFactor = maxDDDollar > 0 ? totalPnL / maxDDDollar : 0;

  // Win/loss stats
  const wins = enriched.filter(t => t.pnl_dollar > 0);
  const losses = enriched.filter(t => t.pnl_dollar <= 0);
  const winRate = n > 0 ? wins.length / n : 0;
  const avgWinDollar = wins.length > 0
    ? wins.reduce((s, t) => s + t.pnl_dollar, 0) / wins.length : 0;
  const avgLossDollar = losses.length > 0
    ? Math.abs(losses.reduce((s, t) => s + t.pnl_dollar, 0) / losses.length) : 0;

  // Kelly criterion: f* = W - (1-W)/R  where R = avg_win/avg_loss
  const winLossRatio = avgLossDollar > 0 ? avgWinDollar / avgLossDollar : 0;
  const kellyCriterion = winLossRatio > 0
    ? winRate - (1 - winRate) / winLossRatio
    : 0;

  // Expectancy: (WR * avg_win) + ((1-WR) * avg_loss)  [avg_loss is negative]
  const avgLossSigned = losses.length > 0
    ? losses.reduce((s, t) => s + t.pnl_dollar, 0) / losses.length : 0;
  const expectancy = (winRate * avgWinDollar) + ((1 - winRate) * avgLossSigned);

  // Consecutive streaks
  let maxCW = 0, maxCL = 0, curW = 0, curL = 0;
  for (const t of enriched) {
    if (t.pnl_dollar > 0) {
      curW++;
      curL = 0;
      if (curW > maxCW) maxCW = curW;
    } else {
      curL++;
      curW = 0;
      if (curL > maxCL) maxCL = curL;
    }
  }
  const currentStreak = curW > 0 ? curW : -curL;
  const currentStreakType = curW > 0 ? `${curW} Wins` : curL > 0 ? `${curL} Losses` : '-';

  const fmtTime = (ts: number) => ts > 0
    ? new Date(ts * 1000).toISOString().replace('T', ' ').slice(0, 19)
    : '-';

  return {
    sharpeRatio, sortinoRatio, var95,
    maxDrawdownPct: maxDDPct,
    maxDrawdownDollar: maxDDDollar,
    maxDDPeakDate: fmtTime(ddPeakTime),
    maxDDTroughDate: fmtTime(ddTroughTime),
    recoveryFactor,
    kellyCriterion,
    expectancy,
    maxConsecutiveWins: maxCW,
    maxConsecutiveLosses: maxCL,
    currentStreak,
    currentStreakType,
  };
}

/* ------------------------------------------------------------------ */
/*  Time analysis computation                                          */
/* ------------------------------------------------------------------ */
interface TimeBucket {
  label: string;
  count: number;
  wins: number;
  winRate: number;
  totalPnl: number;
  avgPnl: number;
}

function computeTimeAnalysis(enriched: EnrichedTrade[]): {
  hourly: TimeBucket[];
  daily: TimeBucket[];
  duration: TimeBucket[];
} {
  // Hourly
  const hourMap = new Map<number, EnrichedTrade[]>();
  for (let h = 0; h < 24; h++) hourMap.set(h, []);
  for (const t of enriched) {
    hourMap.get(t.exitHourUTC)?.push(t);
  }
  const hourly: TimeBucket[] = [];
  for (let h = 0; h < 24; h++) {
    const trades = hourMap.get(h) || [];
    const w = trades.filter(t => t.pnl_dollar > 0).length;
    const total = trades.reduce((s, t) => s + t.pnl_dollar, 0);
    hourly.push({
      label: `${h.toString().padStart(2, '0')}:00 UTC`,
      count: trades.length,
      wins: w,
      winRate: trades.length > 0 ? (w / trades.length) * 100 : 0,
      totalPnl: total,
      avgPnl: trades.length > 0 ? total / trades.length : 0,
    });
  }

  // Day of week
  const dayNames = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];
  const dayMap = new Map<number, EnrichedTrade[]>();
  for (let d = 0; d < 7; d++) dayMap.set(d, []);
  for (const t of enriched) {
    dayMap.get(t.exitDayOfWeek)?.push(t);
  }
  const daily: TimeBucket[] = [];
  for (let d = 0; d < 7; d++) {
    const trades = dayMap.get(d) || [];
    const w = trades.filter(t => t.pnl_dollar > 0).length;
    const total = trades.reduce((s, t) => s + t.pnl_dollar, 0);
    daily.push({
      label: dayNames[d],
      count: trades.length,
      wins: w,
      winRate: trades.length > 0 ? (w / trades.length) * 100 : 0,
      totalPnl: total,
      avgPnl: trades.length > 0 ? total / trades.length : 0,
    });
  }

  // Duration buckets
  const durationBuckets = [
    { label: '1-5 bars', min: 1, max: 5 },
    { label: '6-15 bars', min: 6, max: 15 },
    { label: '16-30 bars', min: 16, max: 30 },
    { label: '31+ bars', min: 31, max: Infinity },
  ];
  const duration: TimeBucket[] = durationBuckets.map(b => {
    const trades = enriched.filter(t => {
      const bh = t.bars_held ?? 0;
      return bh >= b.min && bh <= b.max;
    });
    const w = trades.filter(t => t.pnl_dollar > 0).length;
    const total = trades.reduce((s, t) => s + t.pnl_dollar, 0);
    return {
      label: b.label,
      count: trades.length,
      wins: w,
      winRate: trades.length > 0 ? (w / trades.length) * 100 : 0,
      totalPnl: total,
      avgPnl: trades.length > 0 ? total / trades.length : 0,
    };
  });

  return { hourly, daily, duration };
}

/* ------------------------------------------------------------------ */
/*  Trade quality computation                                          */
/* ------------------------------------------------------------------ */
interface QualityBucket {
  label: string;
  count: number;
  wins: number;
  winRate: number;
  totalPnl: number;
  avgPnl: number;
  avgPortfolioRtn: number;
}

function aggregateBucket(label: string, trades: EnrichedTrade[]): QualityBucket {
  const w = trades.filter(t => t.pnl_dollar > 0).length;
  const totalPnl = trades.reduce((s, t) => s + t.pnl_dollar, 0);
  const avgPortRtn = trades.length > 0
    ? trades.reduce((s, t) => s + t.portfolioRtnPct, 0) / trades.length : 0;
  return {
    label,
    count: trades.length,
    wins: w,
    winRate: trades.length > 0 ? (w / trades.length) * 100 : 0,
    totalPnl,
    avgPnl: trades.length > 0 ? totalPnl / trades.length : 0,
    avgPortfolioRtn: avgPortRtn,
  };
}

function computeTradeQuality(enriched: EnrichedTrade[]): {
  byQuality: QualityBucket[];
  byProbability: QualityBucket[];
  byReason: QualityBucket[];
  byDirection: QualityBucket[];
} {
  // By signal quality
  const qualityGroups: Record<string, EnrichedTrade[]> = { WEAK: [], MEDIUM: [], STRONG: [] };
  for (const t of enriched) {
    qualityGroups[t.quality]?.push(t);
  }
  const byQuality = ['WEAK', 'MEDIUM', 'STRONG'].map(q =>
    aggregateBucket(q, qualityGroups[q] || [])
  );

  // By probability range
  const probRanges = [
    { label: '0.50 - 0.525', min: 0.50, max: 0.525 },
    { label: '0.525 - 0.55', min: 0.525, max: 0.55 },
    { label: '0.55 - 0.60', min: 0.55, max: 0.60 },
    { label: '0.60+', min: 0.60, max: 1.01 },
  ];
  const byProbability = probRanges.map(r => {
    const trades = enriched.filter(t => {
      const p = t.entry_probability ?? 0.5;
      // For SHORT trades, the "away from 0.5" probability is 1-p
      const effectiveP = t.direction === 'SHORT' ? (1 - p) : p;
      return effectiveP >= r.min && effectiveP < r.max;
    });
    return aggregateBucket(r.label, trades);
  });

  // By exit reason
  const reasonMap = new Map<string, EnrichedTrade[]>();
  for (const t of enriched) {
    const reason = t.reason || 'Unknown';
    if (!reasonMap.has(reason)) reasonMap.set(reason, []);
    reasonMap.get(reason)!.push(t);
  }
  const byReason = Array.from(reasonMap.entries())
    .map(([reason, trades]) => aggregateBucket(reason, trades))
    .sort((a, b) => b.count - a.count);

  // By direction
  const longTrades = enriched.filter(t => t.direction === 'LONG');
  const shortTrades = enriched.filter(t => t.direction === 'SHORT');
  const byDirection = [
    aggregateBucket('LONG', longTrades),
    aggregateBucket('SHORT', shortTrades),
  ];

  return { byQuality, byProbability, byReason, byDirection };
}

/* ------------------------------------------------------------------ */
/*  Chart generation helpers (Canvas API -> base64 PNG)                */
/* ------------------------------------------------------------------ */

/** Create a small off-screen canvas and return the 2D context. */
function makeCanvas(w: number, h: number): { canvas: HTMLCanvasElement; ctx: CanvasRenderingContext2D } {
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d')!;
  ctx.fillStyle = '#1e2329';
  ctx.fillRect(0, 0, w, h);
  return { canvas, ctx };
}

/** Win / Loss pie chart -> base64 PNG buffer */
function generateWinLossPieChart(wins: number, losses: number): string {
  const { canvas, ctx } = makeCanvas(360, 280);
  const cx = 140, cy = 140, r = 100;
  const total = wins + losses || 1;
  const winAngle = (wins / total) * Math.PI * 2;

  ctx.beginPath();
  ctx.moveTo(cx, cy);
  ctx.arc(cx, cy, r, -Math.PI / 2, -Math.PI / 2 + winAngle);
  ctx.closePath();
  ctx.fillStyle = '#0ecb81';
  ctx.fill();

  ctx.beginPath();
  ctx.moveTo(cx, cy);
  ctx.arc(cx, cy, r, -Math.PI / 2 + winAngle, -Math.PI / 2 + Math.PI * 2);
  ctx.closePath();
  ctx.fillStyle = '#f6465d';
  ctx.fill();

  ctx.font = 'bold 14px sans-serif';
  ctx.fillStyle = '#eaecef';
  ctx.textAlign = 'left';
  ctx.fillText('Win/Loss Distribution', 10, 20);

  const legendX = 260, legendY = 100;
  ctx.fillStyle = '#0ecb81';
  ctx.fillRect(legendX, legendY, 14, 14);
  ctx.fillStyle = '#eaecef';
  ctx.font = '12px sans-serif';
  ctx.fillText(`Wins: ${wins} (${((wins / total) * 100).toFixed(1)}%)`, legendX + 20, legendY + 12);

  ctx.fillStyle = '#f6465d';
  ctx.fillRect(legendX, legendY + 24, 14, 14);
  ctx.fillStyle = '#eaecef';
  ctx.fillText(`Losses: ${losses} (${((losses / total) * 100).toFixed(1)}%)`, legendX + 20, legendY + 36);

  return canvas.toDataURL('image/png').split(',')[1];
}

/** P&L distribution histogram -> base64 PNG buffer */
function generatePnLHistogram(trades: Trade[]): string {
  const { canvas, ctx } = makeCanvas(480, 280);
  if (trades.length === 0) return canvas.toDataURL('image/png').split(',')[1];

  const values = trades.map(t => t.pnl_pct);
  const min = Math.min(...values), max = Math.max(...values);
  const range = max - min || 1;
  const bins = 20;
  const binWidth = range / bins;
  const counts = new Array(bins).fill(0);
  for (const v of values) {
    const idx = Math.min(Math.floor((v - min) / binWidth), bins - 1);
    counts[idx]++;
  }
  const maxCount = Math.max(...counts, 1);

  const padL = 50, padR = 20, padT = 40, padB = 40;
  const w = canvas.width - padL - padR;
  const h = canvas.height - padT - padB;
  const barW = w / bins;

  ctx.font = 'bold 14px sans-serif';
  ctx.fillStyle = '#eaecef';
  ctx.textAlign = 'left';
  ctx.fillText('P&L % Distribution', 10, 20);

  for (let i = 0; i < bins; i++) {
    const barH = (counts[i] / maxCount) * h;
    const x = padL + i * barW;
    const y = padT + h - barH;
    const binMid = min + (i + 0.5) * binWidth;
    ctx.fillStyle = binMid >= 0 ? '#0ecb81' : '#f6465d';
    ctx.fillRect(x + 1, y, barW - 2, barH);
  }

  ctx.strokeStyle = '#5e6673';
  ctx.beginPath();
  ctx.moveTo(padL, padT);
  ctx.lineTo(padL, padT + h);
  ctx.lineTo(padL + w, padT + h);
  ctx.stroke();

  ctx.fillStyle = '#848e9c';
  ctx.font = '10px sans-serif';
  ctx.textAlign = 'center';
  for (let i = 0; i <= 4; i++) {
    const v = min + (range * i) / 4;
    const x = padL + (w * i) / 4;
    ctx.fillText(v.toFixed(2) + '%', x, padT + h + 16);
  }

  return canvas.toDataURL('image/png').split(',')[1];
}

/** Cumulative equity line chart -> base64 PNG buffer */
function generateEquityCurve(trades: Trade[], startingBalance: number): string {
  const { canvas, ctx } = makeCanvas(480, 280);
  if (trades.length === 0) return canvas.toDataURL('image/png').split(',')[1];

  const equity = [startingBalance];
  let bal = startingBalance;
  for (const t of trades) {
    bal += t.pnl_dollar;
    equity.push(bal);
  }

  const minE = Math.min(...equity), maxE = Math.max(...equity);
  const range = maxE - minE || 1;
  const padL = 60, padR = 20, padT = 40, padB = 30;
  const w = canvas.width - padL - padR;
  const h = canvas.height - padT - padB;

  ctx.font = 'bold 14px sans-serif';
  ctx.fillStyle = '#eaecef';
  ctx.textAlign = 'left';
  ctx.fillText('Equity Curve', 10, 20);

  ctx.strokeStyle = '#f0b90b';
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (let i = 0; i < equity.length; i++) {
    const x = padL + (i / (equity.length - 1 || 1)) * w;
    const y = padT + h - ((equity[i] - minE) / range) * h;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();

  ctx.lineTo(padL + w, padT + h);
  ctx.lineTo(padL, padT + h);
  ctx.closePath();
  ctx.fillStyle = 'rgba(240, 185, 11, 0.12)';
  ctx.fill();

  const startY = padT + h - ((startingBalance - minE) / range) * h;
  ctx.strokeStyle = '#5e6673';
  ctx.setLineDash([4, 4]);
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padL, startY);
  ctx.lineTo(padL + w, startY);
  ctx.stroke();
  ctx.setLineDash([]);

  ctx.strokeStyle = '#5e6673';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padL, padT);
  ctx.lineTo(padL, padT + h);
  ctx.lineTo(padL + w, padT + h);
  ctx.stroke();

  ctx.fillStyle = '#848e9c';
  ctx.font = '10px sans-serif';
  ctx.textAlign = 'right';
  for (let i = 0; i <= 4; i++) {
    const v = minE + (range * i) / 4;
    const y = padT + h - (i / 4) * h;
    ctx.fillText('$' + v.toFixed(2), padL - 5, y + 3);
  }

  return canvas.toDataURL('image/png').split(',')[1];
}

/** Horizontal bar chart for top features -> base64 PNG buffer */
function generateFeatureBarChart(features: { name: string; value: number }[]): string {
  const barH = 18;
  const padL = 200, padR = 60, padT = 40, padB = 20;
  const chartH = features.length * (barH + 4);
  const totalH = padT + chartH + padB;
  const totalW = 600;
  const { canvas, ctx } = makeCanvas(totalW, Math.max(totalH, 200));

  ctx.font = 'bold 14px sans-serif';
  ctx.fillStyle = '#eaecef';
  ctx.textAlign = 'left';
  ctx.fillText('Top 50 Features by Current Value', 10, 20);

  if (features.length === 0) return canvas.toDataURL('image/png').split(',')[1];

  const maxAbs = Math.max(...features.map(f => Math.abs(f.value)), 0.001);
  const barArea = totalW - padL - padR;

  for (let i = 0; i < features.length; i++) {
    const y = padT + i * (barH + 4);
    const val = features[i].value;
    const barW = (Math.abs(val) / maxAbs) * barArea;

    ctx.fillStyle = '#848e9c';
    ctx.font = '10px monospace';
    ctx.textAlign = 'right';
    ctx.fillText(features[i].name, padL - 8, y + barH - 4);

    ctx.fillStyle = val >= 0 ? '#0ecb81' : '#f6465d';
    ctx.fillRect(padL, y, barW, barH);

    ctx.fillStyle = '#eaecef';
    ctx.textAlign = 'left';
    ctx.font = '9px monospace';
    ctx.fillText(val.toFixed(4), padL + barW + 4, y + barH - 4);
  }

  return canvas.toDataURL('image/png').split(',')[1];
}

/* ------------------------------------------------------------------ */
/*  Style helpers                                                      */
/* ------------------------------------------------------------------ */
const HEADER_FILL: import('exceljs').FillPattern = {
  type: 'pattern',
  pattern: 'solid',
  fgColor: { argb: 'FF366092' },
};
const HEADER_FONT: Partial<import('exceljs').Font> = {
  bold: true,
  color: { argb: 'FFFFFFFF' },
  size: 11,
};
const SECTION_FILL: import('exceljs').FillPattern = {
  type: 'pattern',
  pattern: 'solid',
  fgColor: { argb: 'FF2B3E50' },
};
const SECTION_FONT: Partial<import('exceljs').Font> = {
  bold: true,
  color: { argb: 'FFF0B90B' },
  size: 11,
};

const GREEN = 'FF0ECB81';
const RED = 'FFF6465D';

function pnlColor(val: number): string {
  return val >= 0 ? GREEN : RED;
}

function styleHeader(ws: Worksheet) {
  const headerRow = ws.getRow(1);
  headerRow.eachCell(cell => {
    cell.fill = HEADER_FILL;
    cell.font = HEADER_FONT;
    cell.alignment = { horizontal: 'center', vertical: 'middle', wrapText: true };
  });
  ws.views = [{ state: 'frozen', ySplit: 1 }];
}

/* ------------------------------------------------------------------ */
/*  Sheet builders                                                     */
/* ------------------------------------------------------------------ */

// ── Sheet 1: Dashboard ──────────────────────────────────────────────
function buildDashboardSheet(
  wb: Workbook,
  enriched: EnrichedTrade[],
  startingBalance: number,
  risk: RiskMetrics,
  signalData?: ModelSignalData | null,
) {
  const ws = wb.addWorksheet('Dashboard');
  ws.columns = [
    { header: '', key: 'col1', width: 28 },
    { header: '', key: 'col2', width: 18 },
    { header: '', key: 'col3', width: 6 },
    { header: '', key: 'col4', width: 28 },
    { header: '', key: 'col5', width: 18 },
  ];

  const n = enriched.length;
  const wins = enriched.filter(t => t.pnl_dollar > 0).length;
  const totalPnL = n > 0 ? enriched[n - 1].cumulativePnl : 0;
  const finalBalance = startingBalance + totalPnL;
  const totalReturnPct = startingBalance > 0 ? (totalPnL / startingBalance) * 100 : 0;
  const winRate = n > 0 ? (wins / n) * 100 : 0;
  const avgBarsHeld = n > 0 ? enriched.reduce((s, t) => s + (t.bars_held || 0), 0) / n : 0;
  const winningTrades = enriched.filter(t => t.pnl_dollar > 0);
  const losingTrades = enriched.filter(t => t.pnl_dollar <= 0);
  const profitFactor = losingTrades.length > 0
    ? Math.abs(winningTrades.reduce((s, t) => s + t.pnl_dollar, 0) / (losingTrades.reduce((s, t) => s + t.pnl_dollar, 0) || 1))
    : winningTrades.length > 0 ? Infinity : 0;

  const longTrades = enriched.filter(t => t.direction === 'LONG');
  const shortTrades = enriched.filter(t => t.direction === 'SHORT');

  // Title
  const titleRow = ws.addRow({ col1: 'Trading Performance Dashboard', col2: '', col3: '', col4: '', col5: '' });
  titleRow.getCell(1).font = { bold: true, size: 16, color: { argb: 'FFFFFFFF' } };
  titleRow.getCell(1).fill = { type: 'pattern', pattern: 'solid', fgColor: { argb: 'FF1E2329' } };
  ws.mergeCells('A1:E1');

  const tsRow = ws.addRow({
    col1: `Exported: ${new Date().toISOString().replace('T', ' ').slice(0, 19)} UTC`,
    col2: '', col3: '', col4: `Model: ${signalData?.model_info?.version ?? 'v4.15'}`, col5: '',
  });
  tsRow.getCell(1).font = { size: 10, color: { argb: 'FF848E9C' } };
  tsRow.getCell(4).font = { size: 10, color: { argb: 'FF848E9C' } };

  ws.addRow({});

  // ── Account Overview (left) & Performance (right) ──
  const sectionRow = ws.addRow({ col1: 'ACCOUNT OVERVIEW', col2: '', col3: '', col4: 'PERFORMANCE', col5: '' });
  sectionRow.eachCell(c => { c.font = { bold: true, size: 11, color: { argb: 'FFF0B90B' } }; });

  const addKPIRow = (label1: string, val1: string | number, label2: string, val2: string | number) => {
    ws.addRow({ col1: label1, col2: val1, col3: '', col4: label2, col5: val2 });
  };

  addKPIRow('Starting Balance', '$' + startingBalance.toFixed(2), 'Win Rate', winRate.toFixed(1) + '%');
  addKPIRow('Final Balance', '$' + finalBalance.toFixed(2), 'Profit Factor', profitFactor === Infinity ? 'Infinity' : profitFactor.toFixed(2));
  addKPIRow('Total P&L $', (totalPnL >= 0 ? '+$' : '-$') + Math.abs(totalPnL).toFixed(2), 'Max Drawdown', risk.maxDrawdownPct.toFixed(2) + '%');
  addKPIRow('Total Return %', (totalReturnPct >= 0 ? '+' : '') + totalReturnPct.toFixed(2) + '%', 'Avg Bars Held', avgBarsHeld.toFixed(1));
  addKPIRow('Total Trades', n, 'Sharpe Ratio', risk.sharpeRatio.toFixed(2));

  ws.addRow({});

  // ── Status checks ──
  const statusRow = ws.addRow({ col1: 'HEALTH CHECKS', col2: '', col3: '', col4: 'DIRECTION BREAKDOWN', col5: '' });
  statusRow.eachCell(c => { c.font = { bold: true, size: 11, color: { argb: 'FFF0B90B' } }; });

  const checks = [
    { label: 'Win Rate > 50%', pass: winRate > 50 },
    { label: 'Profit Factor > 1.0', pass: profitFactor > 1.0 },
    { label: 'Max Drawdown < 5%', pass: risk.maxDrawdownPct < 5 },
    { label: 'Sharpe > 1.0', pass: risk.sharpeRatio > 1.0 },
    { label: 'Positive Expectancy', pass: risk.expectancy > 0 },
  ];

  const dirLabels = [
    { label: 'LONG Trades', val: longTrades.length.toString() },
    { label: 'LONG Win Rate', val: (longTrades.length > 0 ? (longTrades.filter(t => t.pnl_dollar > 0).length / longTrades.length * 100).toFixed(1) + '%' : '-') },
    { label: 'LONG Total P&L', val: '$' + longTrades.reduce((s, t) => s + t.pnl_dollar, 0).toFixed(2) },
    { label: 'SHORT Trades', val: shortTrades.length.toString() },
    { label: 'SHORT Win Rate', val: (shortTrades.length > 0 ? (shortTrades.filter(t => t.pnl_dollar > 0).length / shortTrades.length * 100).toFixed(1) + '%' : '-') },
  ];

  for (let i = 0; i < Math.max(checks.length, dirLabels.length); i++) {
    const check = checks[i];
    const dir = dirLabels[i];
    const r = ws.addRow({
      col1: check?.label ?? '',
      col2: check ? (check.pass ? 'PASS' : 'FAIL') : '',
      col3: '',
      col4: dir?.label ?? '',
      col5: dir?.val ?? '',
    });
    if (check) {
      r.getCell(2).font = { bold: true, color: { argb: check.pass ? GREEN : RED } };
    }
  }

  // Color key value cells
  for (let rowNum = 5; rowNum <= 9; rowNum++) {
    const row = ws.getRow(rowNum);
    // Color Total P&L and Return cells
    if (rowNum === 7 || rowNum === 8) {
      row.getCell(2).font = { color: { argb: pnlColor(totalPnL) } };
    }
  }
}

// ── Sheet 2: Trade Decisions (FIXED) ────────────────────────────────
function buildTradeDecisionsSheet(wb: Workbook, enriched: EnrichedTrade[]) {
  const ws = wb.addWorksheet('Trade Decisions');
  ws.columns = [
    { header: '#', key: 'num', width: 5 },
    { header: 'Trade Event', key: 'event', width: 14 },
    { header: 'Direction', key: 'direction', width: 10 },
    { header: 'Entry Time', key: 'entry_time', width: 22 },
    { header: 'Exit Time', key: 'exit_time', width: 22 },
    { header: 'Entry Ratio', key: 'entry_price', width: 14 },
    { header: 'Exit Ratio', key: 'exit_price', width: 14 },
    { header: 'Price Chg %', key: 'pnl_pct', width: 12 },
    { header: 'Size %', key: 'size_pct', width: 10 },
    { header: 'Balance Before', key: 'balance_before', width: 14 },
    { header: 'P&L $', key: 'pnl_dollar', width: 10 },
    { header: 'Portfolio Rtn %', key: 'portfolio_rtn_pct', width: 14 },
    { header: 'Cumulative P&L $', key: 'cumulative_pnl', width: 16 },
    { header: 'Cumulative Rtn %', key: 'cumulative_rtn_pct', width: 16 },
    { header: 'Bars Held', key: 'bars_held', width: 10 },
    { header: 'Stop Loss', key: 'stop_loss', width: 14 },
    { header: 'Take Profit', key: 'take_profit', width: 14 },
    { header: 'Probability', key: 'probability', width: 12 },
    { header: 'Strength', key: 'strength', width: 10 },
    { header: 'Signal Quality', key: 'signal_quality', width: 14 },
    { header: 'Decision Reason', key: 'reason', width: 22 },
  ];

  for (const t of enriched) {
    ws.addRow({
      num: t.index + 1,
      event: t.event,
      direction: t.direction,
      entry_time: new Date(t.entry_time * 1000).toISOString().replace('T', ' ').slice(0, 19),
      exit_time: new Date(t.exit_time * 1000).toISOString().replace('T', ' ').slice(0, 19),
      entry_price: Number(t.entry_price.toFixed(4)),
      exit_price: Number(t.exit_price.toFixed(4)),
      pnl_pct: Number(t.pnl_pct.toFixed(4)),
      size_pct: t.position_size_pct ? Number(t.position_size_pct.toFixed(1)) : 0,
      balance_before: Number(t.balanceBefore.toFixed(2)),
      pnl_dollar: Number(t.pnl_dollar.toFixed(2)),
      portfolio_rtn_pct: Number(t.portfolioRtnPct.toFixed(4)),
      cumulative_pnl: Number(t.cumulativePnl.toFixed(2)),
      cumulative_rtn_pct: Number(t.cumulativeRtnPct.toFixed(4)),
      bars_held: t.bars_held ?? 0,
      stop_loss: t.stop_loss ? Number(t.stop_loss.toFixed(4)) : 0,
      take_profit: t.take_profit ? Number(t.take_profit.toFixed(4)) : 0,
      probability: t.entry_probability ? Number(t.entry_probability.toFixed(4)) : 0,
      strength: t.entry_strength ? Number(t.entry_strength.toFixed(4)) : 0,
      signal_quality: t.quality,
      reason: t.reason,
    });

    const row = ws.getRow(t.index + 2);
    const color = pnlColor(t.pnl_dollar);
    row.getCell('pnl_pct').font = { color: { argb: color } };
    row.getCell('pnl_dollar').font = { color: { argb: color } };
    row.getCell('portfolio_rtn_pct').font = { color: { argb: color } };
    row.getCell('cumulative_pnl').font = { color: { argb: pnlColor(t.cumulativePnl) } };
    row.getCell('cumulative_rtn_pct').font = { color: { argb: pnlColor(t.cumulativeRtnPct) } };
    row.getCell('direction').font = { color: { argb: t.direction === 'LONG' ? GREEN : RED } };
  }

  styleHeader(ws);
}

// ── Sheet 3: Trade Summary (FIXED) ─────────────────────────────────
function buildTradeSummarySheet(wb: Workbook, enriched: EnrichedTrade[]) {
  const ws = wb.addWorksheet('Trade Summary');
  ws.columns = [
    { header: '#', key: 'num', width: 5 },
    { header: 'Direction', key: 'direction', width: 10 },
    { header: 'Entry Time', key: 'entry_time', width: 22 },
    { header: 'Exit Time', key: 'exit_time', width: 22 },
    { header: 'Entry Price', key: 'entry_price', width: 14 },
    { header: 'Exit Price', key: 'exit_price', width: 14 },
    { header: 'Price Chg %', key: 'pnl_pct', width: 12 },
    { header: 'Size %', key: 'size_pct', width: 8 },
    { header: 'Balance Before', key: 'balance_before', width: 14 },
    { header: 'P&L $', key: 'pnl_dollar', width: 10 },
    { header: 'Portfolio Rtn %', key: 'portfolio_rtn_pct', width: 14 },
    { header: 'Bars Held', key: 'bars_held', width: 10 },
    { header: 'Stop Loss', key: 'stop_loss', width: 14 },
    { header: 'Take Profit', key: 'take_profit', width: 14 },
    { header: 'Probability', key: 'probability', width: 12 },
    { header: 'Strength', key: 'strength', width: 10 },
    { header: 'Exit Reason', key: 'reason', width: 18 },
  ];

  for (const t of enriched) {
    ws.addRow({
      num: t.index + 1,
      direction: t.direction,
      entry_time: new Date(t.entry_time * 1000).toISOString().replace('T', ' ').slice(0, 19),
      exit_time: new Date(t.exit_time * 1000).toISOString().replace('T', ' ').slice(0, 19),
      entry_price: Number(t.entry_price.toFixed(4)),
      exit_price: Number(t.exit_price.toFixed(4)),
      pnl_pct: Number(t.pnl_pct.toFixed(4)),
      size_pct: t.position_size_pct ? Number(t.position_size_pct.toFixed(1)) : 0,
      balance_before: Number(t.balanceBefore.toFixed(2)),
      pnl_dollar: Number(t.pnl_dollar.toFixed(2)),
      portfolio_rtn_pct: Number(t.portfolioRtnPct.toFixed(4)),
      bars_held: t.bars_held ?? 0,
      stop_loss: t.stop_loss ? Number(t.stop_loss.toFixed(4)) : '-',
      take_profit: t.take_profit ? Number(t.take_profit.toFixed(4)) : '-',
      probability: t.entry_probability ? Number(t.entry_probability.toFixed(4)) : '-',
      strength: t.entry_strength ? Number(t.entry_strength.toFixed(4)) : '-',
      reason: t.reason,
    });

    const row = ws.getRow(t.index + 2);
    const color = pnlColor(t.pnl_dollar);
    row.getCell('pnl_pct').font = { color: { argb: color } };
    row.getCell('pnl_dollar').font = { color: { argb: color } };
    row.getCell('portfolio_rtn_pct').font = { color: { argb: color } };
    row.getCell('direction').font = { color: { argb: t.direction === 'LONG' ? GREEN : RED } };
  }

  styleHeader(ws);
}

// ── Sheet 4: Statistics (FIXED) ─────────────────────────────────────
function buildStatisticsSheet(
  wb: Workbook,
  trades: Trade[],
  enriched: EnrichedTrade[],
  startingBalance: number,
  risk: RiskMetrics,
) {
  const ws = wb.addWorksheet('Statistics');

  const n = trades.length;
  const wins = trades.filter(t => t.pnl_dollar > 0).length;
  const losses = n - wins;
  const totalPnL = trades.reduce((s, t) => s + t.pnl_dollar, 0);
  const finalBalance = startingBalance + totalPnL;
  // FIXED: actual portfolio return instead of summing raw pnl_pct
  const totalReturnPct = startingBalance > 0 ? (totalPnL / startingBalance) * 100 : 0;
  const avgBarsHeld = n > 0 ? trades.reduce((s, t) => s + (t.bars_held || 0), 0) / n : 0;

  const winningTrades = enriched.filter(t => t.pnl_dollar > 0);
  const losingTrades = enriched.filter(t => t.pnl_dollar <= 0);

  // Portfolio-level averages (FIXED)
  const avgPortfolioRtn = n > 0
    ? enriched.reduce((s, t) => s + t.portfolioRtnPct, 0) / n : 0;
  const avgWinPortRtn = winningTrades.length > 0
    ? winningTrades.reduce((s, t) => s + t.portfolioRtnPct, 0) / winningTrades.length : 0;
  const avgLossPortRtn = losingTrades.length > 0
    ? losingTrades.reduce((s, t) => s + t.portfolioRtnPct, 0) / losingTrades.length : 0;

  // Raw price-move averages (kept as secondary reference)
  const avgPriceChg = n > 0 ? trades.reduce((s, t) => s + t.pnl_pct, 0) / n : 0;
  const avgWinPriceChg = winningTrades.length > 0
    ? winningTrades.reduce((s, t) => s + t.pnl_pct, 0) / winningTrades.length : 0;
  const avgLossPriceChg = losingTrades.length > 0
    ? losingTrades.reduce((s, t) => s + t.pnl_pct, 0) / losingTrades.length : 0;

  const profitFactor = losingTrades.length > 0
    ? Math.abs(winningTrades.reduce((s, t) => s + t.pnl_dollar, 0) / (losingTrades.reduce((s, t) => s + t.pnl_dollar, 0) || 1))
    : winningTrades.length > 0 ? Infinity : 0;

  ws.columns = [
    { header: 'Metric', key: 'metric', width: 30 },
    { header: 'Value', key: 'value', width: 18 },
  ];

  const stats: { metric: string; value: string | number }[] = [
    // Account
    { metric: 'Starting Balance', value: '$' + startingBalance.toFixed(2) },
    { metric: 'Final Balance', value: '$' + finalBalance.toFixed(2) },
    { metric: 'Total P&L $', value: Number(totalPnL.toFixed(2)) },
    { metric: 'Total Return %', value: Number(totalReturnPct.toFixed(2)) },
    { metric: '', value: '' },
    // Trade counts
    { metric: 'Total Trades', value: n },
    { metric: 'Wins', value: wins },
    { metric: 'Losses', value: losses },
    { metric: 'Win Rate %', value: n > 0 ? Number(((wins / n) * 100).toFixed(1)) : 0 },
    { metric: 'Profit Factor', value: profitFactor === Infinity ? 'Infinity' : Number(profitFactor.toFixed(2)) },
    { metric: '', value: '' },
    // Portfolio return averages (PRIMARY)
    { metric: 'Avg Portfolio Rtn % / Trade', value: Number(avgPortfolioRtn.toFixed(4)) },
    { metric: 'Avg Win (Portfolio %)', value: Number(avgWinPortRtn.toFixed(4)) },
    { metric: 'Avg Loss (Portfolio %)', value: Number(avgLossPortRtn.toFixed(4)) },
    { metric: '', value: '' },
    // Raw price-move averages (SECONDARY)
    { metric: 'Avg Price Chg %', value: Number(avgPriceChg.toFixed(4)) },
    { metric: 'Avg Win (Price Chg %)', value: Number(avgWinPriceChg.toFixed(4)) },
    { metric: 'Avg Loss (Price Chg %)', value: Number(avgLossPriceChg.toFixed(4)) },
    { metric: '', value: '' },
    // Drawdown & duration
    { metric: 'Max Drawdown %', value: Number(risk.maxDrawdownPct.toFixed(2)) },
    { metric: 'Avg Bars Held', value: Number(avgBarsHeld.toFixed(1)) },
    { metric: '', value: '' },
    // Extremes
    { metric: 'Best Trade $', value: n > 0 ? Number(Math.max(...trades.map(t => t.pnl_dollar)).toFixed(2)) : 0 },
    { metric: 'Worst Trade $', value: n > 0 ? Number(Math.min(...trades.map(t => t.pnl_dollar)).toFixed(2)) : 0 },
    { metric: 'Best Trade (Price Chg %)', value: n > 0 ? Number(Math.max(...trades.map(t => t.pnl_pct)).toFixed(4)) : 0 },
    { metric: 'Worst Trade (Price Chg %)', value: n > 0 ? Number(Math.min(...trades.map(t => t.pnl_pct)).toFixed(4)) : 0 },
  ];

  for (const s of stats) ws.addRow(s);
  styleHeader(ws);

  // Color P&L cells
  for (let rowNum = 2; rowNum <= ws.rowCount; rowNum++) {
    const row = ws.getRow(rowNum);
    const metricVal = String(row.getCell(1).value ?? '');
    const cellVal = row.getCell(2).value;
    if (typeof cellVal === 'number' && (metricVal.includes('P&L') || metricVal.includes('Return') || metricVal.includes('Rtn'))) {
      row.getCell(2).font = { color: { argb: pnlColor(cellVal) } };
    }
  }

  // Embed chart images
  try {
    const pieB64 = generateWinLossPieChart(wins, losses);
    const pieImgId = wb.addImage({ base64: pieB64, extension: 'png' });
    ws.addImage(pieImgId, {
      tl: { col: 3, row: 1 },
      ext: { width: 360, height: 280 },
    });

    const histB64 = generatePnLHistogram(trades);
    const histImgId = wb.addImage({ base64: histB64, extension: 'png' });
    ws.addImage(histImgId, {
      tl: { col: 3, row: 16 },
      ext: { width: 480, height: 280 },
    });

    const eqB64 = generateEquityCurve(trades, startingBalance);
    const eqImgId = wb.addImage({ base64: eqB64, extension: 'png' });
    ws.addImage(eqImgId, {
      tl: { col: 9, row: 1 },
      ext: { width: 480, height: 280 },
    });
  } catch (chartErr) {
    console.warn('Chart generation failed:', chartErr);
  }
}

// ── Sheet 5: Risk Metrics (NEW) ─────────────────────────────────────
function buildRiskMetricsSheet(
  wb: Workbook,
  _enriched: EnrichedTrade[],
  _startingBalance: number,
  risk: RiskMetrics,
) {
  const ws = wb.addWorksheet('Risk Metrics');
  ws.columns = [
    { header: 'Metric', key: 'metric', width: 30 },
    { header: 'Value', key: 'value', width: 18 },
  ];

  const metrics: { metric: string; value: string | number }[] = [
    { metric: 'Sharpe Ratio (annualized)', value: Number(risk.sharpeRatio.toFixed(3)) },
    { metric: 'Sortino Ratio (annualized)', value: Number(risk.sortinoRatio.toFixed(3)) },
    { metric: 'Value at Risk (95%)', value: Number(risk.var95.toFixed(4)) + '%' },
    { metric: '', value: '' },
    { metric: 'Max Drawdown %', value: Number(risk.maxDrawdownPct.toFixed(2)) },
    { metric: 'Max Drawdown $', value: Number(risk.maxDrawdownDollar.toFixed(2)) },
    { metric: 'DD Peak Date', value: risk.maxDDPeakDate },
    { metric: 'DD Trough Date', value: risk.maxDDTroughDate },
    { metric: 'Recovery Factor', value: Number(risk.recoveryFactor.toFixed(2)) },
    { metric: '', value: '' },
    { metric: 'Kelly Criterion', value: Number(risk.kellyCriterion.toFixed(4)) },
    { metric: 'Expectancy ($/trade)', value: Number(risk.expectancy.toFixed(4)) },
    { metric: '', value: '' },
    { metric: 'Max Consecutive Wins', value: risk.maxConsecutiveWins },
    { metric: 'Max Consecutive Losses', value: risk.maxConsecutiveLosses },
    { metric: 'Current Streak', value: risk.currentStreakType },
  ];

  for (const m of metrics) ws.addRow(m);
  styleHeader(ws);

  // Color key metrics
  for (let rowNum = 2; rowNum <= ws.rowCount; rowNum++) {
    const row = ws.getRow(rowNum);
    const metricVal = String(row.getCell(1).value ?? '');
    const cellVal = row.getCell(2).value;
    if (metricVal === 'Kelly Criterion' && typeof cellVal === 'number') {
      row.getCell(2).font = { color: { argb: pnlColor(cellVal) } };
    }
    if (metricVal.includes('Expectancy') && typeof cellVal === 'number') {
      row.getCell(2).font = { color: { argb: pnlColor(cellVal) } };
    }
    if (metricVal.includes('Sharpe') && typeof cellVal === 'number') {
      row.getCell(2).font = { color: { argb: pnlColor(cellVal) } };
    }
    if (metricVal.includes('Sortino') && typeof cellVal === 'number') {
      row.getCell(2).font = { color: { argb: pnlColor(cellVal) } };
    }
  }
}

// ── Sheet 6: Time Analysis (NEW) ────────────────────────────────────
function buildTimeAnalysisSheet(
  wb: Workbook,
  timeData: { hourly: TimeBucket[]; daily: TimeBucket[]; duration: TimeBucket[] },
) {
  const ws = wb.addWorksheet('Time Analysis');
  ws.columns = [
    { header: 'Period', key: 'period', width: 18 },
    { header: 'Trades', key: 'count', width: 10 },
    { header: 'Wins', key: 'wins', width: 8 },
    { header: 'Win Rate %', key: 'win_rate', width: 12 },
    { header: 'Total P&L $', key: 'total_pnl', width: 14 },
    { header: 'Avg P&L $', key: 'avg_pnl', width: 14 },
  ];

  // Section: Hourly
  const hourSection = ws.addRow({ period: 'HOURLY BREAKDOWN (UTC)', count: '', wins: '', win_rate: '', total_pnl: '', avg_pnl: '' });
  hourSection.eachCell(c => { c.fill = SECTION_FILL; c.font = SECTION_FONT; });

  for (const h of timeData.hourly) {
    if (h.count === 0) continue;
    const row = ws.addRow({
      period: h.label,
      count: h.count,
      wins: h.wins,
      win_rate: Number(h.winRate.toFixed(1)),
      total_pnl: Number(h.totalPnl.toFixed(2)),
      avg_pnl: Number(h.avgPnl.toFixed(2)),
    });
    row.getCell('total_pnl').font = { color: { argb: pnlColor(h.totalPnl) } };
    row.getCell('avg_pnl').font = { color: { argb: pnlColor(h.avgPnl) } };
  }

  // Section: Day of week
  ws.addRow({});
  const daySection = ws.addRow({ period: 'DAY OF WEEK', count: '', wins: '', win_rate: '', total_pnl: '', avg_pnl: '' });
  daySection.eachCell(c => { c.fill = SECTION_FILL; c.font = SECTION_FONT; });

  for (const d of timeData.daily) {
    if (d.count === 0) continue;
    const row = ws.addRow({
      period: d.label,
      count: d.count,
      wins: d.wins,
      win_rate: Number(d.winRate.toFixed(1)),
      total_pnl: Number(d.totalPnl.toFixed(2)),
      avg_pnl: Number(d.avgPnl.toFixed(2)),
    });
    row.getCell('total_pnl').font = { color: { argb: pnlColor(d.totalPnl) } };
    row.getCell('avg_pnl').font = { color: { argb: pnlColor(d.avgPnl) } };
  }

  // Section: Duration
  ws.addRow({});
  const durSection = ws.addRow({ period: 'TRADE DURATION', count: '', wins: '', win_rate: '', total_pnl: '', avg_pnl: '' });
  durSection.eachCell(c => { c.fill = SECTION_FILL; c.font = SECTION_FONT; });

  for (const b of timeData.duration) {
    if (b.count === 0) continue;
    const row = ws.addRow({
      period: b.label,
      count: b.count,
      wins: b.wins,
      win_rate: Number(b.winRate.toFixed(1)),
      total_pnl: Number(b.totalPnl.toFixed(2)),
      avg_pnl: Number(b.avgPnl.toFixed(2)),
    });
    row.getCell('total_pnl').font = { color: { argb: pnlColor(b.totalPnl) } };
    row.getCell('avg_pnl').font = { color: { argb: pnlColor(b.avgPnl) } };
  }

  styleHeader(ws);
}

// ── Sheet 7: Trade Quality (NEW) ────────────────────────────────────
function buildTradeQualitySheet(
  wb: Workbook,
  quality: {
    byQuality: QualityBucket[];
    byProbability: QualityBucket[];
    byReason: QualityBucket[];
    byDirection: QualityBucket[];
  },
) {
  const ws = wb.addWorksheet('Trade Quality');
  ws.columns = [
    { header: 'Category', key: 'category', width: 22 },
    { header: 'Trades', key: 'count', width: 10 },
    { header: 'Wins', key: 'wins', width: 8 },
    { header: 'Win Rate %', key: 'win_rate', width: 12 },
    { header: 'Total P&L $', key: 'total_pnl', width: 14 },
    { header: 'Avg P&L $', key: 'avg_pnl', width: 12 },
    { header: 'Avg Port. Rtn %', key: 'avg_port_rtn', width: 14 },
  ];

  const addBucketSection = (title: string, buckets: QualityBucket[]) => {
    const section = ws.addRow({ category: title, count: '', wins: '', win_rate: '', total_pnl: '', avg_pnl: '', avg_port_rtn: '' });
    section.eachCell(c => { c.fill = SECTION_FILL; c.font = SECTION_FONT; });

    for (const b of buckets) {
      if (b.count === 0) continue;
      const row = ws.addRow({
        category: b.label,
        count: b.count,
        wins: b.wins,
        win_rate: Number(b.winRate.toFixed(1)),
        total_pnl: Number(b.totalPnl.toFixed(2)),
        avg_pnl: Number(b.avgPnl.toFixed(2)),
        avg_port_rtn: Number(b.avgPortfolioRtn.toFixed(4)),
      });
      row.getCell('total_pnl').font = { color: { argb: pnlColor(b.totalPnl) } };
      row.getCell('avg_pnl').font = { color: { argb: pnlColor(b.avgPnl) } };
      row.getCell('avg_port_rtn').font = { color: { argb: pnlColor(b.avgPortfolioRtn) } };
    }
    ws.addRow({});
  };

  addBucketSection('BY SIGNAL QUALITY', quality.byQuality);
  addBucketSection('BY PROBABILITY RANGE', quality.byProbability);
  addBucketSection('BY EXIT REASON', quality.byReason);
  addBucketSection('BY DIRECTION', quality.byDirection);

  styleHeader(ws);
}

// ── Sheet 8: Parameters (ENHANCED) ──────────────────────────────────
function buildParametersSheet(
  wb: Workbook,
  trades: Trade[],
  startingBalance: number,
  signalData?: ModelSignalData | null,
) {
  const ws = wb.addWorksheet('Parameters');
  ws.columns = [
    { header: 'Parameter', key: 'param', width: 30 },
    { header: 'Value', key: 'value', width: 32 },
  ];

  const rows: { param: string; value: string | number }[] = [];

  // Model info
  rows.push({ param: 'Model Version', value: signalData?.model_info?.version ?? 'v4.15' });
  rows.push({ param: 'Total Features', value: signalData?.model_info?.n_features ?? 50 });
  rows.push({ param: 'Export Time', value: new Date().toISOString() });
  rows.push({ param: '', value: '' });

  // Signal thresholds
  rows.push({ param: 'Entry Threshold', value: signalData?.signal?.entry_threshold ?? '-' });
  rows.push({ param: 'Exit Threshold', value: signalData?.signal?.exit_threshold ?? '-' });
  rows.push({ param: 'Circuit Breaker Active', value: signalData?.signal?.circuit_breaker_active ? 'YES' : 'NO' });
  rows.push({ param: '', value: '' });

  // Account
  rows.push({ param: 'Starting Balance', value: '$' + startingBalance.toFixed(2) });
  rows.push({ param: 'Current Balance', value: '$' + (signalData?.portfolio?.balance ?? startingBalance).toFixed(2) });
  rows.push({ param: '', value: '' });

  // Trade counts
  rows.push({ param: 'Total Trades', value: trades.length });
  rows.push({ param: 'Long Trades', value: trades.filter(t => t.direction === 'LONG').length });
  rows.push({ param: 'Short Trades', value: trades.filter(t => t.direction === 'SHORT').length });
  rows.push({ param: '', value: '' });

  // Risk management info (from current position meta if available)
  if (signalData?.position_meta) {
    rows.push({ param: 'Current Position Size %', value: Number(signalData.position_meta.position_size_pct.toFixed(1)) });
    rows.push({ param: 'Current Stop Loss', value: signalData.position_meta.stop_loss > 0 ? Number(signalData.position_meta.stop_loss.toFixed(4)) : '-' });
    rows.push({ param: 'Current Take Profit', value: signalData.position_meta.take_profit > 0 ? Number(signalData.position_meta.take_profit.toFixed(4)) : '-' });
    rows.push({ param: 'Effective Min Hold', value: signalData.position_meta.effective_min_hold });
    rows.push({ param: 'Current Bars Held', value: signalData.position_meta.bars_held });
    rows.push({ param: '', value: '' });
  }

  // Position sizing stats from trade history
  if (trades.length > 0) {
    const sizes = trades.filter(t => t.position_size_pct && t.position_size_pct > 0).map(t => t.position_size_pct!);
    if (sizes.length > 0) {
      rows.push({ param: 'Avg Position Size %', value: Number((sizes.reduce((a, b) => a + b, 0) / sizes.length).toFixed(1)) });
      rows.push({ param: 'Min Position Size %', value: Number(Math.min(...sizes).toFixed(1)) });
      rows.push({ param: 'Max Position Size %', value: Number(Math.max(...sizes).toFixed(1)) });
    }
  }

  for (const p of rows) ws.addRow(p);
  styleHeader(ws);
}

// ── Sheet 9: Top Features (UNCHANGED) ───────────────────────────────
async function buildTopFeaturesSheet(
  wb: Workbook,
  signalData?: ModelSignalData | null,
) {
  const ws = wb.addWorksheet('Top Features');
  ws.columns = [
    { header: 'Rank', key: 'rank', width: 6 },
    { header: 'Feature Name', key: 'name', width: 32 },
    { header: 'Current Value', key: 'value', width: 16 },
    { header: 'Abs Value', key: 'abs_value', width: 14 },
  ];

  const featureEntries = signalData?.features
    ? Object.entries(signalData.features)
        .map(([name, value]) => ({ name, value }))
        .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
    : [];

  let importanceMap: Record<string, number> = {};
  try {
    const resp = await fetch(`${MODEL_SERVER_REST_URL}/features/importance`, { headers: getAuthHeaders() });
    if (resp.ok) {
      const data = await resp.json();
      if (data.features) {
        ws.columns = [
          { header: 'Rank', key: 'rank', width: 6 },
          { header: 'Feature Name', key: 'name', width: 32 },
          { header: 'Current Value', key: 'value', width: 16 },
          { header: 'Importance', key: 'importance', width: 14 },
          { header: 'Abs Value', key: 'abs_value', width: 14 },
        ];
        for (const f of data.features) {
          importanceMap[f.name] = f.importance;
        }
      }
    }
  } catch {
    // Importance endpoint not available
  }

  for (let i = 0; i < featureEntries.length; i++) {
    const f = featureEntries[i];
    const rowData: Record<string, unknown> = {
      rank: i + 1,
      name: f.name,
      value: Number(f.value.toFixed(6)),
      abs_value: Number(Math.abs(f.value).toFixed(6)),
    };
    if (importanceMap[f.name] !== undefined) {
      rowData['importance'] = Number(importanceMap[f.name].toFixed(2));
    }
    ws.addRow(rowData);

    const row = ws.getRow(i + 2);
    const valCell = row.getCell('value');
    valCell.font = { color: { argb: f.value >= 0 ? GREEN : RED } };
  }

  styleHeader(ws);

  try {
    const featChartEntries = featureEntries.slice(0, 50);
    const featB64 = generateFeatureBarChart(featChartEntries);
    const featImgId = wb.addImage({ base64: featB64, extension: 'png' });
    ws.addImage(featImgId, {
      tl: { col: 6, row: 1 },
      ext: { width: 600, height: Math.max(featChartEntries.length * 22 + 60, 200) },
    });
  } catch (chartErr) {
    console.warn('Feature chart generation failed:', chartErr);
  }
}

/* ------------------------------------------------------------------ */
/*  Dev-mode validation helper                                         */
/* ------------------------------------------------------------------ */
function validateExport(enriched: EnrichedTrade[], startingBalance: number) {
  if (typeof import.meta !== 'undefined' && !(import.meta as unknown as Record<string, unknown>).env) return;
  try {
    if (!import.meta.env?.DEV) return;
  } catch {
    return;
  }

  const n = enriched.length;
  if (n === 0) return;

  let warnings = 0;

  // Check 1: Per-trade P&L consistency
  for (const t of enriched) {
    const sizeFrac = (t.position_size_pct ?? 0) / 100;
    const priceChgFrac = t.pnl_pct / 100;
    const expectedPnl = t.balanceBefore * sizeFrac * priceChgFrac;
    const diff = Math.abs(t.pnl_dollar - expectedPnl);
    if (diff > 0.02 && sizeFrac > 0) {
      console.warn(
        `[ExcelExport Validation] Trade #${t.index + 1}: ` +
        `P&L$ ${t.pnl_dollar.toFixed(4)} != expected ${expectedPnl.toFixed(4)} ` +
        `(balance=${t.balanceBefore.toFixed(2)}, size=${sizeFrac.toFixed(4)}, priceChg=${priceChgFrac.toFixed(6)})`
      );
      warnings++;
    }
  }

  // Check 2: Cumulative P&L of last row == sum of all P&L$
  const lastCumPnl = enriched[n - 1].cumulativePnl;
  const sumPnl = enriched.reduce((s, t) => s + t.pnl_dollar, 0);
  if (Math.abs(lastCumPnl - sumPnl) > 0.02) {
    console.warn(
      `[ExcelExport Validation] Cumulative P&L mismatch: last=${lastCumPnl.toFixed(4)}, sum=${sumPnl.toFixed(4)}`
    );
    warnings++;
  }

  // Check 3: Cumulative Rtn % matches totalPnL / startingBalance
  const lastCumRtn = enriched[n - 1].cumulativeRtnPct;
  const expectedCumRtn = startingBalance > 0 ? (sumPnl / startingBalance) * 100 : 0;
  if (Math.abs(lastCumRtn - expectedCumRtn) > 0.02) {
    console.warn(
      `[ExcelExport Validation] Cumulative Rtn% mismatch: last=${lastCumRtn.toFixed(4)}, expected=${expectedCumRtn.toFixed(4)}`
    );
    warnings++;
  }

  if (warnings === 0) {
    console.log('[ExcelExport Validation] All checks passed.');
  } else {
    console.warn(`[ExcelExport Validation] ${warnings} warning(s) found.`);
  }
}

/* ------------------------------------------------------------------ */
/*  Main export function                                               */
/* ------------------------------------------------------------------ */
export const exportTradesToExcel = async (options: ExportOptions): Promise<void> => {
  const { trades, signalData, filename = 'trade_history.xlsx' } = options;
  const exceljs = await ensureExcelJS();
  const wb = new exceljs.Workbook();

  const startingBalance = signalData?.portfolio?.starting_balance ?? 1000;

  // ── Pre-compute enriched data ──
  const enriched = enrichTrades(trades, startingBalance);
  const risk = computeRiskMetrics(enriched, startingBalance);
  const timeData = computeTimeAnalysis(enriched);
  const quality = computeTradeQuality(enriched);

  // ── Dev-mode validation ──
  validateExport(enriched, startingBalance);

  // ── Build all sheets in order ──
  // Sheet 1: Dashboard (first tab)
  buildDashboardSheet(wb, enriched, startingBalance, risk, signalData);

  // Sheet 2: Trade Decisions (FIXED)
  buildTradeDecisionsSheet(wb, enriched);

  // Sheet 3: Trade Summary (FIXED)
  buildTradeSummarySheet(wb, enriched);

  // Sheet 4: Statistics (FIXED)
  buildStatisticsSheet(wb, trades, enriched, startingBalance, risk);

  // Sheet 5: Risk Metrics (NEW)
  buildRiskMetricsSheet(wb, enriched, startingBalance, risk);

  // Sheet 6: Time Analysis (NEW)
  buildTimeAnalysisSheet(wb, timeData);

  // Sheet 7: Trade Quality (NEW)
  buildTradeQualitySheet(wb, quality);

  // Sheet 8: Parameters (ENHANCED)
  buildParametersSheet(wb, trades, startingBalance, signalData);

  // Sheet 9: Top Features (UNCHANGED — async for importance fetch)
  await buildTopFeaturesSheet(wb, signalData);

  // ── Write and download ──
  const buffer = await wb.xlsx.writeBuffer();
  const blob = new Blob([buffer], { type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
};

/* ------------------------------------------------------------------ */
/*  Fallback CSV export (no library required)                          */
/* ------------------------------------------------------------------ */
export const exportTradesToCSV = (trades: Trade[]): void => {
  const headers = [
    'Direction', 'Entry Time', 'Exit Time', 'Entry Price', 'Exit Price',
    'Price Chg %', 'P&L $', 'Bars Held', 'Size %', 'Stop Loss', 'Take Profit',
    'Probability', 'Strength', 'Exit Reason'
  ];

  const rows = trades.map(t => [
    t.direction,
    new Date(t.entry_time * 1000).toISOString(),
    new Date(t.exit_time * 1000).toISOString(),
    t.entry_price,
    t.exit_price,
    t.pnl_pct.toFixed(4),
    t.pnl_dollar.toFixed(2),
    t.bars_held ?? '',
    t.position_size_pct?.toFixed(1) ?? '',
    t.stop_loss?.toFixed(4) ?? '',
    t.take_profit?.toFixed(4) ?? '',
    t.entry_probability?.toFixed(4) ?? '',
    t.entry_strength?.toFixed(4) ?? '',
    t.reason,
  ].join(','));

  const csv = [headers.join(','), ...rows].join('\n');
  const blob = new Blob([csv], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'trade_history.csv';
  a.click();
  URL.revokeObjectURL(url);
};
