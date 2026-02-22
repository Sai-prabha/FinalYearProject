import React, { useMemo } from 'react';
import { THEME_COLORS } from '../constants';
import type { ModelSignalData } from '../types';

interface ModelThinkingProps {
  signalData: ModelSignalData | null;
  isConnected: boolean;
  btcPrice?: number;
  ethPrice?: number;
}

export const ModelThinking: React.FC<ModelThinkingProps> = ({ signalData, isConnected, btcPrice = 0, ethPrice = 0 }) => {
  if (!isConnected) {
    return (
      <div className="h-full rounded-lg p-2 flex items-center justify-center" style={{ backgroundColor: THEME_COLORS.CARD_BG }}>
        <div className="text-center">
          <div className="w-6 h-6 border-2 border-t-transparent rounded-full animate-spin mx-auto mb-1"
               style={{ borderColor: `${THEME_COLORS.YELLOW} transparent transparent transparent` }} />
          <p className="text-xs" style={{ color: THEME_COLORS.TEXT_SECONDARY }}>Connecting...</p>
        </div>
      </div>
    );
  }

  if (!signalData) {
    return (
      <div className="h-full rounded-lg p-2 flex items-center justify-center" style={{ backgroundColor: THEME_COLORS.CARD_BG }}>
        <p className="text-xs" style={{ color: THEME_COLORS.TEXT_SECONDARY }}>Waiting for data...</p>
      </div>
    );
  }

  const { signal, features, ratio, portfolio, data_quality, model_info } = signalData;

  // ── Real-time unrealized P/L from live BTC/ETH prices ──
  const liveUnrealized = useMemo(() => {
    if (!portfolio?.position || !portfolio?.entry_price || portfolio.entry_price <= 0) {
      return { pnl: 0, pnlPct: 0 };
    }
    if (btcPrice <= 0 || ethPrice <= 0) {
      return { pnl: portfolio.unrealized_pnl || 0, pnlPct: portfolio.unrealized_pnl_pct || 0 };
    }
    const liveRatio = btcPrice / ethPrice;
    let pnlPct = 0;
    if (portfolio.position === 'LONG') {
      pnlPct = ((liveRatio - portfolio.entry_price) / portfolio.entry_price) * 100;
    } else {
      pnlPct = ((portfolio.entry_price - liveRatio) / portfolio.entry_price) * 100;
    }
    const sizePct = signalData?.position_meta?.position_size_pct || 10;
    const pnl = (portfolio.balance || 1000) * (sizePct / 100) * (pnlPct / 100);
    return { pnl, pnlPct };
  }, [portfolio, btcPrice, ethPrice, signalData]);

  const getSignalColor = () => {
    if (signal.direction === 'LONG') return THEME_COLORS.POSITIVE;
    if (signal.direction === 'SHORT') return THEME_COLORS.NEGATIVE;
    return THEME_COLORS.TEXT_SECONDARY;
  };

  const getPnlColor = (value: number) => value >= 0 ? THEME_COLORS.POSITIVE : THEME_COLORS.NEGATIVE;

  // Probability display helpers
  const probPct = (signal.probability * 100).toFixed(1);
  const entryPct = (signal.entry_threshold * 100).toFixed(1);
  const exitPct = (signal.exit_threshold * 100).toFixed(1);

  // Probability bar position (0-1 mapped to bar width)
  const probBarWidth = Math.max(0, Math.min(100, signal.probability * 100));
  const entryBarPos = signal.entry_threshold * 100;
  const exitBarPos = signal.exit_threshold * 100;

  return (
    <div className="h-full rounded-lg overflow-hidden flex flex-col" style={{ backgroundColor: THEME_COLORS.CARD_BG }}>
      {/* Header */}
      <div className="px-2 py-1.5 border-b flex items-center justify-between" style={{ borderColor: THEME_COLORS.BORDER }}>
        <div className="flex items-center gap-1">
          <span className="text-xs font-semibold" style={{ color: THEME_COLORS.TEXT_PRIMARY }}>Model {model_info?.version || 'v4.15'}</span>
          {model_info && (
            <span className="text-[9px] px-1 rounded" style={{ backgroundColor: THEME_COLORS.CARD_BG_LIGHT, color: THEME_COLORS.TEXT_SECONDARY }}>
              {model_info.calc_time_ms}ms
            </span>
          )}
        </div>
        <div className="flex items-center gap-1">
          {signal.circuit_breaker_active && (
            <span className="text-[9px] px-1 py-0.5 rounded font-bold animate-pulse"
                  style={{ backgroundColor: 'rgba(246, 70, 93, 0.3)', color: THEME_COLORS.NEGATIVE }}>
              CB
            </span>
          )}
          <div className={`w-1.5 h-1.5 rounded-full ${isConnected ? 'bg-green-500' : 'bg-gray-500'}`} />
          <span className="text-xs" style={{ color: THEME_COLORS.TEXT_SECONDARY }}>Live</span>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 p-2 flex flex-col gap-2 overflow-y-auto">

        {/* Portfolio Balance */}
        <div className="p-2 rounded" style={{ backgroundColor: THEME_COLORS.CARD_BG_LIGHT }}>
          <div className="flex justify-between items-center">
            <span className="text-xs" style={{ color: THEME_COLORS.TEXT_SECONDARY }}>Balance</span>
            <span className="text-base font-bold" style={{ color: THEME_COLORS.TEXT_PRIMARY }}>
              ${portfolio?.balance?.toFixed(2) || '1000.00'}
            </span>
          </div>
          <div className="flex justify-between items-center mt-1">
            <span className="text-xs" style={{ color: THEME_COLORS.TEXT_SECONDARY }}>P&L</span>
            <span className="text-xs font-semibold" style={{ color: getPnlColor(portfolio?.total_pnl || 0) }}>
              {(portfolio?.total_pnl || 0) >= 0 ? '+' : ''}{(portfolio?.total_pnl || 0).toFixed(2)} ({(portfolio?.total_pnl_pct || 0).toFixed(2)}%)
            </span>
          </div>
          <div className="flex justify-between text-xs mt-1" style={{ color: THEME_COLORS.TEXT_SECONDARY }}>
            <span>Trades: {portfolio?.total_trades || 0}</span>
            <span>W/L: {portfolio?.wins || 0}/{portfolio?.losses || 0}</span>
            <span>WR: {(portfolio?.win_rate || 0).toFixed(0)}%</span>
          </div>
        </div>

        {/* Probability + Signal */}
        <div className="p-1.5 rounded" style={{ backgroundColor: THEME_COLORS.CARD_BG_LIGHT }}>
          <div className="flex items-center justify-between mb-1">
            <span className="text-base font-bold" style={{ color: getSignalColor() }}>{signal.direction}</span>
            <span className="text-xs font-mono font-bold" style={{ color: THEME_COLORS.TEXT_PRIMARY }}>
              P(up) = {probPct}%
            </span>
          </div>

          {/* Probability bar with thresholds */}
          <div className="relative h-3 rounded-full overflow-hidden" style={{ backgroundColor: THEME_COLORS.BACKGROUND }}>
            {/* Fill bar */}
            <div className="absolute top-0 left-0 h-full rounded-full transition-all duration-300"
                 style={{
                   width: `${probBarWidth}%`,
                   backgroundColor: signal.probability >= signal.entry_threshold
                     ? THEME_COLORS.POSITIVE
                     : signal.probability <= (1 - signal.entry_threshold)
                       ? THEME_COLORS.NEGATIVE
                       : THEME_COLORS.TEXT_SECONDARY,
                   opacity: 0.7,
                 }} />
            {/* Entry threshold markers */}
            <div className="absolute top-0 h-full w-px"
                 style={{ left: `${entryBarPos}%`, backgroundColor: THEME_COLORS.YELLOW, opacity: 0.8 }} />
            <div className="absolute top-0 h-full w-px"
                 style={{ left: `${100 - entryBarPos}%`, backgroundColor: THEME_COLORS.YELLOW, opacity: 0.8 }} />
          </div>
          <div className="flex justify-between text-[9px] mt-0.5" style={{ color: THEME_COLORS.TEXT_SECONDARY }}>
            <span>SHORT &lt;{(100 - parseFloat(entryPct)).toFixed(1)}%</span>
            <span>Exit: {exitPct}%</span>
            <span>LONG &gt;{entryPct}%</span>
          </div>

          {/* Triggered indicator */}
          {signal.triggered && (
            <div className="mt-1 text-center">
              <span className="text-[9px] px-2 py-0.5 rounded animate-pulse font-bold"
                    style={{ backgroundColor: getSignalColor(), color: THEME_COLORS.BACKGROUND }}>
                TRIGGERED
              </span>
            </div>
          )}
        </div>

        {/* Position */}
        <div className="flex gap-1.5">
          <div className="flex-1 p-1.5 rounded" style={{ backgroundColor: THEME_COLORS.CARD_BG_LIGHT }}>
            <div className="text-xs" style={{ color: THEME_COLORS.TEXT_SECONDARY }}>Position</div>
            <div className="text-xs font-bold mt-0.5" style={{
              color: portfolio?.position
                ? (portfolio.position === 'LONG' ? THEME_COLORS.POSITIVE : THEME_COLORS.NEGATIVE)
                : THEME_COLORS.TEXT_SECONDARY
            }}>
              {portfolio?.position || 'NONE'}
            </div>
            {portfolio?.position && (
              <div className="text-xs mt-0.5" style={{ color: getPnlColor(liveUnrealized.pnl) }}>
                {liveUnrealized.pnlPct >= 0 ? '+' : ''}{liveUnrealized.pnlPct.toFixed(2)}% (${liveUnrealized.pnl >= 0 ? '+' : ''}{liveUnrealized.pnl.toFixed(2)})
              </div>
            )}
          </div>

          {/* Ratio */}
          <div className="flex-1 p-1.5 rounded" style={{ backgroundColor: THEME_COLORS.CARD_BG_LIGHT }}>
            <div className="text-xs" style={{ color: THEME_COLORS.TEXT_SECONDARY }}>BTC/ETH</div>
            <div className="text-xs font-bold font-mono mt-0.5" style={{ color: THEME_COLORS.TEXT_PRIMARY }}>
              {ratio?.toFixed(4) || '—'}
            </div>
          </div>
        </div>

        {/* Circuit Breaker Warning */}
        {signal.circuit_breaker_active && (
          <div className="p-1.5 rounded text-xs" style={{ backgroundColor: 'rgba(246, 70, 93, 0.15)' }}>
            <span style={{ color: THEME_COLORS.NEGATIVE }}>
              ⚠️ CIRCUIT BREAKER ACTIVE — Forced NEUTRAL
            </span>
          </div>
        )}

        {/* Blocked By */}
        {signal.blocked_by.length > 0 && !signal.circuit_breaker_active && (
          <div className="p-1.5 rounded text-xs" style={{ backgroundColor: 'rgba(246, 70, 93, 0.15)' }}>
            <span style={{ color: THEME_COLORS.NEGATIVE }}>⚠️ {signal.blocked_by.join(', ')}</span>
          </div>
        )}

        {/* Top Features */}
        {features && Object.keys(features).length > 0 && (
          <div className="p-1.5 rounded" style={{ backgroundColor: THEME_COLORS.CARD_BG_LIGHT }}>
            <div className="text-xs font-medium mb-1" style={{ color: THEME_COLORS.TEXT_SECONDARY }}>
              Top Features ({model_info?.n_features || '?'} total)
            </div>
            <div className="grid grid-cols-1 gap-0.5 text-xs">
              {Object.entries(features).slice(0, 6).map(([name, value]) => {
                const shortName = name
                  .replace('buy_pressure_div_', 'bp_div_')
                  .replace('r_roll_std_', 'rstd_')
                  .replace('btc_net_pressure_', 'btc_np_')
                  .replace('eth_net_pressure_', 'eth_np_')
                  .replace('eu_trading_hours', 'eu_hrs')
                  .replace('trend_slope_', 'tslope_');
                return (
                  <div key={name} className="flex justify-between">
                    <span style={{ color: THEME_COLORS.TEXT_SECONDARY }} title={name}>{shortName}</span>
                    <span className="font-mono font-semibold" style={{
                      color: Math.abs(value) > 2 ? THEME_COLORS.YELLOW : THEME_COLORS.TEXT_PRIMARY
                    }}>
                      {value.toFixed(4)}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Reasoning */}
        <div className="flex-1 p-1.5 rounded flex flex-col" style={{ backgroundColor: THEME_COLORS.CARD_BG_LIGHT }}>
          <div className="text-xs font-medium mb-1" style={{ color: THEME_COLORS.TEXT_SECONDARY }}>Reasoning</div>
          <div className="space-y-0.5 flex-1 overflow-y-auto">
            {signal.reasoning.slice(0, 10).map((reason, idx) => {
              const isPositive = reason.includes('LONG') || reason.includes('Above');
              const isNegative = reason.includes('SHORT') || reason.includes('Below') || reason.includes('CIRCUIT');
              return (
                <div key={idx} className="text-xs leading-relaxed" style={{
                  color: isPositive ? THEME_COLORS.POSITIVE : isNegative ? THEME_COLORS.NEGATIVE : THEME_COLORS.TEXT_SECONDARY
                }}>
                  {reason}
                </div>
              );
            })}
          </div>
        </div>

        {/* Data Status */}
        <div className="flex justify-between text-xs px-0.5" style={{ color: THEME_COLORS.TEXT_SECONDARY }}>
          <span>BTC:{data_quality?.btc_candles || 0} ETH:{data_quality?.eth_candles || 0}</span>
          <span style={{ color: data_quality?.synced ? THEME_COLORS.POSITIVE : THEME_COLORS.NEGATIVE }}>
            {data_quality?.synced ? '✓Sync' : '✗Sync'}
          </span>
        </div>
      </div>
    </div>
  );
};
