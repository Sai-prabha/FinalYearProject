import React, { useEffect, useState, useRef, useCallback, useMemo } from 'react';
import { THEME_COLORS, MODEL_SERVER_REST_URL } from '../constants';
import type { PortfolioState, Trade, ModelSignalData } from '../types';
import { getAuthHeaders } from '../utils/auth';
import { exportTradesToExcel, exportTradesToCSV } from '../utils/excelExport';

const STORAGE_KEY = 'trade_history';

// Grid template for full-width columns
const GRID_COLS = '70px 52px 1fr 1fr 1fr 1fr 70px 70px 60px 55px 55px 55px 1fr';

interface TradeHistoryProps {
  portfolio: PortfolioState | null;
  isModelConnected?: boolean;
  signalData?: ModelSignalData | null;
  btcPrice?: number;
  ethPrice?: number;
}

export const TradeHistory: React.FC<TradeHistoryProps> = ({
  portfolio,
  isModelConnected = false,
  signalData = null,
  btcPrice = 0,
  ethPrice = 0,
}) => {
  // Trades synced from backend — backend is the source of truth
  const [persistedTrades, setPersistedTrades] = useState<Trade[]>([]);
  const hasSyncedRef = useRef(false);
  const lastTradeCountRef = useRef<number>(0);

  // On mount (or when model connects), fetch full trade history from backend
  useEffect(() => {
    if (!isModelConnected) {
      hasSyncedRef.current = false;
      return;
    }
    if (hasSyncedRef.current) return;

    const syncFromBackend = async () => {
      try {
        const resp = await fetch(`${MODEL_SERVER_REST_URL}/trades`, { headers: getAuthHeaders() });
        if (resp.ok) {
          const data = await resp.json();
          const trades: Trade[] = data.trades || [];
          setPersistedTrades(trades);
          lastTradeCountRef.current = trades.length;
          try {
            localStorage.setItem(STORAGE_KEY, JSON.stringify(trades));
          } catch { /* ignore */ }
          hasSyncedRef.current = true;
        }
      } catch {
        // Fallback to localStorage if backend unreachable
        try {
          const stored = localStorage.getItem(STORAGE_KEY);
          if (stored) setPersistedTrades(JSON.parse(stored));
        } catch { /* ignore */ }
        hasSyncedRef.current = true;
      }
    };

    syncFromBackend();
  }, [isModelConnected]);

  // Merge live trades into persisted set whenever portfolio updates
  useEffect(() => {
    if (!hasSyncedRef.current) return;   // wait for backend sync first
    if (!portfolio?.recent_trades) return;

    const liveTrades = portfolio.recent_trades;
    if (liveTrades.length === 0) return;

    // Detect new trades by comparing count
    if (liveTrades.length > lastTradeCountRef.current) {
      const newTrades = liveTrades.slice(lastTradeCountRef.current);
      setPersistedTrades(prev => {
        // Deduplicate by exit_time
        const exitTimes = new Set(prev.map(t => t.exit_time));
        const unique = newTrades.filter(t => !exitTimes.has(t.exit_time));
        if (unique.length === 0) return prev;

        const merged = [...prev, ...unique];
        try {
          localStorage.setItem(STORAGE_KEY, JSON.stringify(merged));
        } catch {
          // Storage full — ignore
        }
        return merged;
      });
    }
    lastTradeCountRef.current = liveTrades.length;
  }, [portfolio?.recent_trades]);

  // ── Compute W/L/WR from persisted trades (not model state) ──
  const { wins, losses, winRate } = useMemo(() => {
    const w = persistedTrades.filter(t => t.pnl_dollar > 0).length;
    const l = persistedTrades.filter(t => t.pnl_dollar <= 0).length;
    const wr = persistedTrades.length > 0 ? (w / persistedTrades.length) * 100 : 0;
    return { wins: w, losses: l, winRate: wr };
  }, [persistedTrades]);

  // ── Real-time unrealized P/L from live BTC/ETH prices ──
  const liveUnrealized = useMemo(() => {
    if (!portfolio?.position || !portfolio?.entry_price || portfolio.entry_price <= 0) {
      return { pnl: 0, pnlPct: 0 };
    }
    if (btcPrice <= 0 || ethPrice <= 0) {
      // Fallback to model server values
      return { pnl: portfolio.unrealized_pnl, pnlPct: portfolio.unrealized_pnl_pct };
    }
    const liveRatio = btcPrice / ethPrice;
    let pnlPct = 0;
    if (portfolio.position === 'LONG') {
      pnlPct = ((liveRatio - portfolio.entry_price) / portfolio.entry_price) * 100;
    } else {
      pnlPct = ((portfolio.entry_price - liveRatio) / portfolio.entry_price) * 100;
    }
    const pnl = (portfolio.balance || 1000) * (signalData?.position_meta?.position_size_pct || 10) / 100 * (pnlPct / 100);
    return { pnl, pnlPct };
  }, [portfolio, btcPrice, ethPrice, signalData]);

  // All trades to display: persisted history, most-recent-first
  const allTrades = [...persistedTrades].reverse();
  const hasOpenPosition = portfolio?.position !== null && portfolio?.position !== undefined;

  // Export handlers
  const handleExportExcel = useCallback(async () => {
    try {
      await exportTradesToExcel({
        trades: persistedTrades,
        signalData,
        filename: `trade_history_${new Date().toISOString().slice(0, 10)}.xlsx`,
      });
    } catch (err) {
      console.error('Excel export failed, falling back to CSV:', err);
      exportTradesToCSV(persistedTrades);
    }
  }, [persistedTrades, signalData]);

  const handleClearHistory = useCallback(async () => {
    if (confirm('Clear all persisted trade history?')) {
      setPersistedTrades([]);
      localStorage.removeItem(STORAGE_KEY);
      lastTradeCountRef.current = 0;

      // Also clear backend trade history so trades don't reappear on refresh
      try {
        await fetch(`${MODEL_SERVER_REST_URL}/trades/clear`, { method: 'DELETE', headers: getAuthHeaders() });
      } catch {
        // Backend may be offline — local clear still succeeded
      }
    }
  }, []);

  return (
    <div 
      className="rounded-lg h-full flex flex-col"
      style={{ backgroundColor: THEME_COLORS.CARD_BG }}
    >
      {/* Header */}
      <div className="px-2 py-1.5 border-b flex items-center justify-between" style={{ borderColor: THEME_COLORS.BORDER }}>
        <div className="flex items-center gap-2">
          <h3 className="text-[#eaecef] font-semibold text-sm">Trade History</h3>
          <div className={`w-1.5 h-1.5 rounded-full ${isModelConnected ? 'bg-green-500' : 'bg-gray-500'}`} />
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs text-[#848e9c]">
            {persistedTrades.length} trades • {wins}W / {losses}L
            {winRate > 0 ? ` • ${winRate.toFixed(0)}% WR` : ''}
          </span>
          <button
            onClick={handleExportExcel}
            disabled={persistedTrades.length === 0}
            className="px-2 py-0.5 text-[10px] rounded transition-colors hover:opacity-80 disabled:opacity-40 disabled:cursor-not-allowed"
            style={{ backgroundColor: THEME_COLORS.YELLOW, color: THEME_COLORS.BACKGROUND }}
            title="Export to Excel"
          >
            Export
          </button>
          {persistedTrades.length > 0 && (
            <button
              onClick={handleClearHistory}
              className="px-2 py-0.5 text-[10px] rounded transition-colors hover:opacity-80"
              style={{ backgroundColor: THEME_COLORS.CARD_BG_LIGHT, color: THEME_COLORS.TEXT_SECONDARY }}
              title="Clear persisted history"
            >
              Clear
            </button>
          )}
        </div>
      </div>
      
      {/* Column Headers — full-width layout */}
      <div 
        className="grid text-xs text-[#848e9c] px-2 py-1 border-b"
        style={{ borderColor: THEME_COLORS.BORDER, gridTemplateColumns: GRID_COLS }}
      >
        <div className="text-left">Time</div>
        <div className="text-left">Type</div>
        <div className="text-right">Entry</div>
        <div className="text-right">Exit</div>
        <div className="text-right">SL</div>
        <div className="text-right">TP</div>
        <div className="text-right">P&L</div>
        <div className="text-right">P&L %</div>
        <div className="text-right">Bars</div>
        <div className="text-right">Size%</div>
        <div className="text-right">Prob</div>
        <div className="text-right">Str</div>
        <div className="text-right">Reason</div>
      </div>

      {/* Trade List */}
      <div className="flex-1 overflow-y-auto">
        {/* Open position row (live) — uses real-time P/L */}
        {hasOpenPosition && portfolio && portfolio.entry_price > 0 && (
          <div 
            className="grid text-xs px-2 py-1 transition-colors"
            style={{ backgroundColor: 'rgba(240, 185, 11, 0.08)', gridTemplateColumns: GRID_COLS }}
          >
            <div className="text-[#f0b90b] text-left font-medium">OPEN</div>
            <div className={`text-left font-medium ${portfolio.position === 'LONG' ? 'text-[#0ecb81]' : 'text-[#f6465d]'}`}>
              {portfolio.position}
            </div>
            <div className="text-[#eaecef] text-right font-mono">{portfolio.entry_price.toFixed(4)}</div>
            <div className="text-[#848e9c] text-right font-mono">---</div>
            <div className="text-[#f6465d] text-right font-mono">
              {signalData?.position_meta?.stop_loss ? signalData.position_meta.stop_loss.toFixed(4) : '-'}
            </div>
            <div className="text-[#0ecb81] text-right font-mono">
              {signalData?.position_meta?.take_profit ? signalData.position_meta.take_profit.toFixed(4) : '-'}
            </div>
            <div 
              className={`text-right font-mono font-medium ${liveUnrealized.pnl >= 0 ? 'text-[#0ecb81]' : 'text-[#f6465d]'}`}
            >
              {liveUnrealized.pnl >= 0 ? '+' : ''}{liveUnrealized.pnl.toFixed(2)}
            </div>
            <div 
              className={`text-right font-mono font-medium ${liveUnrealized.pnlPct >= 0 ? 'text-[#0ecb81]' : 'text-[#f6465d]'}`}
            >
              {liveUnrealized.pnlPct >= 0 ? '+' : ''}{liveUnrealized.pnlPct.toFixed(2)}%
            </div>
            <div className="text-[#848e9c] text-right font-mono">
              {signalData?.position_meta?.bars_held ?? '-'}
            </div>
            <div className="text-[#848e9c] text-right font-mono">
              {signalData?.position_meta?.position_size_pct ? signalData.position_meta.position_size_pct.toFixed(0) : '-'}
            </div>
            <div className="text-[#848e9c] text-right font-mono">-</div>
            <div className="text-[#848e9c] text-right font-mono">-</div>
            <div className="text-[#f0b90b] text-right text-[10px]">Active</div>
          </div>
        )}

        {/* Closed trades */}
        {allTrades.length > 0 ? (
          allTrades.map((trade, index) => {
            const exitTime = new Date(trade.exit_time * 1000).toLocaleTimeString('en-US', { 
              hour: '2-digit', 
              minute: '2-digit',
              second: '2-digit'
            });
            
            return (
              <div 
                key={`${trade.exit_time}-${index}`}
                className="grid text-xs px-2 py-1 hover:bg-[#2b3139] transition-colors"
                style={{ gridTemplateColumns: GRID_COLS }}
              >
                <div className="text-[#848e9c] text-left">{exitTime}</div>
                <div className={`text-left font-medium ${trade.direction === 'LONG' ? 'text-[#0ecb81]' : 'text-[#f6465d]'}`}>
                  {trade.direction}
                </div>
                <div className="text-[#eaecef] text-right font-mono">{trade.entry_price.toFixed(4)}</div>
                <div className="text-[#eaecef] text-right font-mono">{trade.exit_price.toFixed(4)}</div>
                <div className="text-[#f6465d] text-right font-mono">
                  {trade.stop_loss ? trade.stop_loss.toFixed(4) : '-'}
                </div>
                <div className="text-[#0ecb81] text-right font-mono">
                  {trade.take_profit ? trade.take_profit.toFixed(4) : '-'}
                </div>
                <div 
                  className={`text-right font-mono font-medium ${trade.pnl_dollar >= 0 ? 'text-[#0ecb81]' : 'text-[#f6465d]'}`}
                >
                  {trade.pnl_dollar >= 0 ? '+' : ''}{trade.pnl_dollar.toFixed(2)}
                </div>
                <div 
                  className={`text-right font-mono font-medium ${trade.pnl_pct >= 0 ? 'text-[#0ecb81]' : 'text-[#f6465d]'}`}
                >
                  {trade.pnl_pct >= 0 ? '+' : ''}{trade.pnl_pct.toFixed(2)}%
                </div>
                <div className="text-[#848e9c] text-right font-mono">
                  {trade.bars_held ?? '-'}
                </div>
                <div className="text-[#848e9c] text-right font-mono">
                  {trade.position_size_pct ? trade.position_size_pct.toFixed(0) : '-'}
                </div>
                <div className="text-[#848e9c] text-right font-mono">
                  {trade.entry_probability ? (trade.entry_probability * 100).toFixed(0) + '%' : '-'}
                </div>
                <div className="text-[#848e9c] text-right font-mono">
                  {trade.entry_strength ? trade.entry_strength.toFixed(2) : '-'}
                </div>
                <div className="text-[#848e9c] text-right text-[10px] truncate" title={trade.reason}>
                  {trade.reason}
                </div>
              </div>
            );
          })
        ) : !hasOpenPosition ? (
          <div className="flex flex-col items-center justify-center h-full gap-1">
            {!isModelConnected ? (
              <>
                <p className="text-xs text-[#848e9c]">Model server not connected</p>
                <p className="text-[10px] text-[#5e6673]">Connect to ws://localhost:8888 to receive signals</p>
              </>
            ) : (
              <>
                <p className="text-xs text-[#848e9c]">No trades yet</p>
                <p className="text-[10px] text-[#5e6673]">Trades appear after the model opens and closes positions</p>
              </>
            )}
          </div>
        ) : null}
      </div>
    </div>
  );
};
