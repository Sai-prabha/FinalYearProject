import React, { useEffect, useMemo, useState } from 'react';
import { THEME_COLORS } from '../constants';
import { useBrokerConfig } from '../hooks/useBrokerConfig';
import { useBrokerBalance } from '../hooks/useBrokerBalance';
import { useBrokerPositions } from '../hooks/useBrokerPositions';
import { placeOrder, placeTestOrder } from '../services/brokerApi';
import type {
  BrokerConfigSummary,
  ModelSignalData,
  OrderRequestPayload,
  OrderResponsePayload,
} from '../types';

interface BrokerPanelProps {
  signalData: ModelSignalData | null;
}

type Side = 'BUY' | 'SELL';

interface OrderResult {
  ok: boolean;
  text: string;
}

const SYMBOL_RE = /^[A-Z0-9]+$/;

export const BrokerPanel: React.FC<BrokerPanelProps> = ({ signalData }) => {
  const wsBroker = signalData?.model_info?.broker;
  const { config, loading, error, update } = useBrokerConfig(wsBroker);
  const { balance, refresh: refreshBalance, loading: balanceLoading } = useBrokerBalance();
  const { positions, refresh: refreshPositions, loading: positionsLoading } = useBrokerPositions();

  if (loading) {
    return (
      <div
        className="h-full rounded-lg p-2 flex items-center justify-center"
        style={{ backgroundColor: THEME_COLORS.CARD_BG }}
      >
        <p className="text-xs" style={{ color: THEME_COLORS.TEXT_SECONDARY }}>Loading broker…</p>
      </div>
    );
  }

  if (!config) {
    return (
      <div
        className="h-full rounded-lg p-2 flex items-center justify-center"
        style={{ backgroundColor: THEME_COLORS.CARD_BG }}
      >
        <div className="text-center px-2">
          <p className="text-xs mb-1" style={{ color: THEME_COLORS.NEGATIVE }}>Broker unavailable</p>
          <p className="text-[10px]" style={{ color: THEME_COLORS.TEXT_SECONDARY }}>{error ?? 'Unknown error'}</p>
        </div>
      </div>
    );
  }

  return (
    <div
      className="h-full rounded-lg flex flex-col overflow-hidden"
      style={{ backgroundColor: THEME_COLORS.CARD_BG }}
    >
      <Header config={config} />
      <div className="flex-1 overflow-y-auto px-2 pb-2 space-y-2">
        <AutoExecuteRow config={config} update={update} />
        <DefaultsForm config={config} update={update} />
        <TestOrderCard config={config} />
        <ManualOrderCard config={config} onPlaced={refreshPositions} />
        <BalanceCard
          mode={config.mode}
          loading={balanceLoading}
          assets={balance?.assets ?? []}
          onRefresh={refreshBalance}
        />
        <PositionsCard
          mode={config.mode}
          loading={positionsLoading}
          positions={positions?.positions ?? []}
          onRefresh={refreshPositions}
        />
      </div>
    </div>
  );
};

// ── Header & mode badge ──────────────────────────────────────────────────

const Header: React.FC<{ config: BrokerConfigSummary }> = ({ config }) => {
  const isDemo = config.mode === 'demo';
  const badge = isDemo ? 'DEMO' : config.mode.toUpperCase();
  const helper = isDemo ? 'Live Testnet — orders are real' : 'Simulated trades only';

  return (
    <div
      className="px-2 py-2 flex items-center justify-between border-b"
      style={{ borderColor: THEME_COLORS.BORDER }}
    >
      <div>
        <div className="text-xs font-semibold" style={{ color: THEME_COLORS.TEXT_PRIMARY }}>
          Broker Control
        </div>
        <div className="text-[10px]" style={{ color: THEME_COLORS.TEXT_SECONDARY }}>{helper}</div>
      </div>
      <span
        className="text-[10px] font-bold px-2 py-0.5 rounded"
        style={
          isDemo
            ? { backgroundColor: THEME_COLORS.YELLOW, color: THEME_COLORS.BACKGROUND }
            : {
                border: `1px solid ${THEME_COLORS.TEXT_SECONDARY}`,
                color: THEME_COLORS.TEXT_SECONDARY,
              }
        }
      >
        {badge}
      </span>
    </div>
  );
};

// ── Auto-execute row ─────────────────────────────────────────────────────

const AutoExecuteRow: React.FC<{
  config: BrokerConfigSummary;
  update: ReturnType<typeof useBrokerConfig>['update'];
}> = ({ config, update }) => {
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const isDemo = config.mode === 'demo';
  const isOn = config.auto_execute;
  const disabled = !isDemo || busy;

  const onToggle = async () => {
    if (disabled) return;
    setBusy(true);
    setErr(null);
    const result = await update({ auto_execute: !isOn });
    if (!result.ok) setErr(result.error);
    setBusy(false);
    if (err) {
      window.setTimeout(() => setErr(null), 3000);
    }
  };

  return (
    <div className="pt-2">
      <div
        className="flex items-center justify-between px-2 py-1.5 rounded"
        style={{
          backgroundColor: isOn ? `${THEME_COLORS.POSITIVE}22` : THEME_COLORS.CARD_BG_LIGHT,
        }}
      >
        <div>
          <div
            className="text-[11px] font-semibold tracking-wide"
            style={{ color: isOn ? THEME_COLORS.POSITIVE : THEME_COLORS.TEXT_SECONDARY }}
          >
            AUTO EXECUTE: {isOn ? 'ON' : 'OFF'}
          </div>
          <div className="text-[10px]" style={{ color: THEME_COLORS.TEXT_SECONDARY }}>
            {isDemo
              ? 'Forwards triggered signals to Testnet'
              : 'Enable demo mode to use auto-execute'}
          </div>
        </div>
        <button
          type="button"
          onClick={onToggle}
          disabled={disabled}
          aria-pressed={isOn}
          title={!isDemo ? 'Enable demo mode to use auto-execute' : ''}
          className="rounded-full transition-colors"
          style={{
            width: 32,
            height: 18,
            backgroundColor: isOn ? THEME_COLORS.POSITIVE : THEME_COLORS.BORDER,
            opacity: disabled ? 0.5 : 1,
            cursor: disabled ? 'not-allowed' : 'pointer',
            position: 'relative',
          }}
        >
          <span
            style={{
              position: 'absolute',
              top: 2,
              left: isOn ? 16 : 2,
              width: 14,
              height: 14,
              borderRadius: '50%',
              backgroundColor: '#fff',
              transition: 'left 0.15s ease',
            }}
          />
        </button>
      </div>
      {err && (
        <div className="mt-1 text-[10px]" style={{ color: THEME_COLORS.NEGATIVE }}>{err}</div>
      )}
    </div>
  );
};

// ── Defaults form ────────────────────────────────────────────────────────

const DefaultsForm: React.FC<{
  config: BrokerConfigSummary;
  update: ReturnType<typeof useBrokerConfig>['update'];
}> = ({ config, update }) => {
  const [symbol, setSymbol] = useState(config.default_symbol);
  const [qtyText, setQtyText] = useState(String(config.default_qty));
  const [busy, setBusy] = useState(false);
  const [feedback, setFeedback] = useState<{ kind: 'success' | 'error'; text: string } | null>(null);

  // Keep the form in sync with server-truth nudges (WS pushes etc.)
  useEffect(() => { setSymbol(config.default_symbol); }, [config.default_symbol]);
  useEffect(() => { setQtyText(String(config.default_qty)); }, [config.default_qty]);

  const qty = Number(qtyText);
  const symbolValid = SYMBOL_RE.test(symbol);
  const qtyValid = Number.isFinite(qty) && qty > 0;
  const dirty = symbol !== config.default_symbol || qty !== config.default_qty;
  const canSave = symbolValid && qtyValid && dirty && !busy;

  const onSave = async () => {
    setBusy(true);
    setFeedback(null);
    const result = await update({ default_symbol: symbol, default_qty: qty });
    setBusy(false);
    if (result.ok) {
      setFeedback({ kind: 'success', text: 'Saved' });
      window.setTimeout(() => setFeedback(null), 2000);
    } else {
      setFeedback({ kind: 'error', text: result.error });
    }
  };

  const inputStyle: React.CSSProperties = {
    backgroundColor: THEME_COLORS.CARD_BG_LIGHT,
    color: THEME_COLORS.TEXT_PRIMARY,
    border: `1px solid ${THEME_COLORS.BORDER}`,
    borderRadius: 4,
    padding: '4px 6px',
    fontSize: 11,
    width: '100%',
  };

  return (
    <section
      className="rounded p-2"
      style={{ border: `1px solid ${THEME_COLORS.BORDER}` }}
    >
      <div className="text-[11px] font-semibold mb-1" style={{ color: THEME_COLORS.TEXT_PRIMARY }}>
        Defaults
      </div>
      <label className="block mb-1">
        <span className="text-[10px]" style={{ color: THEME_COLORS.TEXT_SECONDARY }}>Symbol</span>
        <input
          type="text"
          value={symbol}
          onChange={e => setSymbol(e.target.value.toUpperCase().trim())}
          placeholder="BTCUSDT"
          style={inputStyle}
        />
        {!symbolValid && (
          <span className="text-[10px]" style={{ color: THEME_COLORS.NEGATIVE }}>
            Letters & digits only
          </span>
        )}
      </label>
      <label className="block mb-1">
        <span className="text-[10px]" style={{ color: THEME_COLORS.TEXT_SECONDARY }}>Quantity</span>
        <input
          type="number"
          value={qtyText}
          step="0.0001"
          min="0"
          onChange={e => setQtyText(e.target.value)}
          style={inputStyle}
        />
        {!qtyValid && (
          <span className="text-[10px]" style={{ color: THEME_COLORS.NEGATIVE }}>
            Must be a positive number
          </span>
        )}
      </label>
      <button
        type="button"
        onClick={onSave}
        disabled={!canSave}
        className="text-[11px] px-3 py-1 rounded transition-colors"
        style={{
          backgroundColor: canSave ? THEME_COLORS.YELLOW : THEME_COLORS.CARD_BG_LIGHT,
          color: canSave ? THEME_COLORS.BACKGROUND : THEME_COLORS.TEXT_SECONDARY,
          cursor: canSave ? 'pointer' : 'not-allowed',
        }}
      >
        {busy ? 'Saving…' : 'Save'}
      </button>
      {feedback && (
        <div
          className="mt-1 text-[10px]"
          style={{ color: feedback.kind === 'success' ? THEME_COLORS.POSITIVE : THEME_COLORS.NEGATIVE }}
        >
          {feedback.text}
        </div>
      )}
    </section>
  );
};

// ── Order action sub-components (test + manual) ──────────────────────────

const useOrderForm = (config: BrokerConfigSummary) => {
  const [symbol, setSymbol] = useState(config.default_symbol);
  const [qtyText, setQtyText] = useState(String(config.default_qty));
  const [side, setSide] = useState<Side>('BUY');
  useEffect(() => { setSymbol(config.default_symbol); }, [config.default_symbol]);
  useEffect(() => { setQtyText(String(config.default_qty)); }, [config.default_qty]);

  const qty = Number(qtyText);
  const symbolValid = SYMBOL_RE.test(symbol);
  const qtyValid = Number.isFinite(qty) && qty > 0;
  const valid = symbolValid && qtyValid;
  const payload: OrderRequestPayload = useMemo(
    () => ({ symbol, side, quantity: qty, order_type: 'MARKET' }),
    [symbol, side, qty],
  );

  return { symbol, setSymbol, side, setSide, qtyText, setQtyText, valid, payload };
};

const SideToggle: React.FC<{ side: Side; onChange: (s: Side) => void }> = ({ side, onChange }) => (
  <div className="flex gap-1">
    {(['BUY', 'SELL'] as Side[]).map(s => (
      <button
        key={s}
        type="button"
        onClick={() => onChange(s)}
        className="text-[10px] px-2 py-0.5 rounded"
        style={{
          backgroundColor:
            side === s
              ? s === 'BUY'
                ? THEME_COLORS.POSITIVE
                : THEME_COLORS.NEGATIVE
              : THEME_COLORS.CARD_BG_LIGHT,
          color: side === s ? '#000' : THEME_COLORS.TEXT_SECONDARY,
        }}
      >
        {s}
      </button>
    ))}
  </div>
);

const SmallInputs: React.FC<{
  symbol: string; setSymbol: (v: string) => void;
  qtyText: string; setQtyText: (v: string) => void;
  side: Side; setSide: (s: Side) => void;
}> = ({ symbol, setSymbol, qtyText, setQtyText, side, setSide }) => {
  const inputStyle: React.CSSProperties = {
    backgroundColor: THEME_COLORS.CARD_BG_LIGHT,
    color: THEME_COLORS.TEXT_PRIMARY,
    border: `1px solid ${THEME_COLORS.BORDER}`,
    borderRadius: 4,
    padding: '3px 5px',
    fontSize: 11,
    width: '100%',
  };
  return (
    <div className="space-y-1">
      <input
        type="text"
        value={symbol}
        placeholder="BTCUSDT"
        onChange={e => setSymbol(e.target.value.toUpperCase().trim())}
        style={inputStyle}
      />
      <input
        type="number"
        value={qtyText}
        step="0.0001"
        min="0"
        onChange={e => setQtyText(e.target.value)}
        style={inputStyle}
      />
      <SideToggle side={side} onChange={setSide} />
    </div>
  );
};

const TestOrderCard: React.FC<{ config: BrokerConfigSummary }> = ({ config }) => {
  const form = useOrderForm(config);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<OrderResult | null>(null);

  const onSend = async () => {
    setBusy(true);
    try {
      const resp = await placeTestOrder(form.payload);
      setResult({ ok: resp.status === 'TEST_OK', text: orderResultText(resp) });
    } catch (e) {
      setResult({ ok: false, text: e instanceof Error ? e.message : String(e) });
    } finally {
      setBusy(false);
    }
  };

  return (
    <section
      className="rounded p-2"
      style={{ border: `1px solid ${THEME_COLORS.BORDER}` }}
    >
      <div className="text-[11px] font-semibold" style={{ color: THEME_COLORS.TEXT_PRIMARY }}>
        Test Order
      </div>
      <div className="text-[10px] mb-1" style={{ color: THEME_COLORS.TEXT_SECONDARY }}>
        Validates signing only — no order placed
      </div>
      <SmallInputs
        symbol={form.symbol}
        setSymbol={form.setSymbol}
        qtyText={form.qtyText}
        setQtyText={form.setQtyText}
        side={form.side}
        setSide={form.setSide}
      />
      <button
        type="button"
        onClick={onSend}
        disabled={!form.valid || busy}
        className="w-full mt-1 text-[11px] py-1 rounded transition-colors"
        style={{
          backgroundColor: 'transparent',
          color: THEME_COLORS.YELLOW,
          border: `1px solid ${THEME_COLORS.YELLOW}`,
          cursor: !form.valid || busy ? 'not-allowed' : 'pointer',
          opacity: !form.valid || busy ? 0.5 : 1,
        }}
      >
        {busy ? 'Sending…' : 'Send Test'}
      </button>
      {result && (
        <div
          className="mt-1 text-[10px]"
          style={{ color: result.ok ? THEME_COLORS.POSITIVE : THEME_COLORS.NEGATIVE }}
        >
          {result.text}
        </div>
      )}
    </section>
  );
};

const ManualOrderCard: React.FC<{ config: BrokerConfigSummary; onPlaced: () => void }> = ({
  config,
  onPlaced,
}) => {
  const form = useOrderForm(config);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<OrderResult | null>(null);
  const isDemo = config.mode === 'demo';

  const onSend = async () => {
    if (isDemo) {
      const ok = window.confirm(
        `Place real Testnet order: ${form.side} ${form.payload.quantity} ${form.symbol}?`,
      );
      if (!ok) return;
    }
    setBusy(true);
    try {
      const resp = await placeOrder(form.payload);
      setResult({ ok: resp.status !== 'REJECTED', text: orderResultText(resp) });
      onPlaced();
    } catch (e) {
      setResult({ ok: false, text: e instanceof Error ? e.message : String(e) });
    } finally {
      setBusy(false);
    }
  };

  return (
    <section
      className="rounded p-2"
      style={{ border: `1px solid ${THEME_COLORS.NEGATIVE}` }}
    >
      <div className="text-[11px] font-semibold" style={{ color: THEME_COLORS.TEXT_PRIMARY }}>
        Place Demo Order
      </div>
      <div className="text-[10px] mb-1 italic" style={{ color: THEME_COLORS.NEGATIVE }}>
        {isDemo ? 'Real Testnet order' : 'Simulated — paper mode'}
      </div>
      <SmallInputs
        symbol={form.symbol}
        setSymbol={form.setSymbol}
        qtyText={form.qtyText}
        setQtyText={form.setQtyText}
        side={form.side}
        setSide={form.setSide}
      />
      <button
        type="button"
        onClick={onSend}
        disabled={!form.valid || busy}
        className="w-full mt-1 text-[11px] py-1 rounded transition-colors font-semibold"
        style={{
          backgroundColor: THEME_COLORS.NEGATIVE,
          color: '#fff',
          cursor: !form.valid || busy ? 'not-allowed' : 'pointer',
          opacity: !form.valid || busy ? 0.5 : 1,
        }}
      >
        {busy ? 'Placing…' : 'Place Demo Order'}
      </button>
      {result && (
        <div
          className="mt-1 text-[10px]"
          style={{ color: result.ok ? THEME_COLORS.POSITIVE : THEME_COLORS.NEGATIVE }}
        >
          {result.text}
        </div>
      )}
    </section>
  );
};

function orderResultText(resp: OrderResponsePayload): string {
  if (resp.status === 'TEST_OK') return 'TEST_OK';
  if (resp.status === 'REJECTED') return `REJECTED — ${resp.message ?? 'unknown'}`;
  if (resp.status === 'FILLED') return `FILLED at ${resp.avg_price || '—'}`;
  return resp.status + (resp.message ? ` — ${resp.message}` : '');
}

// ── Balance + Positions ──────────────────────────────────────────────────

const BalanceCard: React.FC<{
  mode: BrokerConfigSummary['mode'];
  loading: boolean;
  assets: { asset: string; balance: number; available: number }[];
  onRefresh: () => void;
}> = ({ mode, loading, assets, onRefresh }) => {
  const isDemo = mode === 'demo';
  return (
    <section className="rounded p-2" style={{ border: `1px solid ${THEME_COLORS.BORDER}` }}>
      <div className="flex items-center justify-between mb-1">
        <div className="text-[11px] font-semibold" style={{ color: THEME_COLORS.TEXT_PRIMARY }}>
          Balance
        </div>
        <button
          type="button"
          onClick={onRefresh}
          className="text-[10px]"
          style={{ color: THEME_COLORS.TEXT_SECONDARY }}
        >
          {loading ? '…' : 'refresh'}
        </button>
      </div>
      {!isDemo ? (
        <div className="text-[10px] italic" style={{ color: THEME_COLORS.TEXT_SECONDARY }}>
          Paper mode — no live balance
        </div>
      ) : assets.length === 0 ? (
        <div className="text-[10px]" style={{ color: THEME_COLORS.TEXT_SECONDARY }}>No balance data</div>
      ) : (
        <ul className="space-y-0.5">
          {assets.map(a => (
            <li key={a.asset} className="flex justify-between text-[11px] font-mono">
              <span style={{ color: THEME_COLORS.TEXT_PRIMARY }}>{a.asset}</span>
              <span style={{ color: THEME_COLORS.TEXT_SECONDARY }}>
                {a.balance.toFixed(2)}{' '}
                <span style={{ color: THEME_COLORS.TEXT_SECONDARY }}>
                  (avail {a.available.toFixed(2)})
                </span>
              </span>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
};

const PositionsCard: React.FC<{
  mode: BrokerConfigSummary['mode'];
  loading: boolean;
  positions: {
    symbol: string;
    side: 'LONG' | 'SHORT';
    size: number;
    entry_price: number;
    mark_price: number;
    unrealized_pnl: number;
  }[];
  onRefresh: () => void;
}> = ({ mode, loading, positions, onRefresh }) => {
  const isDemo = mode === 'demo';
  return (
    <section className="rounded p-2" style={{ border: `1px solid ${THEME_COLORS.BORDER}` }}>
      <div className="flex items-center justify-between mb-1">
        <div className="text-[11px] font-semibold" style={{ color: THEME_COLORS.TEXT_PRIMARY }}>
          Open positions
        </div>
        <button
          type="button"
          onClick={onRefresh}
          className="text-[10px]"
          style={{ color: THEME_COLORS.TEXT_SECONDARY }}
        >
          {loading ? '…' : 'refresh'}
        </button>
      </div>
      {!isDemo ? (
        <div className="text-[10px] italic" style={{ color: THEME_COLORS.TEXT_SECONDARY }}>
          Paper mode — no live positions
        </div>
      ) : positions.length === 0 ? (
        <div className="text-[10px]" style={{ color: THEME_COLORS.TEXT_SECONDARY }}>No open positions</div>
      ) : (
        <table className="w-full text-[10px] font-mono">
          <thead>
            <tr style={{ color: THEME_COLORS.TEXT_SECONDARY }}>
              <th className="text-left">Sym</th>
              <th className="text-left">Side</th>
              <th className="text-right">Size</th>
              <th className="text-right">Entry</th>
              <th className="text-right">Mark</th>
              <th className="text-right">uPnL</th>
            </tr>
          </thead>
          <tbody>
            {positions.map(p => (
              <tr key={`${p.symbol}-${p.side}`} style={{ color: THEME_COLORS.TEXT_PRIMARY }}>
                <td>{p.symbol}</td>
                <td style={{ color: p.side === 'LONG' ? THEME_COLORS.POSITIVE : THEME_COLORS.NEGATIVE }}>
                  {p.side}
                </td>
                <td className="text-right">{p.size}</td>
                <td className="text-right">{p.entry_price}</td>
                <td className="text-right">{p.mark_price}</td>
                <td
                  className="text-right"
                  style={{
                    color:
                      p.unrealized_pnl >= 0 ? THEME_COLORS.POSITIVE : THEME_COLORS.NEGATIVE,
                  }}
                >
                  {p.unrealized_pnl.toFixed(2)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </section>
  );
};
