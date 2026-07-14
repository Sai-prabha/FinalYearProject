// Candlestick data structure
export interface CandleData {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

// Price data for line charts
export interface PriceData {
  time: number;
  value: number;
}

// Asset data with full state
export interface AssetData {
  symbol: string;
  currentPrice: number;
  change24h: number;
  priceHistory: PriceData[];
  connectionStatus: ConnectionStatus;
  lastUpdate: number;
}

// Connection status type
export type ConnectionStatus = 'connecting' | 'connected' | 'disconnected' | 'error';

// WebSocket kline event structure
export interface KlineEvent {
  e: string; // Event type
  E: number; // Event time
  s: string; // Symbol
  k: {
    t: number; // Kline start time
    T: number; // Kline close time
    s: string; // Symbol
    i: string; // Interval
    o: string; // Open price
    c: string; // Close price
    h: string; // High price
    l: string; // Low price
    v: string; // Volume
    x: boolean; // Is this kline closed?
  };
}

// Keep backward compat alias
export type BinanceKlineEvent = KlineEvent;

// WebSocket service callback types
export type DataCallback = (data: CandleData) => void;
export type StatusCallback = (status: ConnectionStatus) => void;

// V4.15 Model signal types

/** Dynamic feature key-value map (all 50 features sent from model server) */
export type ModelFeatures = Record<string, number>;

export interface ModelSignal {
  direction: 'LONG' | 'SHORT' | 'NEUTRAL';
  strength: number;
  probability: number;
  triggered: boolean;
  blocked_by: string[];
  reasoning: string[];
  circuit_breaker_active: boolean;
  entry_threshold: number;
  exit_threshold: number;
}

export interface PositionMeta {
  stop_loss: number;
  take_profit: number;
  position_size_pct: number;
  effective_min_hold: number;
  bars_held: number;
}

export interface Trade {
  direction: 'LONG' | 'SHORT';
  entry_price: number;
  exit_price: number;
  entry_time: number;
  exit_time: number;
  pnl_pct: number;
  pnl_dollar: number;
  bars_held?: number;
  position_size_pct?: number;
  stop_loss?: number;
  take_profit?: number;
  entry_probability?: number;
  entry_strength?: number;
  reason: string;
}

export interface PortfolioState {
  balance: number;
  starting_balance: number;
  total_pnl: number;
  total_pnl_pct: number;
  unrealized_pnl: number;
  unrealized_pnl_pct: number;
  position: 'LONG' | 'SHORT' | null;
  entry_price: number;
  total_trades: number;
  wins: number;
  losses: number;
  win_rate: number;
  recent_trades: Trade[];
}

export interface DataQuality {
  btc_candles: number;
  eth_candles: number;
  ready: boolean;
  min_required: number;
  last_btc_time: number;
  last_eth_time: number;
  synced: boolean;
}

export interface BrokerExecLeg {
  symbol: string;
  side: 'BUY' | 'SELL';
  qty: number;
  reduce_only: boolean;
  status: string;          // 'NEW' | 'FILLED' | 'REJECTED' | 'ERROR' | ...
  order_id: string;
  filled_qty: number;
  avg_price: number;
  error: string | null;
  reconciled?: boolean;    // true once fill data confirmed via GET /fapi/v1/order
}

export interface BrokerExecEvent {
  timestamp: string;
  prev_pos: -1 | 0 | 1;
  new_pos: -1 | 0 | 1;
  transition: string;      // e.g. "FLAT→LONG"
  legs: BrokerExecLeg[];
  all_ok: boolean;
  reconciled?: boolean;    // true once all legs have been reconciled
}

export interface ExecutionWriterInfo {
  backend: string; // 'railway' | 'local'
  instance_id: string;
  is_writer: boolean;
  role: 'writer' | 'observer';
}

/** Backend-authoritative shared auto-execute state (GET/PATCH /execution/control). */
export interface ExecutionControlState {
  auto_execute: boolean;
  version: number;
  updated_at: string | null;
  updated_by: string | null;
  updated_via: string | null;
  request_id: string | null;
  writer: ExecutionWriterInfo;
  mode: string;
}

export interface BrokerConfigSummary {
  mode: 'paper' | 'demo' | 'unknown';
  auto_execute: boolean;
  default_symbol: string;
  default_qty: number;
  default_btc_qty: number;
  default_eth_qty: number;
}

export interface BrokerBalanceAsset {
  asset: string;
  balance: number;
  available: number;
}

export interface BrokerBalanceResponse {
  mode: string;
  assets: BrokerBalanceAsset[];
}

export interface BrokerPosition {
  symbol: string;
  side: 'LONG' | 'SHORT';
  size: number;
  entry_price: number;
  mark_price: number;
  unrealized_pnl: number;
  leverage: number;
}

export interface BrokerPositionsResponse {
  mode: string;
  positions: BrokerPosition[];
}

export interface OrderRequestPayload {
  symbol: string;
  side: 'BUY' | 'SELL';
  order_type?: 'MARKET' | 'LIMIT';
  quantity: number;
  price?: number;
  reduce_only?: boolean;
  client_id?: string;
}

export interface OrderResponsePayload {
  broker_order_id: string;
  status: string;
  filled_qty: number;
  avg_price: number;
  message?: string | null;
}

export interface ModelInfo {
  version: string;
  n_features: number;
  calc_time_ms: number;
  broker?: BrokerConfigSummary;
  last_exec?: BrokerExecEvent | null;
}

export interface ModelSignalData {
  timestamp: string;
  ratio: number;
  btc_price: number;
  eth_price: number;
  features: ModelFeatures;
  signal: ModelSignal;
  position_meta?: PositionMeta;
  portfolio: PortfolioState;
  data_quality: DataQuality;
  model_info: ModelInfo;
}
