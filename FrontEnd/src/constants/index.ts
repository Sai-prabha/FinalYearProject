// WebSocket URLs for live data streams
export const WS_URLS = {
  BTC: 'wss://stream.binance.com:9443/ws/btcusdt@kline_1m',
  ETH: 'wss://stream.binance.com:9443/ws/ethusdt@kline_1m',
} as const;

// Backward compat alias
export const BINANCE_WS_URLS = WS_URLS;

// Model server URL (use env vars in production)
export const MODEL_SERVER_URL = import.meta.env.VITE_MODEL_SERVER_URL || 'ws://localhost:8888/ws/signals';
export const MODEL_SERVER_REST_URL = import.meta.env.VITE_MODEL_SERVER_REST_URL || 'http://localhost:8888';

// When true, frontend requires login (production)
export const AUTH_REQUIRED = (import.meta.env.VITE_MODEL_SERVER_REST_URL?.startsWith('https://') ?? false);

// API symbols
export const API_SYMBOLS = {
  BTC: 'BTCUSDT',
  ETH: 'ETHUSDT',
} as const;

// Backward compat alias
export const BINANCE_SYMBOLS = API_SYMBOLS;

// Chart colors
export const CHART_COLORS = {
  BTC: '#f0b90b',
  ETH: '#627eea',
  RATIO: '#8b5cf6',
} as const;

// Theme colors – dark trading theme
export const THEME_COLORS = {
  BACKGROUND: '#0b0e11',
  CARD_BG: '#1e2329',
  CARD_BG_LIGHT: '#2b3139',
  TEXT_PRIMARY: '#eaecef',
  TEXT_SECONDARY: '#848e9c',
  POSITIVE: '#0ecb81',
  NEGATIVE: '#f6465d',
  YELLOW: '#f0b90b',
  BORDER: '#2b3139',
  GRID: 'rgba(43, 49, 57, 0.8)',
} as const;

// Configuration constants
export const CONFIG = {
  MAX_DATA_POINTS: 500,
  RECONNECT_MAX_ATTEMPTS: 5,
  RECONNECT_BASE_DELAY: 1000,
  RECONNECT_MAX_DELAY: 16000,
  UPDATE_THROTTLE: 1000,
  VOLATILITY_WINDOW: 30,
} as const;

// REST API base URL
export const REST_API_BASE = 'https://api.binance.com/api/v3';

// Asset symbols
export const SYMBOLS = {
  BTC: 'BTC/USDT',
  ETH: 'ETH/USDT',
  RATIO: 'BTC/ETH',
} as const;

// Price formatting decimals
export const DECIMALS = {
  BTC: 2,
  ETH: 2,
  RATIO: 4,
} as const;
