import type { CandleData } from '../types';
import { REST_API_BASE } from '../constants';

// Fetch historical klines (candles) from REST API
export const fetchHistoricalCandles = async (
  symbol: string,
  interval: string = '1m',
  limit: number = 60
): Promise<CandleData[]> => {
  try {
    const url = `${REST_API_BASE}/klines?symbol=${symbol}&interval=${interval}&limit=${limit}`;
    const response = await fetch(url);
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const data = await response.json();
    
    // Kline format: [openTime, open, high, low, close, volume, closeTime, ...]
    return data.map((kline: (string | number)[]) => ({
      time: Math.floor(Number(kline[0]) / 1000),
      open: parseFloat(kline[1] as string),
      high: parseFloat(kline[2] as string),
      low: parseFloat(kline[3] as string),
      close: parseFloat(kline[4] as string),
      volume: parseFloat(kline[5] as string),
    }));
  } catch (error) {
    console.error(`Error fetching historical candles for ${symbol}:`, error);
    return [];
  }
};

// Fetch 24h ticker stats from REST API
export const fetch24hTicker = async (
  symbol: string
): Promise<{
  priceChange: number;
  priceChangePercent: number;
  highPrice: number;
  lowPrice: number;
  volume: number;
  quoteVolume: number;
} | null> => {
  try {
    const url = `${REST_API_BASE}/ticker/24hr?symbol=${symbol}`;
    const response = await fetch(url);
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const data = await response.json();
    return {
      priceChange: parseFloat(data.priceChange),
      priceChangePercent: parseFloat(data.priceChangePercent),
      highPrice: parseFloat(data.highPrice),
      lowPrice: parseFloat(data.lowPrice),
      volume: parseFloat(data.volume),
      quoteVolume: parseFloat(data.quoteVolume),
    };
  } catch (error) {
    console.error(`Error fetching 24h ticker for ${symbol}:`, error);
    return null;
  }
};

// Order book entry type
export interface OrderBookLevel {
  price: number;
  amount: number;
}

export interface OrderBookData {
  bids: OrderBookLevel[];
  asks: OrderBookLevel[];
  lastUpdateId: number;
}

// Parse order book data from WebSocket
export const parseOrderBookData = (data: {
  lastUpdateId: number;
  bids: [string, string][];
  asks: [string, string][];
}): OrderBookData => {
  return {
    lastUpdateId: data.lastUpdateId,
    bids: data.bids.map(([price, amount]) => ({
      price: parseFloat(price),
      amount: parseFloat(amount),
    })),
    asks: data.asks.map(([price, amount]) => ({
      price: parseFloat(price),
      amount: parseFloat(amount),
    })),
  };
};

// WebSocket URL for order book depth
export const getOrderBookWsUrl = (symbol: string, levels: number = 20): string => {
  return `wss://stream.binance.com:9443/ws/${symbol.toLowerCase()}@depth${levels}@100ms`;
};

