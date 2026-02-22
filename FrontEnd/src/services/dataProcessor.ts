import type { CandleData, PriceData } from '../types';
import { CONFIG } from '../constants';

export class DataProcessor {
  private buffer: PriceData[] = [];
  private prices24hAgo: number[] = [];
  private last24hTimestamp: number = 0;

  constructor() {
    this.buffer = [];
    this.prices24hAgo = [];
  }

  // Add new candle data to buffer
  addCandle(candle: CandleData): PriceData[] {
    const pricePoint: PriceData = {
      time: candle.time,
      value: candle.close,
    };

    // Check if we should update the last entry or add a new one
    if (this.buffer.length > 0) {
      const lastEntry = this.buffer[this.buffer.length - 1];
      
      if (lastEntry.time === pricePoint.time) {
        // Update existing entry
        this.buffer[this.buffer.length - 1] = pricePoint;
      } else {
        // Add new entry
        this.buffer.push(pricePoint);
      }
    } else {
      // First entry
      this.buffer.push(pricePoint);
    }

    // Trim buffer if exceeds max size
    if (this.buffer.length > CONFIG.MAX_DATA_POINTS) {
      this.buffer = this.buffer.slice(this.buffer.length - CONFIG.MAX_DATA_POINTS);
    }

    // Update 24h price tracking
    this.update24hPrices(pricePoint);

    return [...this.buffer];
  }

  // Update 24h price tracking for change calculation
  private update24hPrices(pricePoint: PriceData): void {
    const now = pricePoint.time * 1000; // Convert to ms

    // Initialize if first time
    if (this.last24hTimestamp === 0) {
      this.last24hTimestamp = now;
      this.prices24hAgo.push(pricePoint.value);
      return;
    }

    // Add current price to tracking
    this.prices24hAgo.push(pricePoint.value);

    // Remove prices older than 24h
    this.prices24hAgo = this.prices24hAgo.slice(-1440); // Keep max 1440 minutes (24h of 1m candles)
  }

  // Calculate 24h change percentage
  calculate24hChange(): number {
    if (this.buffer.length < 2 || this.prices24hAgo.length === 0) {
      return 0;
    }

    const currentPrice = this.buffer[this.buffer.length - 1].value;
    const oldPrice = this.prices24hAgo[0];

    if (oldPrice === 0) {
      return 0;
    }

    return ((currentPrice - oldPrice) / oldPrice) * 100;
  }

  // Calculate volatility (standard deviation of last N prices)
  calculateVolatility(): number {
    const window = Math.min(CONFIG.VOLATILITY_WINDOW, this.buffer.length);
    
    if (window < 2) {
      return 0;
    }

    const recentPrices = this.buffer.slice(-window).map(p => p.value);
    
    // Calculate mean
    const mean = recentPrices.reduce((sum, price) => sum + price, 0) / window;
    
    // Calculate variance
    const variance = recentPrices.reduce((sum, price) => {
      const diff = price - mean;
      return sum + diff * diff;
    }, 0) / window;
    
    // Return standard deviation
    return Math.sqrt(variance);
  }

  // Get current price
  getCurrentPrice(): number {
    if (this.buffer.length === 0) {
      return 0;
    }
    return this.buffer[this.buffer.length - 1].value;
  }

  // Get all buffered data
  getBuffer(): PriceData[] {
    return [...this.buffer];
  }

  // Get last update timestamp
  getLastUpdateTime(): number {
    if (this.buffer.length === 0) {
      return 0;
    }
    return this.buffer[this.buffer.length - 1].time * 1000; // Convert to ms
  }

  // Clear all data
  clear(): void {
    this.buffer = [];
    this.prices24hAgo = [];
    this.last24hTimestamp = 0;
  }
}

// Calculate BTC/ETH ratio from two prices
export const calculateRatio = (btcPrice: number, ethPrice: number): number => {
  if (ethPrice === 0) {
    return 0;
  }
  return btcPrice / ethPrice;
};

// Create ratio price data from two price histories
export const createRatioHistory = (btcHistory: PriceData[], ethHistory: PriceData[]): PriceData[] => {
  const ratioHistory: PriceData[] = [];
  
  // Create a map of ETH prices by timestamp for quick lookup
  const ethPriceMap = new Map<number, number>();
  ethHistory.forEach(point => {
    ethPriceMap.set(point.time, point.value);
  });

  // Calculate ratio for each BTC timestamp where we have ETH data
  btcHistory.forEach(btcPoint => {
    const ethPrice = ethPriceMap.get(btcPoint.time);
    if (ethPrice !== undefined && ethPrice !== 0) {
      ratioHistory.push({
        time: btcPoint.time,
        value: calculateRatio(btcPoint.value, ethPrice),
      });
    }
  });

  return ratioHistory;
};
