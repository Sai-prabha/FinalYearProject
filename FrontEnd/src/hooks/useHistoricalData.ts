import React, { useEffect, useState, useRef } from 'react';
import { fetchHistoricalCandles } from '../services/api';
import type { CandleData } from '../types';

interface UseHistoricalDataReturn {
  candles: CandleData[];
  isLoading: boolean;
  error: string | null;
  setCandles: React.Dispatch<React.SetStateAction<CandleData[]>>;
}

// Map UI timeframes to API intervals
const timeframeToInterval = (timeframe: string): string => {
  const map: Record<string, string> = {
    '1m': '1m',
    '5m': '5m',
    '15m': '15m',
    '1h': '1h',
    '4h': '4h',
    '1D': '1d',
    '1W': '1w'
  };
  return map[timeframe] || '1m';
};

// Calculate appropriate limit based on timeframe
const getLimitForTimeframe = (timeframe: string): number => {
  const limits: Record<string, number> = {
    '1m': 500,
    '5m': 500,
    '15m': 500,
    '1h': 500,
    '4h': 500,
    '1D': 365,
    '1W': 200
  };
  return limits[timeframe] || 500;
};

export const useHistoricalData = (
  symbol: string, 
  timeframe: string = '1m'
): UseHistoricalDataReturn => {
  const [candles, setCandles] = useState<CandleData[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const fetchedRef = useRef<string>('');

  useEffect(() => {
    const interval = timeframeToInterval(timeframe);
    const limit = getLimitForTimeframe(timeframe);
    const cacheKey = `${symbol}-${interval}`;

    if (fetchedRef.current !== cacheKey) {
      fetchedRef.current = '';
      setCandles([]);
      setIsLoading(true);
      setError(null);
    }

    const fetchData = async () => {
      if (fetchedRef.current === cacheKey) return;
      fetchedRef.current = cacheKey;

      try {
        console.log(`Fetching historical data for ${symbol} at ${interval}...`);
        const data = await fetchHistoricalCandles(symbol, interval, limit);
        
        if (data.length > 0) {
          console.log(`Loaded ${data.length} historical candles for ${symbol} at ${interval}`);
          setCandles(data);
        } else {
          setError('No historical data available');
        }
      } catch (err) {
        console.error(`Error fetching historical data for ${symbol}:`, err);
        setError('Failed to load historical data');
      } finally {
        setIsLoading(false);
      }
    };

    fetchData();
  }, [symbol, timeframe]);

  return { candles, isLoading, error, setCandles };
};
