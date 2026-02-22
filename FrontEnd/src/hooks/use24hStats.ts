import { useState, useEffect, useRef, useCallback } from 'react';
import { fetch24hTicker } from '../services/api';

interface TickerStats {
  priceChange: number;
  priceChangePercent: number;
  highPrice: number;
  lowPrice: number;
  volume: number;
  quoteVolume: number;
}

interface Use24hStatsReturn {
  btcStats: TickerStats | null;
  ethStats: TickerStats | null;
  ratioStats: {
    change24hPct: number;
    high24h: number;
    low24h: number;
    volume24h: number;
  } | null;
}

/**
 * Hook that fetches real 24h ticker stats from the REST API
 * for BTC and ETH, and derives ratio stats from them.
 * Refreshes every 60 seconds.
 */
export const use24hStats = (): Use24hStatsReturn => {
  const [btcStats, setBtcStats] = useState<TickerStats | null>(null);
  const [ethStats, setEthStats] = useState<TickerStats | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const fetchStats = useCallback(async () => {
    const [btc, eth] = await Promise.all([
      fetch24hTicker('BTCUSDT'),
      fetch24hTicker('ETHUSDT'),
    ]);

    if (btc) setBtcStats(btc);
    if (eth) setEthStats(eth);
  }, []);

  useEffect(() => {
    fetchStats();
    intervalRef.current = setInterval(fetchStats, 60_000);
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [fetchStats]);

  // Derive ratio stats from BTC and ETH
  const ratioStats = btcStats && ethStats ? (() => {
    // Approximate ratio 24h range from BTC/ETH high/low combinations
    const ratioHigh = ethStats.lowPrice > 0 ? btcStats.highPrice / ethStats.lowPrice : 0;
    const ratioLow = ethStats.highPrice > 0 ? btcStats.lowPrice / ethStats.highPrice : 0;
    
    // Derive change from BTC and ETH individual changes
    const btcChangePct = btcStats.priceChangePercent;
    const ethChangePct = ethStats.priceChangePercent;
    // Approximate: ratio_change ≈ btc_change - eth_change (for small changes)
    const ratioChangePct = btcChangePct - ethChangePct;

    const volume = Math.min(btcStats.volume, ethStats.volume);

    return {
      change24hPct: ratioChangePct,
      high24h: ratioHigh,
      low24h: ratioLow,
      volume24h: volume,
    };
  })() : null;

  return { btcStats, ethStats, ratioStats };
};

