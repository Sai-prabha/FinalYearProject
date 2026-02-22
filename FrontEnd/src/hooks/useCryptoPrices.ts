import { useMemo } from 'react';
import { useCryptoPrice } from './useCryptoPrice';
import { WS_URLS, SYMBOLS } from '../constants';
import { createRatioHistory, calculateRatio } from '../services/dataProcessor';
import type { AssetData } from '../types';

// Combined hook that manages all three data streams
export const useCryptoPrices = () => {
  const btcData = useCryptoPrice(WS_URLS.BTC, SYMBOLS.BTC);
  const ethData = useCryptoPrice(WS_URLS.ETH, SYMBOLS.ETH);

  const ratioData: AssetData = useMemo(() => {
    const ratioHistory = createRatioHistory(btcData.priceHistory, ethData.priceHistory);
    const currentRatio = calculateRatio(btcData.currentPrice, ethData.currentPrice);
    
    let ratioChange24h = 0;
    if (ratioHistory.length >= 2) {
      const oldRatio = ratioHistory[0].value;
      if (oldRatio !== 0) {
        ratioChange24h = ((currentRatio - oldRatio) / oldRatio) * 100;
      }
    }

    let connectionStatus: 'connecting' | 'connected' | 'disconnected' | 'error' = 'connecting';
    if (btcData.connectionStatus === 'connected' && ethData.connectionStatus === 'connected') {
      connectionStatus = 'connected';
    } else if (btcData.connectionStatus === 'error' || ethData.connectionStatus === 'error') {
      connectionStatus = 'error';
    } else if (btcData.connectionStatus === 'disconnected' || ethData.connectionStatus === 'disconnected') {
      connectionStatus = 'disconnected';
    }

    return {
      symbol: SYMBOLS.RATIO,
      currentPrice: currentRatio,
      change24h: ratioChange24h,
      priceHistory: ratioHistory,
      connectionStatus,
      lastUpdate: Math.max(btcData.lastUpdate, ethData.lastUpdate),
    };
  }, [btcData, ethData]);

  return {
    btc: btcData,
    eth: ethData,
    ratio: ratioData,
  };
};
