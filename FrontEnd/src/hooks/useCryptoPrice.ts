import { useEffect, useState, useRef, useCallback } from 'react';
import { WebSocketService } from '../services/webSocket';
import { DataProcessor } from '../services/dataProcessor';
import type { AssetData, CandleData, ConnectionStatus } from '../types';
import { CONFIG } from '../constants';

export const useCryptoPrice = (wsUrl: string, symbol: string) => {
  const [assetData, setAssetData] = useState<AssetData>({
    symbol,
    currentPrice: 0,
    change24h: 0,
    priceHistory: [],
    connectionStatus: 'connecting',
    lastUpdate: 0,
  });

  const wsServiceRef = useRef<WebSocketService | null>(null);
  const dataProcessorRef = useRef<DataProcessor>(new DataProcessor());
  const lastUpdateRef = useRef<number>(0);

  const handleData = useCallback((candle: CandleData) => {
    const processor = dataProcessorRef.current;
    const updatedHistory = processor.addCandle(candle);
    
    const now = Date.now();
    if (now - lastUpdateRef.current < CONFIG.UPDATE_THROTTLE) {
      return;
    }
    lastUpdateRef.current = now;

    const currentPrice = processor.getCurrentPrice();
    const change24h = processor.calculate24hChange();
    const lastUpdate = processor.getLastUpdateTime();

    setAssetData(prev => ({
      ...prev,
      currentPrice,
      change24h,
      priceHistory: updatedHistory,
      lastUpdate,
    }));
  }, []);

  const handleStatus = useCallback((status: ConnectionStatus) => {
    setAssetData(prev => ({
      ...prev,
      connectionStatus: status,
    }));
  }, []);

  useEffect(() => {
    wsServiceRef.current = new WebSocketService(wsUrl);
    wsServiceRef.current.connect(handleData, handleStatus);

    return () => {
      if (wsServiceRef.current) {
        wsServiceRef.current.disconnect();
      }
    };
  }, [wsUrl, handleData, handleStatus]);

  return assetData;
};
