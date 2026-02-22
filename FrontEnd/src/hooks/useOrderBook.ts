import { useEffect, useState, useRef, useCallback } from 'react';
import { parseOrderBookData, type OrderBookData, type OrderBookLevel } from '../services/api';

interface UseOrderBookReturn {
  bids: OrderBookLevel[];
  asks: OrderBookLevel[];
  isConnected: boolean;
}

export const useOrderBook = (symbol: string): UseOrderBookReturn => {
  const [orderBook, setOrderBook] = useState<OrderBookData>({
    bids: [],
    asks: [],
    lastUpdateId: 0,
  });
  const [isConnected, setIsConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);

  const handleMessage = useCallback((event: MessageEvent) => {
    try {
      const data = JSON.parse(event.data);
      
      if (data.bids && data.asks) {
        const parsed = parseOrderBookData(data);
        setOrderBook(parsed);
      }
    } catch (error) {
      console.error('Error parsing order book data:', error);
    }
  }, []);

  useEffect(() => {
    const wsUrl = `wss://stream.binance.com:9443/ws/${symbol.toLowerCase()}@depth20@100ms`;
    
    const connect = () => {
      wsRef.current = new WebSocket(wsUrl);

      wsRef.current.onopen = () => {
        console.log(`Order book WebSocket connected for ${symbol}`);
        setIsConnected(true);
      };

      wsRef.current.onmessage = handleMessage;

      wsRef.current.onerror = (error) => {
        console.error('Order book WebSocket error:', error);
        setIsConnected(false);
      };

      wsRef.current.onclose = () => {
        console.log(`Order book WebSocket closed for ${symbol}`);
        setIsConnected(false);
        setTimeout(connect, 3000);
      };
    };

    connect();

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [symbol, handleMessage]);

  return {
    bids: orderBook.bids,
    asks: orderBook.asks,
    isConnected,
  };
};
