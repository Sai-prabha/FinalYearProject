import { useState, useEffect, useRef, useCallback } from 'react';
import { AUTH_REQUIRED, MODEL_SERVER_URL, CONFIG } from '../constants';
import { getAuthToken } from '../utils/auth';
import type { ModelSignalData, ConnectionStatus } from '../types';

interface UseModelSignalsReturn {
  signalData: ModelSignalData | null;
  connectionStatus: ConnectionStatus;
  error: string | null;
}

/**
 * Hook to connect to the model server and receive real-time trading signals.
 */
export const useModelSignals = (): UseModelSignalsReturn => {
  const [signalData, setSignalData] = useState<ModelSignalData | null>(null);
  const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>('connecting');
  const [error, setError] = useState<string | null>(null);
  
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectAttemptsRef = useRef(0);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const isIntentionalCloseRef = useRef(false);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return;
    }

    try {
      setConnectionStatus('connecting');
      setError(null);

      let wsUrl = MODEL_SERVER_URL;
      if (AUTH_REQUIRED) {
        const token = getAuthToken();
        if (token) {
          const sep = wsUrl.includes('?') ? '&' : '?';
          wsUrl = `${wsUrl}${sep}token=${encodeURIComponent(token)}`;
        }
      }
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        console.log('Connected to model server');
        setConnectionStatus('connected');
        reconnectAttemptsRef.current = 0;
        setError(null);

        // Send initial ping
        ws.send('get_latest');
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          
          // Handle pong responses
          if (data.type === 'pong') {
            return;
          }

          // Only full signal snapshots may replace signalData — event-style
          // broadcasts (e.g. type "exec_reconciled" after an auto-execute
          // reconciliation) have no `signal` and would crash consumers.
          if (!data.signal) {
            return;
          }

          setSignalData(data as ModelSignalData);
          
          // Log significant signals
          if (data.signal?.triggered) {
            console.log(`🚨 Signal: ${data.signal.direction} at ratio ${data.ratio.toFixed(4)}`);
          }
        } catch (err) {
          console.error('Error parsing signal data:', err);
        }
      };

      ws.onerror = (event) => {
        console.error('Model server WebSocket error:', event);
        setError('Connection error');
        setConnectionStatus('error');
      };

      ws.onclose = () => {
        console.log('Model server WebSocket closed');
        wsRef.current = null;

        if (!isIntentionalCloseRef.current) {
          setConnectionStatus('disconnected');
          handleReconnect();
        }
      };
    } catch (err) {
      console.error('Failed to connect to model server:', err);
      setError('Failed to connect');
      setConnectionStatus('error');
      handleReconnect();
    }
  }, []);

  const handleReconnect = useCallback(() => {
    if (reconnectAttemptsRef.current >= CONFIG.RECONNECT_MAX_ATTEMPTS) {
      console.error('Max reconnection attempts reached for model server');
      setError('Connection failed');
      setConnectionStatus('error');
      return;
    }

    const delay = Math.min(
      CONFIG.RECONNECT_BASE_DELAY * Math.pow(2, reconnectAttemptsRef.current),
      CONFIG.RECONNECT_MAX_DELAY
    );

    console.log(
      `Reconnecting to model server in ${delay}ms (attempt ${reconnectAttemptsRef.current + 1}/${CONFIG.RECONNECT_MAX_ATTEMPTS})`
    );

    reconnectTimerRef.current = setTimeout(() => {
      reconnectAttemptsRef.current++;
      connect();
    }, delay);
  }, [connect]);

  const disconnect = useCallback(() => {
    isIntentionalCloseRef.current = true;

    if (reconnectTimerRef.current) {
      clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }

    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    reconnectAttemptsRef.current = 0;
  }, []);

  // Connect on mount
  useEffect(() => {
    isIntentionalCloseRef.current = false;
    connect();

    // Heartbeat interval - send ping every 30 seconds
    const heartbeatInterval = setInterval(() => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send('ping');
      }
    }, 30000);

    // Cleanup on unmount
    return () => {
      clearInterval(heartbeatInterval);
      disconnect();
    };
  }, [connect, disconnect]);

  return {
    signalData,
    connectionStatus,
    error,
  };
};
