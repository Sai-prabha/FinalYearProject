import { useCallback, useEffect, useRef, useState } from 'react';
import { fetchBrokerPositions } from '../services/brokerApi';
import type { BrokerPositionsResponse } from '../types';

interface UseBrokerPositionsResult {
  positions: BrokerPositionsResponse | null;
  loading: boolean;
  error: string | null;
  refresh: () => Promise<void>;
}

const POLL_MS = 15000;

export function useBrokerPositions(): UseBrokerPositionsResult {
  const [positions, setPositions] = useState<BrokerPositionsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const aliveRef = useRef(true);

  const refresh = useCallback(async () => {
    try {
      const next = await fetchBrokerPositions();
      if (aliveRef.current) {
        setPositions(next);
        setError(null);
      }
    } catch (e) {
      if (aliveRef.current) setError(e instanceof Error ? e.message : String(e));
    } finally {
      if (aliveRef.current) setLoading(false);
    }
  }, []);

  useEffect(() => {
    aliveRef.current = true;
    refresh();
    const id = window.setInterval(refresh, POLL_MS);
    return () => {
      aliveRef.current = false;
      window.clearInterval(id);
    };
  }, [refresh]);

  return { positions, loading, error, refresh };
}
