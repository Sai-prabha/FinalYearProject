import { useCallback, useEffect, useState } from 'react';
import { fetchBrokerBalance } from '../services/brokerApi';
import type { BrokerBalanceResponse } from '../types';

interface UseBrokerBalanceResult {
  balance: BrokerBalanceResponse | null;
  loading: boolean;
  error: string | null;
  refresh: () => Promise<void>;
}

export function useBrokerBalance(): UseBrokerBalanceResult {
  const [balance, setBalance] = useState<BrokerBalanceResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      setBalance(await fetchBrokerBalance());
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { refresh(); }, [refresh]);

  return { balance, loading, error, refresh };
}
