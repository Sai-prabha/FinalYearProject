import { useCallback, useEffect, useState } from 'react';
import { fetchBrokerConfig, updateBrokerConfig } from '../services/brokerApi';
import type { BrokerConfigSummary } from '../types';

interface UseBrokerConfigResult {
  config: BrokerConfigSummary | null;
  loading: boolean;
  error: string | null;
  update: (
    partial: Partial<Pick<BrokerConfigSummary, 'auto_execute' | 'default_symbol' | 'default_qty' | 'default_btc_qty' | 'default_eth_qty'>>,
  ) => Promise<{ ok: true; config: BrokerConfigSummary } | { ok: false; error: string }>;
  refresh: () => Promise<void>;
}

/**
 * Source-of-truth: the backend. Fetches once on mount, then merges live
 * pushes from the WebSocket payload (`signalDataBroker`) so the UI stays
 * current without polling.
 */
export function useBrokerConfig(signalDataBroker?: BrokerConfigSummary): UseBrokerConfigResult {
  const [config, setConfig] = useState<BrokerConfigSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const next = await fetchBrokerConfig();
      setConfig(next);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  useEffect(() => {
    let alive = true;
    fetchBrokerConfig()
      .then(c => { if (alive) setConfig(c); })
      .catch(e => { if (alive) setError(e instanceof Error ? e.message : String(e)); })
      .finally(() => { if (alive) setLoading(false); });
    return () => { alive = false; };
  }, []);

  // WebSocket → live nudges. Only overwrites when values actually changed so
  // re-renders aren't triggered on every WS message (broker object is a new
  // reference each time even when content is identical).
  useEffect(() => {
    if (!signalDataBroker) return;
    setConfig(prev => {
      if (
        prev &&
        prev.mode === signalDataBroker.mode &&
        prev.auto_execute === signalDataBroker.auto_execute &&
        prev.default_symbol === signalDataBroker.default_symbol &&
        prev.default_qty === signalDataBroker.default_qty &&
        prev.default_btc_qty === signalDataBroker.default_btc_qty &&
        prev.default_eth_qty === signalDataBroker.default_eth_qty
      ) {
        return prev;
      }
      return signalDataBroker;
    });
  }, [signalDataBroker]);

  const update = useCallback<UseBrokerConfigResult['update']>(async (partial) => {
    const previous = config;
    // Optimistic update for toggle path; form submits roll back on error.
    if (config && 'auto_execute' in partial) {
      setConfig({ ...config, ...partial });
    }
    try {
      const next = await updateBrokerConfig(partial);
      setConfig(next);
      setError(null);
      return { ok: true, config: next };
    } catch (e) {
      if (previous) setConfig(previous);
      const msg = e instanceof Error ? e.message : String(e);
      return { ok: false, error: msg };
    }
  }, [config]);

  return { config, loading, error, update, refresh };
}
