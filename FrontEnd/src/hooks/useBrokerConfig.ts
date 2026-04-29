import { useCallback, useEffect, useState } from 'react';
import { fetchBrokerConfig, updateBrokerConfig } from '../services/brokerApi';
import type { BrokerConfigSummary } from '../types';

interface UseBrokerConfigResult {
  config: BrokerConfigSummary | null;
  loading: boolean;
  error: string | null;
  update: (
    partial: Partial<Pick<BrokerConfigSummary, 'auto_execute' | 'default_symbol' | 'default_qty'>>,
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

  // WebSocket → live nudges. Only overwrites when the server actually
  // includes a broker block, so a missing field on older server builds
  // doesn't blank the panel.
  useEffect(() => {
    if (signalDataBroker) setConfig(signalDataBroker);
  }, [signalDataBroker]);

  const update = useCallback<UseBrokerConfigResult['update']>(async (partial) => {
    const previous = config;
    // Optimistic update for the cheap toggle path; form submits roll back on error.
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
