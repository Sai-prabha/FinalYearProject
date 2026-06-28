/**
 * Authenticated REST helpers for the broker control panel.
 *
 * Each helper hits the FastAPI model server, attaches the JWT via
 * `getAuthHeaders()`, and parses the response into the strongly typed
 * payloads declared in `../types`. Errors surface as plain `Error`s with
 * the server's `detail` message when available.
 */
import { MODEL_SERVER_REST_URL } from '../constants';
import { getAuthHeaders } from '../utils/auth';
import type {
  BrokerBalanceResponse,
  BrokerConfigSummary,
  BrokerPositionsResponse,
  OrderRequestPayload,
  OrderResponsePayload,
} from '../types';

async function request<T>(
  path: string,
  init: RequestInit = {},
): Promise<T> {
  const headers: Record<string, string> = {
    ...getAuthHeaders(),
    ...(init.headers as Record<string, string> | undefined),
  };
  if (init.body !== undefined) {
    headers['Content-Type'] = headers['Content-Type'] ?? 'application/json';
  }

  let resp: Response;
  try {
    resp = await fetch(`${MODEL_SERVER_REST_URL}${path}`, { ...init, headers });
  } catch (e) {
    const msg = e instanceof Error ? e.message : 'Network error';
    throw new Error(`Network error: ${msg}`);
  }

  if (!resp.ok) {
    let detail = `HTTP ${resp.status}`;
    try {
      const body = await resp.json();
      detail = body?.detail ?? body?.message ?? detail;
    } catch {
      /* swallow */
    }
    if (resp.status === 401) throw new Error('Session expired — please log in again');
    throw new Error(detail);
  }

  return (await resp.json()) as T;
}

export function fetchBrokerConfig(): Promise<BrokerConfigSummary> {
  return request<BrokerConfigSummary>('/broker/config');
}

export function updateBrokerConfig(
  partial: Partial<Pick<BrokerConfigSummary, 'auto_execute' | 'default_symbol' | 'default_qty' | 'default_btc_qty' | 'default_eth_qty'>>,
): Promise<BrokerConfigSummary> {
  return request<BrokerConfigSummary>('/broker/config', {
    method: 'POST',
    body: JSON.stringify(partial),
  });
}

export function fetchBrokerBalance(): Promise<BrokerBalanceResponse> {
  return request<BrokerBalanceResponse>('/broker/balance');
}

export function fetchBrokerPositions(): Promise<BrokerPositionsResponse> {
  return request<BrokerPositionsResponse>('/broker/positions');
}

export function placeTestOrder(payload: OrderRequestPayload): Promise<OrderResponsePayload> {
  return request<OrderResponsePayload>('/trade/test', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

export function placeOrder(payload: OrderRequestPayload): Promise<OrderResponsePayload> {
  return request<OrderResponsePayload>('/trade', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

export function cancelOrder(orderId: string, symbol: string): Promise<{ ok: boolean }> {
  const url = `/broker/order/${encodeURIComponent(orderId)}?symbol=${encodeURIComponent(symbol)}`;
  return request<{ ok: boolean }>(url, { method: 'DELETE' });
}
