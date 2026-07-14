/**
 * Client for the shared auto-execute control plane (/execution/control).
 *
 * Explicit SET semantics only — the desired final value plus the version we
 * rendered goes to the backend, which rejects stale writes with 409 instead
 * of silently racing another surface. See EXECUTION_CONTROL.md.
 */
import { MODEL_SERVER_REST_URL } from '../constants';
import { getAuthHeaders } from '../utils/auth';
import type { ExecutionControlState } from '../types';

export class ControlConflictError extends Error {
  current: ExecutionControlState;
  constructor(current: ExecutionControlState) {
    super('state changed elsewhere');
    this.current = current;
  }
}

/** 404 from a backend that predates the control plane. */
export class ControlUnsupportedError extends Error {}

export function buildControlPatch(
  control: ExecutionControlState,
  desired: boolean,
): { auto_execute: boolean; expected_version: number } {
  return { auto_execute: desired, expected_version: control.version };
}

/** The Vercel deployment is canonical only when pointed at Railway. */
export function isCanonicalBackend(url: string = MODEL_SERVER_REST_URL): boolean {
  try {
    return new URL(url).hostname.endsWith('.railway.app');
  } catch {
    return false;
  }
}

async function controlRequest(init?: RequestInit): Promise<ExecutionControlState> {
  let resp: Response;
  try {
    resp = await fetch(`${MODEL_SERVER_REST_URL}/execution/control`, {
      ...init,
      headers: {
        ...getAuthHeaders(),
        'X-Control-Surface': 'vercel',
        ...(init?.body ? { 'Content-Type': 'application/json' } : {}),
      },
    });
  } catch (e) {
    throw new Error(`Network error: ${e instanceof Error ? e.message : e}`);
  }
  if (resp.status === 404) throw new ControlUnsupportedError();
  if (resp.status === 409) {
    const body = await resp.json().catch(() => null);
    const current = body?.detail?.current as ExecutionControlState | undefined;
    if (current) throw new ControlConflictError(current);
    throw new Error('Conflict: state changed elsewhere');
  }
  if (!resp.ok) {
    const body = await resp.json().catch(() => null);
    const detail = body?.detail;
    const msg = typeof detail === 'string' ? detail : detail?.message ?? `HTTP ${resp.status}`;
    if (resp.status === 401) throw new Error('Session expired — please log in again');
    throw new Error(msg);
  }
  return (await resp.json()) as ExecutionControlState;
}

export function fetchExecutionControl(): Promise<ExecutionControlState> {
  return controlRequest();
}

export function patchExecutionControl(
  control: ExecutionControlState,
  desired: boolean,
): Promise<ExecutionControlState> {
  return controlRequest({
    method: 'PATCH',
    body: JSON.stringify(buildControlPatch(control, desired)),
  });
}
