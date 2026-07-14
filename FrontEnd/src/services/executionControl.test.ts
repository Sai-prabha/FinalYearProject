/**
 * Contract tests for the shared auto-execute control client:
 * explicit SET (never toggle), version precondition, conflict handling,
 * canonical-backend detection.
 *
 * Run: npm test
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  ControlConflictError,
  ControlUnsupportedError,
  buildControlPatch,
  fetchExecutionControl,
  isCanonicalBackend,
  patchExecutionControl,
} from './executionControl';
import type { ExecutionControlState } from '../types';

const state = (over: Partial<ExecutionControlState> = {}): ExecutionControlState => ({
  auto_execute: false,
  version: 3,
  updated_at: null,
  updated_by: null,
  updated_via: null,
  request_id: null,
  writer: { backend: 'railway', instance_id: 'r-1', is_writer: true, role: 'writer' },
  mode: 'demo',
  ...over,
});

const jsonResponse = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });

afterEach(() => vi.unstubAllGlobals());

describe('buildControlPatch — explicit SET semantics', () => {
  it('carries the desired final value and the rendered version, never a toggle', () => {
    expect(buildControlPatch(state(), true)).toEqual({ auto_execute: true, expected_version: 3 });
    expect(buildControlPatch(state({ auto_execute: true, version: 9 }), false)).toEqual({
      auto_execute: false,
      expected_version: 9,
    });
  });
});

describe('isCanonicalBackend', () => {
  it('only Railway hosts are canonical', () => {
    expect(isCanonicalBackend('https://foo-production.up.railway.app')).toBe(true);
    expect(isCanonicalBackend('http://localhost:8888')).toBe(false);
    expect(isCanonicalBackend('not a url')).toBe(false);
  });
});

describe('patchExecutionControl — wire contract', () => {
  it('PATCHes the explicit desired state with surface header and version', async () => {
    const fetchSpy = vi.fn(async () => jsonResponse(state({ auto_execute: true, version: 4 })));
    vi.stubGlobal('fetch', fetchSpy);
    vi.stubGlobal('localStorage', { getItem: () => null });

    const next = await patchExecutionControl(state(), true);
    expect(next.auto_execute).toBe(true);
    expect(next.version).toBe(4);

    const [url, init] = fetchSpy.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toMatch(/\/execution\/control$/);
    expect(init.method).toBe('PATCH');
    expect(JSON.parse(init.body as string)).toEqual({ auto_execute: true, expected_version: 3 });
    expect((init.headers as Record<string, string>)['X-Control-Surface']).toBe('vercel');
  });

  it('409 → ControlConflictError carrying the authoritative current state', async () => {
    const current = state({ auto_execute: true, version: 5 });
    vi.stubGlobal('fetch', vi.fn(async () =>
      jsonResponse({ detail: { message: 'stale version', current } }, 409),
    ));
    vi.stubGlobal('localStorage', { getItem: () => null });

    const err = await patchExecutionControl(state(), false).catch((e) => e);
    expect(err).toBeInstanceOf(ControlConflictError);
    expect((err as ControlConflictError).current).toEqual(current);
  });

  it('404 → ControlUnsupportedError (backend predates the control plane)', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response('not found', { status: 404 })));
    vi.stubGlobal('localStorage', { getItem: () => null });
    await expect(fetchExecutionControl()).rejects.toBeInstanceOf(ControlUnsupportedError);
  });

  it('GET parses the authoritative state', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => jsonResponse(state({ auto_execute: true }))));
    vi.stubGlobal('localStorage', { getItem: () => null });
    const out = await fetchExecutionControl();
    expect(out.auto_execute).toBe(true);
    expect(out.writer.role).toBe('writer');
  });
});
