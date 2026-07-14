import { useCallback, useEffect, useRef, useState } from 'react';
import {
  ControlConflictError,
  ControlUnsupportedError,
  fetchExecutionControl,
  patchExecutionControl,
} from '../services/executionControl';
import type { ExecutionControlState } from '../types';

interface UseExecutionControlResult {
  control: ExecutionControlState | null;
  /** false when the backend predates the control plane (404) */
  supported: boolean;
  loading: boolean;
  pending: boolean;
  notice: string | null;
  /** Explicit SET of the shared state. Never a toggle, never optimistic —
   * the switch renders backend truth only. */
  set: (desired: boolean) => Promise<void>;
}

const NOTICE_MS = 6000;

export function useExecutionControl(pollMs = 5000): UseExecutionControlResult {
  const [control, setControl] = useState<ExecutionControlState | null>(null);
  const [supported, setSupported] = useState(true);
  const [loading, setLoading] = useState(true);
  const [pending, setPending] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const noticeTimer = useRef<number | undefined>(undefined);

  const flash = useCallback((msg: string) => {
    setNotice(msg);
    window.clearTimeout(noticeTimer.current);
    noticeTimer.current = window.setTimeout(() => setNotice(null), NOTICE_MS);
  }, []);

  useEffect(() => {
    let alive = true;
    let timer: number | undefined;
    const load = async () => {
      try {
        const next = await fetchExecutionControl();
        if (alive) setControl(next);
      } catch (e) {
        if (alive && e instanceof ControlUnsupportedError) {
          setSupported(false);
          window.clearInterval(timer);
        }
        // transient fetch errors: keep showing the last known backend truth
      } finally {
        if (alive) setLoading(false);
      }
    };
    void load();
    timer = window.setInterval(() => void load(), pollMs);
    return () => {
      alive = false;
      window.clearInterval(timer);
      window.clearTimeout(noticeTimer.current);
    };
  }, [pollMs]);

  const set = useCallback(async (desired: boolean) => {
    if (!control) return;
    setPending(true);
    try {
      const next = await patchExecutionControl(control, desired);
      setControl(next); // refresh from backend truth, never assume success
    } catch (e) {
      if (e instanceof ControlConflictError) {
        setControl(e.current);
        flash('State changed elsewhere — refreshed to latest value');
      } else {
        flash(e instanceof Error ? e.message : String(e));
      }
    } finally {
      setPending(false);
    }
  }, [control, flash]);

  return { control, supported, loading, pending, notice, set };
}
