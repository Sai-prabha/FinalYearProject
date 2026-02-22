import React, { createContext, useCallback, useContext, useEffect, useState } from 'react';
import { MODEL_SERVER_REST_URL, AUTH_REQUIRED } from '../constants';
import { clearAuthToken, getAuthToken, setAuthToken } from '../utils/auth';

interface AuthContextValue {
  isAuthenticated: boolean;
  user: string | null;
  isLoading: boolean;
  login: (username: string, password: string) => Promise<boolean>;
  logout: () => void;
}

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [user, setUser] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  const validateToken = useCallback(async () => {
    if (!AUTH_REQUIRED) {
      setIsAuthenticated(true);
      setUser('adm1nFYP');
      setIsLoading(false);
      return;
    }
    const token = getAuthToken();
    if (!token) {
      setIsAuthenticated(false);
      setUser(null);
      setIsLoading(false);
      return;
    }
    try {
      const resp = await fetch(`${MODEL_SERVER_REST_URL}/api/me`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (resp.ok) {
        const data = await resp.json();
        setIsAuthenticated(true);
        setUser(data.user ?? 'adm1nFYP');
      } else {
        clearAuthToken();
        setIsAuthenticated(false);
        setUser(null);
      }
    } catch {
      clearAuthToken();
      setIsAuthenticated(false);
      setUser(null);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    validateToken();
  }, [validateToken]);

  const login = useCallback(async (username: string, password: string): Promise<boolean> => {
    if (!AUTH_REQUIRED) {
      setIsAuthenticated(true);
      setUser(username);
      return true;
    }
    try {
      const resp = await fetch(`${MODEL_SERVER_REST_URL}/api/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password }),
      });
      if (!resp.ok) return false;
      const data = await resp.json();
      setAuthToken(data.token);
      setIsAuthenticated(true);
      setUser(data.user ?? username);
      return true;
    } catch {
      return false;
    }
  }, []);

  const logout = useCallback(() => {
    clearAuthToken();
    setIsAuthenticated(false);
    setUser(null);
    if (AUTH_REQUIRED) {
      fetch(`${MODEL_SERVER_REST_URL}/api/logout`, { method: 'POST' }).catch(() => {});
    }
  }, []);

  return (
    <AuthContext.Provider value={{ isAuthenticated, user, isLoading, login, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error('useAuth must be used within AuthProvider');
  return ctx;
}
