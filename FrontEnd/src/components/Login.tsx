import React, { useState } from 'react';
import { useAuth } from '../context/AuthContext';
import { THEME_COLORS } from '../constants';

export function Login() {
  const { login } = useAuth();
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setLoading(true);
    try {
      const ok = await login(username, password);
      if (!ok) setError('Invalid username or password');
    } catch {
      setError('Login failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div
      className="min-h-screen flex items-center justify-center p-4"
      style={{ backgroundColor: THEME_COLORS.BACKGROUND }}
    >
      <div
        className="w-full max-w-sm rounded-lg border p-6"
        style={{
          backgroundColor: THEME_COLORS.CARD_BG,
          borderColor: THEME_COLORS.BORDER,
        }}
      >
        <h1
          className="text-xl font-bold mb-6 text-center"
          style={{ color: THEME_COLORS.TEXT_PRIMARY }}
        >
          Crypto Dashboard
        </h1>
        <p
          className="text-sm mb-4 text-center"
          style={{ color: THEME_COLORS.TEXT_SECONDARY }}
        >
          Sign in to access the dashboard
        </p>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label
              htmlFor="username"
              className="block text-xs font-medium mb-1"
              style={{ color: THEME_COLORS.TEXT_SECONDARY }}
            >
              Username
            </label>
            <input
              id="username"
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              autoComplete="username"
              required
              className="w-full px-3 py-2 rounded border text-sm focus:outline-none focus:ring-2"
              style={{
                backgroundColor: THEME_COLORS.CARD_BG_LIGHT,
                borderColor: THEME_COLORS.BORDER,
                color: THEME_COLORS.TEXT_PRIMARY,
              }}
            />
          </div>
          <div>
            <label
              htmlFor="password"
              className="block text-xs font-medium mb-1"
              style={{ color: THEME_COLORS.TEXT_SECONDARY }}
            >
              Password
            </label>
            <input
              id="password"
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              autoComplete="current-password"
              required
              className="w-full px-3 py-2 rounded border text-sm focus:outline-none focus:ring-2"
              style={{
                backgroundColor: THEME_COLORS.CARD_BG_LIGHT,
                borderColor: THEME_COLORS.BORDER,
                color: THEME_COLORS.TEXT_PRIMARY,
              }}
            />
          </div>
          {error && (
            <p className="text-sm" style={{ color: THEME_COLORS.NEGATIVE }}>
              {error}
            </p>
          )}
          <button
            type="submit"
            disabled={loading}
            className="w-full py-2 rounded font-medium text-sm disabled:opacity-50"
            style={{
              backgroundColor: THEME_COLORS.YELLOW,
              color: THEME_COLORS.BACKGROUND,
            }}
          >
            {loading ? 'Signing in...' : 'Sign in'}
          </button>
        </form>
      </div>
    </div>
  );
}
