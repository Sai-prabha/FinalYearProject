import { DECIMALS } from '../constants';

// Format price with appropriate decimals and comma separators
export const formatPrice = (price: number, decimals: number): string => {
  return price.toLocaleString('en-US', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
};

// Format BTC price
export const formatBTCPrice = (price: number): string => {
  return `$${formatPrice(price, DECIMALS.BTC)}`;
};

// Format ETH price
export const formatETHPrice = (price: number): string => {
  return `$${formatPrice(price, DECIMALS.ETH)}`;
};

// Format BTC/ETH ratio
export const formatRatio = (ratio: number): string => {
  return formatPrice(ratio, DECIMALS.RATIO);
};

// Format percentage change
export const formatPercentChange = (change: number): string => {
  const sign = change >= 0 ? '+' : '';
  return `${sign}${change.toFixed(2)}%`;
};

// Format timestamp to readable time
export const formatTime = (timestamp: number): string => {
  const date = new Date(timestamp);
  return date.toLocaleTimeString('en-US', {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  });
};

// Format timestamp to relative time (e.g., "5 seconds ago")
export const formatRelativeTime = (timestamp: number): string => {
  const now = Date.now();
  const diffSeconds = Math.floor((now - timestamp) / 1000);

  if (diffSeconds < 60) {
    return `${diffSeconds}s ago`;
  }

  const diffMinutes = Math.floor(diffSeconds / 60);
  if (diffMinutes < 60) {
    return `${diffMinutes}m ago`;
  }

  const diffHours = Math.floor(diffMinutes / 60);
  return `${diffHours}h ago`;
};
