import React from 'react';
import type { AssetData } from '../types';
import { ConnectionStatus } from './ConnectionStatus';
import { formatBTCPrice, formatETHPrice, formatRatio, formatPercentChange, formatRelativeTime } from '../utils/formatters';
import { SYMBOLS } from '../constants';

interface PriceCardProps {
  assetData: AssetData;
}

export const PriceCard: React.FC<PriceCardProps> = ({ assetData }) => {
  const { symbol, currentPrice, change24h, connectionStatus, lastUpdate } = assetData;

  // Format price based on asset type
  const formatPrice = (price: number, symbol: string): string => {
    if (symbol === SYMBOLS.BTC) {
      return formatBTCPrice(price);
    } else if (symbol === SYMBOLS.ETH) {
      return formatETHPrice(price);
    } else if (symbol === SYMBOLS.RATIO) {
      return formatRatio(price);
    }
    return price.toString();
  };

  // Determine change color
  const changeColor = change24h >= 0 ? 'text-green-500' : 'text-red-500';
  const changeIcon = change24h >= 0 ? '↑' : '↓';

  return (
    <div className="bg-slate-800 rounded-lg p-6 space-y-3">
      {/* Header with symbol and status */}
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold text-white">{symbol}</h2>
        <ConnectionStatus status={connectionStatus} />
      </div>

      {/* Current Price */}
      <div className="space-y-1">
        <div className="text-3xl font-bold text-white">
          {currentPrice > 0 ? formatPrice(currentPrice, symbol) : '---'}
        </div>

        {/* 24h Change */}
        <div className={`flex items-center gap-1 text-sm font-medium ${changeColor}`}>
          <span>{changeIcon}</span>
          <span>{formatPercentChange(change24h)}</span>
          <span className="text-gray-500 text-xs ml-1">24h</span>
        </div>
      </div>

      {/* Last Update */}
      {lastUpdate > 0 && (
        <div className="text-xs text-gray-500">
          Updated {formatRelativeTime(lastUpdate)}
        </div>
      )}
    </div>
  );
};
