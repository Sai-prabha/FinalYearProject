import React from 'react';
import { THEME_COLORS } from '../constants';
import { formatPercentChange } from '../utils/formatters';

interface MarketStatsProps {
  symbol: string;
  currentPrice: number;
  change24h: number;
  high24h: number;
  low24h: number;
  volume24h: number;
  isRatio?: boolean;
}

export const MarketStats: React.FC<MarketStatsProps> = ({
  symbol,
  currentPrice,
  change24h,
  high24h,
  low24h,
  volume24h,
  isRatio = false,
}) => {
  const changeColor = change24h >= 0 ? THEME_COLORS.POSITIVE : THEME_COLORS.NEGATIVE;
  
  const formatPrice = (price: number): string => {
    if (isRatio) {
      return price.toFixed(4);
    }
    return `$${price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
  };

  return (
    <div 
      className="rounded-lg px-6 py-4 flex items-center gap-8"
      style={{ backgroundColor: THEME_COLORS.CARD_BG }}
    >
      {/* Symbol and Price */}
      <div className="flex items-center gap-4">
        <div>
          <div className="text-[#eaecef] font-bold text-lg">{symbol}</div>
          <div className="text-[#848e9c] text-xs">
            {isRatio ? 'Price Ratio' : 'Spot Price'}
          </div>
        </div>
        <div className="text-2xl font-bold font-mono" style={{ color: THEME_COLORS.TEXT_PRIMARY }}>
          {currentPrice > 0 ? formatPrice(currentPrice) : '---'}
        </div>
      </div>

      <div className="h-10 w-px" style={{ backgroundColor: THEME_COLORS.BORDER }} />

      {/* 24h Change */}
      <div>
        <div className="text-[#848e9c] text-xs mb-1">24h Change</div>
        <div className="font-mono font-semibold" style={{ color: changeColor }}>
          {formatPercentChange(change24h)}
        </div>
      </div>

      <div className="h-10 w-px" style={{ backgroundColor: THEME_COLORS.BORDER }} />

      {/* 24h High */}
      <div>
        <div className="text-[#848e9c] text-xs mb-1">24h High</div>
        <div className="text-[#eaecef] font-mono">
          {high24h > 0 ? formatPrice(high24h) : '---'}
        </div>
      </div>

      {/* 24h Low */}
      <div>
        <div className="text-[#848e9c] text-xs mb-1">24h Low</div>
        <div className="text-[#eaecef] font-mono">
          {low24h > 0 ? formatPrice(low24h) : '---'}
        </div>
      </div>

      <div className="h-10 w-px" style={{ backgroundColor: THEME_COLORS.BORDER }} />

      {/* 24h Volume */}
      <div>
        <div className="text-[#848e9c] text-xs mb-1">24h Volume</div>
        <div className="text-[#eaecef] font-mono">
          {volume24h > 0 ? volume24h.toLocaleString(undefined, { maximumFractionDigits: 2 }) : '---'}
        </div>
      </div>
    </div>
  );
};
