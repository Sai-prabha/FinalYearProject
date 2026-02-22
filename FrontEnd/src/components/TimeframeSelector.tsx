import React from 'react';
import { THEME_COLORS } from '../constants';

export type Timeframe = '1m' | '5m' | '15m' | '1h' | '4h' | '1D' | '1W';

interface TimeframeSelectorProps {
  selected: Timeframe;
  onSelect: (timeframe: Timeframe) => void;
}

const timeframes: Timeframe[] = ['1m', '5m', '15m', '1h', '4h', '1D', '1W'];

export const TimeframeSelector: React.FC<TimeframeSelectorProps> = ({ selected, onSelect }) => {
  return (
    <div className="flex gap-0.5">
      {timeframes.map(tf => (
        <button
          key={tf}
          onClick={() => onSelect(tf)}
          className="px-2.5 py-1 text-xs font-medium transition-colors rounded"
          style={{
            backgroundColor: selected === tf ? THEME_COLORS.CARD_BG_LIGHT : 'transparent',
            color: selected === tf ? THEME_COLORS.YELLOW : THEME_COLORS.TEXT_SECONDARY,
          }}
        >
          {tf}
        </button>
      ))}
    </div>
  );
};
