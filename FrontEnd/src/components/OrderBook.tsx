import React, { useEffect, useRef, useState, useCallback } from 'react';
import { THEME_COLORS } from '../constants';

interface OrderBookLevel {
  price: number;
  amount: number;
}

interface OrderBookProps {
  symbol: string;
  bids: OrderBookLevel[];
  asks: OrderBookLevel[];
  decimals: number;
  currentPrice?: number;
}

interface FlashState {
  [key: string]: 'up' | 'down' | null;
}

type FilterMode = 'all' | 'bids' | 'asks';

interface HoverStats {
  avgPrice: number;
  sumAmount: number;
  sumUSDT: number;
}

// Approximate height of a single order book row in px
const ROW_HEIGHT = 20;

export const OrderBook: React.FC<OrderBookProps> = ({ 
  symbol, 
  bids, 
  asks, 
  decimals,
  currentPrice 
}) => {
  const [flashState, setFlashState] = useState<FlashState>({});
  const [filterMode, setFilterMode] = useState<FilterMode>('all');
  const [hoveredBidIndex, setHoveredBidIndex] = useState<number | null>(null);
  const [hoveredAskIndex, setHoveredAskIndex] = useState<number | null>(null);
  const [showBarTooltip, setShowBarTooltip] = useState(false);
  const [dynamicRowCount, setDynamicRowCount] = useState(16);
  const prevBidsRef = useRef<OrderBookLevel[]>([]);
  const prevAsksRef = useRef<OrderBookLevel[]>([]);
  const contentRef = useRef<HTMLDivElement>(null);

  // Measure available content height and compute dynamic row count
  const measureRows = useCallback(() => {
    if (!contentRef.current) return;
    const availableHeight = contentRef.current.clientHeight;
    // In 'all' mode: subtract ~36px for the spread/price row, then split between asks & bids
    const spreadHeight = 36;
    const usableHeight = availableHeight - spreadHeight;
    const rowsPerSide = Math.max(4, Math.floor(usableHeight / ROW_HEIGHT / 2));
    setDynamicRowCount(rowsPerSide);
  }, []);

  useEffect(() => {
    measureRows();
    const el = contentRef.current;
    if (!el) return;
    const observer = new ResizeObserver(() => measureRows());
    observer.observe(el);
    return () => observer.disconnect();
  }, [measureRows]);

  // Track changes and trigger flash animations
  useEffect(() => {
    const newFlashState: FlashState = {};

    // Check bid changes
    bids.forEach((bid, index) => {
      const key = `bid-${bid.price}`;
      const prevBid = prevBidsRef.current[index];
      
      if (prevBid && prevBid.price === bid.price) {
        if (bid.amount > prevBid.amount) {
          newFlashState[key] = 'up';
        } else if (bid.amount < prevBid.amount) {
          newFlashState[key] = 'down';
        }
      }
    });

    // Check ask changes
    asks.forEach((ask, index) => {
      const key = `ask-${ask.price}`;
      const prevAsk = prevAsksRef.current[index];
      
      if (prevAsk && prevAsk.price === ask.price) {
        if (ask.amount > prevAsk.amount) {
          newFlashState[key] = 'up';
        } else if (ask.amount < prevAsk.amount) {
          newFlashState[key] = 'down';
        }
      }
    });

    if (Object.keys(newFlashState).length > 0) {
      setFlashState(newFlashState);
      
      // Clear flash state after animation
      const timer = setTimeout(() => {
        setFlashState({});
      }, 300);
      
      return () => clearTimeout(timer);
    }

    prevBidsRef.current = bids;
    prevAsksRef.current = asks;
  }, [bids, asks]);

  // B-S percentage bar always tracks the first 20 levels
  const first20Bids = bids.slice(0, 20);
  const first20Asks = asks.slice(0, 20);
  const barBidTotal = first20Bids.reduce((sum, b) => sum + b.amount, 0);
  const barAskTotal = first20Asks.reduce((sum, a) => sum + a.amount, 0);

  // Calculate max total for depth visualization bars (uses displayed data)
  const maxBidTotal = bids.reduce((sum, b) => sum + b.amount, 0);
  const maxAskTotal = asks.reduce((sum, a) => sum + a.amount, 0);
  const maxTotal = Math.max(maxBidTotal, maxAskTotal);

  // Calculate cumulative stats for hovered row
  const calculateHoverStats = (levels: OrderBookLevel[], hoveredIndex: number): HoverStats => {
    let sumAmount = 0;
    let sumUSDT = 0;
    
    for (let i = 0; i <= hoveredIndex; i++) {
      if (levels[i]) {
        sumAmount += levels[i].amount;
        sumUSDT += levels[i].price * levels[i].amount;
      }
    }
    
    const avgPrice = sumAmount > 0 ? sumUSDT / sumAmount : 0;
    
    return { avgPrice, sumAmount, sumUSDT };
  };

  // Get asset name from symbol (BTC or ETH)
  const assetName = symbol.includes('BTC') ? 'BTC' : 'ETH';

  const renderAskRow = (level: OrderBookLevel, index: number, cumulative: number, originalIndex: number) => {
    const percentage = (cumulative / maxTotal) * 100;
    const flashKey = `ask-${level.price}`;
    const flash = flashState[flashKey];
    const isHovered = hoveredAskIndex === originalIndex;
    const stats = isHovered ? calculateHoverStats(asks, originalIndex) : null;
    
    return (
      <div 
        key={`ask-${index}`} 
        className="relative grid grid-cols-3 text-xs py-0.5 hover:bg-[#2b3139]/50 transition-colors duration-150 cursor-pointer"
        style={{
          backgroundColor: flash === 'up' ? 'rgba(14, 203, 129, 0.1)' : 
                          flash === 'down' ? 'rgba(246, 70, 93, 0.1)' : 
                          'transparent',
          transition: 'background-color 0.2s ease-out'
        }}
        onMouseEnter={() => setHoveredAskIndex(originalIndex)}
        onMouseLeave={() => setHoveredAskIndex(null)}
      >
        {filterMode === 'all' && (
          <div 
            className="absolute inset-0 bg-[#f6465d]/20 transition-all duration-200"
            style={{ width: `${percentage}%`, right: 0 }}
          />
        )}
        {filterMode !== 'all' && (
          <div 
            className="absolute inset-0 bg-[#f6465d]/20 transition-all duration-200"
            style={{ width: `${percentage}%`, left: 0 }}
          />
        )}
        <div className="relative text-[#f6465d] font-mono text-right pr-2">
          {level.price.toFixed(decimals)}
        </div>
        <div className="relative text-[#eaecef] text-right font-mono pr-2">
          {level.amount.toFixed(5)}
        </div>
        <div className="relative text-[#848e9c] text-right font-mono">
          {(level.price * level.amount).toFixed(2)}
        </div>
        
        {/* Hover Popup */}
        {isHovered && stats && (
          <div 
            className="absolute left-0 transform -translate-x-full ml-2 px-3 py-2 rounded shadow-lg z-50 whitespace-nowrap"
            style={{ 
              backgroundColor: THEME_COLORS.CARD_BG_LIGHT,
              border: `1px solid ${THEME_COLORS.BORDER}`,
              top: '50%',
              transform: 'translate(-100%, -50%)'
            }}
          >
            <div className="text-[10px] space-y-1">
              <div className="flex justify-between gap-4">
                <span className="text-[#848e9c]">Avg.Price:</span>
                <span className="text-[#eaecef] font-mono">≈ {stats.avgPrice.toFixed(decimals)}</span>
              </div>
              <div className="flex justify-between gap-4">
                <span className="text-[#848e9c]">Sum {assetName}:</span>
                <span className="text-[#eaecef] font-mono">{stats.sumAmount.toFixed(5)}</span>
              </div>
              <div className="flex justify-between gap-4">
                <span className="text-[#848e9c]">Sum USDT:</span>
                <span className="text-[#eaecef] font-mono">{stats.sumUSDT.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</span>
              </div>
            </div>
          </div>
        )}
      </div>
    );
  };

  const renderBidRow = (level: OrderBookLevel, index: number, cumulative: number, originalIndex: number) => {
    const percentage = (cumulative / maxTotal) * 100;
    const flashKey = `bid-${level.price}`;
    const flash = flashState[flashKey];
    const isHovered = hoveredBidIndex === originalIndex;
    const stats = isHovered ? calculateHoverStats(bids, originalIndex) : null;
    
    return (
      <div 
        key={`bid-${index}`} 
        className="relative grid grid-cols-3 text-xs py-0.5 hover:bg-[#2b3139]/50 transition-colors duration-150 cursor-pointer"
        style={{
          backgroundColor: flash === 'up' ? 'rgba(14, 203, 129, 0.1)' : 
                          flash === 'down' ? 'rgba(246, 70, 93, 0.1)' : 
                          'transparent',
          transition: 'background-color 0.2s ease-out'
        }}
        onMouseEnter={() => setHoveredBidIndex(originalIndex)}
        onMouseLeave={() => setHoveredBidIndex(null)}
      >
        {filterMode === 'all' && (
          <div 
            className="absolute inset-0 bg-[#0ecb81]/20 transition-all duration-200"
            style={{ width: `${percentage}%`, right: 0 }}
          />
        )}
        {filterMode !== 'all' && (
          <div 
            className="absolute inset-0 bg-[#0ecb81]/20 transition-all duration-200"
            style={{ width: `${percentage}%`, left: 0 }}
          />
        )}
        <div className="relative text-[#0ecb81] font-mono text-right pr-2">
          {level.price.toFixed(decimals)}
        </div>
        <div className="relative text-[#eaecef] text-right font-mono pr-2">
          {level.amount.toFixed(5)}
        </div>
        <div className="relative text-[#848e9c] text-right font-mono">
          {(level.price * level.amount).toFixed(2)}
        </div>
        
        {/* Hover Popup */}
        {isHovered && stats && (
          <div 
            className="absolute left-0 transform -translate-x-full ml-2 px-3 py-2 rounded shadow-lg z-50 whitespace-nowrap"
            style={{ 
              backgroundColor: THEME_COLORS.CARD_BG_LIGHT,
              border: `1px solid ${THEME_COLORS.BORDER}`,
              top: '50%',
              transform: 'translate(-100%, -50%)'
            }}
          >
            <div className="text-[10px] space-y-1">
              <div className="flex justify-between gap-4">
                <span className="text-[#848e9c]">Avg.Price:</span>
                <span className="text-[#eaecef] font-mono">≈ {stats.avgPrice.toFixed(decimals)}</span>
              </div>
              <div className="flex justify-between gap-4">
                <span className="text-[#848e9c]">Sum {assetName}:</span>
                <span className="text-[#eaecef] font-mono">{stats.sumAmount.toFixed(5)}</span>
              </div>
              <div className="flex justify-between gap-4">
                <span className="text-[#848e9c]">Sum USDT:</span>
                <span className="text-[#eaecef] font-mono">{stats.sumUSDT.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</span>
              </div>
            </div>
          </div>
        )}
      </div>
    );
  };

  // Calculate cumulative amounts for depth visualization
  let askCumulative = 0;
  let bidCumulative = 0;

  return (
    <div 
      className="rounded-lg h-full flex flex-col"
      style={{ backgroundColor: THEME_COLORS.CARD_BG }}
    >
      {/* Header */}
      <div className="p-2 border-b" style={{ borderColor: THEME_COLORS.BORDER }}>
        <div className="flex items-center justify-between mb-1.5">
          <h3 className="text-[#eaecef] font-semibold text-sm">Order Book</h3>
          <span className="text-xs text-[#848e9c]">{symbol}</span>
        </div>
        {/* Filter Buttons */}
        <div className="flex gap-1">
          <button
            onClick={() => setFilterMode('all')}
            className="flex-1 py-1 px-2 text-xs rounded transition-colors"
            style={{
              backgroundColor: filterMode === 'all' ? THEME_COLORS.CARD_BG_LIGHT : 'transparent',
              color: filterMode === 'all' ? THEME_COLORS.TEXT_PRIMARY : THEME_COLORS.TEXT_SECONDARY,
              border: `1px solid ${THEME_COLORS.BORDER}`
            }}
          >
            All
          </button>
          <button
            onClick={() => setFilterMode('bids')}
            className="flex-1 py-1 px-2 text-xs rounded transition-colors"
            style={{
              backgroundColor: filterMode === 'bids' ? 'rgba(14, 203, 129, 0.2)' : 'transparent',
              color: filterMode === 'bids' ? THEME_COLORS.POSITIVE : THEME_COLORS.TEXT_SECONDARY,
              border: `1px solid ${filterMode === 'bids' ? THEME_COLORS.POSITIVE : THEME_COLORS.BORDER}`
            }}
          >
            Buy
          </button>
          <button
            onClick={() => setFilterMode('asks')}
            className="flex-1 py-1 px-2 text-xs rounded transition-colors"
            style={{
              backgroundColor: filterMode === 'asks' ? 'rgba(246, 70, 93, 0.2)' : 'transparent',
              color: filterMode === 'asks' ? THEME_COLORS.NEGATIVE : THEME_COLORS.TEXT_SECONDARY,
              border: `1px solid ${filterMode === 'asks' ? THEME_COLORS.NEGATIVE : THEME_COLORS.BORDER}`
            }}
          >
            Sell
          </button>
        </div>
      </div>
      
      {/* Column Headers */}
      <div 
        className="grid grid-cols-3 text-xs text-[#848e9c] px-2 py-1.5 border-b"
        style={{ borderColor: THEME_COLORS.BORDER }}
      >
        <div className="text-right pr-2">Price(USDT)</div>
        <div className="text-right pr-2">Amount</div>
        <div className="text-right">Total</div>
      </div>

      {/* Order Book Content */}
      <div ref={contentRef} className="flex-1 overflow-hidden flex flex-col px-2">
        {/* Asks (sell orders) - shown in reverse, red */}
        {(filterMode === 'all' || filterMode === 'asks') && (
          <div className={`${filterMode === 'asks' ? 'flex-1 overflow-y-auto' : 'flex flex-col justify-end overflow-hidden'}`}>
            {asks.slice(0, filterMode === 'asks' ? 50 : dynamicRowCount).reverse().map((ask, index, arr) => {
              const originalIndex = arr.length - 1 - index;
              askCumulative += ask.amount;
              return renderAskRow(ask, index, askCumulative, originalIndex);
            })}
          </div>
        )}

        {/* Spread / Current Price */}
        <div 
          className="py-1 border-y flex-shrink-0"
          style={{ borderColor: THEME_COLORS.BORDER }}
        >
          <div className="flex items-center justify-center gap-2">
            <span 
              className="text-lg font-bold font-mono"
              style={{ color: THEME_COLORS.POSITIVE }}
            >
              {currentPrice ? currentPrice.toFixed(decimals) : (asks[0]?.price.toFixed(decimals) || '---')}
            </span>
            {currentPrice && (
              <span className="text-xs text-[#848e9c]">
                ≈ ${currentPrice.toFixed(2)}
              </span>
            )}
          </div>
        </div>

        {/* Bids (buy orders) - green */}
        {(filterMode === 'all' || filterMode === 'bids') && (
          <div className={`${filterMode === 'bids' ? 'flex-1 overflow-y-auto' : 'overflow-hidden'}`}>
            {bids.slice(0, filterMode === 'bids' ? 50 : dynamicRowCount).map((bid, index) => {
              bidCumulative += bid.amount;
              return renderBidRow(bid, index, bidCumulative, index);
            })}
          </div>
        )}
      </div>

      {/* Footer with bid/ask ratio bar */}
      <div className="px-2 py-1 border-t relative" style={{ borderColor: THEME_COLORS.BORDER }}>
        {/* Bid/Ask Ratio Bar - hoverable for tooltip */}
        <div 
          className="cursor-pointer"
          onMouseEnter={() => setShowBarTooltip(true)}
          onMouseLeave={() => setShowBarTooltip(false)}
        >
          <div className="flex items-center gap-2 text-[10px]">
            <span className="text-[#0ecb81] font-mono">B {((barBidTotal / (barBidTotal + barAskTotal)) * 100).toFixed(2)}%</span>
            <div className="flex-1 h-1 rounded-full overflow-hidden" style={{ backgroundColor: THEME_COLORS.CARD_BG_LIGHT }}>
              <div 
                className="h-full transition-all duration-300"
                style={{ 
                  width: `${(barBidTotal / (barBidTotal + barAskTotal)) * 100}%`,
                  backgroundColor: THEME_COLORS.POSITIVE
                }}
              />
            </div>
            <span className="text-[#f6465d] font-mono">{((barAskTotal / (barBidTotal + barAskTotal)) * 100).toFixed(2)}% S</span>
          </div>
        </div>
        
        {/* Tooltip on hover */}
        {showBarTooltip && (
          <div
            className="absolute left-1/2 -translate-x-1/2 bottom-full mb-1 px-3 py-1.5 rounded shadow-lg z-50 whitespace-nowrap"
            style={{
              backgroundColor: THEME_COLORS.CARD_BG_LIGHT,
              border: `1px solid ${THEME_COLORS.BORDER}`,
            }}
          >
            <span className="text-[10px] text-[#848e9c]">
          Track the contents of the first 20 data tranches of the Spot Order book and update the data in real time.
            </span>
        </div>
        )}
      </div>
    </div>
  );
};
