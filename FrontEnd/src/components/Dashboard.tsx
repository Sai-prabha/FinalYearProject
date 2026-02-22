import React, { useState, useEffect, useMemo } from 'react';
import { useCryptoPrices } from '../hooks/useCryptoPrices';
import { useOrderBook } from '../hooks/useOrderBook';
import { useHistoricalData } from '../hooks/useHistoricalData';
import { useModelSignals } from '../hooks/useModelSignals';
import { use24hStats } from '../hooks/use24hStats';
import { MarketStats } from './MarketStats';
import { CandlestickChart } from './CandlestickChart';
import { OrderBook } from './OrderBook';
import { RatioOrderBook } from './RatioOrderBook';
import { ConnectionStatus } from './ConnectionStatus';
import { ModelThinking } from './ModelThinking';
import { TradeHistory } from './TradeHistory';
import { AUTH_REQUIRED, THEME_COLORS, API_SYMBOLS } from '../constants';
import { useAuth } from '../context/AuthContext';
import type { CandleData, Trade } from '../types';
import type { Timeframe } from './TimeframeSelector';

type ActivePair = 'BTC' | 'ETH' | 'RATIO';

export const Dashboard: React.FC = () => {
  const { btc, eth, ratio } = useCryptoPrices();
  const [activePair, setActivePair] = useState<ActivePair>('BTC');
  const [timeframe, setTimeframe] = useState<Timeframe>('1m');
  const [maVisibility, setMAVisibility] = useState({ ma7: true, ma25: true, ma99: true });
  const [showModelPanel, setShowModelPanel] = useState<boolean>(true);
  
  // Connect to model server
  const { signalData, connectionStatus: modelConnectionStatus } = useModelSignals();
  const { logout } = useAuth();

  // 24h stats from REST API (accurate)
  const { btcStats, ethStats, ratioStats } = use24hStats();

  // Get the API symbol for the active pair
  const apiSymbol = activePair === 'BTC' ? API_SYMBOLS.BTC : 
                    activePair === 'ETH' ? API_SYMBOLS.ETH : '';

  // Fetch historical data with selected timeframe
  const { candles: historicalCandles, isLoading: isLoadingHistory } = useHistoricalData(
    apiSymbol,
    timeframe
  );

  // Always fetch BTC and ETH historical data for ratio calculation
  const { candles: btcHistoricalCandles, isLoading: isLoadingBtcHistory } = useHistoricalData(
    API_SYMBOLS.BTC,
    timeframe
  );
  const { candles: ethHistoricalCandles, isLoading: isLoadingEthHistory } = useHistoricalData(
    API_SYMBOLS.ETH,
    timeframe
  );

  // Get real order book data
  const btcOrderBook = useOrderBook('BTCUSDT');
  const ethOrderBook = useOrderBook('ETHUSDT');
  
  const { bids, asks, isConnected: orderBookConnected } = activePair === 'BTC' ? btcOrderBook : 
                                                          activePair === 'ETH' ? ethOrderBook : 
                                                          { bids: [], asks: [], isConnected: btcOrderBook.isConnected && ethOrderBook.isConnected };

  // Get current active data
  const activeData = activePair === 'BTC' ? btc : activePair === 'ETH' ? eth : ratio;

  // Combine historical data with real-time updates
  const [chartData, setChartData] = useState<CandleData[]>([]);

  useEffect(() => {
    if (historicalCandles.length > 0) {
      setChartData(historicalCandles);
    }
  }, [historicalCandles]);

  // Update chart with real-time data
  useEffect(() => {
    if (activeData.priceHistory.length > 0 && chartData.length > 0) {
      const latestPrice = activeData.priceHistory[activeData.priceHistory.length - 1];
      
      setChartData(prev => {
        if (prev.length === 0) return prev;
        
        const lastCandle = prev[prev.length - 1];
        const currentTime = Math.floor(Date.now() / 60000) * 60;
        
        if (lastCandle.time === currentTime) {
          const updated = [...prev];
          updated[updated.length - 1] = {
            ...lastCandle,
            high: Math.max(lastCandle.high, latestPrice.value),
            low: Math.min(lastCandle.low, latestPrice.value),
            close: latestPrice.value,
          };
          return updated;
        } else if (currentTime > lastCandle.time) {
          const newCandle: CandleData = {
            time: currentTime,
            open: latestPrice.value,
            high: latestPrice.value,
            low: latestPrice.value,
            close: latestPrice.value,
            volume: 0,
          };
          const updated = [...prev, newCandle];
          if (updated.length > 500) {
            return updated.slice(-500);
          }
          return updated;
        }
        return prev;
      });
    }
  }, [activeData.priceHistory, chartData.length]);

  // Use REST API 24h stats (accurate) instead of computing from limited chart data
  const stats24h = useMemo(() => {
    if (activePair === 'BTC' && btcStats) {
      return {
        high: btcStats.highPrice,
        low: btcStats.lowPrice,
        volume: btcStats.volume,
        change: btcStats.priceChangePercent,
      };
    }
    if (activePair === 'ETH' && ethStats) {
      return {
        high: ethStats.highPrice,
        low: ethStats.lowPrice,
        volume: ethStats.volume,
        change: ethStats.priceChangePercent,
      };
    }
    if (activePair === 'RATIO' && ratioStats) {
      return {
        high: ratioStats.high24h,
        low: ratioStats.low24h,
        volume: ratioStats.volume24h,
        change: ratioStats.change24hPct,
      };
    }
    // Fallback to chart data
    if (chartData.length === 0) return { high: 0, low: 0, volume: 0, change: 0 };
    return {
      high: Math.max(...chartData.map(c => c.high)),
      low: Math.min(...chartData.map(c => c.low)),
      volume: chartData.reduce((sum, c) => sum + c.volume, 0),
      change: activeData.change24h,
    };
  }, [activePair, btcStats, ethStats, ratioStats, chartData, activeData.change24h]);

  // Generate ratio chart data from BTC and ETH historical candles
  const ratioChartData = useMemo(() => {
    if (btcHistoricalCandles.length === 0 || ethHistoricalCandles.length === 0) return [];
    
    const ethCandleMap = new Map<number, CandleData>();
    ethHistoricalCandles.forEach(candle => {
      ethCandleMap.set(candle.time, candle);
    });

    const ratioCandles: CandleData[] = [];
    btcHistoricalCandles.forEach(btcCandle => {
      const ethCandle = ethCandleMap.get(btcCandle.time);
      if (ethCandle && ethCandle.open > 0 && ethCandle.high > 0 && ethCandle.low > 0 && ethCandle.close > 0) {
        ratioCandles.push({
          time: btcCandle.time,
          open: btcCandle.open / ethCandle.open,
          high: btcCandle.high / ethCandle.low,
          low: btcCandle.low / ethCandle.high,
          close: btcCandle.close / ethCandle.close,
          volume: Math.min(btcCandle.volume, ethCandle.volume),
        });
      }
    });

    return ratioCandles;
  }, [btcHistoricalCandles, ethHistoricalCandles]);

  const displayChartData = activePair === 'RATIO' ? ratioChartData : chartData;
  const isRatioLoading = activePair === 'RATIO' && (isLoadingBtcHistory || isLoadingEthHistory);

  // Extract trades for chart markers
  const chartTrades: Trade[] = useMemo(() => {
    return signalData?.portfolio?.recent_trades || [];
  }, [signalData?.portfolio?.recent_trades]);

  // Open position info for SL/TP price lines
  const openPosition = signalData?.portfolio?.position && signalData?.portfolio?.entry_price > 0
    ? { direction: signalData.portfolio.position, entry_price: signalData.portfolio.entry_price }
    : null;

  const positionMeta = signalData?.position_meta || null;

  // Use accurate 24h change from REST API
  const display24hChange = stats24h.change || activeData.change24h;

  return (
    <div className="h-screen flex flex-col" style={{ backgroundColor: THEME_COLORS.BACKGROUND }}>
      {/* Top Navigation */}
      <header 
        className="border-b px-4 py-3 flex-shrink-0"
        style={{ backgroundColor: THEME_COLORS.CARD_BG, borderColor: THEME_COLORS.BORDER }}
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-6">
            <h1 className="text-xl font-bold" style={{ color: THEME_COLORS.YELLOW }}>
              Crypto Dashboard
            </h1>
            
            {/* Pair Selector */}
            <div className="flex">
              {(['BTC', 'ETH', 'RATIO'] as ActivePair[]).map(pair => (
                <button
                  key={pair}
                  onClick={() => setActivePair(pair)}
                  className="px-4 py-2 font-medium transition-colors text-sm"
                  style={{
                    backgroundColor: activePair === pair ? THEME_COLORS.CARD_BG_LIGHT : 'transparent',
                    color: activePair === pair ? THEME_COLORS.TEXT_PRIMARY : THEME_COLORS.TEXT_SECONDARY,
                    borderBottom: activePair === pair ? `2px solid ${THEME_COLORS.YELLOW}` : '2px solid transparent',
                  }}
                >
                  {pair === 'BTC' ? 'BTC/USDT' : pair === 'ETH' ? 'ETH/USDT' : 'BTC/ETH'}
                </button>
              ))}
            </div>
          </div>

          {/* Connection Status */}
          <div className="flex items-center gap-4">
            <ConnectionStatus status={activeData.connectionStatus} />
            <div className="flex items-center gap-2">
              <div 
                className={`w-2 h-2 rounded-full ${orderBookConnected ? 'bg-green-500' : 'bg-gray-500'}`}
              />
              <span className="text-xs" style={{ color: THEME_COLORS.TEXT_SECONDARY }}>
                Order Book {orderBookConnected ? 'Live' : 'Offline'}
              </span>
            </div>
            <div className="flex items-center gap-2">
              <div 
                className={`w-2 h-2 rounded-full ${modelConnectionStatus === 'connected' ? 'bg-green-500' : 'bg-gray-500'}`}
              />
              <span className="text-xs" style={{ color: THEME_COLORS.TEXT_SECONDARY }}>
                Model {modelConnectionStatus === 'connected' ? 'Live' : 'Offline'}
              </span>
            </div>
            <button
              onClick={() => setShowModelPanel(!showModelPanel)}
              className="px-3 py-1 text-xs rounded transition-colors"
              style={{
                backgroundColor: showModelPanel ? THEME_COLORS.YELLOW : THEME_COLORS.CARD_BG_LIGHT,
                color: showModelPanel ? THEME_COLORS.BACKGROUND : THEME_COLORS.TEXT_SECONDARY
              }}
            >
              {showModelPanel ? 'Hide' : 'Show'} Model
            </button>
            {AUTH_REQUIRED && (
              <button
                onClick={logout}
                className="px-3 py-1 text-xs rounded transition-colors"
                style={{
                  backgroundColor: THEME_COLORS.CARD_BG_LIGHT,
                  color: THEME_COLORS.TEXT_SECONDARY
                }}
              >
                Logout
              </button>
            )}
          </div>
        </div>
      </header>

      {/* Market Stats Bar */}
      <div className="p-4 flex-shrink-0">
        <MarketStats
          symbol={activePair === 'BTC' ? 'BTC/USDT' : activePair === 'ETH' ? 'ETH/USDT' : 'BTC/ETH'}
          currentPrice={activeData.currentPrice}
          change24h={display24hChange}
          high24h={stats24h.high}
          low24h={stats24h.low}
          volume24h={stats24h.volume}
          isRatio={activePair === 'RATIO'}
        />
      </div>

      {/* Main Trading View */}
      <div className="flex-1 flex flex-col gap-2 px-4 pb-2 min-h-0 overflow-hidden">
        {/* Top Section: Chart + Panels */}
        <div className="flex-1 flex gap-2 min-h-0 overflow-hidden">
          {/* Chart Section */}
          <div className="flex-1 min-w-0 h-full overflow-hidden">
            <CandlestickChart
              title={activePair === 'BTC' ? 'BTC/USDT' : activePair === 'ETH' ? 'ETH/USDT' : 'BTC/ETH Ratio'}
              data={displayChartData}
              isLoading={activePair === 'RATIO' ? isRatioLoading : isLoadingHistory}
              selectedTimeframe={timeframe}
              onTimeframeChange={setTimeframe}
              maVisibility={maVisibility}
              onMAVisibilityChange={setMAVisibility}
              trades={chartTrades}
              openPosition={openPosition}
              positionMeta={positionMeta}
            />
          </div>

          {/* Model Thinking Panel */}
          {showModelPanel && (
            <div className="flex-shrink-0 h-full" style={{ width: '280px' }}>
              <ModelThinking
                signalData={signalData}
                isConnected={modelConnectionStatus === 'connected'}
                btcPrice={btc.currentPrice}
                ethPrice={eth.currentPrice}
              />
            </div>
          )}

          {/* Order Book Section */}
          <div className="flex-shrink-0 h-full" style={{ width: '260px' }}>
            {activePair === 'RATIO' ? (
              <RatioOrderBook
                btcBids={btcOrderBook.bids}
                btcAsks={btcOrderBook.asks}
                ethBids={ethOrderBook.bids}
                ethAsks={ethOrderBook.asks}
                currentRatio={ratio.currentPrice}
              />
            ) : (
              <OrderBook
                symbol={activePair === 'BTC' ? 'BTC/USDT' : 'ETH/USDT'}
                bids={bids}
                asks={asks}
                decimals={activePair === 'ETH' ? 2 : 2}
                currentPrice={activeData.currentPrice}
              />
            )}
          </div>
        </div>

        {/* Bottom Section: Trade History */}
        <div className="flex-shrink-0" style={{ height: '150px' }}>
          <TradeHistory
            portfolio={signalData?.portfolio || null}
            isModelConnected={modelConnectionStatus === 'connected'}
            signalData={signalData}
            btcPrice={btc.currentPrice}
            ethPrice={eth.currentPrice}
          />
        </div>
      </div>

      {/* Footer */}
      <footer 
        className="p-2 text-center text-[10px] border-t flex-shrink-0"
        style={{ borderColor: THEME_COLORS.BORDER, color: THEME_COLORS.TEXT_SECONDARY }}
      >
        <p>Real-time data via WebSocket API · Charts powered by TradingView Lightweight Charts · {signalData?.model_info?.version?.toUpperCase() || 'V4.15'} Model</p>
      </footer>
    </div>
  );
};

// Backward compat alias
export { Dashboard as BinanceDashboard };
