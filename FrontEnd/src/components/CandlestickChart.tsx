import React, { useEffect, useRef, memo, useState, useMemo } from 'react';
import { createChart, ColorType, type IChartApi, type Time, type CandlestickData } from 'lightweight-charts';
import type { CandleData, Trade, PositionMeta } from '../types';
import { THEME_COLORS } from '../constants';
import { useWindowResize } from '../hooks/useWindowResize';
import { TimeframeSelector, type Timeframe } from './TimeframeSelector';

export interface MAVisibility {
  ma7: boolean;
  ma25: boolean;
  ma99: boolean;
}

interface CandlestickChartProps {
  title: string;
  data: CandleData[];
  height?: number;
  isLoading?: boolean;
  onTimeframeChange?: (timeframe: Timeframe) => void;
  selectedTimeframe?: Timeframe;
  maVisibility?: MAVisibility;
  onMAVisibilityChange?: (visibility: MAVisibility) => void;
  trades?: Trade[];
  openPosition?: { direction: 'LONG' | 'SHORT'; entry_price: number } | null;
  positionMeta?: PositionMeta | null;
}

interface HoverData {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  quoteVolume: number;
  change: number;
}

// Format large numbers with K/M/B suffixes
const formatVolume = (v: number): string => {
  if (v >= 1_000_000_000) return `${(v / 1_000_000_000).toFixed(2)}B`;
  if (v >= 1_000_000) return `${(v / 1_000_000).toFixed(2)}M`;
  if (v >= 1_000) return `${(v / 1_000).toFixed(2)}K`;
  return v.toFixed(2);
};

// Format date as DD/MM/YYYY
const formatDate = (timestamp: number): string => {
  const d = new Date(timestamp * 1000);
  const day = String(d.getDate()).padStart(2, '0');
  const month = String(d.getMonth() + 1).padStart(2, '0');
  const year = d.getFullYear();
  return `${day}/${month}/${year}`;
};

export const CandlestickChart: React.FC<CandlestickChartProps> = memo(({ 
  title, 
  data, 
  height = 500,
  isLoading = false,
  onTimeframeChange,
  selectedTimeframe = '1m',
  maVisibility = { ma7: true, ma25: true, ma99: true },
  onMAVisibilityChange,
  trades = [],
  openPosition = null,
  positionMeta = null,
}) => {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const candleSeriesRef = useRef<any>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const volumeSeriesRef = useRef<any>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const volMA7SeriesRef = useRef<any>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const volMA25SeriesRef = useRef<any>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const ma7SeriesRef = useRef<any>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const ma25SeriesRef = useRef<any>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const ma99SeriesRef = useRef<any>(null);
  const windowSize = useWindowResize();
  const [timeframe, setTimeframe] = useState<Timeframe>(selectedTimeframe);
  const [hoverData, setHoverData] = useState<HoverData | null>(null);
  
  // Track if this is the initial load to avoid fitting content on updates
  const isInitialLoadRef = useRef(true);
  const prevDataFirstTimeRef = useRef<number>(0);
  const prevTitleRef = useRef(title);
  const prevTimeframeRef = useRef(selectedTimeframe);
  
  // Store current data in ref for crosshair callback access
  const dataRef = useRef<CandleData[]>(data);
  dataRef.current = data;

  // Map timeframes to a sensible default visible candle count
  const getVisibleCandleCount = (tf: Timeframe): number => {
    const map: Record<Timeframe, number> = {
      '1m': 60,
      '5m': 60,
      '15m': 60,
      '1h': 48,
      '4h': 42,
      '1D': 60,
      '1W': 52,
    };
    return map[tf] || 60;
  };

  // Apply a timeframe-aware visible range instead of fitContent
  const applyDefaultZoom = () => {
    if (!chartRef.current || data.length === 0) return;
    const visibleCount = getVisibleCandleCount(selectedTimeframe);
    const from = Math.max(data.length - visibleCount, 0);
    const to = data.length - 1;
    chartRef.current.timeScale().setVisibleLogicalRange({ from, to });
  };
  
  // Reset initial load when title (symbol) or timeframe changes
  useEffect(() => {
    if (prevTitleRef.current !== title || prevTimeframeRef.current !== selectedTimeframe) {
      isInitialLoadRef.current = true;
      prevTitleRef.current = title;
      prevTimeframeRef.current = selectedTimeframe;
    }
  }, [title, selectedTimeframe]);

  // Calculate Moving Averages
  const calculateMA = (data: CandleData[], period: number): { time: Time; value: number }[] => {
    const result: { time: Time; value: number }[] = [];
    for (let i = period - 1; i < data.length; i++) {
      const sum = data.slice(i - period + 1, i + 1).reduce((acc, candle) => acc + candle.close, 0);
      result.push({
        time: data[i].time as Time,
        value: sum / period
      });
    }
    return result;
  };

  // Calculate MA values at specific index
  const calculateMAAtIndex = (data: CandleData[], index: number, period: number): number | null => {
    if (index < period - 1) return null;
    const sum = data.slice(index - period + 1, index + 1).reduce((acc, candle) => acc + candle.close, 0);
    return sum / period;
  };

  // Calculate Volume Moving Averages
  const calculateVolumeMA = (data: CandleData[], period: number): { time: Time; value: number }[] => {
    const result: { time: Time; value: number }[] = [];
    for (let i = period - 1; i < data.length; i++) {
      const sum = data.slice(i - period + 1, i + 1).reduce((acc, candle) => acc + candle.volume, 0);
      result.push({
        time: data[i].time as Time,
        value: sum / period
      });
    }
    return result;
  };

  // Calculate Volume MA at specific index
  const calculateVolumeMAAtIndex = (data: CandleData[], index: number, period: number): number | null => {
    if (index < period - 1) return null;
    const sum = data.slice(index - period + 1, index + 1).reduce((acc, candle) => acc + candle.volume, 0);
    return sum / period;
  };

  // Memoize MA calculations
  const ma7Data = useMemo(() => calculateMA(data, 7), [data]);
  const ma25Data = useMemo(() => calculateMA(data, 25), [data]);
  const ma99Data = useMemo(() => calculateMA(data, 99), [data]);

  // Memoize Volume MA calculations
  const volumeMA7Data = useMemo(() => calculateVolumeMA(data, 7), [data]);
  const volumeMA25Data = useMemo(() => calculateVolumeMA(data, 25), [data]);

  // Get current MA values for the hovered or latest candle
  const currentMAValues = useMemo(() => {
    if (data.length === 0) return { ma7: null, ma25: null, ma99: null };
    
    let index = data.length - 1;
    if (hoverData) {
      index = data.findIndex(d => d.time === hoverData.time);
      if (index === -1) index = data.length - 1;
    }

    return {
      ma7: calculateMAAtIndex(data, index, 7),
      ma25: calculateMAAtIndex(data, index, 25),
      ma99: calculateMAAtIndex(data, index, 99)
    };
  }, [data, hoverData]);

  // Get current Volume MA values for the hovered or latest candle
  const currentVolumeMAValues = useMemo(() => {
    if (data.length === 0) return { volumeMA7: null, volumeMA25: null };
    
    let index = data.length - 1;
    if (hoverData) {
      index = data.findIndex(d => d.time === hoverData.time);
      if (index === -1) index = data.length - 1;
    }

    return {
      volumeMA7: calculateVolumeMAAtIndex(data, index, 7),
      volumeMA25: calculateVolumeMAAtIndex(data, index, 25)
    };
  }, [data, hoverData]);

  // Build trade markers for the chart
  const tradeMarkers = useMemo(() => {
    if (!trades || trades.length === 0) return [];
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const markers: any[] = [];
    trades.forEach(trade => {
      // Entry marker
      markers.push({
        time: trade.entry_time as Time,
        position: trade.direction === 'LONG' ? 'belowBar' : 'aboveBar',
        color: trade.direction === 'LONG' ? THEME_COLORS.POSITIVE : THEME_COLORS.NEGATIVE,
        shape: trade.direction === 'LONG' ? 'arrowUp' : 'arrowDown',
        text: `${trade.direction} ${trade.position_size_pct ? trade.position_size_pct.toFixed(0) + '%' : ''}`,
      });
      // Exit marker
      if (trade.exit_time && trade.exit_price) {
        markers.push({
          time: trade.exit_time as Time,
          position: 'inBar',
          color: trade.pnl_pct >= 0 ? THEME_COLORS.POSITIVE : THEME_COLORS.NEGATIVE,
          shape: 'circle',
          text: `${trade.reason || 'Exit'} ${trade.pnl_pct >= 0 ? '+' : ''}${trade.pnl_pct.toFixed(2)}%`,
        });
      }
    });
    // Sort by time (required by lightweight-charts)
    markers.sort((a, b) => Number(a.time) - Number(b.time));
    return markers;
  }, [trades]);

  // Initialize chart
  useEffect(() => {
    if (!chartContainerRef.current) return;

    try {
      const containerHeight = chartContainerRef.current.clientHeight || height;
      const chart = createChart(chartContainerRef.current, {
        layout: {
          background: { type: ColorType.Solid, color: THEME_COLORS.CARD_BG },
          textColor: THEME_COLORS.TEXT_PRIMARY,
        },
        grid: {
          vertLines: { color: THEME_COLORS.GRID },
          horzLines: { color: THEME_COLORS.GRID },
        },
        width: chartContainerRef.current.clientWidth,
        height: containerHeight,
        timeScale: {
          timeVisible: true,
          secondsVisible: false,
          borderColor: THEME_COLORS.BORDER,
        },
        rightPriceScale: {
          borderColor: THEME_COLORS.BORDER,
          scaleMargins: {
            top: 0.05,
            bottom: 0.25,
          },
        },
        crosshair: {
          mode: 1,
          vertLine: {
            width: 1,
            color: THEME_COLORS.TEXT_SECONDARY,
            style: 3,
            labelBackgroundColor: THEME_COLORS.CARD_BG_LIGHT,
          },
          horzLine: {
            width: 1,
            color: THEME_COLORS.TEXT_SECONDARY,
            style: 3,
            labelBackgroundColor: THEME_COLORS.CARD_BG_LIGHT,
          },
        },
      });

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const candleSeries = (chart as any).addCandlestickSeries({
        upColor: THEME_COLORS.POSITIVE,
        downColor: THEME_COLORS.NEGATIVE,
        borderVisible: false,
        wickUpColor: THEME_COLORS.POSITIVE,
        wickDownColor: THEME_COLORS.NEGATIVE,
      });

      // Volume series - solid bars matching exchange style
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const volumeSeries = (chart as any).addHistogramSeries({
        color: '#485563',
        priceFormat: { type: 'volume' },
        priceScaleId: 'volume',
      });

      // Volume scale: slightly larger pane
      volumeSeries.priceScale().applyOptions({
        scaleMargins: {
          top: 0.80,
          bottom: 0,
        },
        borderVisible: true,
        borderColor: THEME_COLORS.BORDER,
      });

      // Volume MA lines overlaid on volume pane
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const volMA7Series = (chart as any).addLineSeries({
        color: '#f8bbd0', // Pink
        lineWidth: 1,
        priceScaleId: 'volume',
        lastValueVisible: false,
        priceLineVisible: false,
      });
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const volMA25Series = (chart as any).addLineSeries({
        color: '#80deea', // Cyan
        lineWidth: 1,
        priceScaleId: 'volume',
        lastValueVisible: false,
        priceLineVisible: false,
      });

      // Price Moving Average series
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const ma7Series = (chart as any).addLineSeries({
        color: '#ffa726',
        lineWidth: 1,
        title: '',
        visible: maVisibility.ma7,
        lastValueVisible: false,
        priceLineVisible: false,
      });
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const ma25Series = (chart as any).addLineSeries({
        color: '#ab47bc',
        lineWidth: 1,
        title: '',
        visible: maVisibility.ma25,
        lastValueVisible: false,
        priceLineVisible: false,
      });
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const ma99Series = (chart as any).addLineSeries({
        color: '#42a5f5',
        lineWidth: 1,
        title: '',
        visible: maVisibility.ma99,
        lastValueVisible: false,
        priceLineVisible: false,
      });

      chartRef.current = chart;
      candleSeriesRef.current = candleSeries;
      volumeSeriesRef.current = volumeSeries;
      volMA7SeriesRef.current = volMA7Series;
      volMA25SeriesRef.current = volMA25Series;
      ma7SeriesRef.current = ma7Series;
      ma25SeriesRef.current = ma25Series;
      ma99SeriesRef.current = ma99Series;

      // Subscribe to crosshair move
      chart.subscribeCrosshairMove((param) => {
        if (param.time) {
          const candleData = param.seriesData.get(candleSeries) as CandlestickData | undefined;
          if (candleData) {
            const currentData = dataRef.current;
            const currentIndex = currentData.findIndex(d => d.time === Number(param.time));
            const currentCandle = currentIndex >= 0 ? currentData[currentIndex] : null;
            const prevCandle = currentIndex > 0 ? currentData[currentIndex - 1] : null;
            const change = prevCandle ? candleData.close - prevCandle.close : 0;
            
            const volume = currentCandle?.volume || 0;
            const quoteVolume = volume * candleData.close;
            
            setHoverData({
              time: Number(param.time),
              open: candleData.open,
              high: candleData.high,
              low: candleData.low,
              close: candleData.close,
              volume,
              quoteVolume,
              change
            });
          }
        } else {
          setHoverData(null);
        }
      });

      return () => {
        try {
          chart.remove();
        } catch (error) {
          console.error('Error removing chart:', error);
        }
        chartRef.current = null;
        candleSeriesRef.current = null;
        volumeSeriesRef.current = null;
        volMA7SeriesRef.current = null;
        volMA25SeriesRef.current = null;
        ma7SeriesRef.current = null;
        ma25SeriesRef.current = null;
        ma99SeriesRef.current = null;
      };
    } catch (error) {
      console.error(`Error initializing chart ${title}:`, error);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [height, title]);

  // Update chart data
  useEffect(() => {
    if (!candleSeriesRef.current || !volumeSeriesRef.current || data.length === 0) return;

    try {
      const candleData = data.map(candle => ({
        time: candle.time as Time,
        open: candle.open,
        high: candle.high,
        low: candle.low,
        close: candle.close,
      }));

      // Volume bars - solid green/red matching exchange style
      const volumeData = data.map(candle => ({
        time: candle.time as Time,
        value: candle.volume,
        color: candle.close >= candle.open 
          ? THEME_COLORS.POSITIVE + '99'
          : THEME_COLORS.NEGATIVE + '99',
      }));

      candleSeriesRef.current.setData(candleData);
      volumeSeriesRef.current.setData(volumeData);

      // Update price MA series
      if (ma7SeriesRef.current && ma7Data.length > 0) ma7SeriesRef.current.setData(ma7Data);
      if (ma25SeriesRef.current && ma25Data.length > 0) ma25SeriesRef.current.setData(ma25Data);
      if (ma99SeriesRef.current && ma99Data.length > 0) ma99SeriesRef.current.setData(ma99Data);

      // Update volume MA overlay lines
      if (volMA7SeriesRef.current && volumeMA7Data.length > 0) volMA7SeriesRef.current.setData(volumeMA7Data);
      if (volMA25SeriesRef.current && volumeMA25Data.length > 0) volMA25SeriesRef.current.setData(volumeMA25Data);

      // Set trade markers on candle series
      if (tradeMarkers.length > 0) {
        candleSeriesRef.current.setMarkers(tradeMarkers);
      } else {
        candleSeriesRef.current.setMarkers([]);
      }

      // Remove any stale price lines (SL/TP/Entry were previously drawn here)
      try {
        const allLines = candleSeriesRef.current.priceLines?.() || [];
        allLines.forEach((line: unknown) => candleSeriesRef.current.removePriceLine(line));
      } catch {
        // priceLines may not exist
      }

      // Zoom management
      if (chartRef.current && data.length > 0) {
        const currentFirstTime = data[0].time;
        const dataSourceChanged = prevDataFirstTimeRef.current !== 0 && 
                                   prevDataFirstTimeRef.current !== currentFirstTime;
        
        if (isInitialLoadRef.current) {
          applyDefaultZoom();
          isInitialLoadRef.current = false;
          prevDataFirstTimeRef.current = currentFirstTime;
        } else if (dataSourceChanged) {
          applyDefaultZoom();
          prevDataFirstTimeRef.current = currentFirstTime;
        } else {
          prevDataFirstTimeRef.current = currentFirstTime;
        }
      }
    } catch (error) {
      console.error(`Error updating chart data for ${title}:`, error);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data, title, ma7Data, ma25Data, ma99Data, volumeMA7Data, volumeMA25Data, tradeMarkers, selectedTimeframe, openPosition, positionMeta]);

  // Update MA visibility when toggles change
  useEffect(() => {
    if (ma7SeriesRef.current) ma7SeriesRef.current.applyOptions({ visible: maVisibility.ma7 });
    if (ma25SeriesRef.current) ma25SeriesRef.current.applyOptions({ visible: maVisibility.ma25 });
    if (ma99SeriesRef.current) ma99SeriesRef.current.applyOptions({ visible: maVisibility.ma99 });
  }, [maVisibility]);

  // Handle window resize — update both width and height to fit container
  useEffect(() => {
    if (!chartRef.current || !chartContainerRef.current) return;
    try {
      const width = chartContainerRef.current.clientWidth;
      const height = chartContainerRef.current.clientHeight;
      chartRef.current.applyOptions({ width, height });
    } catch (error) {
      console.error('Error resizing chart:', error);
    }
  }, [windowSize]);

  // Get display data (hover or latest)
  const displayData = hoverData || (data.length > 0 ? {
    time: data[data.length - 1].time,
    open: data[data.length - 1].open,
    high: data[data.length - 1].high,
    low: data[data.length - 1].low,
    close: data[data.length - 1].close,
    volume: data[data.length - 1].volume,
    quoteVolume: data[data.length - 1].volume * data[data.length - 1].close,
    change: data.length > 1 ? data[data.length - 1].close - data[data.length - 2].close : 0
  } : null);

  const handleTimeframeChange = (tf: Timeframe) => {
    setTimeframe(tf);
    if (onTimeframeChange) onTimeframeChange(tf);
  };

  const handleMAToggle = (ma: 'ma7' | 'ma25' | 'ma99') => {
    if (onMAVisibilityChange) {
      onMAVisibilityChange({ ...maVisibility, [ma]: !maVisibility[ma] });
    }
  };

  // Determine if candle is bullish/bearish for coloring
  const isBullish = displayData ? displayData.close >= displayData.open : true;
  const ohlcColor = isBullish ? THEME_COLORS.POSITIVE : THEME_COLORS.NEGATIVE;

  // Calculate CHANGE % and RANGE %
  const changePct = displayData && displayData.open !== 0
    ? ((displayData.close - displayData.open) / displayData.open * 100)
    : 0;
  const rangePct = displayData && displayData.low !== 0
    ? ((displayData.high - displayData.low) / displayData.low * 100)
    : 0;

  // Determine price decimals based on title
  const priceDecimals = title.includes('BTC/ETH') || title.includes('Ratio') ? 4 : 2;
  const fmtPrice = (v: number) => v.toLocaleString(undefined, { minimumFractionDigits: priceDecimals, maximumFractionDigits: priceDecimals });

  return (
    <div 
      className="rounded-lg flex flex-col h-full"
      style={{ backgroundColor: THEME_COLORS.CARD_BG }}
    >
      {/* Header */}
      <div 
        className="px-4 py-3 border-b"
        style={{ borderColor: THEME_COLORS.BORDER }}
      >
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-4">
            <h3 className="text-[#eaecef] font-semibold">{title}</h3>
            <span className="text-[#848e9c] text-xs">Latest Data · {selectedTimeframe}</span>
          </div>
          <TimeframeSelector selected={timeframe} onSelect={handleTimeframeChange} />
        </div>
        
        {/* OHLC Values - Exchange style: Date Open High Low Close CHANGE RANGE */}
        {displayData && (
          <div className="flex items-center gap-2 text-xs flex-wrap">
            {/* Date (no label) */}
            <span className="text-[#848e9c] font-mono">{formatDate(displayData.time)}</span>
            {/* Open */}
            <span className="text-[#848e9c]">Open</span>
            <span className="font-mono" style={{ color: ohlcColor }}>{fmtPrice(displayData.open)}</span>
            {/* High */}
            <span className="text-[#848e9c]">High</span>
            <span className="font-mono" style={{ color: ohlcColor }}>{fmtPrice(displayData.high)}</span>
            {/* Low */}
            <span className="text-[#848e9c]">Low</span>
            <span className="font-mono" style={{ color: ohlcColor }}>{fmtPrice(displayData.low)}</span>
            {/* Close */}
            <span className="text-[#848e9c]">Close</span>
            <span className="font-mono" style={{ color: ohlcColor }}>{fmtPrice(displayData.close)}</span>
            {/* CHANGE */}
            <span className="text-[#848e9c]">CHANGE</span>
            <span className="font-mono" style={{ color: changePct >= 0 ? THEME_COLORS.POSITIVE : THEME_COLORS.NEGATIVE }}>
              {changePct >= 0 ? '+' : ''}{changePct.toFixed(2)}%
            </span>
            {/* RANGE */}
            <span className="text-[#848e9c]">Range</span>
            <span className="font-mono" style={{ color: ohlcColor }}>
              {rangePct.toFixed(2)}%
            </span>
          </div>
        )}

        {/* MA Values - color-coded to match chart lines, no "close" label */}
        <div className="flex items-center gap-4 text-xs mt-1">
          <button 
            onClick={() => handleMAToggle('ma7')}
            className="flex items-center gap-1 hover:opacity-80 transition-opacity"
            style={{ opacity: maVisibility.ma7 ? 1 : 0.4 }}
          >
            <div 
              className="w-3 h-3 rounded-sm border"
              style={{ 
                backgroundColor: maVisibility.ma7 ? '#ffa726' : 'transparent',
                borderColor: '#ffa726'
              }}
            />
            <span style={{ color: '#ffa726' }}>MA(7)</span>
            {currentMAValues.ma7 !== null && maVisibility.ma7 && (
              <span className="font-mono font-semibold" style={{ color: '#ffa726' }}>
                {fmtPrice(currentMAValues.ma7)}
              </span>
            )}
          </button>
          
          <button 
            onClick={() => handleMAToggle('ma25')}
            className="flex items-center gap-1 hover:opacity-80 transition-opacity"
            style={{ opacity: maVisibility.ma25 ? 1 : 0.4 }}
          >
            <div 
              className="w-3 h-3 rounded-sm border"
              style={{ 
                backgroundColor: maVisibility.ma25 ? '#ab47bc' : 'transparent',
                borderColor: '#ab47bc'
              }}
            />
            <span style={{ color: '#ab47bc' }}>MA(25)</span>
            {currentMAValues.ma25 !== null && maVisibility.ma25 && (
              <span className="font-mono font-semibold" style={{ color: '#ab47bc' }}>
                {fmtPrice(currentMAValues.ma25)}
              </span>
            )}
          </button>
          
          <button 
            onClick={() => handleMAToggle('ma99')}
            className="flex items-center gap-1 hover:opacity-80 transition-opacity"
            style={{ opacity: maVisibility.ma99 ? 1 : 0.4 }}
          >
            <div 
              className="w-3 h-3 rounded-sm border"
              style={{ 
                backgroundColor: maVisibility.ma99 ? '#42a5f5' : 'transparent',
                borderColor: '#42a5f5'
              }}
            />
            <span style={{ color: '#42a5f5' }}>MA(99)</span>
            {currentMAValues.ma99 !== null && maVisibility.ma99 && (
              <span className="font-mono font-semibold" style={{ color: '#42a5f5' }}>
                {fmtPrice(currentMAValues.ma99)}
              </span>
            )}
          </button>
        </div>
      </div>

      {/* Volume Header - Exchange style with vol MAs */}
      <div 
        className="px-4 py-1 text-xs border-b flex items-center gap-3"
        style={{ borderColor: THEME_COLORS.BORDER }}
      >
        {displayData ? (
          <>
            <span className="text-[#848e9c]">
              Vol({title.includes('ETH') && !title.includes('BTC/ETH') ? 'ETH' : 'BTC'})
            </span>
            <span className="text-[#eaecef] font-mono font-semibold">
              {formatVolume(displayData.volume)}
            </span>
            <span className="text-[#848e9c]">
              Vol(USDT)
            </span>
            <span className="text-[#eaecef] font-mono font-semibold">
              {formatVolume(displayData.quoteVolume)}
            </span>
            {currentVolumeMAValues.volumeMA7 !== null && (
              <span className="font-mono" style={{ color: '#f8bbd0' }}>
                {formatVolume(currentVolumeMAValues.volumeMA7)}
              </span>
            )}
            {currentVolumeMAValues.volumeMA25 !== null && (
              <span className="font-mono" style={{ color: '#80deea' }}>
                {formatVolume(currentVolumeMAValues.volumeMA25)}
              </span>
            )}
          </>
        ) : (
          <span className="text-[#848e9c]">Vol: -</span>
        )}
      </div>

      {/* Chart Container */}
      <div className="flex-1 relative overflow-hidden">
        {isLoading && (
          <div className="absolute inset-0 flex items-center justify-center bg-[#1e2329]/80 z-10">
            <div className="text-[#848e9c]">Loading chart data...</div>
          </div>
        )}
        <div ref={chartContainerRef} className="w-full h-full overflow-hidden" />
      </div>
    </div>
  );
});

CandlestickChart.displayName = 'CandlestickChart';
