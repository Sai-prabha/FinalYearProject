import React, { useEffect, useRef, memo } from 'react';
import { createChart, ColorType, type IChartApi, type Time } from 'lightweight-charts';
import type { PriceData } from '../types';
import { THEME_COLORS } from '../constants';
import { useWindowResize } from '../hooks/useWindowResize';

interface PriceChartProps {
  title: string;
  data: PriceData[];
  color: string;
  height?: number;
}

export const PriceChart: React.FC<PriceChartProps> = memo(({ title, data, color, height = 300 }) => {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const seriesRef = useRef<any>(null);
  const windowSize = useWindowResize();

  // Initialize chart
  useEffect(() => {
    if (!chartContainerRef.current) return;

    try {
      console.log(`Initializing chart: ${title}`);
      
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
        height: height,
        timeScale: {
          timeVisible: true,
          secondsVisible: false,
          borderColor: THEME_COLORS.TEXT_SECONDARY,
        },
        rightPriceScale: {
          borderColor: THEME_COLORS.TEXT_SECONDARY,
        },
        crosshair: {
          mode: 1,
          vertLine: {
            width: 1,
            color: THEME_COLORS.TEXT_SECONDARY,
            style: 3,
          },
          horzLine: {
            width: 1,
            color: THEME_COLORS.TEXT_SECONDARY,
            style: 3,
          },
        },
      });

      // Create line series using v4 API
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const lineSeries = (chart as any).addLineSeries({
        color: color,
        lineWidth: 2,
        crosshairMarkerVisible: true,
        crosshairMarkerRadius: 4,
        lastValueVisible: true,
        priceLineVisible: true,
      });

      chartRef.current = chart;
      seriesRef.current = lineSeries;

      console.log(`Chart initialized successfully: ${title}`);

      return () => {
        try {
          chart.remove();
        } catch (error) {
          console.error('Error removing chart:', error);
        }
        chartRef.current = null;
        seriesRef.current = null;
      };
    } catch (error) {
      console.error(`Error initializing chart ${title}:`, error);
    }
  }, [color, height, title]);

  // Update chart data
  useEffect(() => {
    if (!seriesRef.current || data.length === 0) return;

    try {
      // Convert our data to the format expected by Lightweight Charts
      const chartData = data.map(point => ({
        time: point.time as Time,
        value: point.value,
      }));

      console.log(`Updating chart data for ${title}, points:`, chartData.length);
      seriesRef.current.setData(chartData);
    } catch (error) {
      console.error(`Error updating chart data for ${title}:`, error);
    }
  }, [data, title]);

  // Handle window resize
  useEffect(() => {
    if (!chartRef.current || !chartContainerRef.current) return;

    try {
      const width = chartContainerRef.current.clientWidth;
      chartRef.current.applyOptions({ width });
    } catch (error) {
      console.error('Error resizing chart:', error);
    }
  }, [windowSize]);

  return (
    <div className="bg-slate-800 rounded-lg p-4">
      <h3 className="text-white font-semibold mb-3">{title}</h3>
      <div ref={chartContainerRef} className="w-full" />
    </div>
  );
});

PriceChart.displayName = 'PriceChart';
