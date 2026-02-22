import React from 'react';

interface SkeletonProps {
  className?: string;
}

export const Skeleton: React.FC<SkeletonProps> = ({ className = '' }) => {
  return (
    <div className={`animate-pulse bg-slate-700 rounded ${className}`}></div>
  );
};

export const ChartSkeleton: React.FC = () => {
  return (
    <div className="w-full h-[300px] bg-slate-800 rounded-lg flex items-center justify-center">
      <div className="text-gray-400">Loading chart...</div>
    </div>
  );
};

export const PriceCardSkeleton: React.FC = () => {
  return (
    <div className="bg-slate-800 rounded-lg p-4 space-y-3">
      <Skeleton className="h-4 w-24" />
      <Skeleton className="h-8 w-32" />
      <Skeleton className="h-4 w-20" />
    </div>
  );
};
