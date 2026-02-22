# Live Cryptocurrency Trend Dashboard

A real-time cryptocurrency dashboard built with React, TypeScript, and Vite that displays live BTC and ETH prices along with their ratio using WebSocket connections to Binance.

## Features

- **Real-time Price Tracking**: Live BTC/USDT and ETH/USDT price updates via Binance WebSocket API
- **BTC/ETH Ratio**: Automatically calculated and displayed ratio between Bitcoin and Ethereum
- **Interactive Charts**: Beautiful, responsive charts powered by Lightweight Charts library
- **24h Change Tracking**: Shows percentage change over the last 24 hours for each asset
- **Connection Status**: Visual indicators showing WebSocket connection health
- **Auto-reconnection**: Automatic reconnection with exponential backoff on connection loss
- **Responsive Design**: Fully responsive layout that works on desktop, tablet, and mobile
- **Dark Theme**: Modern dark theme optimized for extended viewing

## Tech Stack

- **React 18** - UI library
- **TypeScript** - Type safety
- **Vite** - Build tool and dev server
- **Tailwind CSS** - Styling
- **Lightweight Charts** - Chart visualization
- **Binance WebSocket API** - Real-time data source

## Prerequisites

- Node.js 18+ and npm

## Installation

1. Navigate to the FrontEnd directory:
```bash
cd FrontEnd
```

2. Install dependencies:
```bash
npm install
```

## Running the Application

### Development Mode

Start the development server:
```bash
npm run dev
```

The application will be available at `http://localhost:5173`

### Production Build

Build for production:
```bash
npm run build
```

Preview the production build:
```bash
npm run preview
```

## Project Structure

```
FrontEnd/
├── src/
│   ├── components/          # React components
│   │   ├── Dashboard.tsx    # Main dashboard layout
│   │   ├── PriceCard.tsx    # Price display card
│   │   ├── PriceChart.tsx   # Chart component
│   │   ├── ConnectionStatus.tsx  # Connection indicator
│   │   └── Skeleton.tsx     # Loading skeletons
│   ├── hooks/              # Custom React hooks
│   │   ├── useCryptoPrice.ts    # Single asset hook
│   │   ├── useCryptoPrices.ts   # Combined assets hook
│   │   └── useWindowResize.ts   # Window resize hook
│   ├── services/           # Business logic
│   │   ├── binanceWebSocket.ts  # WebSocket service
│   │   └── dataProcessor.ts     # Data processing utilities
│   ├── types/              # TypeScript definitions
│   │   └── index.ts
│   ├── constants/          # Configuration constants
│   │   └── index.ts
│   ├── utils/              # Utility functions
│   │   └── formatters.ts
│   ├── App.tsx
│   ├── main.tsx
│   └── index.css
├── public/
├── index.html
├── package.json
├── vite.config.ts
├── tsconfig.json
└── tailwind.config.js
```

## How It Works

### WebSocket Connection

The application connects to two Binance WebSocket streams:
- `wss://stream.binance.com:9443/ws/btcusdt@kline_1m` for Bitcoin prices
- `wss://stream.binance.com:9443/ws/ethusdt@kline_1m` for Ethereum prices

Each stream provides 1-minute candlestick data with open, high, low, close, and volume information.

### Data Processing

- **Buffer Management**: Maintains a rolling buffer of the last 500 data points for each asset
- **Ratio Calculation**: Automatically calculates BTC/ETH ratio from live prices
- **24h Change**: Tracks price changes over 24 hours
- **Volatility**: Calculates standard deviation of the last 30 prices

### Auto-reconnection

If a WebSocket connection drops, the application automatically attempts to reconnect with exponential backoff:
- 1st attempt: 1 second
- 2nd attempt: 2 seconds
- 3rd attempt: 4 seconds
- 4th attempt: 8 seconds
- 5th attempt: 16 seconds

After 5 failed attempts, the connection enters an error state.

### Chart Updates

Charts are updated in real-time as new data arrives, with throttling to prevent excessive re-renders (1 second throttle). The Lightweight Charts library provides smooth animations and excellent performance.

## Configuration

Key configuration values can be modified in `src/constants/index.ts`:

- `MAX_DATA_POINTS`: Maximum number of data points to keep in memory (default: 500)
- `RECONNECT_MAX_ATTEMPTS`: Maximum reconnection attempts (default: 5)
- `UPDATE_THROTTLE`: Throttle interval for updates in ms (default: 1000)
- `CHART_COLORS`: Colors for BTC, ETH, and ratio charts

## Browser Support

- Chrome/Edge 90+
- Firefox 88+
- Safari 14+

## Performance

- Efficient data buffering with fixed-size arrays
- Throttled updates to prevent excessive re-renders
- Debounced window resize events
- Memoized components to avoid unnecessary renders
- WebSocket connection pooling

## License

MIT

## Contributing

Feel free to open issues or submit pull requests with improvements.

## Acknowledgments

- Data provided by [Binance API](https://binance-docs.github.io/apidocs/spot/en/)
- Charts powered by [Lightweight Charts](https://www.tradingview.com/lightweight-charts/)
