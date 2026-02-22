# Quick Start Guide

## Installation

```bash
cd FrontEnd
npm install
```

## Development

Start the development server:

```bash
npm run dev
```

The application will be available at `http://localhost:5173`

## Production Build

```bash
npm run build
npm run preview
```

## What to Expect

Once the application starts, you will see:

1. **Three Price Cards** showing:
   - BTC/USDT current price and 24h change
   - ETH/USDT current price and 24h change
   - BTC/ETH ratio and 24h change

2. **Three Real-time Charts** displaying:
   - Bitcoin price trend (orange line)
   - Ethereum price trend (purple line)
   - BTC/ETH ratio trend (blue line)

3. **Connection Status Indicators** showing:
   - Green dot: Connected to Binance WebSocket
   - Yellow dot (pulsing): Connecting...
   - Red dot: Connection error
   - Gray dot: Disconnected

## Features

- **Real-time Updates**: Data updates every minute via WebSocket
- **Auto-reconnection**: Automatically reconnects if connection drops
- **Responsive Design**: Works on desktop, tablet, and mobile
- **Dark Theme**: Easy on the eyes for extended viewing
- **Interactive Charts**: Hover to see precise values at any point

## Troubleshooting

### Charts not loading?
- Check your internet connection
- Ensure Binance API is accessible (not blocked by firewall)
- Check the browser console for error messages

### Data not updating?
- Connection status should show green
- If red/yellow, wait for auto-reconnection (up to 5 attempts)
- Refresh the page if problem persists

### Build errors?
- Delete `node_modules` and `package-lock.json`
- Run `npm install` again
- Make sure you have Node.js 18+ installed
