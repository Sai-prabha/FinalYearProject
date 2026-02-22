import type { KlineEvent, CandleData, ConnectionStatus, DataCallback, StatusCallback } from '../types';
import { CONFIG } from '../constants';

export class WebSocketService {
  private ws: WebSocket | null = null;
  private reconnectAttempts = 0;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private url: string;
  private dataCallback: DataCallback | null = null;
  private statusCallback: StatusCallback | null = null;
  private isIntentionalClose = false;

  constructor(url: string) {
    this.url = url;
  }

  connect(onData: DataCallback, onStatus?: StatusCallback): void {
    this.dataCallback = onData;
    this.statusCallback = onStatus || null;
    this.isIntentionalClose = false;
    this.createConnection();
  }

  private createConnection(): void {
    try {
      this.updateStatus('connecting');
      this.ws = new WebSocket(this.url);

      this.ws.onopen = this.handleOpen.bind(this);
      this.ws.onmessage = this.handleMessage.bind(this);
      this.ws.onerror = this.handleError.bind(this);
      this.ws.onclose = this.handleClose.bind(this);
    } catch (error) {
      console.error('WebSocket connection error:', error);
      this.updateStatus('error');
      this.handleReconnect();
    }
  }

  private handleOpen(): void {
    console.log(`WebSocket connected: ${this.url}`);
    this.reconnectAttempts = 0;
    this.updateStatus('connected');
  }

  private handleMessage(event: MessageEvent): void {
    try {
      const data: KlineEvent = JSON.parse(event.data);
      
      if (data.e === 'kline' && data.k) {
        const candle: CandleData = {
          time: data.k.t / 1000,
          open: parseFloat(data.k.o),
          high: parseFloat(data.k.h),
          low: parseFloat(data.k.l),
          close: parseFloat(data.k.c),
          volume: parseFloat(data.k.v),
        };

        if (this.dataCallback) {
          try {
            this.dataCallback(candle);
          } catch (callbackError) {
            console.error('Error in data callback:', callbackError);
          }
        }
      }
    } catch (error) {
      console.error('Error parsing WebSocket message:', error, event.data);
    }
  }

  private handleError(event: Event): void {
    console.error('WebSocket error:', event);
    this.updateStatus('error');
  }

  private handleClose(): void {
    console.log('WebSocket closed');
    this.ws = null;

    if (!this.isIntentionalClose) {
      this.updateStatus('disconnected');
      this.handleReconnect();
    }
  }

  private handleReconnect(): void {
    if (this.reconnectAttempts >= CONFIG.RECONNECT_MAX_ATTEMPTS) {
      console.error('Max reconnection attempts reached');
      this.updateStatus('error');
      return;
    }

    const delay = Math.min(
      CONFIG.RECONNECT_BASE_DELAY * Math.pow(2, this.reconnectAttempts),
      CONFIG.RECONNECT_MAX_DELAY
    );

    console.log(`Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts + 1}/${CONFIG.RECONNECT_MAX_ATTEMPTS})`);

    this.reconnectTimer = setTimeout(() => {
      this.reconnectAttempts++;
      this.createConnection();
    }, delay);
  }

  private updateStatus(status: ConnectionStatus): void {
    if (this.statusCallback) {
      this.statusCallback(status);
    }
  }

  disconnect(): void {
    this.isIntentionalClose = true;

    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }

    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }

    this.dataCallback = null;
    this.statusCallback = null;
    this.reconnectAttempts = 0;
  }

  isConnected(): boolean {
    return this.ws !== null && this.ws.readyState === WebSocket.OPEN;
  }
}

// Backward compat alias
export { WebSocketService as BinanceWebSocketService };

