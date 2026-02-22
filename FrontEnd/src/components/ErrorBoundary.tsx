import { Component, type ErrorInfo, type ReactNode } from 'react';

interface Props {
  children: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
}

export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
    };
  }

  static getDerivedStateFromError(error: Error): State {
    return {
      hasError: true,
      error,
      errorInfo: null,
    };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo): void {
    console.error('Error caught by boundary:', error, errorInfo);
    this.setState({
      error,
      errorInfo,
    });
  }

  render(): ReactNode {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen bg-slate-900 p-8 text-white">
          <div className="max-w-4xl mx-auto">
            <h1 className="text-3xl font-bold text-red-500 mb-4">
              Something went wrong
            </h1>
            <div className="bg-slate-800 p-6 rounded-lg">
              <h2 className="text-xl font-semibold mb-2">Error Details:</h2>
              <pre className="bg-slate-900 p-4 rounded overflow-auto text-sm">
                {this.state.error?.toString()}
              </pre>
              {this.state.errorInfo && (
                <>
                  <h3 className="text-lg font-semibold mt-4 mb-2">
                    Component Stack:
                  </h3>
                  <pre className="bg-slate-900 p-4 rounded overflow-auto text-sm">
                    {this.state.errorInfo.componentStack}
                  </pre>
                </>
              )}
              <button
                onClick={() => window.location.reload()}
                className="mt-4 bg-blue-500 hover:bg-blue-600 px-6 py-2 rounded font-semibold"
              >
                Reload Page
              </button>
            </div>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
