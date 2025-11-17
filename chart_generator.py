# Visualization module for creating charts and reports
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime
import config


class ChartGenerator:
    """Generate charts for stock analysis"""
    
    def __init__(self):
        self.style = 'seaborn-v0_8'
    
    def plot_price_chart(self, data: pd.DataFrame, symbol: str, 
                        technical_indicators: dict = None,
                        save_path: str = None):
        """Create comprehensive price chart with indicators"""
        
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.5, 0.15, 0.15, 0.2],
            subplot_titles=(f'{symbol} - Price & Indicators', 'Volume', 'RSI', 'MACD')
        )
        
        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=data['date'],
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'],
                name='Price'
            ),
            row=1, col=1
        )
        
        # Add moving averages if provided
        if technical_indicators:
            if 'sma_short' in technical_indicators:
                fig.add_trace(
                    go.Scatter(
                        x=data['date'],
                        y=technical_indicators['sma_short'],
                        name='SMA 7',
                        line=dict(color='orange', width=1)
                    ),
                    row=1, col=1
                )
            
            if 'sma_medium' in technical_indicators:
                fig.add_trace(
                    go.Scatter(
                        x=data['date'],
                        y=technical_indicators['sma_medium'],
                        name='SMA 20',
                        line=dict(color='blue', width=1)
                    ),
                    row=1, col=1
                )
            
            # Bollinger Bands
            if 'bb_upper' in technical_indicators:
                fig.add_trace(
                    go.Scatter(
                        x=data['date'],
                        y=technical_indicators['bb_upper'],
                        name='BB Upper',
                        line=dict(color='gray', width=1, dash='dash')
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=data['date'],
                        y=technical_indicators['bb_lower'],
                        name='BB Lower',
                        line=dict(color='gray', width=1, dash='dash'),
                        fill='tonexty'
                    ),
                    row=1, col=1
                )
        
        # Volume
        colors = ['red' if close < open else 'green' 
                 for close, open in zip(data['close'], data['open'])]
        
        fig.add_trace(
            go.Bar(x=data['date'], y=data['volume'], name='Volume', marker_color=colors),
            row=2, col=1
        )
        
        # RSI
        if technical_indicators and 'rsi' in technical_indicators:
            fig.add_trace(
                go.Scatter(
                    x=data['date'],
                    y=technical_indicators['rsi'],
                    name='RSI',
                    line=dict(color='purple', width=2)
                ),
                row=3, col=1
            )
            
            # RSI levels
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
        
        # MACD
        if technical_indicators and 'macd' in technical_indicators:
            fig.add_trace(
                go.Scatter(
                    x=data['date'],
                    y=technical_indicators['macd'],
                    name='MACD',
                    line=dict(color='blue', width=2)
                ),
                row=4, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=data['date'],
                    y=technical_indicators['macd_signal'],
                    name='Signal',
                    line=dict(color='orange', width=2)
                ),
                row=4, col=1
            )
            
            # MACD histogram
            colors = ['green' if val > 0 else 'red' 
                     for val in technical_indicators['macd_histogram']]
            
            fig.add_trace(
                go.Bar(
                    x=data['date'],
                    y=technical_indicators['macd_histogram'],
                    name='Histogram',
                    marker_color=colors
                ),
                row=4, col=1
            )
        
        # Update layout
        fig.update_layout(
            title=f'{symbol} Technical Analysis',
            xaxis_rangeslider_visible=False,
            height=1000,
            showlegend=True,
            hovermode='x unified'
        )
        
        fig.update_xaxes(title_text="Date", row=4, col=1)
        fig.update_yaxes(title_text="Price (NPR)", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        fig.update_yaxes(title_text="RSI", row=3, col=1)
        fig.update_yaxes(title_text="MACD", row=4, col=1)
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def plot_comparison_chart(self, comparison_df: pd.DataFrame, save_path: str = None):
        """Create comparison chart for multiple stocks"""
        
        fig = go.Figure()
        
        # Create bar chart
        fig.add_trace(go.Bar(
            x=comparison_df['Symbol'],
            y=comparison_df['Probability'],
            name='Profitability Probability',
            marker_color='lightblue'
        ))
        
        fig.update_layout(
            title='Stock Comparison - Profitability Probability',
            xaxis_title='Stock Symbol',
            yaxis_title='Probability (%)',
            height=500
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig


if __name__ == "__main__":
    print("Chart generator module loaded successfully")
