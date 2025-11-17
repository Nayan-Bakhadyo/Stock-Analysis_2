"""Visualization of the complete analysis pipeline and data flow"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
import numpy as np
from typing import Dict
import seaborn as sns

class AnalysisVisualizer:
    """Visualize the complete analysis pipeline"""
    
    def __init__(self):
        self.colors = {
            'technical': '#3498db',
            'fundamental': '#2ecc71',
            'sentiment': '#e74c3c',
            'momentum': '#f39c12',
            'broker': '#9b59b6',
            'ml': '#1abc9c',
            'final': '#34495e',
            'scraper': '#95a5a6',
            'database': '#e67e22',
            'analyzer': '#16a085'
        }
    
    def visualize_process_flow(self, symbol: str = "IGI"):
        """
        Visualize the complete data flow from scraping to analysis
        
        Shows the entire pipeline:
        1. Data Sources (ShareSansar, NepalAlpha, Database)
        2. Scrapers
        3. Database Storage
        4. Analysis Components (Technical, Fundamental, Sentiment, ML)
        5. Insights Engine
        6. Final Output
        """
        
        fig, ax = plt.subplots(figsize=(18, 12))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        fig.suptitle(f'NEPSE Stock Analysis Pipeline: Complete Data Flow', 
                     fontsize=18, fontweight='bold', y=0.98)
        
        # Layer 1: Data Sources (Top)
        y_sources = 9
        sources = [
            ('ShareSansar\n(Price + News)', 1.5, '#e74c3c'),
            ('NepalAlpha\n(Fundamentals)', 5, '#2ecc71'),
            ('SQLite DB\n(Cache)', 8.5, '#e67e22')
        ]
        
        for source, x, color in sources:
            self._draw_box(ax, x, y_sources, 1.2, 0.6, source, color, shape='round')
        
        # Layer 2: Scrapers
        y_scrapers = 7.5
        scrapers = [
            ('Price Scraper\n(556 days)', 1, '#3498db'),
            ('News Scraper\n(10 articles)', 2, '#9b59b6'),
            ('Fundamental\nScraper', 5, '#16a085')
        ]
        
        for scraper, x, color in scrapers:
            self._draw_box(ax, x, y_scrapers, 0.9, 0.5, scraper, color)
        
        # Arrows: Sources to Scrapers
        self._draw_arrow(ax, 1.5, y_sources-0.3, 1, y_scrapers+0.25)
        self._draw_arrow(ax, 1.5, y_sources-0.3, 2, y_scrapers+0.25)
        self._draw_arrow(ax, 5, y_sources-0.3, 5, y_scrapers+0.25)
        
        # Layer 3: Database Storage
        y_db = 6.2
        self._draw_box(ax, 5, y_db, 2.5, 0.6, 
                      'Database Storage\n(price_history, news_cache, fundamentals)', 
                      '#e67e22', shape='cylinder')
        
        # Arrows: Scrapers to DB
        self._draw_arrow(ax, 1, y_scrapers-0.25, 4, y_db+0.3)
        self._draw_arrow(ax, 2, y_scrapers-0.25, 4.5, y_db+0.3)
        self._draw_arrow(ax, 5, y_scrapers-0.25, 5, y_db+0.3)
        
        # Bidirectional arrow: DB to Cache
        self._draw_arrow(ax, 8.5, y_sources-0.3, 6.2, y_db+0.3, style='dashed')
        self._draw_arrow(ax, 6.2, y_db+0.3, 8.5, y_sources-0.3, style='dashed')
        
        # Layer 4: Analysis Components
        y_analyzers = 4.5
        analyzers = [
            ('Technical\nAnalyzer\n(12 indicators)', 1, '#3498db'),
            ('Fundamental\nAnalyzer\n(7 ratios)', 2.5, '#2ecc71'),
            ('Sentiment\nAnalyzer\n(NLP)', 4, '#e74c3c'),
            ('Momentum\nAnalyzer\n(trends)', 5.5, '#f39c12'),
            ('ML Predictor\n(LSTM)', 7, '#1abc9c')
        ]
        
        for analyzer, x, color in analyzers:
            self._draw_box(ax, x, y_analyzers, 0.8, 0.6, analyzer, color)
        
        # Arrows: DB to Analyzers
        for analyzer, x, _ in analyzers:
            self._draw_arrow(ax, 5, y_db-0.3, x, y_analyzers+0.3)
        
        # Layer 5: Insights Engine
        y_engine = 2.8
        self._draw_box(ax, 5, y_engine, 2, 0.7, 
                      'Trading Insights Engine\n(Weighted Score Calculation)', 
                      '#34495e', shape='round')
        
        # Arrows: Analyzers to Engine
        for analyzer, x, _ in analyzers:
            self._draw_arrow(ax, x, y_analyzers-0.3, 5, y_engine+0.35)
        
        # Layer 6: Final Outputs
        y_outputs = 1.2
        outputs = [
            ('Probability\n41%', 2.5, '#34495e'),
            ('Recommendation\nHOLD', 4, '#f39c12'),
            ('ML Predictions\n+5.9%', 5.5, '#1abc9c'),
            ('Risk/Reward\n12.65', 7, '#e74c3c')
        ]
        
        for output, x, color in outputs:
            self._draw_box(ax, x, y_outputs, 0.9, 0.5, output, color, shape='round')
        
        # Arrows: Engine to Outputs
        for output, x, _ in outputs:
            self._draw_arrow(ax, 5, y_engine-0.35, x, y_outputs+0.25)
        
        # Add data flow annotations
        self._add_annotation(ax, 0.5, 9, 'DATA SOURCES', fontsize=12, weight='bold')
        self._add_annotation(ax, 0.5, 7.5, 'SCRAPERS', fontsize=12, weight='bold')
        self._add_annotation(ax, 0.5, 6.2, 'STORAGE', fontsize=12, weight='bold')
        self._add_annotation(ax, 0.5, 4.5, 'ANALYZERS', fontsize=12, weight='bold')
        self._add_annotation(ax, 0.5, 2.8, 'ENGINE', fontsize=12, weight='bold')
        self._add_annotation(ax, 0.5, 1.2, 'OUTPUTS', fontsize=12, weight='bold')
        
        # Add metrics
        metrics_text = f"""
        Data Points: 556 days
        News Articles: 10
        Training Samples: 379
        Model Accuracy: 91.5%
        Prediction Confidence: 99.6%
        """
        ax.text(9.5, 4.5, metrics_text, fontsize=9, 
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3),
               verticalalignment='center')
        
        plt.tight_layout()
        return fig
    
    def _draw_box(self, ax, x, y, width, height, text, color, shape='rect'):
        """Draw a box with text"""
        if shape == 'round':
            box = FancyBboxPatch((x-width/2, y-height/2), width, height,
                                boxstyle="round,pad=0.05", 
                                facecolor=color, edgecolor='black', 
                                linewidth=2, alpha=0.7)
        elif shape == 'cylinder':
            box = FancyBboxPatch((x-width/2, y-height/2), width, height,
                                boxstyle="round,pad=0.1", 
                                facecolor=color, edgecolor='black', 
                                linewidth=2, alpha=0.6)
        else:
            box = Rectangle((x-width/2, y-height/2), width, height,
                           facecolor=color, edgecolor='black', 
                           linewidth=2, alpha=0.7)
        
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', 
               fontsize=8, fontweight='bold', color='white')
    
    def _draw_arrow(self, ax, x1, y1, x2, y2, style='solid'):
        """Draw an arrow between two points"""
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                               arrowstyle='->', mutation_scale=20,
                               linewidth=2, color='#34495e', alpha=0.6,
                               linestyle=style)
        ax.add_patch(arrow)
    
    def _add_annotation(self, ax, x, y, text, fontsize=10, weight='normal'):
        """Add text annotation"""
        ax.text(x, y, text, fontsize=fontsize, fontweight=weight,
               color='#2c3e50', verticalalignment='center')
    
    def visualize_complete_analysis(self, analysis_result: Dict, symbol: str):
        """
        Create comprehensive visualization of the entire analysis
        
        Shows:
        1. Data sources and inputs
        2. Individual analysis components
        3. Score calculations
        4. Weight distribution
        5. Final probability calculation
        """
        
        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
        
        # Title
        fig.suptitle(f'Complete Analysis Pipeline: {symbol}', 
                     fontsize=20, fontweight='bold', y=0.98)
        
        # 1. Data Sources (Top Left)
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_data_sources(ax1, analysis_result)
        
        # 2. Component Scores (Top Middle)
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_component_scores(ax2, analysis_result)
        
        # 3. Weight Distribution (Top Right)
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_weight_distribution(ax3, analysis_result)
        
        # 4. Technical Analysis Details (Row 2 Left)
        ax4 = fig.add_subplot(gs[1, 0])
        self._plot_technical_details(ax4, analysis_result)
        
        # 5. Sentiment Analysis Details (Row 2 Middle)
        ax5 = fig.add_subplot(gs[1, 1])
        self._plot_sentiment_details(ax5, analysis_result)
        
        # 6. Fundamental Details (Row 2 Right)
        ax6 = fig.add_subplot(gs[1, 2])
        self._plot_fundamental_details(ax6, analysis_result)
        
        # 7. Score Calculation Flow (Row 3 - Full Width)
        ax7 = fig.add_subplot(gs[2, :])
        self._plot_calculation_flow(ax7, analysis_result)
        
        # 8. Final Probability Breakdown (Row 4 Left + Middle)
        ax8 = fig.add_subplot(gs[3, 0:2])
        self._plot_probability_breakdown(ax8, analysis_result)
        
        # 9. Risk-Reward Visual (Row 4 Right)
        ax9 = fig.add_subplot(gs[3, 2])
        self._plot_risk_reward(ax9, analysis_result)
        
        plt.tight_layout()
        
        # Save the visualization
        filename = f'analysis_visualization_{symbol}_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\n✓ Visualization saved as: {filename}")
        
        plt.show()
        
        return filename
    
    def _plot_data_sources(self, ax, result):
        """Show all data sources used in analysis"""
        ax.set_title('Data Sources', fontweight='bold', fontsize=12)
        ax.axis('off')
        
        # Data sources
        sources = [
            ('📊 Price History', f"{result.get('data_points', 'N/A')} days", 'ShareSansar DB'),
            ('📰 News Articles', f"{result.get('sentiment_details', {}).get('total_articles', 0)} articles", 'ShareSansar'),
            ('🏢 Broker Data', f"{result.get('broker_trades', 'N/A')} trades", 'Floorsheet'),
            ('💰 Fundamentals', 'Mock Data', 'Generated'),
        ]
        
        y_pos = 0.9
        for icon_name, count, source in sources:
            ax.text(0.05, y_pos, f'{icon_name}', fontsize=10, fontweight='bold')
            ax.text(0.05, y_pos - 0.08, f'  • Count: {count}', fontsize=9)
            ax.text(0.05, y_pos - 0.15, f'  • Source: {source}', fontsize=9, color='gray')
            y_pos -= 0.25
    
    def _plot_component_scores(self, ax, result):
        """Bar chart of individual component scores"""
        ax.set_title('Component Scores', fontweight='bold', fontsize=12)
        
        scores = result.get('component_scores', {})
        components = ['Technical', 'Fundamental', 'Sentiment', 'Momentum', 'Broker']
        values = [
            scores.get('technical', 0),
            scores.get('fundamental', 0),
            scores.get('sentiment', 0),
            scores.get('momentum', 0),
            scores.get('broker', 50)  # Default broker score
        ]
        
        colors = [self.colors['technical'], self.colors['fundamental'], 
                 self.colors['sentiment'], self.colors['momentum'], 
                 self.colors['broker']]
        
        bars = ax.barh(components, values, color=colors, alpha=0.7)
        ax.set_xlim(0, 100)
        ax.set_xlabel('Score (0-100)', fontsize=10)
        ax.axvline(x=50, color='gray', linestyle='--', alpha=0.5, label='Neutral')
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, values)):
            ax.text(val + 2, i, f'{val:.1f}', va='center', fontsize=9)
        
        ax.grid(axis='x', alpha=0.3)
    
    def _plot_weight_distribution(self, ax, result):
        """Pie chart showing weight distribution"""
        ax.set_title('Weight Distribution', fontweight='bold', fontsize=12)
        
        weights = result.get('weights_used', {})
        labels = []
        values = []
        colors = []
        
        for key in ['technical', 'fundamental', 'sentiment', 'momentum', 'broker']:
            weight = weights.get(key, 0)
            if weight > 0:
                labels.append(key.capitalize())
                values.append(weight * 100)
                colors.append(self.colors[key])
        
        wedges, texts, autotexts = ax.pie(values, labels=labels, colors=colors,
                                           autopct='%1.1f%%', startangle=90)
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(9)
    
    def _plot_technical_details(self, ax, result):
        """Technical analysis breakdown"""
        ax.set_title('Technical Analysis Breakdown', fontweight='bold', fontsize=11)
        ax.axis('off')
        
        tech = result.get('technical_analysis', {})
        
        details = [
            ('Trend', tech.get('trend', 'N/A')),
            ('RSI', f"{tech.get('rsi', 0):.1f}"),
            ('MACD Signal', tech.get('macd_signal', 'N/A')),
            ('MA Signal', tech.get('moving_avg_signal', 'N/A')),
            ('Pattern', tech.get('pattern', 'None')),
            ('Support', f"NPR {tech.get('support', 0):.2f}"),
            ('Resistance', f"NPR {tech.get('resistance', 0):.2f}"),
        ]
        
        y_pos = 0.9
        for label, value in details:
            ax.text(0.05, y_pos, f'{label}:', fontweight='bold', fontsize=9)
            ax.text(0.5, y_pos, str(value), fontsize=9, color='#2c3e50')
            y_pos -= 0.13
    
    def _plot_sentiment_details(self, ax, result):
        """Sentiment analysis breakdown"""
        ax.set_title('Sentiment Analysis Breakdown', fontweight='bold', fontsize=11)
        ax.axis('off')
        
        sent = result.get('sentiment_details', {})
        
        # Sentiment distribution
        total = sent.get('total_articles', 0)
        pos = sent.get('positive_count', 0)
        neg = sent.get('negative_count', 0)
        neu = sent.get('neutral_count', 0)
        
        y_pos = 0.9
        ax.text(0.05, y_pos, f'Total Articles: {total}', fontweight='bold', fontsize=10)
        y_pos -= 0.15
        
        if total > 0:
            ax.text(0.05, y_pos, f'✅ Positive: {pos} ({pos/total*100:.0f}%)', 
                   fontsize=9, color='green')
            y_pos -= 0.12
            ax.text(0.05, y_pos, f'❌ Negative: {neg} ({neg/total*100:.0f}%)', 
                   fontsize=9, color='red')
            y_pos -= 0.12
            ax.text(0.05, y_pos, f'⚪ Neutral: {neu} ({neu/total*100:.0f}%)', 
                   fontsize=9, color='gray')
            y_pos -= 0.15
            
            ax.text(0.05, y_pos, f'Avg Score: {sent.get("average_score", 0):.3f}', 
                   fontsize=9)
            y_pos -= 0.12
            ax.text(0.05, y_pos, f'Weighted Score: {sent.get("weighted_average_score", 0):.3f}', 
                   fontsize=9)
            y_pos -= 0.12
            ax.text(0.05, y_pos, f'Overall: {sent.get("overall_sentiment", "N/A").upper()}', 
                   fontsize=9, fontweight='bold')
        else:
            ax.text(0.05, y_pos, 'No news articles found', fontsize=9, color='gray')
    
    def _plot_fundamental_details(self, ax, result):
        """Fundamental analysis breakdown"""
        ax.set_title('Fundamental Analysis Breakdown', fontweight='bold', fontsize=11)
        ax.axis('off')
        
        fund = result.get('fundamental_analysis', {})
        
        details = [
            ('P/E Ratio', f"{fund.get('pe_ratio', 0):.2f}"),
            ('P/B Ratio', f"{fund.get('pb_ratio', 0):.2f}"),
            ('EPS', f"{fund.get('eps', 0):.2f}"),
            ('Book Value', f"{fund.get('book_value', 0):.2f}"),
            ('ROE', f"{fund.get('roe', 0):.1f}%"),
            ('Debt/Equity', f"{fund.get('debt_equity', 0):.2f}"),
        ]
        
        y_pos = 0.9
        for label, value in details:
            ax.text(0.05, y_pos, f'{label}:', fontweight='bold', fontsize=9)
            ax.text(0.6, y_pos, str(value), fontsize=9, color='#2c3e50')
            y_pos -= 0.13
    
    def _plot_calculation_flow(self, ax, result):
        """Show how final probability is calculated"""
        ax.set_title('Probability Calculation Flow', fontweight='bold', fontsize=12)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 5)
        ax.axis('off')
        
        # Get scores and weights
        scores = result.get('component_scores', {})
        weights = result.get('weights_used', {})
        
        components = [
            ('Technical', scores.get('technical', 0), weights.get('technical', 0)),
            ('Fundamental', scores.get('fundamental', 0), weights.get('fundamental', 0)),
            ('Sentiment', scores.get('sentiment', 0), weights.get('sentiment', 0)),
            ('Momentum', scores.get('momentum', 0), weights.get('momentum', 0)),
            ('Broker', scores.get('broker', 50), weights.get('broker', 0)),
        ]
        
        # Draw calculation boxes
        x_start = 0.5
        y_pos = 3.5
        
        for i, (name, score, weight) in enumerate(components):
            color = self.colors.get(name.lower(), 'gray')
            
            # Score box
            rect = FancyBboxPatch((x_start, y_pos - 0.3), 1.2, 0.6,
                                 boxstyle="round,pad=0.05", 
                                 edgecolor=color, facecolor=color, alpha=0.3)
            ax.add_patch(rect)
            ax.text(x_start + 0.6, y_pos, f'{score:.1f}', 
                   ha='center', va='center', fontweight='bold', fontsize=9)
            ax.text(x_start + 0.6, y_pos - 0.6, name, 
                   ha='center', va='top', fontsize=8)
            
            # Weight
            ax.text(x_start + 0.6, y_pos + 0.5, f'×{weight:.2f}', 
                   ha='center', va='bottom', fontsize=8, color=color)
            
            # Arrow to contribution
            if i < len(components) - 1:
                ax.annotate('', xy=(x_start + 1.5, y_pos), xytext=(x_start + 1.2, y_pos),
                           arrowprops=dict(arrowstyle='->', color='gray', lw=1))
            
            x_start += 1.8
        
        # Sum all weighted scores
        final_prob = result.get('probability', 0)
        
        # Final result box
        rect_final = FancyBboxPatch((8.5, 2.5), 1.3, 1.5,
                                   boxstyle="round,pad=0.1", 
                                   edgecolor=self.colors['final'], 
                                   facecolor=self.colors['final'], alpha=0.3)
        ax.add_patch(rect_final)
        ax.text(9.15, 3.5, f'{final_prob:.2f}%', 
               ha='center', va='center', fontweight='bold', fontsize=14)
        ax.text(9.15, 2.8, 'Final\nProbability', 
               ha='center', va='center', fontsize=9)
        
        # Arrow to final
        ax.annotate('', xy=(8.5, 3.25), xytext=(7.5, 3.25),
                   arrowprops=dict(arrowstyle='->', color='black', lw=2))
        ax.text(8, 3.6, 'Σ', ha='center', fontsize=16, fontweight='bold')
    
    def _plot_probability_breakdown(self, ax, result):
        """Waterfall chart showing contribution of each component"""
        ax.set_title('Probability Contribution Breakdown', fontweight='bold', fontsize=12)
        
        scores = result.get('component_scores', {})
        weights = result.get('weights_used', {})
        
        components = ['Technical', 'Fundamental', 'Sentiment', 'Momentum', 'Broker']
        contributions = [
            (scores.get('technical', 0) / 100) * weights.get('technical', 0) * 100,
            (scores.get('fundamental', 0) / 100) * weights.get('fundamental', 0) * 100,
            (scores.get('sentiment', 0) / 100) * weights.get('sentiment', 0) * 100,
            (scores.get('momentum', 0) / 100) * weights.get('momentum', 0) * 100,
            (scores.get('broker', 50) / 100) * weights.get('broker', 0) * 100,
        ]
        
        # Create waterfall
        x_pos = np.arange(len(components) + 1)
        cumulative = [0] + list(np.cumsum(contributions))
        
        colors_list = [self.colors['technical'], self.colors['fundamental'], 
                      self.colors['sentiment'], self.colors['momentum'], 
                      self.colors['broker']]
        
        for i, (comp, contrib) in enumerate(zip(components, contributions)):
            ax.bar(i, contrib, bottom=cumulative[i], color=colors_list[i], 
                  alpha=0.7, edgecolor='black', linewidth=1)
            ax.text(i, cumulative[i] + contrib/2, f'{contrib:.2f}%', 
                   ha='center', va='center', fontweight='bold', fontsize=9)
        
        # Final bar
        final_prob = result.get('probability', 0)
        ax.bar(len(components), final_prob, color=self.colors['final'], 
              alpha=0.5, edgecolor='black', linewidth=2, linestyle='--')
        ax.text(len(components), final_prob/2, f'{final_prob:.2f}%', 
               ha='center', va='center', fontweight='bold', fontsize=11)
        
        ax.set_xticks(range(len(components) + 1))
        ax.set_xticklabels(components + ['TOTAL'], rotation=45, ha='right')
        ax.set_ylabel('Contribution to Probability (%)', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, max(final_prob * 1.2, 100))
    
    def _plot_risk_reward(self, ax, result):
        """Visual representation of risk vs reward"""
        ax.set_title('Risk-Reward Analysis', fontweight='bold', fontsize=12)
        
        rr_ratio = result.get('risk_reward_ratio', 1)
        potential_profit = result.get('potential_profit_pct', 0)
        potential_loss = result.get('potential_loss_pct', 0)
        
        # Create diverging bar
        categories = ['Potential\nProfit', 'Potential\nLoss']
        values = [potential_profit, -potential_loss]
        colors_bars = ['green', 'red']
        
        bars = ax.barh(categories, values, color=colors_bars, alpha=0.6)
        ax.axvline(x=0, color='black', linewidth=2)
        ax.set_xlabel('Percentage (%)', fontsize=10)
        
        # Add value labels
        for bar, val in zip(bars, [potential_profit, potential_loss]):
            x_pos = bar.get_width()
            label = f'{abs(val):.2f}%'
            if x_pos > 0:
                ax.text(x_pos + 1, bar.get_y() + bar.get_height()/2, label, 
                       va='center', fontweight='bold', fontsize=10)
            else:
                ax.text(x_pos - 1, bar.get_y() + bar.get_height()/2, label, 
                       va='center', ha='right', fontweight='bold', fontsize=10)
        
        # Add R:R ratio
        ax.text(0.5, 0.95, f'Risk:Reward = 1:{rr_ratio:.2f}', 
               transform=ax.transAxes, ha='center', fontsize=11, 
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        
        ax.grid(axis='x', alpha=0.3)


if __name__ == "__main__":
    # Test with sample data
    sample_result = {
        'symbol': 'IGI',
        'probability': 44.72,
        'data_points': 223,
        'broker_trades': 0,
        'component_scores': {
            'technical': 42.0,
            'fundamental': 62.5,
            'sentiment': 70.4,
            'momentum': 35.54,
            'broker': 50.0
        },
        'weights_used': {
            'technical': 0.25,
            'fundamental': 0.25,
            'sentiment': 0.2,
            'momentum': 0.15,
            'broker': 0.15
        },
        'technical_analysis': {
            'trend': 'Sideways',
            'rsi': 45.2,
            'macd_signal': 'Neutral',
            'moving_avg_signal': 'Hold',
            'pattern': 'Head and Shoulders',
            'support': 396.20,
            'resistance': 570.91
        },
        'sentiment_details': {
            'total_articles': 10,
            'positive_count': 9,
            'negative_count': 0,
            'neutral_count': 1,
            'average_score': 0.408,
            'weighted_average_score': 0.333,
            'overall_sentiment': 'positive'
        },
        'fundamental_analysis': {
            'pe_ratio': 20.0,
            'pb_ratio': 2.5,
            'eps': 20.45,
            'book_value': 163.60,
            'roe': 12.5,
            'debt_equity': 0.3
        },
        'risk_reward_ratio': 12.65,
        'potential_profit_pct': 39.59,
        'potential_loss_pct': 3.13
    }
    
    visualizer = AnalysisVisualizer()
    visualizer.visualize_complete_analysis(sample_result, 'IGI')
