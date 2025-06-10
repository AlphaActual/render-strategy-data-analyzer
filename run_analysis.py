#!/usr/bin/env python3
"""
🚀 Comprehensive Multi-Page Performance Analysis Runner
Automated script to execute the complete analysis pipeline
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from IPython.display import display, HTML
import warnings
import os
import json
from pathlib import Path
import time
warnings.filterwarnings('ignore')

def setup_environment():
    """Set up the analysis environment"""
    print("Setting up analysis environment...")
    
    # Set style for better visualizations
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Configure plotly
    import plotly.io as pio
    pio.renderers.default = "png"  # Use PNG for script execution
    
    # Create output directory structure
    output_dir = Path('output')
    pages = ['blog', 'about', 'blogPost']
    
    for page in pages:
        page_dir = output_dir / page
        page_dir.mkdir(parents=True, exist_ok=True)
    
    # Create comparison directory
    comparison_dir = output_dir / 'cross_page_comparison'
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    print("Environment setup complete!")
    return output_dir

def load_data():
    """Load all input data files"""
    print("Loading data files...")
    
    input_files = {
        'blog': 'inputs/master_blog_comparison_slow4g_2025-06-10T09-41-32-118Z.csv',
        'about': 'inputs/master_about_comparison_slow4g_2025-06-10T10-11-32-511Z.csv',
        'blogPost': 'inputs/master_blogPost_comparison_slow4g_2025-06-10T10-38-29-481Z.csv'
    }
    
    datasets = {}
    for page_name, file_path in input_files.items():
        try:
            df = pd.read_csv(file_path)
            df['Page_Type'] = page_name
            datasets[page_name] = df
            print(f"SUCCESS {page_name.upper()}: {df.shape}")
        except Exception as e:
            print(f"ERROR loading {page_name}: {e}")
    
    print(f"Total pages loaded: {len(datasets)}")
    return datasets

def process_data(datasets):
    """Process and clean all datasets"""
    print("🔧 Processing data...")
    
    def process_page_data(df, page_name):
        metrics_data = []
        
        for _, row in df.iterrows():
            base_info = {
                'Page_Type': page_name,
                'App_Name': row['App_Name'],
                'Framework': row['Framework'],
                'Strategy': row['Strategy'],
                'Total_Runs': row['Total_Runs']
            }
            
            metrics = {
                'FCP': {'value': row['First Contentful Paint_Avg_Value'], 'score': row['First Contentful Paint_Avg_Score_%']},
                'LCP': {'value': row['Largest Contentful Paint_Avg_Value'], 'score': row['Largest Contentful Paint_Avg_Score_%']},
                'SI': {'value': row['Speed Index_Avg_Value'], 'score': row['Speed Index_Avg_Score_%']},
                'TTI': {'value': row['Interactive_Avg_Value'], 'score': row['Interactive_Avg_Score_%']},
                'TBT': {'value': row['Total Blocking Time_Avg_Value'], 'score': row['Total Blocking Time_Avg_Score_%']},
                'CLS': {'value': row['Cumulative Layout Shift_Avg_Value'], 'score': row['Cumulative Layout Shift_Avg_Score_%']}
            }
            
            for metric_name, metric_data in metrics.items():
                metrics_data.append({
                    **base_info,
                    'Metric': metric_name,
                    'Value': metric_data['value'],
                    'Score': metric_data['score']
                })
        
        return pd.DataFrame(metrics_data)
    
    cleaned_datasets = {}
    for page_name, df in datasets.items():
        cleaned_df = process_page_data(df, page_name)
        cleaned_datasets[page_name] = cleaned_df
        print(f"✅ {page_name.upper()} processed: {cleaned_df.shape}")
    
    # Combine all data
    all_data = pd.concat(cleaned_datasets.values(), ignore_index=True)
    print(f"🔗 Combined dataset: {all_data.shape}")
    
    return cleaned_datasets, all_data

def analyze_individual_pages(cleaned_datasets):
    """Analyze each page individually"""
    print("📊 Analyzing individual pages...")
    
    def analyze_page_performance(df, page_name):
        print(f"  🔍 Analyzing {page_name.upper()}...")
        
        # Create pivot tables
        values_pivot = df.pivot_table(
            index=['Framework', 'Strategy'], 
            columns='Metric', 
            values='Value', 
            aggfunc='mean'
        ).round(3)
        
        scores_pivot = df.pivot_table(
            index=['Framework', 'Strategy'], 
            columns='Metric', 
            values='Score', 
            aggfunc='mean'
        ).round(1)
        
        # Calculate rankings
        framework_rankings = df.groupby('Framework')['Score'].mean().sort_values(ascending=False)
        strategy_rankings = df.groupby('Strategy')['Score'].mean().sort_values(ascending=False)
        combination_rankings = df.groupby(['Framework', 'Strategy'])['Score'].mean().sort_values(ascending=False)
        
        # Best performers by metric
        metric_leaders = {}
        metrics_list = ['FCP', 'LCP', 'SI', 'TTI', 'TBT', 'CLS']
        for metric in metrics_list:
            metric_data = df[df['Metric'] == metric]
            best = metric_data.loc[metric_data['Score'].idxmax()]
            metric_leaders[metric] = {
                'framework': best['Framework'],
                'strategy': best['Strategy'],
                'score': best['Score'],
                'value': best['Value']
            }
        
        return {
            'values_pivot': values_pivot,
            'scores_pivot': scores_pivot,
            'framework_rankings': framework_rankings,
            'strategy_rankings': strategy_rankings,
            'combination_rankings': combination_rankings,
            'metric_leaders': metric_leaders
        }
    
    page_analyses = {}
    for page_name, df in cleaned_datasets.items():
        analysis = analyze_page_performance(df, page_name)
        page_analyses[page_name] = analysis
    
    return page_analyses

def create_visualizations(cleaned_datasets, page_analyses):
    """Create visualizations for all pages"""
    print("📈 Creating visualizations...")
    
    def create_page_visualizations(df, page_name, analysis_results):
        print(f"  📊 Creating {page_name.upper()} visualizations...")
        
        # 1. Performance Heatmap
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Scores heatmap
        sns.heatmap(analysis_results['scores_pivot'], annot=True, cmap='RdYlGn', center=75, 
                   fmt='.1f', ax=ax1, cbar_kws={'label': 'Score (%)'}, linewidths=0.5)
        ax1.set_title(f'🎯 {page_name.upper()} Performance Scores\n(Higher is Better)', 
                     fontsize=14, fontweight='bold')
        ax1.set_xlabel('Core Web Vitals Metrics', fontweight='bold')
        ax1.set_ylabel('Framework + Strategy', fontweight='bold')
        
        # Values heatmap
        sns.heatmap(analysis_results['values_pivot'], annot=True, cmap='RdYlGn_r', 
                   fmt='.2f', ax=ax2, cbar_kws={'label': 'Value (seconds/units)'}, linewidths=0.5)
        ax2.set_title(f'⏱️ {page_name.upper()} Performance Values\n(Lower is Generally Better)', 
                     fontsize=14, fontweight='bold')
        ax2.set_xlabel('Core Web Vitals Metrics', fontweight='bold')
        ax2.set_ylabel('Framework + Strategy', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'output/{page_name}/{page_name}_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Framework Comparison
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        metrics_list = ['FCP', 'LCP', 'SI', 'TTI', 'TBT', 'CLS']
        
        for i, metric in enumerate(metrics_list):
            row = i // 3
            col = i % 3
            
            metric_data = df[df['Metric'] == metric]
            
            sns.boxplot(data=metric_data, x='Framework', y='Score', ax=axes[row, col])
            axes[row, col].set_title(f'{metric} Scores Distribution', fontweight='bold')
            axes[row, col].set_ylabel('Score (%)')
            axes[row, col].grid(axis='y', alpha=0.3)
            
            # Add mean values
            means = metric_data.groupby('Framework')['Score'].mean()
            for j, (framework, mean_score) in enumerate(means.items()):
                axes[row, col].text(j, mean_score + 5, f'{mean_score:.1f}%', 
                                   ha='center', fontweight='bold', color='red')
        
        plt.suptitle(f'🏗️ {page_name.upper()} Framework Performance Distribution', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(f'output/{page_name}/{page_name}_framework_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Create visualizations for each page
    for page_name, df in cleaned_datasets.items():
        create_page_visualizations(df, page_name, page_analyses[page_name])

def cross_page_analysis(all_data):
    """Perform cross-page analysis"""
    print("🔄 Performing cross-page analysis...")
    
    # Overall page performance comparison
    page_performance = all_data.groupby('Page_Type')['Score'].agg(['mean', 'std', 'min', 'max']).round(2)
    page_performance.columns = ['Average_Score', 'Std_Dev', 'Min_Score', 'Max_Score']
    page_performance = page_performance.sort_values('Average_Score', ascending=False)
    
    # Framework performance across pages
    framework_across_pages = all_data.groupby(['Page_Type', 'Framework'])['Score'].mean().unstack().round(1)
    
    # Strategy performance across pages
    strategy_across_pages = all_data.groupby(['Page_Type', 'Strategy'])['Score'].mean().unstack().round(1)
    
    # Metric performance across pages
    metric_across_pages = all_data.groupby(['Page_Type', 'Metric'])['Score'].mean().unstack().round(1)
    
    print("✅ Cross-page analysis complete!")
    
    return page_performance, framework_across_pages, strategy_across_pages, metric_across_pages

def create_cross_page_visualizations(all_data, page_performance, framework_across_pages, strategy_across_pages):
    """Create cross-page comparison visualizations"""
    print("📊 Creating cross-page visualizations...")
    
    # Page Performance Comparison
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
    
    # Overall page scores
    page_scores = all_data.groupby('Page_Type')['Score'].mean().sort_values(ascending=False)
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    bars = ax1.bar(page_scores.index, page_scores.values, color=colors, alpha=0.8)
    ax1.set_title('📊 Overall Page Performance', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Average Score (%)')
    ax1.set_ylim(0, 100)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, score in zip(bars, page_scores.values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{score:.1f}%', ha='center', fontweight='bold')
    
    # Framework performance across pages
    framework_across_pages.plot(kind='bar', ax=ax2, color=['#FF9999', '#66B2FF', '#99FF99'])
    ax2.set_title('🏗️ Framework Performance Across Pages', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Average Score (%)')
    ax2.legend(title='Framework', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(axis='y', alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    # Strategy performance across pages
    strategy_across_pages.plot(kind='bar', ax=ax3)
    ax3.set_title('🔄 Strategy Performance Across Pages', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Average Score (%)')
    ax3.legend(title='Strategy', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.grid(axis='y', alpha=0.3)
    ax3.tick_params(axis='x', rotation=45)
    
    # Page consistency
    page_consistency = all_data.groupby('Page_Type')['Score'].std().sort_values()
    bars = ax4.bar(page_consistency.index, page_consistency.values, color=colors, alpha=0.8)
    ax4.set_title('📈 Page Performance Consistency\n(Lower = More Consistent)', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Standard Deviation')
    ax4.grid(axis='y', alpha=0.3)
    
    for bar, std in zip(bars, page_consistency.values):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{std:.1f}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('output/cross_page_comparison/cross_page_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Cross-page visualizations created!")

def save_results(cleaned_datasets, page_analyses, all_data, page_performance, 
                framework_across_pages, strategy_across_pages, metric_across_pages):
    """Save all analysis results"""
    print("💾 Saving results...")
    
    # Save individual page results
    def save_page_results(page_name, analysis_results, df):
        output_path = Path(f'output/{page_name}')
        
        # Save pivot tables
        analysis_results['values_pivot'].to_csv(output_path / f'{page_name}_performance_values.csv')
        analysis_results['scores_pivot'].to_csv(output_path / f'{page_name}_performance_scores.csv')
        
        # Save rankings
        analysis_results['framework_rankings'].to_csv(output_path / f'{page_name}_framework_rankings.csv')
        analysis_results['strategy_rankings'].to_csv(output_path / f'{page_name}_strategy_rankings.csv')
        analysis_results['combination_rankings'].to_csv(output_path / f'{page_name}_combination_rankings.csv')
          # Save detailed report
        with open(output_path / f'{page_name}_performance_report.txt', 'w', encoding='utf-8') as f:
            f.write(f"{page_name.upper()} PAGE PERFORMANCE REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("FRAMEWORK RANKINGS\n")
            f.write("-" * 20 + "\n")
            for i, (framework, score) in enumerate(analysis_results['framework_rankings'].items(), 1):
                f.write(f"{i}. {framework}: {score:.1f}%\n")
            
            f.write("\nSTRATEGY RANKINGS\n")
            f.write("-" * 20 + "\n")
            for i, (strategy, score) in enumerate(analysis_results['strategy_rankings'].items(), 1):
                f.write(f"{i}. {strategy}: {score:.1f}%\n")
            
            f.write("\nBEST COMBINATIONS\n")
            f.write("-" * 20 + "\n")
            for i, ((framework, strategy), score) in enumerate(analysis_results['combination_rankings'].head(5).items(), 1):
                f.write(f"{i}. {framework} + {strategy}: {score:.1f}%\n")
            
            f.write("\nMETRIC LEADERS\n")
            f.write("-" * 20 + "\n")
            for metric, leader in analysis_results['metric_leaders'].items():
                f.write(f"{metric}: {leader['framework']} + {leader['strategy']} ({leader['score']:.1f}%, {leader['value']:.3f})\n")
        
        # Save raw cleaned data
        df.to_csv(output_path / f'{page_name}_cleaned_data.csv', index=False)
        
        print(f"  ✅ {page_name.upper()} results saved")
    
    # Save results for each page
    for page_name in cleaned_datasets.keys():
        save_page_results(page_name, page_analyses[page_name], cleaned_datasets[page_name])
    
    # Save cross-page analysis results
    comparison_path = Path('output/cross_page_comparison')
    
    page_performance.to_csv(comparison_path / 'overall_page_performance.csv')
    framework_across_pages.to_csv(comparison_path / 'framework_across_pages.csv')
    strategy_across_pages.to_csv(comparison_path / 'strategy_across_pages.csv')
    metric_across_pages.to_csv(comparison_path / 'metric_across_pages.csv')
    all_data.to_csv(comparison_path / 'complete_dataset.csv', index=False)
    
    print("✅ All results saved!")

def generate_final_report(cleaned_datasets, page_analyses, all_data, page_performance):
    """Generate comprehensive final report"""
    print("Generating final report...")
    
    with open('output/COMPREHENSIVE_PERFORMANCE_REPORT.txt', 'w', encoding='utf-8') as f:
        f.write("COMPREHENSIVE MULTI-PAGE PERFORMANCE ANALYSIS REPORT\n")
        f.write("=" * 70 + "\n\n")
        
        # Executive Summary
        f.write("EXECUTIVE SUMMARY\n")
        f.write("-" * 20 + "\n")
        f.write(f"Pages Analyzed: {len(cleaned_datasets)}\n")
        f.write(f"Frameworks Tested: {', '.join(all_data['Framework'].unique())}\n")
        f.write(f"Strategies Tested: {len(all_data['Strategy'].unique())}\n")
        f.write(f"Total Test Combinations: {len(all_data.groupby(['Page_Type', 'Framework', 'Strategy']))}\n")
        f.write(f"Overall Average Score: {all_data['Score'].mean():.1f}%\n\n")
        
        # Page Rankings
        f.write("OVERALL PAGE PERFORMANCE RANKINGS\n")
        f.write("-" * 40 + "\n")
        for i, (page, stats) in enumerate(page_performance.iterrows(), 1):
            f.write(f"{i}. {page.upper()}: {stats['Average_Score']:.1f}% (±{stats['Std_Dev']:.1f})\n")
        f.write("\n")
        
        # Individual Page Analysis
        for page_name, analysis in page_analyses.items():
            f.write(f"{page_name.upper()} PAGE ANALYSIS\n")
            f.write("-" * 30 + "\n")
            best_combo = analysis['combination_rankings'].index[0]
            f.write(f"Best Framework: {analysis['framework_rankings'].index[0]} ({analysis['framework_rankings'].iloc[0]:.1f}%)\n")
            f.write(f"Best Strategy: {analysis['strategy_rankings'].index[0]} ({analysis['strategy_rankings'].iloc[0]:.1f}%)\n")
            f.write(f"Best Combination: {best_combo[0]} + {best_combo[1]} ({analysis['combination_rankings'].iloc[0]:.1f}%)\n")
            
            f.write("\nMetric Leaders:\n")
            for metric, leader in analysis['metric_leaders'].items():
                f.write(f"  • {metric}: {leader['framework']} + {leader['strategy']} ({leader['score']:.1f}%)\n")
            f.write("\n")
        
        # Key Recommendations
        overall_best_framework = all_data.groupby('Framework')['Score'].mean().sort_values(ascending=False)
        overall_best_strategy = all_data.groupby('Strategy')['Score'].mean().sort_values(ascending=False)
        
        f.write("KEY RECOMMENDATIONS\n")
        f.write("-" * 25 + "\n")
        f.write(f"1. FOR MAXIMUM PERFORMANCE: Use {overall_best_framework.index[0]} with {overall_best_strategy.index[0]}\n")
        f.write(f"2. BEST PAGE TYPE: {page_performance.index[0]} performs best overall\n")
        f.write(f"3. FOCUS AREA: Optimize {page_performance.index[-1]} page (lowest performer)\n")
        f.write(f"4. METRIC PRIORITIES: Address TBT and LCP metrics across all pages\n")
        f.write(f"5. TESTING APPROACH: Validate performance gains with A/B testing\n\n")
    
    # Generate summary JSON
    summary_stats = {
        'total_pages': len(cleaned_datasets),
        'total_frameworks': len(all_data['Framework'].unique()),
        'total_strategies': len(all_data['Strategy'].unique()),
        'overall_average_score': float(all_data['Score'].mean()),
        'best_page': page_performance.index[0],
        'best_framework': overall_best_framework.index[0],
        'best_strategy': overall_best_strategy.index[0],
        'analysis_timestamp': pd.Timestamp.now().isoformat()
    }
    
    with open('output/analysis_summary.json', 'w') as f:
        json.dump(summary_stats, f, indent=2)
    
    print("Final report generated!")

def main():
    """Main execution function"""
    start_time = time.time()
    
    print("STARTING COMPREHENSIVE MULTI-PAGE PERFORMANCE ANALYSIS")
    print("=" * 60)
    
    try:
        # Step 1: Setup
        output_dir = setup_environment()
        
        # Step 2: Load data
        datasets = load_data()
        
        # Step 3: Process data
        cleaned_datasets, all_data = process_data(datasets)
        
        # Step 4: Individual page analysis
        page_analyses = analyze_individual_pages(cleaned_datasets)
        
        # Step 5: Create visualizations
        create_visualizations(cleaned_datasets, page_analyses)
        
        # Step 6: Cross-page analysis
        page_performance, framework_across_pages, strategy_across_pages, metric_across_pages = cross_page_analysis(all_data)
        
        # Step 7: Cross-page visualizations
        create_cross_page_visualizations(all_data, page_performance, framework_across_pages, strategy_across_pages)
        
        # Step 8: Save results
        save_results(cleaned_datasets, page_analyses, all_data, page_performance, 
                    framework_across_pages, strategy_across_pages, metric_across_pages)
        
        # Step 9: Generate final report
        generate_final_report(cleaned_datasets, page_analyses, all_data, page_performance)
        
        # Summary
        elapsed_time = time.time() - start_time
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE!")
        print(f"Total execution time: {elapsed_time:.2f} seconds")
        print("All results saved to the output/ directory")
        print("Check individual page folders for detailed analysis")
        print("Check cross_page_comparison/ for comparative insights")
        print("Read COMPREHENSIVE_PERFORMANCE_REPORT.txt for executive summary")
        print("=" * 60)
        
        # Display key insights
        best_page = page_performance.index[0]
        overall_best_framework = all_data.groupby('Framework')['Score'].mean().sort_values(ascending=False)
        overall_best_strategy = all_data.groupby('Strategy')['Score'].mean().sort_values(ascending=False)
        
        print("\nKEY INSIGHTS:")
        print(f"• Best performing page: {best_page.upper()}")
        print(f"• Best framework overall: {overall_best_framework.index[0]}")
        print(f"• Best strategy overall: {overall_best_strategy.index[0]}")
        print(f"• Overall average score: {all_data['Score'].mean():.1f}%")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        raise

if __name__ == "__main__":
    main()
