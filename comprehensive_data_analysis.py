#!/usr/bin/env python3
"""
Comprehensive Render Strategy Performance Analysis
==================================================

This script analyzes performance data from 4 CSV files:
1. scripting_times.csv - Raw scripting performance times
2. scripting_times_percentage.csv - Scripting performance as percentages
3. build_times.csv - Build time measurements
4. bundle_sizes.csv - JavaScript bundle sizes

Generates scientific-quality visualizations and comprehensive analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set scientific plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class RenderStrategyAnalyzer:
    def __init__(self, input_dir="inputs", output_dir="output"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for organized output
        (self.output_dir / "figures").mkdir(exist_ok=True)
        (self.output_dir / "tables").mkdir(exist_ok=True)
        (self.output_dir / "analysis").mkdir(exist_ok=True)
        
        self.frameworks = ['Next.js', 'Nuxt.js', 'SvelteKit']
        self.strategies = ['CSR', 'SSR', 'SSG', 'ISR']
        self.colors = {
            'Next.js': '#0070f3',
            'Nuxt.js': '#00dc82', 
            'SvelteKit': '#ff3e00'
        }
        
        self.data = {}
        self.insights = []
        
    def load_data(self):
        """Load all CSV files"""
        print("Loading data files...")
        
        # Load scripting times (raw values)
        scripting_file = self.input_dir / "scripting_times.csv"
        if scripting_file.exists():
            self.data['scripting_raw'] = pd.read_csv(scripting_file)
            print(f"✓ Loaded {scripting_file}")
        
        # Load scripting percentages
        scripting_pct_file = self.input_dir / "scripting_times_percentage.csv"
        if scripting_pct_file.exists():
            self.data['scripting_pct'] = pd.read_csv(scripting_pct_file)
            print(f"✓ Loaded {scripting_pct_file}")
        
        # Load build times
        build_file = self.input_dir / "build_times.csv"
        if build_file.exists():
            self.data['build_times'] = pd.read_csv(build_file)
            print(f"✓ Loaded {build_file}")
        
        # Load bundle sizes
        bundle_file = self.input_dir / "bundle_sizes.csv"
        if bundle_file.exists():
            self.data['bundle_sizes'] = pd.read_csv(bundle_file)
            print(f"✓ Loaded {bundle_file}")
    
    def process_scripting_data(self):
        """Process scripting times data for analysis"""
        if 'scripting_raw' not in self.data:
            return None
            
        df = self.data['scripting_raw'].copy()
        
        # Skip URL row and convert to numeric
        numeric_data = df.iloc[2:].copy()  # Skip header and URL rows
        
        # Convert all columns to numeric
        for col in numeric_data.columns:
            numeric_data[col] = pd.to_numeric(numeric_data[col], errors='coerce')
        
        # Calculate statistics
        stats = {
            'mean': numeric_data.mean(),
            'median': numeric_data.median(),
            'std': numeric_data.std(),
            'min': numeric_data.min(),
            'max': numeric_data.max()
        }
        
        return stats
    
    def process_build_data(self):
        """Process build times data"""
        if 'build_times' not in self.data:
            return None
            
        df = self.data['build_times'].copy()
        
        # Parse framework and strategy from the first column
        df['Framework'] = df['Framework'].str.extract(r'(Next\.js|Nuxt\.js|SvelteKit)')
        df['Strategy'] = df['Framework'].str.extract(r'(CSR|SSR|SSG|ISR)')
        
        return df
    
    def create_performance_overview(self):
        """Create comprehensive performance overview visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Render Strategy Performance Analysis Overview', fontsize=16, fontweight='bold')
        
        # 1. Bundle Sizes Comparison
        if 'bundle_sizes' in self.data:
            ax1 = axes[0, 0]
            bundle_df = self.data['bundle_sizes']
            
            # Group by framework
            frameworks = []
            sizes = []
            strategies = []
            
            for _, row in bundle_df.iterrows():
                frameworks.append(row['Framework'])
                sizes.append(row['JS bundle size(kB)'])
                # Extract strategy from app name
                strategy = 'CSR' if 'CSR' in row['App_Name'] else \
                          'SSR' if 'SSR' in row['App_Name'] else \
                          'SSG' if 'SSG' in row['App_Name'] else 'ISR'
                strategies.append(strategy)
            
            # Create grouped bar plot
            df_bundles = pd.DataFrame({
                'Framework': frameworks,
                'Bundle Size (kB)': sizes,
                'Strategy': strategies
            })
            
            # Pivot for grouped bar chart
            pivot_bundles = df_bundles.pivot(index='Strategy', columns='Framework', values='Bundle Size (kB)')
            pivot_bundles.plot(kind='bar', ax=ax1, color=[self.colors[fw] for fw in pivot_bundles.columns])
            ax1.set_title('JavaScript Bundle Sizes by Framework and Strategy', fontweight='bold')
            ax1.set_ylabel('Bundle Size (kB)')
            ax1.legend(title='Framework')
            ax1.tick_params(axis='x', rotation=45)
        
        # 2. Build Times Analysis
        if 'build_times' in self.data:
            ax2 = axes[0, 1]
            build_df = self.data['build_times']
            
            # Extract framework and strategy info
            frameworks = []
            avg_times = []
            strategies = []
            
            for _, row in build_df.iterrows():
                framework_name = row['Framework']
                if 'Next.js' in framework_name:
                    fw = 'Next.js'
                elif 'Nuxt.js' in framework_name:
                    fw = 'Nuxt.js'
                elif 'SvelteKit' in framework_name:
                    fw = 'SvelteKit'
                else:
                    continue
                    
                strategy = 'CSR' if 'CSR' in framework_name else \
                          'SSR' if 'SSR' in framework_name else \
                          'SSG' if 'SSG' in framework_name else 'ISR'
                
                frameworks.append(fw)
                avg_times.append(row['Average'])
                strategies.append(strategy)
            
            df_builds = pd.DataFrame({
                'Framework': frameworks,
                'Build Time (s)': avg_times,
                'Strategy': strategies
            })
            
            pivot_builds = df_builds.pivot(index='Strategy', columns='Framework', values='Build Time (s)')
            pivot_builds.plot(kind='bar', ax=ax2, color=[self.colors[fw] for fw in pivot_builds.columns])
            ax2.set_title('Average Build Times by Framework and Strategy', fontweight='bold')
            ax2.set_ylabel('Build Time (seconds)')
            ax2.legend(title='Framework')
            ax2.tick_params(axis='x', rotation=45)
        
        # 3. Scripting Performance Distribution
        if 'scripting_raw' in self.data:
            ax3 = axes[1, 0]
            scripting_stats = self.process_scripting_data()
            
            if scripting_stats:
                strategies = list(scripting_stats['mean'].index)
                means = list(scripting_stats['mean'].values)
                stds = list(scripting_stats['std'].values)
                
                # Group by framework
                next_strategies = [s for s in strategies if 'Next.js' in s]
                nuxt_strategies = [s for s in strategies if 'Nuxt.js' in s]
                svelte_strategies = [s for s in strategies if 'SvelteKit' in s]
                
                x_pos = np.arange(len(self.strategies))
                width = 0.25
                
                # Get means for each framework and strategy
                next_means = [scripting_stats['mean'][f'Next.js {s}'] for s in self.strategies if f'Next.js {s}' in scripting_stats['mean'].index]
                nuxt_means = [scripting_stats['mean'][f'Nuxt.js {s}'] for s in self.strategies if f'Nuxt.js {s}' in scripting_stats['mean'].index]
                svelte_means = [scripting_stats['mean'][f'SvelteKit {s}'] for s in self.strategies if f'SvelteKit {s}' in scripting_stats['mean'].index]
                
                if next_means:
                    ax3.bar(x_pos - width, next_means, width, label='Next.js', color=self.colors['Next.js'], alpha=0.8)
                if nuxt_means:
                    ax3.bar(x_pos, nuxt_means, width, label='Nuxt.js', color=self.colors['Nuxt.js'], alpha=0.8)
                if svelte_means:
                    ax3.bar(x_pos + width, svelte_means, width, label='SvelteKit', color=self.colors['SvelteKit'], alpha=0.8)
                
                ax3.set_title('Average Scripting Performance Times', fontweight='bold')
                ax3.set_ylabel('Time (ms)')
                ax3.set_xlabel('Render Strategy')
                ax3.set_xticks(x_pos)
                ax3.set_xticklabels(self.strategies[:len(x_pos)])
                ax3.legend()
        
        # 4. Performance Efficiency Ratio (Bundle Size vs Performance)
        ax4 = axes[1, 1]
        if 'bundle_sizes' in self.data and 'scripting_raw' in self.data:
            # Create efficiency metric: Performance/Bundle Size ratio
            bundle_df = self.data['bundle_sizes']
            scripting_stats = self.process_scripting_data()
            
            if scripting_stats:
                efficiency_data = []
                
                for _, row in bundle_df.iterrows():
                    app_name = row['App_Name']
                    bundle_size = row['JS bundle size(kB)']
                    
                    # Find corresponding performance data
                    matching_col = None
                    for col in scripting_stats['mean'].index:
                        if app_name.replace(' ', '').replace('.', '') in col.replace(' ', '').replace('.', ''):
                            matching_col = col
                            break
                    
                    if matching_col:
                        performance = scripting_stats['mean'][matching_col]
                        efficiency = performance / bundle_size  # Higher is worse (more time per kB)
                        
                        framework = row['Framework']
                        efficiency_data.append({
                            'Framework': framework,
                            'Bundle Size': bundle_size,
                            'Performance': performance,
                            'Efficiency': efficiency,
                            'Strategy': app_name.split()[-1] if len(app_name.split()) > 1 else 'Unknown'
                        })
                
                if efficiency_data:
                    eff_df = pd.DataFrame(efficiency_data)
                    
                    # Scatter plot
                    for fw in self.frameworks:
                        fw_data = eff_df[eff_df['Framework'] == fw]
                        if not fw_data.empty:
                            ax4.scatter(fw_data['Bundle Size'], fw_data['Performance'], 
                                      label=fw, color=self.colors[fw], s=100, alpha=0.7)
                    
                    ax4.set_title('Performance vs Bundle Size Trade-off', fontweight='bold')
                    ax4.set_xlabel('Bundle Size (kB)')
                    ax4.set_ylabel('Performance Time (ms)')
                    ax4.legend()
                    
                    # Add trend line
                    if len(eff_df) > 1:
                        z = np.polyfit(eff_df['Bundle Size'], eff_df['Performance'], 1)
                        p = np.poly1d(z)
                        ax4.plot(eff_df['Bundle Size'], p(eff_df['Bundle Size']), "r--", alpha=0.8)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "figures" / "performance_overview.png", dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / "figures" / "performance_overview.pdf", bbox_inches='tight')
        return fig
    
    def create_detailed_analysis(self):
        """Create detailed analysis charts"""
        
        # 1. Detailed Bundle Size Analysis
        if 'bundle_sizes' in self.data:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            fig.suptitle('JavaScript Bundle Size Analysis', fontsize=14, fontweight='bold')
            
            bundle_df = self.data['bundle_sizes']
            
            # Box plot by framework
            frameworks = []
            sizes = []
            for _, row in bundle_df.iterrows():
                frameworks.append(row['Framework'])
                sizes.append(row['JS bundle size(kB)'])
            
            df_plot = pd.DataFrame({'Framework': frameworks, 'Bundle Size (kB)': sizes})
            
            sns.boxplot(data=df_plot, x='Framework', y='Bundle Size (kB)', ax=ax1)
            ax1.set_title('Bundle Size Distribution by Framework')
            
            # Strategy comparison
            strategies = []
            for _, row in bundle_df.iterrows():
                app_name = row['App_Name']
                strategy = 'CSR' if 'CSR' in app_name else \
                          'SSR' if 'SSR' in app_name else \
                          'SSG' if 'SSG' in app_name else 'ISR'
                strategies.append(strategy)
            
            df_plot['Strategy'] = strategies
            pivot_strategy = df_plot.pivot_table(values='Bundle Size (kB)', index='Strategy', columns='Framework', aggfunc='mean')
            
            pivot_strategy.plot(kind='bar', ax=ax2, color=[self.colors[fw] for fw in pivot_strategy.columns])
            ax2.set_title('Average Bundle Size by Strategy')
            ax2.set_ylabel('Bundle Size (kB)')
            ax2.tick_params(axis='x', rotation=45)
            ax2.legend(title='Framework')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / "figures" / "bundle_analysis.png", dpi=300, bbox_inches='tight')
        
        # 2. Build Performance Analysis
        if 'build_times' in self.data:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Build Performance Analysis', fontsize=14, fontweight='bold')
            
            build_df = self.data['build_times']
            
            # Extract framework info
            frameworks = []
            strategies = []
            for _, row in build_df.iterrows():
                framework_str = row['Framework']
                if 'Next.js' in framework_str:
                    fw = 'Next.js'
                elif 'Nuxt.js' in framework_str:
                    fw = 'Nuxt.js'
                elif 'SvelteKit' in framework_str:
                    fw = 'SvelteKit'
                else:
                    continue
                
                strategy = 'CSR' if 'CSR' in framework_str else \
                          'SSR' if 'SSR' in framework_str else \
                          'SSG' if 'SSG' in framework_str else 'ISR'
                
                frameworks.append(fw)
                strategies.append(strategy)
            
            build_df['Framework_Clean'] = frameworks
            build_df['Strategy_Clean'] = strategies
            
            # 1. Average build times
            ax1 = axes[0, 0]
            pivot_avg = build_df.pivot_table(values='Average', index='Strategy_Clean', columns='Framework_Clean', aggfunc='mean')
            pivot_avg.plot(kind='bar', ax=ax1, color=[self.colors[fw] for fw in pivot_avg.columns])
            ax1.set_title('Average Build Times')
            ax1.set_ylabel('Time (seconds)')
            ax1.tick_params(axis='x', rotation=45)
            
            # 2. Build time variability
            ax2 = axes[0, 1]
            build_variance = []
            framework_list = []
            strategy_list = []
            
            for _, row in build_df.iterrows():
                test_times = [row['Test1'], row['Test2'], row['Test3']]
                variance = np.std(test_times)
                build_variance.append(variance)
                framework_list.append(row['Framework_Clean'])
                strategy_list.append(row['Strategy_Clean'])
            
            var_df = pd.DataFrame({
                'Framework': framework_list,
                'Strategy': strategy_list,
                'Variance': build_variance
            })
            
            pivot_var = var_df.pivot_table(values='Variance', index='Strategy', columns='Framework', aggfunc='mean')
            pivot_var.plot(kind='bar', ax=ax2, color=[self.colors[fw] for fw in pivot_var.columns])
            ax2.set_title('Build Time Consistency (Lower is Better)')
            ax2.set_ylabel('Standard Deviation (seconds)')
            ax2.tick_params(axis='x', rotation=45)
            
            # 3. Heatmap of build times
            ax3 = axes[1, 0]
            pivot_heat = build_df.pivot_table(values='Average', index='Framework_Clean', columns='Strategy_Clean', aggfunc='mean')
            sns.heatmap(pivot_heat, annot=True, fmt='.1f', cmap='RdYlBu_r', ax=ax3)
            ax3.set_title('Build Times Heatmap (seconds)')
            
            # 4. Framework efficiency comparison
            ax4 = axes[1, 1]
            framework_avg = build_df.groupby('Framework_Clean')['Average'].mean().sort_values()
            framework_avg.plot(kind='bar', ax=ax4, color=[self.colors[fw] for fw in framework_avg.index])
            ax4.set_title('Overall Framework Build Performance')
            ax4.set_ylabel('Average Build Time (seconds)')
            ax4.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / "figures" / "build_analysis.png", dpi=300, bbox_inches='tight')
    
    def create_correlation_analysis(self):
        """Create correlation analysis between different metrics"""
        if not all(key in self.data for key in ['bundle_sizes', 'build_times']):
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Cross-Metric Correlation Analysis', fontsize=14, fontweight='bold')
        
        # Prepare combined dataset
        combined_data = []
        
        bundle_df = self.data['bundle_sizes']
        build_df = self.data['build_times']
        
        # Match bundle sizes with build times
        for _, bundle_row in bundle_df.iterrows():
            app_name = bundle_row['App_Name']
            framework = bundle_row['Framework']
            bundle_size = bundle_row['JS bundle size(kB)']
            
            # Find matching build time
            for _, build_row in build_df.iterrows():
                build_framework = build_row['Framework']
                if framework.replace('.', '').replace(' ', '').lower() in build_framework.replace('.', '').replace(' ', '').lower():
                    # Check strategy match
                    strategy_bundle = 'CSR' if 'CSR' in app_name else \
                                    'SSR' if 'SSR' in app_name else \
                                    'SSG' if 'SSG' in app_name else 'ISR'
                    strategy_build = 'CSR' if 'CSR' in build_framework else \
                                   'SSR' if 'SSR' in build_framework else \
                                   'SSG' if 'SSG' in build_framework else 'ISR'
                    
                    if strategy_bundle == strategy_build:
                        combined_data.append({
                            'Framework': framework,
                            'Strategy': strategy_bundle,
                            'Bundle Size': bundle_size,
                            'Build Time': build_row['Average']
                        })
                        break
        
        if combined_data:
            combined_df = pd.DataFrame(combined_data)
            
            # Correlation scatter plot
            ax1 = axes[0]
            for fw in self.frameworks:
                fw_data = combined_df[combined_df['Framework'] == fw]
                if not fw_data.empty:
                    ax1.scatter(fw_data['Bundle Size'], fw_data['Build Time'], 
                              label=fw, color=self.colors[fw], s=100, alpha=0.7)
            
            # Add correlation line
            if len(combined_df) > 1:
                correlation = combined_df['Bundle Size'].corr(combined_df['Build Time'])
                z = np.polyfit(combined_df['Bundle Size'], combined_df['Build Time'], 1)
                p = np.poly1d(z)
                ax1.plot(combined_df['Bundle Size'], p(combined_df['Bundle Size']), 
                        "r--", alpha=0.8, label=f'Correlation: {correlation:.3f}')
            
            ax1.set_xlabel('Bundle Size (kB)')
            ax1.set_ylabel('Build Time (seconds)')
            ax1.set_title('Bundle Size vs Build Time Correlation')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Strategy efficiency matrix
            ax2 = axes[1]
            strategy_metrics = combined_df.groupby(['Framework', 'Strategy']).agg({
                'Bundle Size': 'mean',
                'Build Time': 'mean'
            }).reset_index()
            
            # Create efficiency score (lower is better)
            strategy_metrics['Efficiency Score'] = (
                (strategy_metrics['Bundle Size'] / strategy_metrics['Bundle Size'].max()) + 
                (strategy_metrics['Build Time'] / strategy_metrics['Build Time'].max())
            ) / 2
            
            # Pivot for heatmap
            efficiency_pivot = strategy_metrics.pivot_table(
                values='Efficiency Score', 
                index='Framework', 
                columns='Strategy', 
                aggfunc='mean'
            )
            
            sns.heatmap(efficiency_pivot, annot=True, fmt='.3f', cmap='RdYlGn_r', ax=ax2,
                       cbar_kws={'label': 'Efficiency Score (Lower is Better)'})
            ax2.set_title('Framework-Strategy Efficiency Matrix')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "figures" / "correlation_analysis.png", dpi=300, bbox_inches='tight')
    
    def generate_summary_tables(self):
        """Generate summary tables for all metrics"""
        tables = {}
        
        # Bundle size summary
        if 'bundle_sizes' in self.data:
            bundle_df = self.data['bundle_sizes'].copy()
            
            # Add strategy column
            strategies = []
            for _, row in bundle_df.iterrows():
                app_name = row['App_Name']
                strategy = 'CSR' if 'CSR' in app_name else \
                          'SSR' if 'SSR' in app_name else \
                          'SSG' if 'SSG' in app_name else 'ISR'
                strategies.append(strategy)
            
            bundle_df['Strategy'] = strategies
            
            bundle_summary = bundle_df.pivot_table(
                values='JS bundle size(kB)', 
                index='Framework', 
                columns='Strategy', 
                aggfunc='mean'
            )
            tables['bundle_sizes'] = bundle_summary
        
        # Build time summary
        if 'build_times' in self.data:
            build_df = self.data['build_times'].copy()
            
            frameworks = []
            strategies = []
            for _, row in build_df.iterrows():
                framework_str = row['Framework']
                if 'Next.js' in framework_str:
                    fw = 'Next.js'
                elif 'Nuxt.js' in framework_str:
                    fw = 'Nuxt.js'
                elif 'SvelteKit' in framework_str:
                    fw = 'SvelteKit'
                else:
                    continue
                
                strategy = 'CSR' if 'CSR' in framework_str else \
                          'SSR' if 'SSR' in framework_str else \
                          'SSG' if 'SSG' in framework_str else 'ISR'
                
                frameworks.append(fw)
                strategies.append(strategy)
            
            build_df['Framework_Clean'] = frameworks
            build_df['Strategy_Clean'] = strategies
            
            build_summary = build_df.pivot_table(
                values='Average', 
                index='Framework_Clean', 
                columns='Strategy_Clean', 
                aggfunc='mean'
            )
            tables['build_times'] = build_summary
        
        # Save tables
        for table_name, table_df in tables.items():
            table_df.to_csv(self.output_dir / "tables" / f"{table_name}_summary.csv")
            
            # Create formatted table visualization
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.axis('tight')
            ax.axis('off')
            
            table = ax.table(cellText=table_df.round(2).values,
                           rowLabels=table_df.index,
                           colLabels=table_df.columns,
                           cellLoc='center',
                           loc='center')
            
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)
            
            # Style the table
            for i in range(len(table_df.columns)):
                table[(0, i)].set_facecolor('#4CAF50')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            for i in range(len(table_df.index)):
                table[(i+1, -1)].set_facecolor('#E0E0E0')
                table[(i+1, -1)].set_text_props(weight='bold')
            
            plt.title(f'{table_name.replace("_", " ").title()} Summary Table', 
                     fontsize=14, fontweight='bold', pad=20)
            plt.savefig(self.output_dir / "figures" / f"{table_name}_table.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        return tables
    
    def generate_insights(self):
        """Generate comprehensive insights from the analysis"""
        insights = []
        insights.append("# Render Strategy Performance Analysis - Comprehensive Insights\n")
        insights.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        insights.append("=" * 80 + "\n")
        
        # Bundle Size Analysis
        if 'bundle_sizes' in self.data:
            bundle_df = self.data['bundle_sizes']
            
            insights.append("## JavaScript Bundle Size Analysis\n")
            
            # Framework comparison
            framework_bundles = {}
            for _, row in bundle_df.iterrows():
                fw = row['Framework']
                size = row['JS bundle size(kB)']
                if fw not in framework_bundles:
                    framework_bundles[fw] = []
                framework_bundles[fw].append(size)
            
            for fw, sizes in framework_bundles.items():
                avg_size = np.mean(sizes)
                min_size = np.min(sizes)
                max_size = np.max(sizes)
                insights.append(f"- **{fw}**: Average {avg_size:.1f}kB (Range: {min_size:.1f}-{max_size:.1f}kB)\n")
            
            # Best and worst performers
            best_fw = min(framework_bundles.keys(), key=lambda x: np.mean(framework_bundles[x]))
            worst_fw = max(framework_bundles.keys(), key=lambda x: np.mean(framework_bundles[x]))
            
            insights.append(f"\n**Bundle Size Winner**: {best_fw} ({np.mean(framework_bundles[best_fw]):.1f}kB average)\n")
            insights.append(f"**Largest Bundles**: {worst_fw} ({np.mean(framework_bundles[worst_fw]):.1f}kB average)\n")
            
            # Bundle size impact
            size_diff = np.mean(framework_bundles[worst_fw]) - np.mean(framework_bundles[best_fw])
            perc_diff = (size_diff / np.mean(framework_bundles[best_fw])) * 100
            insights.append(f"**Size Difference**: {size_diff:.1f}kB ({perc_diff:.1f}% larger)\n\n")
        
        # Build Time Analysis
        if 'build_times' in self.data:
            build_df = self.data['build_times']
            
            insights.append("## Build Performance Analysis\n")
            
            framework_builds = {}
            for _, row in build_df.iterrows():
                framework_str = row['Framework']
                if 'Next.js' in framework_str:
                    fw = 'Next.js'
                elif 'Nuxt.js' in framework_str:
                    fw = 'Nuxt.js'
                elif 'SvelteKit' in framework_str:
                    fw = 'SvelteKit'
                else:
                    continue
                
                if fw not in framework_builds:
                    framework_builds[fw] = []
                framework_builds[fw].append(row['Average'])
            
            for fw, times in framework_builds.items():
                avg_time = np.mean(times)
                min_time = np.min(times)
                max_time = np.max(times)
                insights.append(f"- **{fw}**: Average {avg_time:.1f}s (Range: {min_time:.1f}-{max_time:.1f}s)\n")
            
            # Build performance rankings
            fastest_fw = min(framework_builds.keys(), key=lambda x: np.mean(framework_builds[x]))
            slowest_fw = max(framework_builds.keys(), key=lambda x: np.mean(framework_builds[x]))
            
            insights.append(f"\n**Fastest Builds**: {fastest_fw} ({np.mean(framework_builds[fastest_fw]):.1f}s average)\n")
            insights.append(f"**Slowest Builds**: {slowest_fw} ({np.mean(framework_builds[slowest_fw]):.1f}s average)\n")
            
            # Build consistency analysis
            consistency_scores = {}
            for fw, times in framework_builds.items():
                consistency_scores[fw] = np.std(times)
            
            most_consistent = min(consistency_scores.keys(), key=lambda x: consistency_scores[x])
            insights.append(f"**Most Consistent**: {most_consistent} (σ = {consistency_scores[most_consistent]:.2f}s)\n\n")
        
        # Strategy-specific insights
        insights.append("## Rendering Strategy Analysis\n")
        
        strategy_performance = {}
        
        if 'bundle_sizes' in self.data:
            for _, row in self.data['bundle_sizes'].iterrows():
                app_name = row['App_Name']
                strategy = 'CSR' if 'CSR' in app_name else \
                          'SSR' if 'SSR' in app_name else \
                          'SSG' if 'SSG' in app_name else 'ISR'
                
                if strategy not in strategy_performance:
                    strategy_performance[strategy] = {'bundle_sizes': [], 'build_times': []}
                strategy_performance[strategy]['bundle_sizes'].append(row['JS bundle size(kB)'])
        
        if 'build_times' in self.data:
            for _, row in self.data['build_times'].iterrows():
                framework_str = row['Framework']
                strategy = 'CSR' if 'CSR' in framework_str else \
                          'SSR' if 'SSR' in framework_str else \
                          'SSG' if 'SSG' in framework_str else 'ISR'
                
                if strategy in strategy_performance:
                    strategy_performance[strategy]['build_times'].append(row['Average'])
        
        for strategy, metrics in strategy_performance.items():
            if metrics['bundle_sizes'] and metrics['build_times']:
                avg_bundle = np.mean(metrics['bundle_sizes'])
                avg_build = np.mean(metrics['build_times'])
                insights.append(f"- **{strategy}**: {avg_bundle:.1f}kB bundles, {avg_build:.1f}s builds\n")
        
        # Recommendations
        insights.append("\n## Key Recommendations\n")
        
        if 'bundle_sizes' in self.data:
            # Bundle size recommendations
            smallest_bundles = min(framework_bundles.keys(), key=lambda x: np.mean(framework_bundles[x]))
            insights.append(f"1. **Bundle Optimization**: Consider {smallest_bundles} for minimal bundle sizes\n")
        
        if 'build_times' in self.data:
            # Build time recommendations  
            fastest_builds = min(framework_builds.keys(), key=lambda x: np.mean(framework_builds[x]))
            insights.append(f"2. **Build Performance**: {fastest_builds} offers fastest build times\n")
        
        # Strategy recommendations
        insights.append("3. **Strategy Selection**:\n")
        insights.append("   - SSG: Best for static content with predictable data\n")
        insights.append("   - ISR: Good balance for dynamic content with caching\n")
        insights.append("   - SSR: Necessary for real-time dynamic content\n")
        insights.append("   - CSR: Suitable for highly interactive applications\n\n")
        
        # Performance trade-offs
        insights.append("## Performance Trade-offs Summary\n")
        insights.append("- **Bundle Size vs Build Time**: Smaller bundles don't always mean faster builds\n")
        insights.append("- **Framework Choice**: Each framework optimizes different aspects\n")
        insights.append("- **Strategy Impact**: Rendering strategy significantly affects both metrics\n")
        insights.append("- **Consistency**: Consider build time variability for CI/CD pipelines\n\n")
          # Save insights
        with open(self.output_dir / "analysis" / "comprehensive_insights.md", 'w', encoding='utf-8') as f:
            f.writelines(insights)
        
        return insights
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("🚀 Starting Comprehensive Render Strategy Analysis...")
        print("=" * 60)
        
        # Load data
        self.load_data()
        
        # Generate visualizations
        print("\n📊 Generating performance overview...")
        self.create_performance_overview()
        
        print("📈 Creating detailed analysis charts...")
        self.create_detailed_analysis()
        
        print("🔄 Analyzing correlations...")
        self.create_correlation_analysis()
        
        # Generate tables
        print("📋 Generating summary tables...")
        tables = self.generate_summary_tables()
        
        # Generate insights
        print("💡 Generating comprehensive insights...")
        insights = self.generate_insights()
        
        # Create final summary
        summary = {
            'analysis_date': datetime.now().isoformat(),
            'files_analyzed': list(self.data.keys()),
            'frameworks_compared': self.frameworks,
            'strategies_analyzed': self.strategies,
            'outputs_generated': {
                'figures': ['performance_overview.png', 'bundle_analysis.png', 'build_analysis.png', 'correlation_analysis.png'],
                'tables': [f"{name}_summary.csv" for name in tables.keys()],
                'insights': 'comprehensive_insights.md'
            }
        }
        
        with open(self.output_dir / "analysis_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("\n✅ Analysis Complete!")
        print(f"📁 Results saved to: {self.output_dir}")
        print("📊 Generated:")
        print("   - Performance overview charts")
        print("   - Detailed analysis visualizations")
        print("   - Correlation analysis")
        print("   - Summary tables")
        print("   - Comprehensive insights report")
        print("\n🎯 Check the 'analysis' folder for detailed insights!")

if __name__ == "__main__":
    analyzer = RenderStrategyAnalyzer()
    analyzer.run_complete_analysis()
