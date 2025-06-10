#!/usr/bin/env python3
"""
Quick Analysis Summary Generator
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

def main():
    print("Generating quick analysis summary...")
    
    # Load data files
    input_files = {
        'blog': 'inputs/master_blog_comparison_slow4g_2025-06-10T09-41-32-118Z.csv',
        'about': 'inputs/master_about_comparison_slow4g_2025-06-10T10-11-32-511Z.csv',
        'blogPost': 'inputs/master_blogPost_comparison_slow4g_2025-06-10T10-38-29-481Z.csv'
    }
    
    all_results = {}
    
    for page_name, file_path in input_files.items():
        print(f"Processing {page_name}...")
        df = pd.read_csv(file_path)
        
        # Extract metrics
        metrics_data = []
        for _, row in df.iterrows():
            metrics = {
                'FCP': row['First Contentful Paint_Avg_Score_%'],
                'LCP': row['Largest Contentful Paint_Avg_Score_%'],
                'SI': row['Speed Index_Avg_Score_%'],
                'TTI': row['Interactive_Avg_Score_%'],
                'TBT': row['Total Blocking Time_Avg_Score_%'],
                'CLS': row['Cumulative Layout Shift_Avg_Score_%']
            }
            
            framework = row['Framework']
            strategy = row['Strategy']
            avg_score = np.mean(list(metrics.values()))
            
            metrics_data.append({
                'page': page_name,
                'framework': framework,
                'strategy': strategy,
                'avg_score': avg_score,
                'metrics': metrics
            })
        
        all_results[page_name] = metrics_data
    
    # Generate summary
    print("\n" + "="*60)
    print("COMPREHENSIVE ANALYSIS SUMMARY")
    print("="*60)
    
    # Overall page performance
    page_scores = {}
    for page_name, results in all_results.items():
        page_avg = np.mean([r['avg_score'] for r in results])
        page_scores[page_name] = page_avg
        print(f"{page_name.upper()} average score: {page_avg:.1f}%")
    
    best_page = max(page_scores, key=page_scores.get)
    print(f"\nBEST PERFORMING PAGE: {best_page.upper()} ({page_scores[best_page]:.1f}%)")
    
    # Best combinations per page
    print("\nBEST COMBINATIONS PER PAGE:")
    for page_name, results in all_results.items():
        best_combo = max(results, key=lambda x: x['avg_score'])
        print(f"{page_name.upper()}: {best_combo['framework']} + {best_combo['strategy']} ({best_combo['avg_score']:.1f}%)")
    
    # Overall best framework and strategy
    all_data = []
    for results in all_results.values():
        all_data.extend(results)
    
    framework_scores = {}
    strategy_scores = {}
    
    for item in all_data:
        fw = item['framework']
        st = item['strategy']
        score = item['avg_score']
        
        if fw not in framework_scores:
            framework_scores[fw] = []
        framework_scores[fw].append(score)
        
        if st not in strategy_scores:
            strategy_scores[st] = []
        strategy_scores[st].append(score)
    
    # Calculate averages
    fw_avg = {fw: np.mean(scores) for fw, scores in framework_scores.items()}
    st_avg = {st: np.mean(scores) for st, scores in strategy_scores.items()}
    
    best_framework = max(fw_avg, key=fw_avg.get)
    best_strategy = max(st_avg, key=st_avg.get)
    
    print(f"\nOVERALL BEST FRAMEWORK: {best_framework} ({fw_avg[best_framework]:.1f}%)")
    print(f"OVERALL BEST STRATEGY: {best_strategy} ({st_avg[best_strategy]:.1f}%)")
    
    print("\nFRAMEWORK RANKINGS:")
    for i, (fw, score) in enumerate(sorted(fw_avg.items(), key=lambda x: x[1], reverse=True), 1):
        print(f"{i}. {fw}: {score:.1f}%")
    
    print("\nSTRATEGY RANKINGS:")
    for i, (st, score) in enumerate(sorted(st_avg.items(), key=lambda x: x[1], reverse=True), 1):
        print(f"{i}. {st}: {score:.1f}%")
    
    # Save summary to file
    summary = {
        'page_scores': page_scores,
        'best_page': best_page,
        'framework_averages': fw_avg,
        'strategy_averages': st_avg,
        'best_framework': best_framework,
        'best_strategy': best_strategy,
        'overall_average': np.mean([item['avg_score'] for item in all_data])
    }
    
    with open('output/quick_analysis_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save text report
    with open('output/QUICK_ANALYSIS_REPORT.txt', 'w', encoding='utf-8') as f:
        f.write("COMPREHENSIVE MULTI-PAGE PERFORMANCE ANALYSIS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Best Performing Page: {best_page.upper()} ({page_scores[best_page]:.1f}%)\n")
        f.write(f"Best Framework Overall: {best_framework} ({fw_avg[best_framework]:.1f}%)\n")
        f.write(f"Best Strategy Overall: {best_strategy} ({st_avg[best_strategy]:.1f}%)\n")
        f.write(f"Overall Average Score: {summary['overall_average']:.1f}%\n\n")
        
        f.write("PAGE PERFORMANCE:\n")
        for page, score in sorted(page_scores.items(), key=lambda x: x[1], reverse=True):
            f.write(f"- {page.upper()}: {score:.1f}%\n")
        
        f.write("\nFRAMEWORK RANKINGS:\n")
        for i, (fw, score) in enumerate(sorted(fw_avg.items(), key=lambda x: x[1], reverse=True), 1):
            f.write(f"{i}. {fw}: {score:.1f}%\n")
        
        f.write("\nSTRATEGY RANKINGS:\n")
        for i, (st, score) in enumerate(sorted(st_avg.items(), key=lambda x: x[1], reverse=True), 1):
            f.write(f"{i}. {st}: {score:.1f}%\n")
    
    print(f"\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("Results saved to:")
    print("- output/quick_analysis_summary.json")
    print("- output/QUICK_ANALYSIS_REPORT.txt")
    print("- Individual page results in output/[page]/ folders")
    print("="*60)

if __name__ == "__main__":
    main()
