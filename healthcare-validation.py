# validation_visualization.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, Tuple
from resource_optimizer import ResourceOptimizer
from data_generator import HealthcareDataGenerator
import datetime

class OptimizationValidator:
    """Validates and visualizes resource optimization results"""
    
    def __init__(self):
        self.data_generator = HealthcareDataGenerator()
        self.optimizer = ResourceOptimizer()
        
    def generate_and_validate(self, n_records: int = 1000) -> Tuple[pd.DataFrame, Dict]:
        """Generates data and runs validation tests"""
        data = self.data_generator.generate_utilization_data(n_records=n_records)
        report = self.optimizer.get_optimization_report(data)
        validation_results = self._validate_results(data, report)
        self._create_visualizations(data, report)
        return data, validation_results
    
    def _validate_results(self, data: pd.DataFrame, report: Dict) -> Dict:
        """Performs statistical validation of optimization results"""
        validation = {
            'data_quality': self._validate_data_quality(data),
            'pattern_validation': self._validate_patterns(data, report),
            'utilization_tests': self._validate_utilization(data, report),
            'statistical_tests': self._perform_statistical_tests(data)
        }
        return validation
    
    def _validate_data_quality(self, data: pd.DataFrame) -> Dict:
        """Validates data quality metrics"""
        return {
            'completeness': {
                'missing_values': data.isnull().sum().to_dict(),
                'completion_rate': (1 - data.isnull().sum() / len(data)).to_dict()
            },
            'consistency': {
                'value_ranges': {
                    col: {'min': data[col].min(), 'max': data[col].max()}
                    for col in data.select_dtypes(include=[np.number]).columns
                },
                'unique_values': {
                    col: data[col].nunique()
                    for col in data.columns
                }
            }
        }
    
    def _validate_patterns(self, data: pd.DataFrame, report: Dict) -> Dict:
        """Validates identified patterns against known data patterns"""
        pattern_validation = {}
        
        # Validate emergency department night patterns
        night_mask = data['timestamp'].dt.hour.between(22, 6)
        emergency_mask = data['department'] == 'Emergency'
        
        night_usage = data[night_mask & emergency_mask]['usage_time'].mean()
        day_usage = data[~night_mask & emergency_mask]['usage_time'].mean()
        
        pattern_validation['emergency_night_pattern'] = {
            'night_vs_day_ratio': night_usage / day_usage,
            'statistically_significant': self._check_significance(
                data[night_mask & emergency_mask]['usage_time'],
                data[~night_mask & emergency_mask]['usage_time']
            )
        }
        
        return pattern_validation
    
    def _validate_utilization(self, data: pd.DataFrame, report: Dict) -> Dict:
        """Validates utilization metrics and recommendations"""
        utilization_validation = {}
        
        # Validate department-level metrics
        for dept in data['department'].unique():
            dept_data = data[data['department'] == dept]
            reported_metrics = report['department_metrics'][dept]
            
            # Calculate actual metrics for validation
            actual_total_usage = dept_data['usage_time'].sum()
            actual_avg_usage = dept_data.groupby('resource_id')['usage_time'].mean().mean()
            
            utilization_validation[dept] = {
                'metric_accuracy': {
                    'total_usage': {
                        'reported': reported_metrics['total_usage_hours'],
                        'actual': actual_total_usage,
                        'error_percentage': abs(reported_metrics['total_usage_hours'] - actual_total_usage) / actual_total_usage * 100
                    },
                    'avg_usage': {
                        'reported': reported_metrics['avg_usage_per_resource'],
                        'actual': actual_avg_usage,
                        'error_percentage': abs(reported_metrics['avg_usage_per_resource'] - actual_avg_usage) / actual_avg_usage * 100
                    }
                }
            }
        
        return utilization_validation
    
    def _perform_statistical_tests(self, data: pd.DataFrame) -> Dict:
        """Performs statistical tests on the data"""
        tests = {}
        
        # Test for normality of usage times
        _, p_value = stats.normaltest(data['usage_time'])
        tests['usage_time_normality'] = {
            'test': 'normaltest',
            'p_value': p_value,
            'is_normal': p_value > 0.05
        }
        
        # Test for differences between departments
        f_stat, p_value = stats.f_oneway(*[
            group['usage_time'].values 
            for name, group in data.groupby('department')
        ])
        tests['department_differences'] = {
            'test': 'one_way_anova',
            'f_statistic': f_stat,
            'p_value': p_value,
            'significant_differences': p_value < 0.05
        }
        
        return tests
    
    def _check_significance(self, group1: pd.Series, group2: pd.Series) -> Dict:
        """Performs t-test to check for significant differences between groups"""
        t_stat, p_value = stats.ttest_ind(group1, group2)
        return {
            'test': 't_test',
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    def _create_visualizations(self, data: pd.DataFrame, report: Dict):
        """Creates and saves visualization plots"""
        self._plot_utilization_patterns(data)
        self._plot_department_comparisons(data)
        self._plot_temporal_patterns(data)
        self._plot_resource_distribution(data)
        plt.close('all')
    
    def _plot_utilization_patterns(self, data: pd.DataFrame):
        """Plots utilization patterns"""
        plt.figure(figsize=(15, 8))
        
        # Utilization by hour
        plt.subplot(2, 2, 1)
        hourly_usage = data.groupby(data['timestamp'].dt.hour)['usage_time'].mean()
        sns.lineplot(x=hourly_usage.index, y=hourly_usage.values)
        plt.title('Average Utilization by Hour')
        plt.xlabel('Hour of Day')
        plt.ylabel('Average Usage Time (minutes)')
        
        # Utilization by department
        plt.subplot(2, 2, 2)
        sns.boxplot(x='department', y='usage_time', data=data)
        plt.xticks(rotation=45)
        plt.title('Usage Time Distribution by Department')
        
        # Resource utilization heatmap
        plt.subplot(2, 2, 3)
        pivot_data = data.pivot_table(
            values='usage_time',
            index=data['timestamp'].dt.hour,
            columns='department',
            aggfunc='mean'
        )
        sns.heatmap(pivot_data, cmap='YlOrRd')
        plt.title('Utilization Heatmap')
        
        # Usage time distribution
        plt.subplot(2, 2, 4)
        sns.histplot(data=data, x='usage_time', bins=30)
        plt.title('Usage Time Distribution')
        
        plt.tight_layout()
        plt.savefig('utilization_patterns.png')
    
    def _plot_department_comparisons(self, data: pd.DataFrame):
        """Plots department comparison metrics"""
        plt.figure(figsize=(15, 6))
        
        # Department efficiency metrics
        dept_metrics = data.groupby('department').agg({
            'usage_time': ['mean', 'std'],
            'patient_count': 'mean',
            'staff_assigned': 'mean'
        }).round(2)
        
        # Efficiency ratio (patients per hour of usage)
        efficiency = (dept_metrics[('patient_count', 'mean')] / 
                     (dept_metrics[('usage_time', 'mean')] / 60))
        
        plt.subplot(1, 2, 1)
        efficiency.plot(kind='bar')
        plt.title('Department Efficiency (Patients/Hour)')
        plt.xticks(rotation=45)
        
        # Staff utilization
        plt.subplot(1, 2, 2)
        staff_util = (dept_metrics[('patient_count', 'mean')] / 
                     dept_metrics[('staff_assigned', 'mean')])
        staff_util.plot(kind='bar')
        plt.title('Patients per Staff Member')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('department_comparisons.png')
    
    def _plot_temporal_patterns(self, data: pd.DataFrame):
        """Plots temporal patterns in the data"""
        plt.figure(figsize=(15, 6))
        
        # Daily patterns
        plt.subplot(1, 2, 1)
        daily_usage = data.groupby(data['timestamp'].dt.date)['usage_time'].mean()
        sns.lineplot(x=daily_usage.index, y=daily_usage.values)
        plt.title('Daily Usage Patterns')
        plt.xticks(rotation=45)
        
        # Weekly patterns
        plt.subplot(1, 2, 2)
        weekly_usage = data.groupby(data['timestamp'].dt.dayofweek)['usage_time'].mean()
        sns.barplot(x=weekly_usage.index, y=weekly_usage.values)
        plt.title('Weekly Usage Patterns')
        plt.xlabel('Day of Week')
        
        plt.tight_layout()
        plt.savefig('temporal_patterns.png')
    
    def _plot_resource_distribution(self, data: pd.DataFrame):
        """Plots resource distribution and utilization"""
        plt.figure(figsize=(15, 6))
        
        # Resource usage distribution
        plt.subplot(1, 2, 1)
        resource_usage = data.groupby('resource_id')['usage_time'].mean().sort_values()
        sns.barplot(x=resource_usage.values, y=resource_usage.index)
        plt.title('Average Usage by Resource')
        
        # Resource utilization density
        plt.subplot(1, 2, 2)
        sns.kdeplot(data=data, x='usage_time', hue='department')
        plt.title('Usage Time Density by Department')
        
        plt.tight_layout()
        plt.savefig('resource_distribution.png')

def run_validation():
    """Runs the validation and visualization process"""
    validator = OptimizationValidator()
    data, validation_results = validator.generate_and_validate()
    
    print("\nValidation Results:")
    print("==================")
    
    print("\nData Quality Metrics:")
    for metric, results in validation_results['data_quality'].items():
        print(f"\n{metric.title()}:")
        for key, value in results.items():
            print(f"  {key}: {value}")
    
    print("\nPattern Validation:")
    for pattern, results in validation_results['pattern_validation'].items():
        print(f"\n{pattern}:")
        for key, value in results.items():
            print(f"  {key}: {value}")
    
    print("\nStatistical Tests:")
    for test, results in validation_results['statistical_tests'].items():
        print(f"\n{test}:")
        for key, value in results.items():
            print(f"  {key}: {value}")
    
    print("\nVisualizations have been saved to:")
    print("- utilization_patterns.png")
    print("- department_comparisons.png")
    print("- temporal_patterns.png")
    print("- resource_distribution.png")

if __name__ == "__main__":
    run_validation()
