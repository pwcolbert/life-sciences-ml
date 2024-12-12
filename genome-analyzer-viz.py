import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd
from Bio import SeqIO
from genome_analyzer import SequenceProcessor, FeatureExtractor
import numpy as np
from pathlib import Path

class GenomeAnalysisValidator:
    """Class for validating and visualizing genomic analysis results."""
    
    def __init__(self, output_dir="analysis_results"):
        """
        Initialize the validator.
        
        Args:
            output_dir (str): Directory to save visualization results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def calculate_statistics(self, features_dict):
        """
        Calculate statistical measures for features.
        
        Args:
            features_dict (dict): Dictionary of feature arrays
            
        Returns:
            dict: Statistical measures
        """
        stats_dict = {}
        for feature_name, values in features_dict.items():
            stats_dict[feature_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'median': np.median(values),
                'skewness': stats.skew(values),
                'kurtosis': stats.kurtosis(values),
                'shapiro_test': stats.shapiro(values)
            }
        return stats_dict
    
    def plot_feature_distributions(self, features_dict, sequence_id):
        """
        Plot feature distributions.
        
        Args:
            features_dict (dict): Dictionary of feature arrays
            sequence_id (str): Identifier for the sequence
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Feature Distributions for {sequence_id}')
        
        # Flatten axes for easier iteration
        axes = axes.flatten()
        
        for idx, (feature_name, values) in enumerate(features_dict.items()):
            # Histogram
            sns.histplot(values, kde=True, ax=axes[idx])
            axes[idx].set_title(f'{feature_name} Distribution')
            axes[idx].set_xlabel(feature_name)
            axes[idx].set_ylabel('Count')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{sequence_id}_distributions.png')
        plt.close()
    
    def plot_windowed_analysis(self, window_features, sequence_id):
        """
        Plot windowed feature analysis.
        
        Args:
            window_features (dict): Dictionary of windowed features
            sequence_id (str): Identifier for the sequence
        """
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        fig.suptitle(f'Windowed Analysis for {sequence_id}')
        
        # Plot GC content variation
        axes[0].plot(window_features['gc_content'])
        axes[0].set_title('GC Content Variation')
        axes[0].set_xlabel('Window Position')
        axes[0].set_ylabel('GC Content (%)')
        
        # Plot complexity variation
        axes[1].plot(window_features['complexity'])
        axes[1].set_title('Sequence Complexity Variation')
        axes[1].set_xlabel('Window Position')
        axes[1].set_ylabel('Complexity Score')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{sequence_id}_windowed_analysis.png')
        plt.close()
    
    def create_summary_report(self, stats_dict, sequence_id):
        """
        Create a summary report of the analysis.
        
        Args:
            stats_dict (dict): Dictionary of statistical measures
            sequence_id (str): Identifier for the sequence
            
        Returns:
            str: Summary report
        """
        report = [f"Statistical Analysis Report for {sequence_id}"]
        report.append("-" * 50)
        
        for feature_name, stats_values in stats_dict.items():
            report.append(f"\n{feature_name} Statistics:")
            report.append(f"Mean: {stats_values['mean']:.3f}")
            report.append(f"Standard Deviation: {stats_values['std']:.3f}")
            report.append(f"Median: {stats_values['median']:.3f}")
            report.append(f"Skewness: {stats_values['skewness']:.3f}")
            report.append(f"Kurtosis: {stats_values['kurtosis']:.3f}")
            
            # Add Shapiro-Wilk test results
            statistic, p_value = stats_values['shapiro_test']
            report.append(f"Normality Test (Shapiro-Wilk):")
            report.append(f"  Statistic: {statistic:.3f}")
            report.append(f"  p-value: {p_value:.3f}")
            report.append(f"  Distribution is {'normal' if p_value > 0.05 else 'non-normal'}")
        
        return "\n".join(report)

def run_enhanced_analysis():
    """
    Run enhanced analysis with visualization and validation.
    """
    # Generate test data
    test_file = "test_sequences.fasta"
    create_test_fasta(test_file)
    
    # Initialize components
    processor = SequenceProcessor()
    extractor = FeatureExtractor(window_size=100)
    validator = GenomeAnalysisValidator()
    
    # Load and analyze sequences
    sequences = processor.load_fasta(test_file)
    
    for seq_id, sequence in sequences.items():
        print(f"\nAnalyzing {seq_id}...")
        
        # Extract features
        window_features = extractor.sliding_window_features(sequence)
        
        # Calculate statistics
        stats_dict = validator.calculate_statistics(window_features)
        
        # Generate visualizations
        validator.plot_feature_distributions(window_features, seq_id)
        validator.plot_windowed_analysis(window_features, seq_id)
        
        # Create and save report
        report = validator.create_summary_report(stats_dict, seq_id)
        report_path = validator.output_dir / f"{seq_id}_report.txt"
        with open(report_path, "w") as f:
            f.write(report)
        
        print(f"Analysis complete for {seq_id}")
        print(f"Results saved in {validator.output_dir}")

if __name__ == "__main__":
    # Set style for better-looking plots
    plt.style.use('seaborn')
    sns.set_palette("husl")
    
    # Run analysis
    run_enhanced_analysis()
