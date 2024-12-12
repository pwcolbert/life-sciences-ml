# genome_analyzer/
# __init__.py
from .sequence_processor import SequenceProcessor
from .feature_extractor import FeatureExtractor

# sequence_processor.py
from Bio import SeqIO
from Bio.SeqUtils import GC
import numpy as np

class SequenceProcessor:
    """Class for processing genomic sequences."""
    
    def __init__(self):
        """Initialize the sequence processor."""
        self.sequences = {}
        
    def load_fasta(self, filepath):
        """
        Load sequences from a FASTA file.
        
        Args:
            filepath (str): Path to the FASTA file
            
        Returns:
            dict: Dictionary of loaded sequences
        """
        try:
            for record in SeqIO.parse(filepath, "fasta"):
                self.sequences[record.id] = str(record.seq)
            return self.sequences
        except Exception as e:
            raise Exception(f"Error loading FASTA file: {e}")
    
    def get_sequence(self, sequence_id):
        """
        Retrieve a specific sequence by ID.
        
        Args:
            sequence_id (str): ID of the sequence to retrieve
            
        Returns:
            str: The requested sequence
        """
        return self.sequences.get(sequence_id)

# feature_extractor.py
from Bio.SeqUtils import GC
import numpy as np
from collections import Counter

class FeatureExtractor:
    """Class for extracting features from genomic sequences."""
    
    def __init__(self, window_size=100):
        """
        Initialize the feature extractor.
        
        Args:
            window_size (int): Size of the sliding window for feature calculation
        """
        self.window_size = window_size
    
    def calculate_gc_content(self, sequence):
        """
        Calculate GC content of a sequence.
        
        Args:
            sequence (str): Input DNA sequence
            
        Returns:
            float: GC content percentage
        """
        return GC(sequence)
    
    def calculate_sequence_complexity(self, sequence):
        """
        Calculate sequence complexity using k-mer diversity.
        
        Args:
            sequence (str): Input DNA sequence
            
        Returns:
            float: Sequence complexity score
        """
        kmers = [sequence[i:i+3] for i in range(len(sequence)-2)]
        kmer_counts = Counter(kmers)
        total_kmers = len(kmers)
        
        # Calculate Shannon entropy
        entropy = 0
        for count in kmer_counts.values():
            probability = count / total_kmers
            entropy -= probability * np.log2(probability)
            
        return entropy
    
    def sliding_window_features(self, sequence):
        """
        Calculate features using a sliding window approach.
        
        Args:
            sequence (str): Input DNA sequence
            
        Returns:
            dict: Dictionary containing windowed features
        """
        features = {
            'gc_content': [],
            'complexity': []
        }
        
        for i in range(0, len(sequence) - self.window_size + 1):
            window = sequence[i:i + self.window_size]
            features['gc_content'].append(self.calculate_gc_content(window))
            features['complexity'].append(self.calculate_sequence_complexity(window))
            
        return features

# Example usage
if __name__ == "__main__":
    # Initialize processors
    processor = SequenceProcessor()
    extractor = FeatureExtractor(window_size=100)
    
    # Load sequences from FASTA file
    sequences = processor.load_fasta("example.fasta")
    
    # Process first sequence
    first_seq_id = list(sequences.keys())[0]
    sequence = processor.get_sequence(first_seq_id)
    
    # Extract features
    gc_content = extractor.calculate_gc_content(sequence)
    complexity = extractor.calculate_sequence_complexity(sequence)
    windowed_features = extractor.sliding_window_features(sequence)
    
    print(f"GC Content: {gc_content:.2f}%")
    print(f"Sequence Complexity: {complexity:.2f}")
