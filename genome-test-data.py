import random
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import numpy as np
from genome_analyzer import SequenceProcessor, FeatureExtractor

def generate_random_dna(length, gc_bias=0.5):
    """
    Generate a random DNA sequence with specified GC content bias.
    
    Args:
        length (int): Length of sequence to generate
        gc_bias (float): Probability of generating G or C (0.0 to 1.0)
    
    Returns:
        str: Generated DNA sequence
    """
    bases = []
    for _ in range(length):
        if random.random() < gc_bias:
            bases.append(random.choice(['G', 'C']))
        else:
            bases.append(random.choice(['A', 'T']))
    return ''.join(bases)

def generate_sequence_with_motifs(length, motif, num_motifs):
    """
    Generate a sequence with specific motifs inserted.
    
    Args:
        length (int): Length of sequence to generate
        motif (str): Motif to insert
        num_motifs (int): Number of motifs to insert
    
    Returns:
        str: Generated DNA sequence
    """
    # Generate background sequence
    sequence = generate_random_dna(length)
    sequence = list(sequence)
    
    # Insert motifs at random positions
    for _ in range(num_motifs):
        pos = random.randint(0, length - len(motif))
        sequence[pos:pos + len(motif)] = motif
    
    return ''.join(sequence)

def create_test_fasta(filename, num_sequences=5):
    """
    Create a FASTA file with test sequences.
    
    Args:
        filename (str): Output FASTA filename
        num_sequences (int): Number of sequences to generate
    """
    sequences = []
    
    # Generate different types of test sequences
    for i in range(num_sequences):
        if i % 3 == 0:
            # High GC content sequence
            seq = generate_random_dna(1000, gc_bias=0.7)
            desc = "High GC content sequence"
        elif i % 3 == 1:
            # Sequence with motifs
            seq = generate_sequence_with_motifs(1000, "ATCGATCG", 5)
            desc = "Sequence with ATCGATCG motifs"
        else:
            # Random sequence
            seq = generate_random_dna(1000, gc_bias=0.5)
            desc = "Random sequence"
        
        # Create SeqRecord
        record = SeqRecord(
            Seq(seq),
            id=f"seq_{i+1}",
            description=desc
        )
        sequences.append(record)
    
    # Write sequences to FASTA file
    SeqIO.write(sequences, filename, "fasta")

def test_genome_analyzer():
    """
    Test the GenomeAnalyzer package with generated data.
    """
    # Generate test data
    test_file = "test_sequences.fasta"
    create_test_fasta(test_file)
    
    # Initialize analyzers
    processor = SequenceProcessor()
    extractor = FeatureExtractor(window_size=100)
    
    # Load and analyze sequences
    sequences = processor.load_fasta(test_file)
    
    print("Analysis Results:")
    print("-" * 50)
    
    for seq_id, sequence in sequences.items():
        print(f"\nAnalyzing {seq_id}:")
        
        # Calculate basic features
        gc_content = extractor.calculate_gc_content(sequence)
        complexity = extractor.calculate_sequence_complexity(sequence)
        
        print(f"GC Content: {gc_content:.2f}%")
        print(f"Sequence Complexity: {complexity:.2f}")
        
        # Calculate windowed features
        window_features = extractor.sliding_window_features(sequence)
        
        print("Window Analysis:")
        print(f"Average Window GC: {np.mean(window_features['gc_content']):.2f}%")
        print(f"Average Window Complexity: {np.mean(window_features['complexity']):.2f}")

if __name__ == "__main__":
    test_genome_analyzer()
