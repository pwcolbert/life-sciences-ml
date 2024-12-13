genome-analyzer-README

1. SequenceProcessor Class:
   - Handles FASTA file loading and parsing using BioPython
   - Stores sequences in a dictionary for easy access
   - Provides methods to retrieve specific sequences

2. FeatureExtractor Class:
   - Calculates GC content using BioPython's utilities
   - Implements sequence complexity calculation using k-mer diversity and Shannon entropy
   - Supports sliding window analysis for feature extraction

Key features:
- Efficient sequence handling using BioPython
- Error handling for file operations
- Sliding window approach for local feature analysis
- Extensible architecture for adding new features