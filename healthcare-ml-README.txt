healthcare-ml-README

1. `ResourceUtilization`: Handles pattern analysis through:
   - Preprocessing of utilization data
   - Clustering to identify usage patterns
   - Peak hour detection
   - Pattern analysis within clusters

2. `ResourceScheduler`: Manages resource allocation with:
   - Resource capacity tracking
   - Availability window management
   - Scheduling logic with conflict prevention
   - Current load monitoring

3. `ResourceOptimizer`: Main interface providing:
   - Department-level metrics analysis
   - Utilization pattern identification
   - Automated recommendations generation
   - Comprehensive optimization reporting

The package uses scikit-learn for clustering and provides detailed metrics and recommendations based on:
- Resource utilization rates
- Peak usage patterns
- Department-specific patterns
- Cross-department optimization opportunities
