clinical-test-data-README

1. VitalsAnalyzer:
- Processes time-series vital signs data
- Calculates rolling statistics (mean, std) for each vital sign
- Detects anomalies using z-score method
- Handles multiple vital signs (heart rate, blood pressure, temperature, etc.)

2. OutcomeTracker:
- Analyzes different types of outcomes (mortality, readmission, complications)
- Calculates outcome distributions
- Identifies risk factors through statistical analysis
- Analyzes temporal patterns in outcomes

3. MedicationResponseAnalyzer:
- Evaluates medication effectiveness using pre/post vitals
- Tracks side effects and their severity
- Monitors adherence patterns
- Analyzes dosage-response relationships
- Calculates optimal dosage ranges

4. Vitals Data Generation:
- Creates time-series vital signs with realistic ranges
- Includes heart rate, blood pressure, temperature, etc.
- Adds random variations to simulate real measurements
- Maintains patient-specific baseline values

5. Medication Data Generation:
- Generates medication responses with realistic dosages
- Includes pre/post vitals measurements
- Simulates adherence patterns and side effects
- Creates time-stamped administration records

6. Outcomes Data Generation:
- Produces patient demographic information
- Generates various outcome types (mortality, readmission, etc.)
- Includes comorbidities and length of stay
- Creates admission dates and patient characteristics

Key Features:
- Comprehensive vital signs monitoring
- Statistical analysis of outcomes
- Detailed medication response tracking
- Time-series analysis capabilities
- Anomaly detection in vital signs
- Configurable number of patients and time periods
- Realistic value ranges for all parameters
- Consistent patient IDs across datasets
- Built-in test function to demonstrate usage