import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

class ClinicalDataGenerator:
    def __init__(self, num_patients=100, time_periods=30):
        self.num_patients = num_patients
        self.time_periods = time_periods
        self.patient_ids = [f'P{str(i).zfill(4)}' for i in range(num_patients)]
        
        # Define normal ranges for vital signs
        self.vital_ranges = {
            'heart_rate': (60, 100),
            'blood_pressure_systolic': (90, 140),
            'blood_pressure_diastolic': (60, 90),
            'temperature': (36.1, 37.2),
            'respiratory_rate': (12, 20),
            'oxygen_saturation': (95, 100)
        }
        
        # Define common medications
        self.medications = [
            'Lisinopril', 'Metformin', 'Amlodipine', 'Metoprolol',
            'Omeprazole', 'Simvastatin', 'Levothyroxine', 'Gabapentin'
        ]
        
        self.side_effects = [
            'Nausea', 'Headache', 'Dizziness', 'Fatigue',
            'Rash', 'Dry mouth', 'Insomnia', 'None'
        ]

    def generate_vitals_data(self):
        """Generate time-series vitals data for patients"""
        data_rows = []
        
        for patient_id in self.patient_ids:
            # Generate baseline values for this patient
            baselines = {
                vital: np.random.uniform(low, high)
                for vital, (low, high) in self.vital_ranges.items()
            }
            
            # Generate time series data
            start_date = datetime(2024, 1, 1)
            for day in range(self.time_periods):
                timestamp = start_date + timedelta(days=day)
                
                # Add some random variation to baseline values
                vitals = {
                    vital: baseline + np.random.normal(0, (high - low) * 0.05)
                    for vital, baseline in baselines.items()
                    for vital_name, (low, high) in self.vital_ranges.items()
                    if vital == vital_name
                }
                
                data_rows.append({
                    'patient_id': patient_id,
                    'timestamp': timestamp,
                    **vitals
                })
        
        return pd.DataFrame(data_rows)

    def generate_medication_data(self):
        """Generate medication response data"""
        data_rows = []
        
        for patient_id in self.patient_ids:
            # Assign 1-3 medications per patient
            num_meds = random.randint(1, 3)
            patient_meds = random.sample(self.medications, num_meds)
            
            start_date = datetime(2024, 1, 1)
            for medication in patient_meds:
                # Generate baseline effectiveness
                base_effectiveness = random.uniform(0.6, 0.9)
                
                for day in range(self.time_periods):
                    timestamp = start_date + timedelta(days=day)
                    
                    # Generate medication response data
                    adherence_score = random.uniform(0.7, 1.0)
                    effectiveness = base_effectiveness + random.uniform(-0.1, 0.1)
                    
                    # Simulate pre/post vitals
                    vitals_pre = random.uniform(0.8, 1.2)
                    vitals_post = vitals_pre * (1 + effectiveness * adherence_score)
                    
                    data_rows.append({
                        'patient_id': patient_id,
                        'medication_name': medication,
                        'dosage': random.choice([50, 100, 150, 200]),
                        'administration_time': timestamp,
                        'vitals_pre': vitals_pre,
                        'vitals_post': vitals_post,
                        'reported_side_effects': random.choice(self.side_effects),
                        'side_effect_severity': random.uniform(0, 3),
                        'adherence_score': adherence_score,
                        'effect_time': timestamp + timedelta(hours=random.uniform(0.5, 4))
                    })
        
        return pd.DataFrame(data_rows)

    def generate_outcomes_data(self):
        """Generate patient outcomes data"""
        data_rows = []
        
        for patient_id in self.patient_ids:
            # Generate demographic and baseline data
            age = random.randint(25, 85)
            gender = random.choice(['M', 'F'])
            comorbidities = random.randint(0, 4)
            
            # Generate outcomes
            mortality = random.choice(['deceased', 'survived'])
            readmission = random.choice(['readmitted', 'no_readmission'])
            complications = random.choice(['none', 'minor', 'major'])
            length_of_stay = random.choice(['short', 'medium', 'long'])
            
            data_rows.append({
                'patient_id': patient_id,
                'age': age,
                'gender': gender,
                'comorbidities': comorbidities,
                'mortality': mortality,
                'readmission': readmission,
                'complications': complications,
                'length_of_stay': length_of_stay,
                'admission_date': datetime(2024, 1, 1) + timedelta(days=random.randint(0, 30))
            })
        
        return pd.DataFrame(data_rows)

def test_clinical_modules():
    """Test function to demonstrate usage of generated data"""
    # Initialize data generator
    generator = ClinicalDataGenerator(num_patients=100, time_periods=30)
    
    # Generate test data
    vitals_data = generator.generate_vitals_data()
    medication_data = generator.generate_medication_data()
    outcomes_data = generator.generate_outcomes_data()
    
    # Example usage with VitalsAnalyzer
    from clinical.vitals_analysis import VitalsAnalyzer
    vitals_analyzer = VitalsAnalyzer()
    processed_vitals = vitals_analyzer.process_vitals(vitals_data)
    print("\nProcessed Vitals Shape:", processed_vitals.shape)
    
    # Example usage with MedicationResponseAnalyzer
    from clinical.medication_response import MedicationResponseAnalyzer
    med_analyzer = MedicationResponseAnalyzer()
    med_analysis = med_analyzer.analyze_medication_response(medication_data)
    print("\nMedication Analysis Keys:", med_analysis.keys())
    
    # Example usage with OutcomeTracker
    from clinical.patient_outcomes import OutcomeTracker
    outcome_tracker = OutcomeTracker()
    mortality_analysis = outcome_tracker.analyze_outcomes(outcomes_data, 'mortality')
    print("\nMortality Analysis Keys:", mortality_analysis.keys())
    
    return {
        'vitals_data': vitals_data,
        'medication_data': medication_data,
        'outcomes_data': outcomes_data
    }

if __name__ == "__main__":
    # Generate and test data
    test_data = test_clinical_modules()
    
    # Display sample data
    print("\nVitals Data Sample:")
    print(test_data['vitals_data'].head())
    
    print("\nMedication Data Sample:")
    print(test_data['medication_data'].head())
    
    print("\nOutcomes Data Sample:")
    print(test_data['outcomes_data'].head())
