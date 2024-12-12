# data_generator.py
import pandas as pd
import numpy as np
import datetime
from typing import List, Tuple
from resource_optimizer import ResourceOptimizer, ResourceUtilization, ResourceScheduler

class HealthcareDataGenerator:
    """Generates sample healthcare resource utilization data"""
    
    def __init__(self):
        self.departments = ['Emergency', 'Surgery', 'Radiology', 'ICU', 'Outpatient']
        self.resource_types = ['MRI', 'CT Scanner', 'X-Ray', 'Ventilator', 'Operating Room']
        self.current_date = datetime.datetime.now()
        
    def generate_resource_ids(self, n_resources: int) -> List[str]:
        """Generates unique resource IDs"""
        return [f"{np.random.choice(self.resource_types)}-{i:03d}" 
                for i in range(n_resources)]
    
    def generate_utilization_data(self, 
                                n_records: int = 1000,
                                n_resources: int = 20,
                                days_back: int = 30) -> pd.DataFrame:
        """
        Generates sample utilization data
        
        Parameters:
        n_records: Number of utilization records to generate
        n_resources: Number of unique resources
        days_back: Number of days of historical data
        
        Returns:
        DataFrame with utilization records
        """
        resource_ids = self.generate_resource_ids(n_resources)
        
        # Generate random timestamps within the specified date range
        end_date = self.current_date
        start_date = end_date - datetime.timedelta(days=days_back)
        timestamps = pd.date_range(start=start_date, end=end_date, periods=n_records)
        
        # Generate synthetic data
        data = {
            'resource_id': np.random.choice(resource_ids, n_records),
            'department': np.random.choice(self.departments, n_records),
            'timestamp': timestamps,
            'usage_time': np.random.normal(120, 30, n_records),  # minutes
            'patient_count': np.random.poisson(3, n_records),
            'staff_assigned': np.random.randint(1, 5, n_records)
        }
        
        # Add some realistic patterns
        df = pd.DataFrame(data)
        
        # Make emergency department more active during nights
        night_mask = df['timestamp'].dt.hour.between(22, 6)
        emergency_mask = df['department'] == 'Emergency'
        df.loc[night_mask & emergency_mask, 'usage_time'] *= 1.5
        
        # Make surgery department more active during weekdays
        weekday_mask = df['timestamp'].dt.weekday < 5
        surgery_mask = df['department'] == 'Surgery'
        df.loc[weekday_mask & surgery_mask, 'usage_time'] *= 1.3
        
        # Add some seasonal patterns
        df['usage_time'] *= 1 + 0.2 * np.sin(df['timestamp'].dt.month * np.pi / 6)
        
        return df
    
    def generate_availability_windows(self) -> List[Tuple[datetime.time, datetime.time]]:
        """Generates random availability windows for resources"""
        n_windows = np.random.randint(1, 4)
        windows = []
        
        for _ in range(n_windows):
            start_hour = np.random.randint(0, 23)
            duration = np.random.randint(4, 12)
            end_hour = (start_hour + duration) % 24
            
            windows.append((
                datetime.time(start_hour, 0),
                datetime.time(end_hour, 0)
            ))
            
        return windows

def test_resource_optimization():
    """Test function to demonstrate usage of the resource optimization package"""
    
    # Initialize data generator and create sample data
    data_gen = HealthcareDataGenerator()
    utilization_data = data_gen.generate_utilization_data()
    
    print("Generated sample data shape:", utilization_data.shape)
    print("\nSample of generated data:")
    print(utilization_data.head())
    
    # Initialize ResourceOptimizer
    optimizer = ResourceOptimizer()
    
    # Generate and print optimization report
    report = optimizer.get_optimization_report(utilization_data)
    
    print("\nOptimization Report:")
    print("==================")
    
    print("\nUtilization Patterns:")
    for cluster, data in report['utilization_patterns'].items():
        print(f"\n{cluster}:")
        for metric, value in data.items():
            print(f"  {metric}: {value}")
    
    print("\nDepartment Metrics:")
    for dept, metrics in report['department_metrics'].items():
        print(f"\n{dept}:")
        for metric, value in metrics.items():
            if metric != 'recommendations':
                print(f"  {metric}: {value}")
        print("  Recommendations:")
        for rec in metrics['recommendations']:
            print(f"    - {rec}")
    
    print("\nGlobal Recommendations:")
    for rec in report['global_recommendations']:
        print(f"- {rec}")
    
    # Test resource scheduling
    print("\nTesting Resource Scheduling:")
    scheduler = ResourceScheduler()
    
    # Add some resources with availability windows
    for resource_id in utilization_data['resource_id'].unique()[:5]:
        availability = data_gen.generate_availability_windows()
        scheduler.add_resource(resource_id, capacity=3, availability=availability)
        print(f"\nAdded resource {resource_id} with availability windows:")
        for start, end in availability:
            print(f"  {start} - {end}")
    
    # Test scheduling some resources
    test_schedule_times = [
        datetime.datetime.now() + datetime.timedelta(hours=i)
        for i in range(24)
    ]
    
    for time in test_schedule_times:
        resource_id = np.random.choice(list(scheduler.resources.keys()))
        success = scheduler.schedule_resource(
            resource_id=resource_id,
            start_time=time,
            duration=60,
            department=np.random.choice(data_gen.departments)
        )
        if success:
            print(f"Successfully scheduled {resource_id} at {time}")

if __name__ == "__main__":
    test_resource_optimization()
