# resource_optimizer.py
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Tuple
import datetime

class ResourceUtilization:
    """Analyzes and tracks resource utilization patterns"""
    
    def __init__(self, n_clusters: int = 3):
        self.scaler = StandardScaler()
        self.clustering_model = KMeans(n_clusters=n_clusters, random_state=42)
        
    def preprocess_data(self, data: pd.DataFrame) -> np.ndarray:
        """Preprocesses utilization data for analysis"""
        numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
        return self.scaler.fit_transform(data[numeric_cols])
    
    def identify_patterns(self, data: pd.DataFrame) -> Dict:
        """
        Identifies resource utilization patterns using clustering
        
        Parameters:
        data: DataFrame with columns like 'resource_id', 'usage_time', 'department', etc.
        
        Returns:
        Dictionary containing pattern analysis and cluster assignments
        """
        processed_data = self.preprocess_data(data)
        clusters = self.clustering_model.fit_predict(processed_data)
        
        # Analyze patterns within each cluster
        pattern_analysis = {}
        for cluster in range(self.clustering_model.n_clusters):
            cluster_data = data[clusters == cluster]
            pattern_analysis[f'cluster_{cluster}'] = {
                'size': len(cluster_data),
                'avg_utilization': cluster_data['usage_time'].mean(),
                'peak_hours': self._identify_peak_hours(cluster_data),
                'common_departments': cluster_data['department'].value_counts().head(3).to_dict()
            }
            
        return pattern_analysis
    
    def _identify_peak_hours(self, data: pd.DataFrame) -> List[int]:
        """Identifies peak usage hours from utilization data"""
        if 'timestamp' in data.columns:
            hourly_usage = data.groupby(data['timestamp'].dt.hour)['usage_time'].mean()
            return hourly_usage.nlargest(3).index.tolist()
        return []

class ResourceScheduler:
    """Handles resource scheduling and allocation"""
    
    def __init__(self):
        self.schedule = {}
        self.resources = {}
        
    def add_resource(self, resource_id: str, capacity: int, availability: List[Tuple[datetime.time, datetime.time]]):
        """Adds a new resource with its capacity and availability windows"""
        self.resources[resource_id] = {
            'capacity': capacity,
            'availability': availability,
            'current_load': 0
        }
    
    def schedule_resource(self, resource_id: str, start_time: datetime.datetime, 
                         duration: int, department: str) -> bool:
        """
        Attempts to schedule a resource
        
        Parameters:
        resource_id: Unique identifier for the resource
        start_time: Requested start time
        duration: Duration in minutes
        department: Requesting department
        
        Returns:
        Boolean indicating if scheduling was successful
        """
        if resource_id not in self.resources:
            return False
            
        resource = self.resources[resource_id]
        end_time = start_time + datetime.timedelta(minutes=duration)
        
        # Check availability and capacity
        if not self._is_available(resource, start_time, end_time):
            return False
            
        # Schedule the resource
        if start_time not in self.schedule:
            self.schedule[start_time] = {}
        
        self.schedule[start_time][resource_id] = {
            'department': department,
            'duration': duration,
            'end_time': end_time
        }
        
        resource['current_load'] += 1
        return True
    
    def _is_available(self, resource: Dict, start_time: datetime.datetime, 
                     end_time: datetime.datetime) -> bool:
        """Checks if resource is available for the requested time slot"""
        if resource['current_load'] >= resource['capacity']:
            return False
            
        # Check if requested time is within availability windows
        request_time = start_time.time()
        for avail_start, avail_end in resource['availability']:
            if avail_start <= request_time <= avail_end:
                return True
        return False

class ResourceOptimizer:
    """Main class for resource optimization and analysis"""
    
    def __init__(self):
        self.utilization_analyzer = ResourceUtilization()
        self.scheduler = ResourceScheduler()
        
    def analyze_department_metrics(self, data: pd.DataFrame) -> Dict:
        """
        Analyzes department-level resource utilization metrics
        
        Parameters:
        data: DataFrame with department utilization data
        
        Returns:
        Dictionary containing department-level metrics and recommendations
        """
        metrics = {}
        
        for dept in data['department'].unique():
            dept_data = data[data['department'] == dept]
            
            metrics[dept] = {
                'total_usage_hours': dept_data['usage_time'].sum(),
                'avg_usage_per_resource': dept_data.groupby('resource_id')['usage_time'].mean().mean(),
                'utilization_rate': (dept_data['usage_time'].sum() / 
                                   (len(dept_data['resource_id'].unique()) * 24 * 60)),
                'recommendations': self._generate_recommendations(dept_data)
            }
            
        return metrics
    
    def _generate_recommendations(self, dept_data: pd.DataFrame) -> List[str]:
        """Generates optimization recommendations based on utilization patterns"""
        recommendations = []
        
        # Analyze utilization patterns
        avg_utilization = dept_data['usage_time'].mean()
        peak_usage = self.utilization_analyzer._identify_peak_hours(dept_data)
        
        if avg_utilization < 0.3:  # Less than 30% utilization
            recommendations.append("Consider reducing resource allocation during off-peak hours")
        elif avg_utilization > 0.8:  # More than 80% utilization
            recommendations.append("Consider increasing resource capacity to prevent bottlenecks")
            
        if peak_usage:
            recommendations.append(f"Peak usage occurs during hours: {peak_usage}. "
                                "Consider load balancing during these times")
            
        return recommendations
    
    def get_optimization_report(self, data: pd.DataFrame) -> Dict:
        """
        Generates comprehensive optimization report
        
        Parameters:
        data: DataFrame with resource utilization data
        
        Returns:
        Dictionary containing analysis results and recommendations
        """
        patterns = self.utilization_analyzer.identify_patterns(data)
        dept_metrics = self.analyze_department_metrics(data)
        
        return {
            'utilization_patterns': patterns,
            'department_metrics': dept_metrics,
            'global_recommendations': self._generate_global_recommendations(patterns, dept_metrics),
            'timestamp': datetime.datetime.now()
        }
    
    def _generate_global_recommendations(self, patterns: Dict, dept_metrics: Dict) -> List[str]:
        """Generates organization-wide optimization recommendations"""
        recommendations = []
        
        # Analyze cross-department patterns
        avg_utilization = np.mean([m['utilization_rate'] for m in dept_metrics.values()])
        
        if avg_utilization < 0.4:
            recommendations.append("Overall resource utilization is low. Consider resource consolidation")
        
        # Analyze cluster patterns
        for cluster, data in patterns.items():
            if data['size'] > 100 and data['avg_utilization'] > 0.8:
                recommendations.append(f"High utilization cluster identified in {cluster}. "
                                    "Consider capacity expansion")
                
        return recommendations
