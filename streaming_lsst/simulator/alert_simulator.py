"""
LSST Alert Stream Simulator for benchmarking and testing.
Generates synthetic alerts following LSST alert schema.
"""

import numpy as np
from typing import Dict, Iterator, Tuple
import time


class LSSTAlertSimulator:
    """Simulates LSST alert streams with realistic distributions."""
    
    def __init__(self, seed: int = 42, anomaly_rate: float = 0.05):
        """
        Args:
            seed: Random seed for reproducibility
            anomaly_rate: Fraction of alerts that are anomalies
        """
        np.random.seed(seed)
        self.anomaly_rate = anomaly_rate
        self.alert_counter = 0
        
        # LSST survey parameters
        self.ra_min, self.ra_max = 0, 360
        self.dec_min, self.dec_max = -90, 90
        
        # Typical magnitude range (g, r, i, z bands)
        self.mag_min, self.mag_max = 12, 23
        
        # Spatial clusters for GNN testing
        self.cluster_centers = [
            (np.random.uniform(0, 360), np.random.uniform(-90, 90))
            for _ in range(10)
        ]
        
    def generate_alert(self) -> Dict:
        """Generate a single synthetic LSST alert."""
        
        alert_id = self.alert_counter
        self.alert_counter += 1
        
        # Determine if this is a spatial anomaly (alert doesn't match its cluster)
        is_spatial_anomaly = np.random.random() < (self.anomaly_rate * 0.5)
        is_anomaly = (np.random.random() < self.anomaly_rate) or is_spatial_anomaly
        
        # Pick a cluster
        center_ra, center_dec = self.cluster_centers[np.random.randint(len(self.cluster_centers))]
        ra = center_ra + np.random.normal(0, 0.5)
        dec = center_dec + np.random.normal(0, 0.5)
        
        # Base features
        alert = {
            'alert': {
                'objectId': alert_id,
                'ra': ra,
                'dec': dec,
                'candidate': self._generate_candidate(is_anomaly),
                'prv_candidates': self._generate_historical(is_anomaly),
                'is_spatial_anomaly': is_spatial_anomaly
            }
        }
        
        return alert
    
    def _generate_candidate(self, is_anomaly: bool) -> Dict:
        """Generate candidate (current detection) properties."""
        
        if is_anomaly:
            # Anomalous features
            mag = np.random.normal(18, 1.5)  # Brighter or dimmer
            flux = 10 ** (-mag / 2.5)  # Unusual flux
            mag_err = np.random.exponential(0.2)  # Higher uncertainty
            flux_err = flux * 0.3
            nd_hist = np.random.randint(5, 50)  # Many detections
            nc_gen = np.random.randint(10, 100)
            nn_gen = np.random.randint(5, 50)
            dist_psn = np.random.uniform(0.1, 5.0)  # Far from reference
            sg_score = np.random.uniform(0.3, 0.9)  # Ambiguous
        else:
            # Normal features
            mag = np.random.normal(18.5, 0.5)
            flux = 10 ** (-mag / 2.5)
            mag_err = np.random.exponential(0.05)
            flux_err = flux * 0.1
            nd_hist = np.random.randint(1, 10)
            nc_gen = np.random.randint(1, 5)
            nn_gen = np.random.randint(0, 3)
            dist_psn = np.random.normal(0.5, 0.2)
            sg_score = np.random.normal(0.1, 0.15)  # Stellar/Galactic
        
        candidate = {
            'jd': time.time() / 86400 + 2400000.5,  # Julian date
            'mag': max(self.mag_min, min(self.mag_max, mag)),
            'magerr': abs(mag_err),
            'flux': abs(flux),
            'fluxerr': abs(flux_err),
            'ndethist': int(nd_hist),
            'ncandgn': int(nc_gen),
            'nnegn': int(nn_gen),
            'distpsnr1': dist_psn,
            'rmag': np.random.normal(18.0, 1.0),
            'imag': np.random.normal(17.5, 1.0),
            'zmag': np.random.normal(17.0, 1.0),
            'sgscore': np.clip(sg_score, 0, 1),
            'ra': np.random.uniform(self.ra_min, self.ra_max),
            'dec': np.random.uniform(self.dec_min, self.dec_max),
            'ranr': np.random.uniform(self.ra_min, self.ra_max),
            'decnr': np.random.uniform(self.dec_min, self.dec_max),
            'diffmaglim': np.random.normal(20.5, 0.5),
            'isdiffpos': np.random.choice([-1, 1]),
            'classtar': np.random.random(),
            'pixelscore': np.random.random(),
        }
        
        return candidate
    
    def _generate_historical(self, is_anomaly: bool) -> list:
        """Generate historical candidates (previous detections)."""
        
        num_prev = np.random.randint(0, 5) if not is_anomaly else np.random.randint(3, 15)
        history = []
        
        for i in range(num_prev):
            mag_base = np.random.normal(18.5, 0.5)
            if is_anomaly:
                mag_base += np.random.normal(1.0, 0.5)  # Variable
            
            history.append({
                'jd': time.time() / 86400 + 2400000.5 - (i * 10),
                'mag': max(self.mag_min, min(self.mag_max, mag_base)),
                'magerr': np.random.exponential(0.05),
                'flux': 10 ** (-mag_base / 2.5),
                'fluxerr': 0.1 * 10 ** (-mag_base / 2.5),
            })
        
        return history
    
    def stream_alerts(self, duration_sec: float = 60.0, rate_hz: float = 100.0) -> Iterator[Tuple[Dict, float]]:
        """
        Generate alert stream at specified rate.
        
        Args:
            duration_sec: Duration of stream in seconds
            rate_hz: Alert rate in Hz
        Yields:
            (alert, timestamp)
        """
        interval = 1.0 / rate_hz
        start_time = time.perf_counter()
        
        while time.perf_counter() - start_time < duration_sec:
            alert = self.generate_alert()
            yield alert, time.perf_counter()
            
            # Rate-limiting
            elapsed = time.perf_counter() - start_time
            expected = self.alert_counter * interval
            if elapsed < expected:
                time.sleep(expected - elapsed)


class HighVariabilityAlertSimulator(LSSTAlertSimulator):
    """Generates alerts with high-variability sources (supernovae, TDE, etc)."""
    
    def _generate_candidate(self, is_anomaly: bool) -> Dict:
        """Generate candidate with high-variability characteristics."""
        
        if is_anomaly:
            # High variability signatures
            # Rapidly brightening
            mag = np.random.normal(16, 1.5)
            dm_dt = np.random.uniform(0.5, 3.0)  # magnitudes per day
            
            flux = 10 ** (-mag / 2.5)
            mag_err = np.random.exponential(0.15)
            flux_err = flux * 0.15
        else:
            mag = np.random.normal(19, 0.5)
            dm_dt = np.random.normal(0.01, 0.02)
            flux = 10 ** (-mag / 2.5)
            mag_err = np.random.exponential(0.05)
            flux_err = flux * 0.1
        
        candidate = super()._generate_candidate(is_anomaly)
        candidate['mag'] = max(self.mag_min, min(self.mag_max, mag))
        candidate['magerr'] = abs(mag_err)
        candidate['flux'] = abs(flux)
        candidate['fluxerr'] = abs(flux_err)
        candidate['dmag_dt'] = dm_dt
        
        return candidate


class BurstAlertSimulator(LSSTAlertSimulator):
    """Generates alert streams with bursty behavior (realistic for real LSST)."""
    
    def __init__(self, seed: int = 42, anomaly_rate: float = 0.05, 
                 burst_probability: float = 0.1, burst_size_scale: float = 10.0):
        super().__init__(seed, anomaly_rate)
        self.burst_probability = burst_probability
        self.burst_size_scale = burst_size_scale
        self.burst_counter = 0
        self.burst_remaining = 0
    
    def stream_alerts(self, duration_sec: float = 60.0, base_rate_hz: float = 50.0) -> Iterator[Tuple[Dict, float]]:
        """
        Generate bursty alert stream.
        
        Args:
            duration_sec: Duration of stream
            base_rate_hz: Base alert rate
        Yields:
            (alert, timestamp)
        """
        start_time = time.perf_counter()
        alert_times = []
        
        while time.perf_counter() - start_time < duration_sec:
            current_rate = base_rate_hz
            
            # Determine if we should start a burst
            if self.burst_remaining == 0:
                if np.random.random() < self.burst_probability:
                    self.burst_remaining = int(np.random.exponential(self.burst_size_scale))
                    self.burst_counter += 1
            
            # Increase rate during burst
            if self.burst_remaining > 0:
                current_rate = base_rate_hz * 5  # 5x normal rate
                self.burst_remaining -= 1
            
            # Generate alert
            alert = self.generate_alert()
            timestamp = time.perf_counter()
            alert_times.append(timestamp)
            
            yield alert, timestamp
            
            # Rate limiting
            interval = 1.0 / current_rate
            elapsed = time.perf_counter() - start_time
            expected = self.alert_counter * (1.0 / base_rate_hz)
            
            if elapsed < expected:
                time.sleep(min(interval * 0.1, expected - elapsed))
