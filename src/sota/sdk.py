import logging
from typing import Dict, List, Tuple
from datetime import datetime, timezone

try:
    from astroquery.simbad import Simbad
    from alerce.core import Alerce
    alerce_client = Alerce()
    SOTA_SDK_AVAILABLE = True
except ImportError:
    SOTA_SDK_AVAILABLE = False

class SotaSDK:
    """
    Unified interface for professional astronomical SDKs.
    """
    @staticmethod
    def resolve_object(name: str) -> Dict:
        if not SOTA_SDK_AVAILABLE: return {"resolved": False}
        try:
            result = Simbad.query_object(name)
            if result is None: return {"resolved": False}
            row = result[0]
            return {
                "resolved": True,
                "main_id": str(row['MAIN_ID']),
                "ra": float(row['RA_PREC']),
                "dec": float(row['DEC_PREC'])
            }
        except Exception as e:
            logging.error(f"Simbad resolution error: {e}")
            return {"resolved": False}

    @staticmethod
    def fetch_alerce_transients(limit: int = 20) -> List[Dict]:
        if not SOTA_SDK_AVAILABLE: return []
        try:
            alerts = alerce_client.query_alerts(limit=limit, format="pandas")
            return [
                {
                    "id": str(row['oid']),
                    "ra": float(row['ra']),
                    "dec": float(row['dec']),
                    "mag": float(row['magpsf']),
                    "time": datetime.fromtimestamp(row['mjd'], timezone.utc).isoformat()
                }
                for _, row in alerts.iterrows()
            ]
        except Exception as e:
            logging.error(f"ALeRCE query error: {e}")
            return []
