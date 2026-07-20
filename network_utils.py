# network_utils.py
import urllib.request
import json

def get_public_ip() -> str:
    """Fetch public IP using multiple reliable services with fallback."""
    services = [
        "https://api.ipify.org",
        "https://checkip.amazonaws.com",
        "https://icanhazip.com"
    ]
    for service in services:
        try:
            with urllib.request.urlopen(service, timeout=3) as resp:
                return resp.read().decode().strip()
        except Exception:
            continue
    return "0.0.0.0" # Fallback if all services fail