import datetime
from datetime import datetime
from zoneinfo import ZoneInfo
from timezonefinder import TimezoneFinder
import requests
from typing import Any, Optional, Union
import re
import httpx
from tools._tool_label import tool_label
import asyncio

geo_url = "https://nominatim.openstreetmap.org/search"


def find_city_coordinates(city: str):
    """
    take a city name and find the coordinates
    :param city: City name ex. Tokyo
    :return:
    """
    params = {"q": city, "format": "json", "limit": 1}

    response = requests.get(geo_url, params=params, headers={"User-Agent": "python-app"})
    response.raise_for_status()
    geo_data = response.json()

    if not geo_data:
        raise ValueError("City not found")

    lat = float(geo_data[0]["lat"])
    lon = float(geo_data[0]["lon"])

    return lat, lon

@tool_label
def get_current_time(city: str) -> dict:
    """
    Get the current local time for a city.

    :param city: City name (e.g., "Tokyo")
    :return: Dictionary with local time info
    """
    lat, lon = find_city_coordinates(city)

    # Find timezone
    tf = TimezoneFinder()
    timezone_str = tf.timezone_at(lat=lat, lng=lon)

    if not timezone_str:
        raise ValueError("Timezone not found")

    local_time = datetime.now(ZoneInfo(timezone_str))

    return {
        "city": city,
        "timezone": timezone_str,
        "local_time": local_time.strftime("%Y-%m-%d %H:%M:%S")
    }



# Constants
NWS_API_BASE = "https://api.weather.gov"
USER_AGENT = "ChiefGuapoCodes"  # <-- put your contact

# weather request
async def make_nws_request(url: str) -> dict[str, Any] | None:
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/geo+json",
    }
    async with httpx.AsyncClient(follow_redirects=True) as client:
        try:
            resp = await client.get(url, headers=headers, timeout=60.0)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            # This will tell you if it's 403/429/etc.
            print(f"NWS HTTP error {e.response.status_code} for {url}")
            print(e.response.text[:300])
            return None
        except Exception as e:
            print(f"NWS request failed for {url}: {e}")
            return None

def format_alert(feature: dict) -> dict[str, Any]:
    """Format an alert feature into a readable string."""
    props = feature["properties"]
    return {
        "Event": props.get('event', 'Unknown'),
        "Area": props.get('areaDesc', 'Unknown'),
        "Severity": props.get('severity', 'Unknown'),
        "Description": props.get('description', 'No description available'),
        "Instructions": props.get('instruction', 'No specific instructions provided')
    }


@tool_label
async def get_alerts(state: str) -> Union[str, list[dict[str, Any]]]:
    """Get weather alerts for a US state.

    Args:
        state: Two-letter US state code (e.g. CA, NY)
    """
    url = f"{NWS_API_BASE}/alerts/active/area/{state}"
    data = await make_nws_request(url)

    if not data or "features" not in data:
        return "Unable to fetch alerts or no alerts found."

    if not data["features"]:
        return "No active alerts for this state."

    alerts = [format_alert(feature) for feature in data["features"]]
    return alerts


@tool_label
async def get_forecast(city: str) -> str:
    """Get weather forecast for a location.

    Args:
        city: the name of the city to get the forecast for (include state if necessary) e.g. Seattle, WA
    """

    lat, lon = find_city_coordinates(city)
    print(lat, lon)

    # First get the forecast grid endpoint
    points_url = f"{NWS_API_BASE}/points/{lat},{lon}"
    print(points_url)
    points_data = await make_nws_request(points_url)

    if not points_data:
        return "Unable to fetch forecast data for this location."

    # Get the forecast URL from the points response
    forecast_url = points_data["properties"]["forecast"]
    forecast_data = await make_nws_request(forecast_url)

    if not forecast_data:
        return "Unable to fetch detailed forecast."

    # Format the periods into a readable forecast
    periods = forecast_data["properties"]["periods"]
    forecasts = []
    for period in periods[:5]:  # Only show next 5 periods
        temp = period['temperature']
        tunit = period['temperatureUnit']
        wind = period['windSpeed']
        forecast = f"""
            {period['name']}:
            Temperature: {temp}°{tunit}
            Wind: {wind} {period['windDirection']}
            Forecast: {period['detailedForecast']}
        """
        forecasts.append(forecast)

    return "\n---\n".join(forecasts)


async def main():
    print(get_current_time("Reston, Virginia, USA"))
    forecast = await get_forecast("Reston, Virginia, USA")
    print(forecast)

if __name__ == "__main__":
    asyncio.run(main())