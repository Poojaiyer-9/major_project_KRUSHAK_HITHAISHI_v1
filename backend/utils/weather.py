import httpx


async def get_weather(lat: float, lon: float):
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "current": "temperature_2m,relative_humidity_2m,precipitation",
                },
            )
            response.raise_for_status()
            data = response.json()
            current = data.get("current", {})
            return {
                "temperature_2m": current.get("temperature_2m", 28),
                "relative_humidity_2m": current.get("relative_humidity_2m", 65),
                "precipitation": current.get("precipitation", 0),
            }
    except Exception:
        return {
            "temperature_2m": 28,
            "relative_humidity_2m": 65,
            "precipitation": 0,
        }
