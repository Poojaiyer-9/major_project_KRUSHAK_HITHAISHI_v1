from fastapi import APIRouter
from pydantic import BaseModel
from sqlalchemy.orm import Session

from db.database import SessionLocal, Shop
from utils.distance import haversine_distance

router = APIRouter(prefix="/shops", tags=["shops"])


class NearbyRequest(BaseModel):
    lat: float
    lon: float
    medicine_name: str | None = None


@router.post("/nearby")
def nearby_shops(payload: NearbyRequest):
    session: Session = SessionLocal()
    try:
        shops = session.query(Shop).all()
        results = []
        for shop in shops:
            distance = haversine_distance(
                payload.lat, payload.lon, float(shop.latitude), float(shop.longitude)
            )
            medicines = shop.medicines_available or []
            if payload.medicine_name and payload.medicine_name not in medicines:
                continue
            results.append(
                {
                    "shop_name": shop.shop_name,
                    "phone_number": shop.phone_number,
                    "distance_km": distance,
                    "address": shop.address,
                    "medicines_available": medicines,
                    "last_verified_date": (
                        shop.last_verified_date.strftime("%Y-%m-%d")
                        if shop.last_verified_date
                        else None
                    ),
                }
            )

        results.sort(key=lambda item: item["distance_km"])
        return results[:3]
    finally:
        session.close()
