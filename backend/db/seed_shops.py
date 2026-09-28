import csv
import json
from datetime import datetime

from db.database import SessionLocal, Shop, create_tables


def seed_shops(csv_path: str):
    create_tables()
    session = SessionLocal()
    try:
        with open(csv_path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                raw_medicines = row.get("medicines_available", "[]") or "[]"
                try:
                    medicines = json.loads(raw_medicines)
                except (json.JSONDecodeError, TypeError):
                    medicines = [
                        m.strip() for m in raw_medicines.replace(";", ",").split(",") if m.strip()
                    ]

                verified_raw = row.get("last_verified_date")
                last_verified = (
                    datetime.strptime(verified_raw, "%Y-%m-%d")
                    if verified_raw
                    else None
                )

                shop = Shop(
                    shop_name=row.get("shop_name", "Unknown"),
                    owner_name=row.get("owner_name"),
                    phone_number=row.get("phone_number", ""),
                    latitude=float(row.get("latitude") or 0),
                    longitude=float(row.get("longitude") or 0),
                    address=row.get("address"),
                    medicines_available=medicines,
                    last_verified_date=last_verified,
                )
                session.add(shop)
        session.commit()
        print("Seeded shops from", csv_path)
    finally:
        session.close()


if __name__ == "__main__":
    import os

    default_csv = os.path.join(os.path.dirname(__file__), "shops_sample.csv")
    seed_shops(default_csv)
