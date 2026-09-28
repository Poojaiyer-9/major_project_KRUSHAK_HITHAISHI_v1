import os

from sqlalchemy import Column, DateTime, Float, Integer, String, JSON, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# Defaults to a local SQLite file so the project runs with zero setup.
# Set DATABASE_URL (e.g. postgresql://user:pass@host:5432/krushak) to use Postgres.
DEFAULT_URL = "sqlite:///" + os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "krushak.db"
)
DATABASE_URL = os.environ.get("DATABASE_URL", DEFAULT_URL)

engine = create_engine(DATABASE_URL, connect_args={} if DATABASE_URL.startswith("sqlite") else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Shop(Base):
    __tablename__ = "shops"

    id = Column(Integer, primary_key=True, index=True)
    shop_name = Column(String, nullable=False)
    owner_name = Column(String, nullable=True)
    phone_number = Column(String, nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    address = Column(String, nullable=True)
    medicines_available = Column(JSON, nullable=True)
    last_verified_date = Column(DateTime, nullable=True)


def create_tables():
    Base.metadata.create_all(bind=engine)
