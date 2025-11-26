from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import String, Float

class Base(DeclarativeBase):
    pass

class Video(Base):
    __tablename__ = "videos"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String)
    path: Mapped[str] = mapped_column(String)
    fps: Mapped[float] = mapped_column(Float)