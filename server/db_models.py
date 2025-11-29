from __future__ import annotations

from sqlalchemy import Date, ForeignKey, Integer, Numeric, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base declarative class for SQLAlchemy models."""


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    client_name: Mapped[str] = mapped_column(String(255), nullable=False)
    start_date: Mapped[Date]
    end_date: Mapped[Date | None]
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    budget_eur: Mapped[Numeric] = mapped_column(Numeric(12, 2), nullable=False)

    cost_items: Mapped[list["CostItem"]] = relationship(back_populates="project")
    deliveries: Mapped[list["Delivery"]] = relationship(back_populates="project")


class CostItem(Base):
    __tablename__ = "cost_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), nullable=False)
    category: Mapped[str] = mapped_column(String(64), nullable=False)
    amount_eur: Mapped[Numeric] = mapped_column(Numeric(12, 2), nullable=False)
    cost_date: Mapped[Date]
    supplier_name: Mapped[str] = mapped_column(String(255), nullable=False)

    project: Mapped[Project] = relationship(back_populates="cost_items")


class Delivery(Base):
    __tablename__ = "deliveries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), nullable=False)
    material_name: Mapped[str] = mapped_column(String(255), nullable=False)
    quantity: Mapped[int]
    unit: Mapped[str] = mapped_column(String(32))
    unit_cost_eur: Mapped[Numeric] = mapped_column(Numeric(12, 2), nullable=False)
    delivery_date: Mapped[Date]
    supplier_name: Mapped[str] = mapped_column(String(255), nullable=False)

    project: Mapped[Project] = relationship(back_populates="deliveries")
