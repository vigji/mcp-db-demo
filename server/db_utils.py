from __future__ import annotations

import os
from datetime import date, timedelta
from decimal import Decimal
from random import choice, randint

from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import Session

from db_models import Base, CostItem, Delivery, Project

load_dotenv()

DB_URL_ADMIN = os.environ.get("DB_URL_ADMIN")
DB_URL_RO = os.environ.get("DB_URL_RO")


def get_admin_engine():
    if not DB_URL_ADMIN:
        raise RuntimeError("DB_URL_ADMIN environment variable is required")
    return create_engine(DB_URL_ADMIN, future=True)


def get_ro_engine():
    if not DB_URL_RO:
        raise RuntimeError("DB_URL_RO environment variable is required")
    return create_engine(DB_URL_RO, future=True)


def init_schema_and_seed():
    admin_engine = get_admin_engine()
    with admin_engine.begin() as conn:
        Base.metadata.create_all(bind=conn)

        inspector = inspect(conn)
        if not inspector.has_table("projects"):
            raise RuntimeError("projects table missing after create_all")

        result = conn.execute(text("SELECT COUNT(*) FROM projects"))
        (count_projects,) = result.one()
        if count_projects == 0:
            seed_data(conn)

        create_readonly_user(conn)


def seed_data(conn):
    projects = [
        Project(
            name="Linea 4 Metropolitana",
            client_name="Comune di Milano",
            start_date=date(2023, 1, 10),
            end_date=None,
            status="active",
            budget_eur=Decimal("25000000.00"),
        ),
        Project(
            name="Ospedale San Luca Ristrutturazione",
            client_name="Regione Lombardia",
            start_date=date(2022, 5, 1),
            end_date=None,
            status="active",
            budget_eur=Decimal("18000000.00"),
        ),
        Project(
            name="Complesso Residenziale Verde",
            client_name="EdilGreen S.p.A.",
            start_date=date(2021, 3, 15),
            end_date=date(2024, 2, 28),
            status="completed",
            budget_eur=Decimal("12000000.00"),
        ),
        Project(
            name="Ponte Sul Fiume Dora",
            client_name="Comune di Torino",
            start_date=date(2024, 2, 1),
            end_date=None,
            status="planned",
            budget_eur=Decimal("9000000.00"),
        ),
        Project(
            name="Scuola Primaria Galileo",
            client_name="Comune di Bologna",
            start_date=date(2023, 9, 1),
            end_date=None,
            status="active",
            budget_eur=Decimal("6000000.00"),
        ),
    ]

    session = Session(conn)
    session.add_all(projects)
    session.flush()

    categories = ["materials", "labor", "equipment", "subcontractor"]
    suppliers = [
        "ItalCementi",
        "Ferrovie Forniture",
        "Impresa Rossi",
        "EdilNord",
        "TecnoScavi",
    ]

    for project in projects:
        for _ in range(15):
            cost_item = CostItem(
                project_id=project.id,
                category=choice(categories),
                amount_eur=Decimal(str(randint(5_000, 150_000))),
                cost_date=project.start_date + timedelta(days=randint(0, 365)),
                supplier_name=choice(suppliers),
            )
            session.add(cost_item)

        for _ in range(10):
            delivery = Delivery(
                project_id=project.id,
                material_name=choice(
                    [
                        "Calcestruzzo C25/30",
                        "Acciaio B450C",
                        "Mattoni Forati",
                        "Isolante EPS",
                    ]
                ),
                quantity=randint(10, 500),
                unit=choice(["m3", "kg", "pallet"]),
                unit_cost_eur=Decimal(str(randint(50, 500))),
                delivery_date=project.start_date + timedelta(days=randint(0, 365)),
                supplier_name=choice(suppliers),
            )
            session.add(delivery)

    session.commit()
    session.close()


def create_readonly_user(conn):
    conn.execute(
        text(
            """
            DO $$
            BEGIN
                IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'readonly') THEN
                    CREATE ROLE readonly LOGIN PASSWORD 'readonlypass';
                END IF;
            END
            $$;
            """
        )
    )
    conn.execute(text("GRANT CONNECT ON DATABASE construction TO readonly;"))
    conn.execute(text("GRANT USAGE ON SCHEMA public TO readonly;"))
    conn.execute(text("GRANT SELECT ON ALL TABLES IN SCHEMA public TO readonly;"))
    conn.execute(text("ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO readonly;"))
