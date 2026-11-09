from app.db.postgres import Base, engine
from app.models.transaction import Transaction


def initialize_database():
    print("Initializing FraudGraph database...")

    Base.metadata.create_all(bind=engine)

    print("Database tables created successfully.")


if __name__ == "__main__":
    initialize_database()