from app.db.neo4j import neo4j_connection

CONSTRAINTS = [
    "CREATE CONSTRAINT user_id_unique IF NOT EXISTS FOR (n:User) REQUIRE n.user_id IS UNIQUE",
    "CREATE CONSTRAINT account_id_unique IF NOT EXISTS FOR (n:Account) REQUIRE n.account_id IS UNIQUE",
    "CREATE CONSTRAINT card_id_unique IF NOT EXISTS FOR (n:Card) REQUIRE n.card_id IS UNIQUE",
    "CREATE CONSTRAINT device_id_unique IF NOT EXISTS FOR (n:Device) REQUIRE n.device_id IS UNIQUE",
    "CREATE CONSTRAINT ip_id_unique IF NOT EXISTS FOR (n:IP) REQUIRE n.ip_id IS UNIQUE",
    "CREATE CONSTRAINT merchant_id_unique IF NOT EXISTS FOR (n:Merchant) REQUIRE n.merchant_id IS UNIQUE",
    "CREATE CONSTRAINT transaction_id_unique IF NOT EXISTS FOR (n:Transaction) REQUIRE n.transaction_id IS UNIQUE",
]


def initialize_neo4j():
    print("Initializing FraudGraph graph schema...")

    with neo4j_connection.driver.session() as session:
        for statement in CONSTRAINTS:
            session.run(statement)

    print("Neo4j constraints created successfully.")


if __name__ == "__main__":
    initialize_neo4j()
    neo4j_connection.close()