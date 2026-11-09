from app.db.neo4j import neo4j_connection
from app.schemas.transaction import TransactionCreate


def create_transaction_graph(payload: TransactionCreate):
    location = payload.location

    query = """
    // Core entities
    MERGE (u:User {user_id: $user_id})
    MERGE (src:Account {account_id: $source_account_id})
    MERGE (dst:Account {account_id: $destination_account_id})
    MERGE (t:Transaction {transaction_id: $transaction_id})

    // Core relationships
    MERGE (u)-[:OWNS]->(src)

    MERGE (src)-[:TRANSFERRED_TO {
        transaction_id: $transaction_id
    }]->(dst)

    MERGE (src)-[:INITIATED]->(t)

    // Transaction properties
    SET t.timestamp = datetime($timestamp),
        t.amount = $amount,
        t.currency = $currency,
        t.country = $country,
        t.region = $region

    // Optional Card
    FOREACH (_ IN CASE WHEN $card_id IS NOT NULL THEN [1] ELSE [] END |
        MERGE (card:Card {card_id: $card_id})
        MERGE (src)-[:USES_CARD]->(card)
    )

    // Optional Device
    FOREACH (_ IN CASE WHEN $device_id IS NOT NULL THEN [1] ELSE [] END |
        MERGE (device:Device {device_id: $device_id})
        MERGE (u)-[:USES_DEVICE]->(device)
        MERGE (t)-[:FROM_DEVICE]->(device)
    )

    // Optional IP
    FOREACH (_ IN CASE WHEN $ip_id IS NOT NULL THEN [1] ELSE [] END |
        MERGE (ip:IP {ip_id: $ip_id})
        MERGE (u)-[:LOGGED_IN_FROM]->(ip)
        MERGE (t)-[:FROM_IP]->(ip)
    )

    // Optional Merchant
    FOREACH (_ IN CASE WHEN $merchant_id IS NOT NULL THEN [1] ELSE [] END |
        MERGE (merchant:Merchant {merchant_id: $merchant_id})
        MERGE (t)-[:TO_MERCHANT]->(merchant)
    )

    RETURN t.transaction_id AS transaction_id
    """

    parameters = {
        "user_id": payload.user_id,
        "source_account_id": payload.source_account_id,
        "destination_account_id": payload.destination_account_id,
        "transaction_id": payload.transaction_id,
        "timestamp": payload.timestamp.isoformat(),
        "amount": float(payload.amount),
        "currency": payload.currency,
        "country": location.country if location else None,
        "region": location.region if location else None,
        "card_id": payload.card_id,
        "device_id": payload.device_id,
        "ip_id": payload.ip_id,
        "merchant_id": payload.merchant_id,
    }

    with neo4j_connection.driver.session() as session:
        result = session.run(query, parameters)
        record = result.single()

        if record is None:
            raise RuntimeError("Neo4j transaction graph creation failed")

        return record["transaction_id"]