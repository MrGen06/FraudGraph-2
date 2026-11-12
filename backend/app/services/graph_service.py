import logging
from typing import Any, Dict, List, Optional

from app.db.neo4j import neo4j_connection
from app.schemas.transaction import TransactionCreate

logger = logging.getLogger(__name__)


import logging
from typing import Any, Dict, List, Optional

from app.db.neo4j import neo4j_connection
from app.schemas.transaction import TransactionCreate

logger = logging.getLogger(__name__)

GENERIC_DEVICE_OS = {
    "windows",
    "ios device",
    "macos",
    "desktop",
    "mobile",
    "linux",
    "android",
    "unknown",
    "missing",
    "nan",
    "other",
    "windows 10/chrome",
    "desktop_chrome",
}

GENERIC_EMAIL_DOMAINS = {
    "gmail.com",
    "yahoo.com",
    "hotmail.com",
    "aol.com",
    "comcast.net",
    "icloud.com",
    "msn.com",
    "outlook.com",
    "sbcglobal.net",
    "verizon.net",
    "att.net",
    "bellsouth.net",
    "ymail.com",
    "mail.com",
    "anonymous.com",
}


def create_transaction_graph(payload: TransactionCreate) -> bool:
    """
    Persist transaction and entity nodes into Neo4j property graph for investigation.
    Filters generic OS identifiers and generic email domains to prevent mega-hub graph collapse.
    Designed with partial-failure resiliency: logs warning on Neo4j failure
    without aborting the primary transaction record.
    """
    location = payload.location
    country = payload.country or (location.country if location else None)
    region = payload.region or (location.region if location else None)

    # Filter out generic OS identifiers that collapse the graph
    raw_device = (payload.device_id or "").strip()
    clean_device_id = None if raw_device.lower() in GENERIC_DEVICE_OS else raw_device

    # Distinguish generic default account IDs
    src_account = payload.source_account_id if payload.source_account_id != "ACC-DEFAULT-01" else None
    dst_account = payload.destination_account_id if payload.destination_account_id != "ACC-DEFAULT-02" else None

    query = """
    // Core transaction node
    MERGE (t:Transaction {transaction_id: $transaction_id})
    SET t.timestamp = datetime($timestamp),
        t.amount = $amount,
        t.currency = $currency,
        t.country = $country,
        t.region = $region

    // User ownership
    MERGE (u:User {user_id: $user_id})
    MERGE (u)-[:INITIATED]->(t)

    // Optional Verified Source Account (if genuine account identifier provided)
    FOREACH (_ IN CASE WHEN $source_account_id IS NOT NULL AND $source_account_id <> '' THEN [1] ELSE [] END |
        MERGE (src:Account {account_id: $source_account_id})
        MERGE (u)-[:OWNS]->(src)
        MERGE (src)-[:DEBIT_FROM]->(t)
    )

    // Optional Verified Destination Account
    FOREACH (_ IN CASE WHEN $destination_account_id IS NOT NULL AND $destination_account_id <> '' THEN [1] ELSE [] END |
        MERGE (dst:Account {account_id: $destination_account_id})
        MERGE (t)-[:TRANSFERRED_TO]->(dst)
    )

    // Optional Card Instrument
    FOREACH (_ IN CASE WHEN $card_id IS NOT NULL AND $card_id <> '' THEN [1] ELSE [] END |
        MERGE (card:Card {card_id: $card_id})
        MERGE (t)-[:USES_CARD]->(card)
    )

    // Optional Specific Device Fingerprint (excluding generic OS identifiers)
    FOREACH (_ IN CASE WHEN $device_id IS NOT NULL AND $device_id <> '' THEN [1] ELSE [] END |
        MERGE (device:Device {device_id: $device_id})
        MERGE (u)-[:USES_DEVICE]->(device)
        MERGE (t)-[:FROM_DEVICE]->(device)
    )

    // Optional IP address
    FOREACH (_ IN CASE WHEN $ip_id IS NOT NULL AND $ip_id <> '' THEN [1] ELSE [] END |
        MERGE (ip:IP {ip_id: $ip_id})
        MERGE (u)-[:LOGGED_IN_FROM]->(ip)
        MERGE (t)-[:FROM_IP]->(ip)
    )

    // Optional Merchant
    FOREACH (_ IN CASE WHEN $merchant_id IS NOT NULL AND $merchant_id <> '' THEN [1] ELSE [] END |
        MERGE (merchant:Merchant {merchant_id: $merchant_id})
        MERGE (t)-[:TO_MERCHANT]->(merchant)
    )

    RETURN t.transaction_id AS transaction_id
    """

    parameters = {
        "user_id": payload.user_id,
        "source_account_id": src_account,
        "destination_account_id": dst_account,
        "transaction_id": payload.transaction_id,
        "timestamp": payload.timestamp.isoformat(),
        "amount": float(payload.amount),
        "currency": payload.currency,
        "country": country,
        "region": region,
        "card_id": payload.card_id,
        "device_id": clean_device_id,
        "ip_id": payload.ip_id,
        "merchant_id": payload.merchant_id,
    }

    try:
        driver = neo4j_connection.driver
        with driver.session() as session:
            result = session.run(query, parameters)
            record = result.single()
            if record:
                logger.info(f"Neo4j: successfully synchronized transaction {payload.transaction_id}")
                return True
    except Exception as e:
        logger.warning(
            f"Neo4j partial sync failure for transaction {payload.transaction_id}: {e}. "
            f"Transaction remains safely committed in PostgreSQL."
        )
    return False


def get_transaction_subgraph(transaction_id: str, db_tx: Optional[Any] = None) -> Dict[str, Any]:
    """
    Retrieve real entity graph for a transaction from Neo4j AuraDB.
    Falls back to relational database attributes if Neo4j is offline.
    """
    nodes: List[Dict[str, Any]] = []
    links: List[Dict[str, Any]] = []
    seen_nodes = set()

    # Try querying Neo4j live driver first
    try:
        query = """
        MATCH (t:Transaction {transaction_id: $transaction_id})
        OPTIONAL MATCH (t)-[r]-(neighbor)
        OPTIONAL MATCH (neighbor)-[r2]-(second_hop:Transaction)
        RETURN t, collect(DISTINCT neighbor) as neighbors, collect(DISTINCT second_hop) as second_hops
        """
        driver = neo4j_connection.driver
        with driver.session() as session:
            result = session.run(query, {"transaction_id": transaction_id})
            record = result.single()
            if record and record["t"]:
                t = record["t"]
                t_id = t["transaction_id"]
                nodes.append({
                    "id": t_id,
                    "label": f"Tx {t_id}",
                    "type": "transaction",
                    "is_center": True,
                    "risk_flag": False,
                })
                seen_nodes.add(t_id)

                for n in record["neighbors"]:
                    labels = list(n.labels)
                    ntype = labels[0].lower() if labels else "entity"
                    nid = n.get(f"{ntype}_id") or str(n.element_id)
                    full_id = f"{ntype}_{nid}"
                    if full_id not in seen_nodes:
                        nodes.append({
                            "id": full_id,
                            "label": f"{ntype.capitalize()}: {nid[:16]}",
                            "type": ntype,
                            "is_center": False,
                            "risk_flag": False,
                        })
                        seen_nodes.add(full_id)
                    links.append({"source": t_id, "target": full_id, "relationship": f"USES_{ntype.upper()}"})

                return {
                    "transaction_id": transaction_id,
                    "nodes": nodes,
                    "links": links,
                    "source": "neo4j_live",
                }
    except Exception as e:
        logger.info(f"Neo4j query unavailable ({e}), using relational transaction structure.")

    # Fallback to PostgreSQL transaction attributes
    t_id = transaction_id
    nodes.append({
        "id": t_id,
        "label": f"Tx {t_id}",
        "type": "transaction",
        "is_center": True,
        "risk_flag": False,
    })
    seen_nodes.add(t_id)

    if db_tx:
        if getattr(db_tx, "card_id", None):
            cid = f"card_{db_tx.card_id}"
            nodes.append({"id": cid, "label": f"Card: {db_tx.card_id[:16]}", "type": "card", "is_center": False, "risk_flag": False})
            links.append({"source": t_id, "target": cid, "relationship": "USES_CARD"})

        if getattr(db_tx, "device_id", None):
            did = f"dev_{db_tx.device_id}"
            nodes.append({"id": did, "label": f"Device: {db_tx.device_id[:16]}", "type": "device", "is_center": False, "risk_flag": False})
            links.append({"source": t_id, "target": did, "relationship": "ORIGINATES_FROM"})

        if getattr(db_tx, "source_account_id", None):
            aid = f"acc_{db_tx.source_account_id}"
            nodes.append({"id": aid, "label": f"Account: {db_tx.source_account_id[:16]}", "type": "account", "is_center": False, "risk_flag": False})
            links.append({"source": t_id, "target": aid, "relationship": "DEBIT_FROM"})

        if getattr(db_tx, "merchant_id", None):
            mid = f"merch_{db_tx.merchant_id}"
            nodes.append({"id": mid, "label": f"Merchant: {db_tx.merchant_id[:16]}", "type": "merchant", "is_center": False, "risk_flag": False})
            links.append({"source": t_id, "target": mid, "relationship": "TO_MERCHANT"})

    return {
        "transaction_id": transaction_id,
        "nodes": nodes,
        "links": links,
        "source": "relational_fallback",
    }
