
import csv
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

SEED = 42
TOTAL_TRANSACTIONS = 10_000
FRAUD_COUNT = 500

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "synthetic" / "training_transactions.csv"

COLUMNS = [
    "transaction_id",
    "timestamp",
    "user_id",
    "source_account_id",
    "destination_account_id",
    "device_id",
    "ip_id",
    "merchant_id",
    "amount",
    "currency",
    "country",
    "fraud_label",
]


def generate():
    rng = random.Random(SEED)
    anchor = datetime(2026, 1, 1, tzinfo=timezone.utc)

    accounts = [f"A{i:04d}" for i in range(1, 501)]
    users = [f"U{i:04d}" for i in range(1, 351)]
    devices = [f"D{i:04d}" for i in range(1, 151)]
    ips = [f"IP{i:04d}" for i in range(1, 101)]
    merchants = [f"M{i:04d}" for i in range(1, 101)]

    # Stable but independently assigned account ownership.
    account_owner = {
        account: rng.choice(users) for account in accounts
    }

    rows = []

    def add_transaction(
        source,
        destination,
        timestamp,
        amount,
        fraud_label,
        device,
        ip,
    ):
        rows.append({
            "transaction_id": f"TX{len(rows) + 1:06d}",
            "timestamp": timestamp.isoformat(),
            "user_id": account_owner[source],
            "source_account_id": source,
            "destination_account_id": destination,
            "device_id": device,
            "ip_id": ip,
            "merchant_id": rng.choice(merchants),
            "amount": round(amount, 2),
            "currency": "INR",
            "country": "IN",
            "fraud_label": fraud_label,
        })

    # All transactions occupy the same 180-day window.
    # Both classes use overlapping accounts, devices and IPs.
    def transaction_time():
        return anchor + timedelta(
            minutes=rng.randint(0, 180 * 24 * 60)
        )

    # Legitimate transactions.
    for _ in range(TOTAL_TRANSACTIONS - FRAUD_COUNT):
        source, destination = rng.sample(accounts, 2)

        add_transaction(
            source,
            destination,
            transaction_time(),
            rng.uniform(100, 20000),
            0,
            rng.choice(devices),
            rng.choice(ips),
        )

    # Fraud scenario 1: fan-in.
    mule = rng.choice(accounts)
    fanin_sources = rng.sample(
        [a for a in accounts if a != mule], 100
    )
    shared_device = rng.choice(devices)
    shared_ip = rng.choice(ips)

    for source in fanin_sources:
        add_transaction(
            source, mule, transaction_time(),
            rng.uniform(5000, 30000), 1,
            shared_device, shared_ip,
        )

    # Fraud scenario 2: fan-out.
    destinations = rng.sample(
        [a for a in accounts if a != mule], 100
    )

    for destination in destinations:
        add_transaction(
            mule, destination, transaction_time(),
            rng.uniform(4000, 25000), 1,
            shared_device, shared_ip,
        )

    # Fraud scenario 3: shared device across accounts.
    shared_device = rng.choice(devices)
    for source, destination in [
        rng.sample(accounts, 2) for _ in range(100)
    ]:
        add_transaction(
            source, destination, transaction_time(),
            rng.uniform(3000, 20000), 1,
            shared_device, rng.choice(ips),
        )

    # Fraud scenario 4: shared IP across accounts.
    shared_ip = rng.choice(ips)
    for source, destination in [
        rng.sample(accounts, 2) for _ in range(100)
    ]:
        add_transaction(
            source, destination, transaction_time(),
            rng.uniform(3000, 20000), 1,
            rng.choice(devices), shared_ip,
        )

    # Fraud scenario 5: twenty four-account circular networks.
    for _ in range(25):
        cycle = rng.sample(accounts, 4)
        cycle_device = rng.choice(devices)
        cycle_ip = rng.choice(ips)

        for i in range(4):
            add_transaction(
                cycle[i],
                cycle[(i + 1) % 4],
                transaction_time(),
                rng.uniform(5000, 15000),
                1,
                cycle_device,
                cycle_ip,
            )

    # Shuffle and regenerate neutral transaction IDs.
    rng.shuffle(rows)

    for index, row in enumerate(rows, start=1):
        row["transaction_id"] = f"TX{index:06d}"

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    fraud_count = sum(r["fraud_label"] == 1 for r in rows)

    print(f"Dataset: {OUTPUT}")
    print(f"Total: {len(rows)}")
    print(f"Legitimate: {len(rows) - fraud_count}")
    print(f"Fraudulent: {fraud_count}")
    print(f"Fraud rate: {fraud_count / len(rows) * 100:.2f}%")
    print(f"Unique transaction IDs: {len(set(r['transaction_id'] for r in rows))}")
    print(f"Random seed: {SEED}")


if __name__ == "__main__":
    generate()
