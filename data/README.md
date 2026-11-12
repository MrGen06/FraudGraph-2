# Data Contract & Synthetic Scenarios

This directory contains datasets used for graph-based fraud detection, including raw inputs, processed datasets, and synthetic demonstration data.

## Directory Structure

- `raw/`: Unprocessed source files or ingestion dumps.
- `processed/`: Cleaned, normalized, or graph-ready datasets.
- `synthetic/`: Deterministically generated demonstration datasets for development, testing, and demos.

## CSV Schema Contract

All transaction datasets follow the column contract detailed below:

| Column | Type / Format | Description |
| :--- | :--- | :--- |
| `transaction_id` | String | Unique transaction identifier |
| `timestamp` | ISO 8601 (Timestamp) | Transaction timestamp |
| `user_id` | String | Initiating user identifier |
| `source_account_id` | String | Sender account identifier |
| `destination_account_id` | String | Receiver account identifier |
| `device_id` | String | Associated device identifier |
| `ip_id` | String | Source IP address or identifier |
| `merchant_id` | String (nullable) | Merchant identifier (optional) |
| `amount` | Float / Decimal | Transaction amount |
| `currency` | String (ISO 4217) | Currency code (e.g., `USD`) |
| `country` | String (ISO 3166-1) | Transaction country code |
| `fraud_label` | Integer (`0` or `1`) | Target: `0` = legitimate, `1` = fraudulent |

## Deterministic Fraud Scenarios

Synthetic data generation supports deterministic scenarios to test graph heuristics and ML models:

- **Normal transactions**: Baseline peer-to-peer and merchant activity with typical patterns.
- **Rapid fan-in / fan-out**: Multiple accounts routing into a single hub or rapid dispersal to many destinations.
- **Shared-device activity**: Distinct accounts and users transacting from the same physical device or IP.
- **Circular transfers**: Multi-hop fund routing forming closed cycles across accounts.

> [!IMPORTANT]
> **Synthetic Demonstration Data**: These labels are synthetic demonstration labels for testing and validation, not real-world fraud ground truth.
