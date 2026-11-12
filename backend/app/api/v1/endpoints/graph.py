from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.postgres import get_db
from app.services.graph_service import get_transaction_subgraph
from app.services.transaction_service import get_transaction_by_id

router = APIRouter(prefix="/graph", tags=["Graph Visualizations"])


@router.get(
    "/{transaction_id}",
    status_code=status.HTTP_200_OK,
)
def get_graph_by_transaction_id(
    transaction_id: str,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """
    Retrieve real 1-hop and 2-hop entity relationships for an ingested transaction.
    Returns nodes and edges formatted for Cytoscape.js.
    """
    # Look up transaction in relational DB to provide fallback attributes if Neo4j is offline
    db_tx = get_transaction_by_id(db, transaction_id)
    subgraph = get_transaction_subgraph(transaction_id, db_tx=db_tx)

    if not subgraph["nodes"]:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No graph entities found for transaction '{transaction_id}'.",
        )

    return subgraph

