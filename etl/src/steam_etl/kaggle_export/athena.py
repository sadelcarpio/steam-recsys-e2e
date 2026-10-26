"""Minimal Athena client: run a statement in the workgroup and wait for it."""

from __future__ import annotations

import time
from typing import Any


class AthenaError(RuntimeError):
    pass


class Athena:
    def __init__(self, client: Any, work_group: str, *, poll_seconds: float = 2.0) -> None:
        self.client = client
        self.work_group = work_group
        self.poll_seconds = poll_seconds

    def run(self, sql: str) -> str:
        """Runs `sql` and waits for it; returns the query execution id."""
        query_id = self.client.start_query_execution(QueryString=sql, WorkGroup=self.work_group)[
            "QueryExecutionId"
        ]
        while True:
            status = self.client.get_query_execution(QueryExecutionId=query_id)["QueryExecution"][
                "Status"
            ]
            state = status["State"]
            if state == "SUCCEEDED":
                return query_id
            if state in ("FAILED", "CANCELLED"):
                reason = status.get("StateChangeReason", "")
                raise AthenaError(f"query {query_id} {state}: {reason}")
            time.sleep(self.poll_seconds)

    def scalar(self, sql: str) -> str:
        """The first column of the first row (Athena returns strings)."""
        query_id = self.run(sql)
        rows = self.client.get_query_results(QueryExecutionId=query_id)["ResultSet"]["Rows"]
        if len(rows) < 2:  # the first row is the header
            raise AthenaError(f"query {query_id} returned no rows")
        return rows[1]["Data"][0]["VarCharValue"]
