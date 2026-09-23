from types import SimpleNamespace

import pytest
import sqlglot

from app.northwind import NorthwindSource
from app.tool_errors import SQLInputError, SQLPolicyError, classify_tool_error


@pytest.mark.parametrize("sql", [
    "DELETE FROM orders", "UPDATE orders SET freight=0", "SELECT * FROM orders; SELECT * FROM customers",
    "SELECT * FROM private.customers", "SELECT * FROM runs",
    "WITH changed AS (DELETE FROM orders RETURNING *) SELECT * FROM changed",
])
def test_disallowed_query_never_reaches_database(sql):
    source = object.__new__(NorthwindSource)
    with pytest.raises(SQLPolicyError):
        source.validate_sql(sql)


@pytest.mark.parametrize("sql,expected", [
    ("SELECT * FROM orders", 200), ("SELECT * FROM orders LIMIT 9999", 200),
    ("SELECT * FROM orders LIMIT 5", 5), ("SELECT * FROM orders LIMIT 0", 0),
])
def test_result_limit_is_enforced(sql, expected):
    source = object.__new__(NorthwindSource)
    result = sqlglot.parse_one(source.validate_sql(sql), read="postgres")
    assert int(result.args["limit"].expression.this) == expected


@pytest.mark.parametrize("state,kind,retryable", [
    ("42703", "schema_error", True), ("42P01", "schema_error", True),
    ("42601", "sql_syntax", True), ("57014", "query_timeout", True),
    ("42501", "permission_denied", False), ("25006", "permission_denied", False),
    ("08006", "database_unavailable", True),
])
def test_wrapped_postgres_errors_have_recovery_policy(state, kind, retryable):
    exc = RuntimeError("database operation failed")
    exc.orig = SimpleNamespace(sqlstate=state)
    result = classify_tool_error(exc)
    assert result["error_type"] == kind
    assert result["retryable"] is retryable


def test_parse_errors_are_repairable_and_policy_errors_are_not():
    assert classify_tool_error(SQLInputError("syntax"))["retryable"] is True
    assert classify_tool_error(SQLPolicyError("permission"))["retryable"] is False
