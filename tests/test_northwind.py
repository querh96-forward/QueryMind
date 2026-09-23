import pytest

from app.northwind import NorthwindSource


def test_northwind_counts_and_views():
    source = NorthwindSource()
    counts = source.row_counts()
    assert sum(counts.values()) == 3204
    assert counts["orders"] == 830
    assert counts["order_details"] == 2155
    names = {item["name"] for item in source.schema()}
    assert "v_order_summary" in names
    assert "v_inventory_status" in names


def test_sql_is_read_only():
    with pytest.raises(ValueError, match="只允许"):
        NorthwindSource().query("DELETE FROM orders")


def test_business_dashboard():
    data = NorthwindSource().dashboard()
    assert data["cards"]["order_count"] == 830
    assert len(data["monthly"]) > 10
