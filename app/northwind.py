from __future__ import annotations

import csv
import time
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import sqlglot
from sqlglot import exp
from sqlalchemy import create_engine, text

from app.config import ROOT, get_settings
from app.tool_errors import SQLInputError, SQLPolicyError

TABLES = [
    "categories", "customers", "employees", "orders", "order_details",
    "products", "shippers", "suppliers",
]
VIEWS = [
    "v_order_line_sales", "v_order_summary", "v_product_sales",
    "v_customer_sales", "v_inventory_status",
]
ALLOWED_OBJECTS = set(TABLES + VIEWS)

METRIC_DEFINITIONS = [
    {"name": "销售额", "formula": "unit_price × quantity × (1 - discount)", "description": "订单明细扣除折扣后的实际销售金额"},
    {"name": "订单量", "formula": "COUNT(DISTINCT order_id)", "description": "去重订单数"},
    {"name": "客单价", "formula": "销售额 ÷ 订单量", "description": "每张订单平均贡献的销售额"},
    {"name": "销售数量", "formula": "SUM(quantity)", "description": "订单明细中的商品销售件数"},
    {"name": "按期发货率", "formula": "shipped_date <= required_date 的订单占比", "description": "在要求日期前完成发货的订单比例"},
    {"name": "运费率", "formula": "freight ÷ sales_amount", "description": "运费占订单销售额的比例"},
    {"name": "补货风险", "formula": "units_in_stock <= reorder_level", "description": "库存不高于补货点时判定存在补货风险"},
]

SCHEMA_SQL = """
CREATE SCHEMA IF NOT EXISTS northwind;

CREATE TABLE IF NOT EXISTS northwind.categories (
  category_id INTEGER PRIMARY KEY,
  category_name TEXT NOT NULL,
  description TEXT
);
CREATE TABLE IF NOT EXISTS northwind.customers (
  customer_id VARCHAR(5) PRIMARY KEY,
  company_name TEXT NOT NULL,
  contact_name TEXT,
  contact_title TEXT,
  city TEXT,
  region TEXT,
  country TEXT
);
CREATE TABLE IF NOT EXISTS northwind.employees (
  employee_id INTEGER PRIMARY KEY,
  first_name TEXT NOT NULL,
  last_name TEXT NOT NULL,
  title TEXT,
  hire_date DATE,
  city TEXT,
  country TEXT,
  reports_to INTEGER
);
CREATE TABLE IF NOT EXISTS northwind.shippers (
  shipper_id INTEGER PRIMARY KEY,
  company_name TEXT NOT NULL,
  phone TEXT
);
CREATE TABLE IF NOT EXISTS northwind.suppliers (
  supplier_id INTEGER PRIMARY KEY,
  company_name TEXT NOT NULL,
  contact_name TEXT,
  city TEXT,
  country TEXT
);
CREATE TABLE IF NOT EXISTS northwind.products (
  product_id INTEGER PRIMARY KEY,
  product_name TEXT NOT NULL,
  supplier_id INTEGER REFERENCES northwind.suppliers(supplier_id),
  category_id INTEGER REFERENCES northwind.categories(category_id),
  quantity_per_unit TEXT,
  unit_price NUMERIC(12,2) NOT NULL DEFAULT 0,
  units_in_stock INTEGER NOT NULL DEFAULT 0,
  units_on_order INTEGER NOT NULL DEFAULT 0,
  reorder_level INTEGER NOT NULL DEFAULT 0,
  discontinued BOOLEAN NOT NULL DEFAULT FALSE
);
CREATE TABLE IF NOT EXISTS northwind.orders (
  order_id INTEGER PRIMARY KEY,
  customer_id VARCHAR(5) REFERENCES northwind.customers(customer_id),
  employee_id INTEGER REFERENCES northwind.employees(employee_id),
  order_date DATE,
  required_date DATE,
  shipped_date DATE,
  ship_via INTEGER REFERENCES northwind.shippers(shipper_id),
  freight NUMERIC(12,2) NOT NULL DEFAULT 0,
  ship_city TEXT,
  ship_country TEXT
);
CREATE TABLE IF NOT EXISTS northwind.order_details (
  order_id INTEGER REFERENCES northwind.orders(order_id),
  product_id INTEGER REFERENCES northwind.products(product_id),
  unit_price NUMERIC(12,2) NOT NULL,
  quantity INTEGER NOT NULL,
  discount NUMERIC(8,4) NOT NULL DEFAULT 0,
  PRIMARY KEY(order_id, product_id)
);

CREATE INDEX IF NOT EXISTS idx_orders_order_date ON northwind.orders(order_date);
CREATE INDEX IF NOT EXISTS idx_orders_customer_id ON northwind.orders(customer_id);
CREATE INDEX IF NOT EXISTS idx_order_details_product_id ON northwind.order_details(product_id);
CREATE INDEX IF NOT EXISTS idx_products_category_id ON northwind.products(category_id);
"""

VIEW_SQL = """
CREATE OR REPLACE VIEW northwind.v_order_line_sales AS
SELECT od.order_id, o.order_date, o.customer_id, c.company_name AS customer_name,
       o.employee_id, concat_ws(' ', e.first_name, e.last_name) AS employee_name,
       o.ship_country, od.product_id, p.product_name, p.category_id, cat.category_name,
       od.unit_price, od.quantity, od.discount,
       ROUND(od.unit_price * od.quantity, 2) AS gross_sales,
       ROUND(od.unit_price * od.quantity * od.discount, 2) AS discount_amount,
       ROUND(od.unit_price * od.quantity * (1 - od.discount), 2) AS sales_amount
FROM northwind.order_details od
JOIN northwind.orders o ON o.order_id = od.order_id
JOIN northwind.products p ON p.product_id = od.product_id
JOIN northwind.categories cat ON cat.category_id = p.category_id
LEFT JOIN northwind.customers c ON c.customer_id = o.customer_id
LEFT JOIN northwind.employees e ON e.employee_id = o.employee_id;

CREATE OR REPLACE VIEW northwind.v_order_summary AS
SELECT o.order_id, o.order_date, o.required_date, o.shipped_date, o.customer_id,
       c.company_name AS customer_name, o.employee_id,
       concat_ws(' ', e.first_name, e.last_name) AS employee_name,
       o.ship_country, o.freight,
       ROUND(SUM(od.unit_price * od.quantity * (1 - od.discount)), 2) AS sales_amount,
       SUM(od.quantity) AS item_quantity,
       ROUND(CASE WHEN SUM(od.unit_price * od.quantity * (1 - od.discount)) = 0 THEN 0
                  ELSE o.freight / SUM(od.unit_price * od.quantity * (1 - od.discount)) END, 4) AS freight_rate,
       (o.shipped_date - o.order_date) AS shipping_days,
       CASE WHEN o.shipped_date IS NOT NULL AND o.required_date IS NOT NULL
                 AND o.shipped_date <= o.required_date THEN 1 ELSE 0 END AS on_time
FROM northwind.orders o
JOIN northwind.order_details od ON od.order_id = o.order_id
LEFT JOIN northwind.customers c ON c.customer_id = o.customer_id
LEFT JOIN northwind.employees e ON e.employee_id = o.employee_id
GROUP BY o.order_id, c.company_name, e.first_name, e.last_name;

CREATE OR REPLACE VIEW northwind.v_product_sales AS
SELECT p.product_id, p.product_name, c.category_name, s.company_name AS supplier_name,
       p.units_in_stock, p.units_on_order, p.reorder_level, p.discontinued,
       COUNT(DISTINCT od.order_id) AS order_count,
       COALESCE(SUM(od.quantity), 0) AS sales_quantity,
       ROUND(COALESCE(SUM(od.unit_price * od.quantity * (1 - od.discount)), 0), 2) AS sales_amount
FROM northwind.products p
LEFT JOIN northwind.categories c ON c.category_id = p.category_id
LEFT JOIN northwind.suppliers s ON s.supplier_id = p.supplier_id
LEFT JOIN northwind.order_details od ON od.product_id = p.product_id
GROUP BY p.product_id, c.category_name, s.company_name;

CREATE OR REPLACE VIEW northwind.v_customer_sales AS
SELECT c.customer_id, c.company_name, c.country,
       COUNT(DISTINCT o.order_id) AS order_count,
       ROUND(COALESCE(SUM(od.unit_price * od.quantity * (1 - od.discount)), 0), 2) AS sales_amount,
       ROUND(CASE WHEN COUNT(DISTINCT o.order_id) = 0 THEN 0
                  ELSE COALESCE(SUM(od.unit_price * od.quantity * (1 - od.discount)), 0)
                       / COUNT(DISTINCT o.order_id) END, 2) AS avg_order_value
FROM northwind.customers c
LEFT JOIN northwind.orders o ON o.customer_id = c.customer_id
LEFT JOIN northwind.order_details od ON od.order_id = o.order_id
GROUP BY c.customer_id;

CREATE OR REPLACE VIEW northwind.v_inventory_status AS
SELECT p.product_id, p.product_name, c.category_name, p.units_in_stock,
       p.units_on_order, p.reorder_level, p.discontinued,
       CASE WHEN p.discontinued THEN '已停售'
            WHEN p.units_in_stock <= p.reorder_level AND p.units_on_order = 0 THEN '需要补货'
            WHEN p.units_in_stock <= p.reorder_level THEN '在途补货'
            ELSE '库存正常' END AS inventory_status
FROM northwind.products p
LEFT JOIN northwind.categories c ON c.category_id = p.category_id;
"""


def json_value(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return value


class NorthwindSource:
    """Read-only PostgreSQL adapter for the bundled Northwind schema."""

    def __init__(self) -> None:
        settings = get_settings()
        self.engine = create_engine(
            settings.business_database_url,
            future=True,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=5,
        )

    def validate_sql(self, sql: str) -> str:
        sql = sql.strip()
        if not sql:
            raise SQLInputError("SQL不能为空")
        try:
            statements = [tree for tree in sqlglot.parse(sql, read="postgres") if tree is not None]
        except Exception as exc:
            raise SQLInputError(f"SQL语法无法解析：{exc}") from exc
        if len(statements) != 1:
            raise SQLPolicyError("只允许执行一条只读SELECT查询")
        tree = statements[0]

        forbidden = (exp.Insert, exp.Update, exp.Delete, exp.Create, exp.Drop, exp.Alter, exp.Command, exp.Merge)
        if isinstance(tree, forbidden) or any(tree.find(item) for item in forbidden):
            raise SQLPolicyError("只允许执行只读SELECT查询")
        if not isinstance(tree, (exp.Select, exp.Union, exp.Intersect, exp.Except, exp.Subquery)):
            raise SQLPolicyError("只允许执行只读SELECT查询")

        tables = list(tree.find_all(exp.Table))
        if any(table.catalog or table.db.lower() not in {"", "northwind"} for table in tables):
            raise SQLPolicyError("只允许访问northwind业务数据")
        names = {table.name.lower() for table in tables}
        unknown = names - ALLOWED_OBJECTS
        if unknown:
            raise SQLPolicyError(f"SQL访问了未授权对象：{', '.join(sorted(unknown))}")
        limit = tree.args.get("limit")
        if limit is None:
            tree = tree.limit(200)
        else:
            value = limit.expression
            if not isinstance(value, exp.Literal) or not value.is_int or int(value.this) < 0:
                raise SQLInputError("LIMIT必须为非负整数")
            if int(value.this) > 200:
                tree = tree.limit(200)
        return tree.sql(dialect="postgres")

    def query(self, sql: str) -> dict[str, Any]:
        safe_sql = self.validate_sql(sql)
        started = time.perf_counter()
        with self.engine.connect() as conn:
            with conn.begin():
                conn.execute(text("SET LOCAL TRANSACTION READ ONLY"))
                conn.execute(text("SET LOCAL search_path TO northwind, public"))
                conn.execute(text("SET LOCAL statement_timeout = '15s'"))
                result = conn.execute(text(safe_sql))
                rows = [{key: json_value(value) for key, value in row.items()} for row in result.mappings().all()]
                columns = list(result.keys())
        return {
            "sql": safe_sql,
            "columns": columns,
            "rows": rows,
            "row_count": len(rows),
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }

    def schema(self, compact: bool = False) -> list[dict[str, Any]]:
        sql = """
        SELECT table_name, column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = 'northwind'
        ORDER BY table_name, ordinal_position
        """
        with self.engine.connect() as conn:
            rows = conn.execute(text(sql)).mappings().all()
        grouped: dict[str, list[dict[str, str]]] = {}
        for row in rows:
            grouped.setdefault(row["table_name"], []).append(
                {"name": row["column_name"], "type": row["data_type"]}
            )
        output = []
        for name, columns in grouped.items():
            output.append({"name": name, "columns": columns[:8] if compact else columns})
        return output

    def schema_text(self) -> str:
        lines = []
        for item in self.schema(compact=False):
            columns = ", ".join(f"{col['name']}({col['type']})" for col in item["columns"])
            lines.append(f"- {item['name']}: {columns}")
        return "\n".join(lines)

    def row_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        with self.engine.connect() as conn:
            conn.execute(text("SET search_path TO northwind, public"))
            for table in TABLES:
                counts[table] = int(conn.execute(text(f'SELECT COUNT(*) FROM "{table}"')).scalar_one())
        return counts

    def dashboard(self) -> dict[str, Any]:
        cards = self.query("""
            SELECT ROUND(SUM(sales_amount), 2) AS sales_amount,
                   COUNT(*) AS order_count,
                   ROUND(AVG(sales_amount), 2) AS avg_order_value,
                   ROUND(AVG(on_time) * 100, 2) AS on_time_rate
            FROM v_order_summary
        """)["rows"][0]
        monthly = self.query("""
            SELECT to_char(order_date, 'YYYY-MM') AS month,
                   ROUND(SUM(sales_amount), 2) AS sales_amount,
                   COUNT(*) AS order_count
            FROM v_order_summary
            GROUP BY month ORDER BY month
        """)["rows"]
        categories = self.query("""
            SELECT category_name, ROUND(SUM(sales_amount), 2) AS sales_amount
            FROM v_order_line_sales GROUP BY category_name ORDER BY sales_amount DESC
        """)["rows"]
        countries = self.query("""
            SELECT ship_country AS country, ROUND(SUM(sales_amount), 2) AS sales_amount,
                   COUNT(*) AS order_count
            FROM v_order_summary GROUP BY ship_country ORDER BY sales_amount DESC LIMIT 10
        """)["rows"]
        inventory = self.query("""
            SELECT inventory_status, COUNT(*) AS product_count
            FROM v_inventory_status GROUP BY inventory_status ORDER BY product_count DESC
        """)["rows"]
        return {
            "cards": cards,
            "monthly": monthly,
            "categories": categories,
            "countries": countries,
            "inventory": inventory,
            "row_counts": self.row_counts(),
        }


def csv_rows(table: str) -> list[dict[str, str]]:
    path = ROOT / "data" / "northwind" / f"{table}.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return [{key: (value if value != "" else None) for key, value in row.items()} for row in csv.DictReader(file)]
