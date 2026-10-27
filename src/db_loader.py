"""
Database Loader & Relational Engine for ZentroMart VoiceOps 2026.

Loads all 7 core CSV database tables with dtype=str:
- customers (300 rows)
- orders (1,000 rows)
- order_items (1,783 rows)
- products (70 rows)
- pincodes (36 rows)
- holidays (4 rows)
- support_logs (244 rows)

Implements relational joins:
- customers -> orders -> order_items -> products
- orders.delivery_pincode -> pincodes.pincode
- customers.pincode -> pincodes.pincode
- support_logs -> orders / customers

Provides business day calendar calculations based on:
- Fixed reference date: 20 October 2026 (2026-10-20)
- Business days: Monday-Saturday, excluding holidays in data/holidays.csv
- Calendar days: for return eligibility windows
"""
from collections import defaultdict
from datetime import date, datetime, timedelta
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

# Fixed reference date for the 2026 Capstone dataset
REFERENCE_DATE = date(2026, 10, 20)
REFERENCE_DATE_STR = "2026-10-20"


class DatabaseLoader:
    """Manages CSV database loading, relational joins, and calendar date calculations."""

    def __init__(self, data_dir: str | Path | None = None) -> None:
        if data_dir is None:
            # Default to data/ relative to project root
            base_dir = Path(__file__).resolve().parent.parent
            self.data_dir = base_dir / "data"
        else:
            self.data_dir = Path(data_dir)

        self.reference_date: date = REFERENCE_DATE

        # DataFrames
        self.customers_df: pd.DataFrame = pd.DataFrame()
        self.orders_df: pd.DataFrame = pd.DataFrame()
        self.order_items_df: pd.DataFrame = pd.DataFrame()
        self.products_df: pd.DataFrame = pd.DataFrame()
        self.pincodes_df: pd.DataFrame = pd.DataFrame()
        self.holidays_df: pd.DataFrame = pd.DataFrame()
        self.support_logs_df: pd.DataFrame = pd.DataFrame()

        # In-memory dictionary lookups for fast retrieval
        self.customers: Dict[str, Dict[str, str]] = {}
        self.orders: Dict[str, Dict[str, str]] = {}
        self.order_items: Dict[str, List[Dict[str, str]]] = defaultdict(list)
        self.products: Dict[str, Dict[str, str]] = {}
        self.pincodes: Dict[str, Dict[str, str]] = {}
        self.holidays: Set[str] = set()

        self.load_tables()

    def load_tables(self) -> None:
        """Load all 7 core CSV tables using Pandas with dtype=str to preserve IDs."""
        self.customers_df = pd.read_csv(self.data_dir / "customers.csv", dtype=str, keep_default_na=False)
        self.orders_df = pd.read_csv(self.data_dir / "orders.csv", dtype=str, keep_default_na=False)
        self.order_items_df = pd.read_csv(self.data_dir / "order_items.csv", dtype=str, keep_default_na=False)
        self.products_df = pd.read_csv(self.data_dir / "products.csv", dtype=str, keep_default_na=False)
        self.pincodes_df = pd.read_csv(self.data_dir / "pincodes.csv", dtype=str, keep_default_na=False)
        self.holidays_df = pd.read_csv(self.data_dir / "holidays.csv", dtype=str, keep_default_na=False)
        self.support_logs_df = pd.read_csv(self.data_dir / "support_logs.csv", dtype=str, keep_default_na=False)

        # Index dictionaries
        self.customers = {row["customer_id"]: row.to_dict() for _, row in self.customers_df.iterrows()}
        self.orders = {row["order_id"]: row.to_dict() for _, row in self.orders_df.iterrows()}
        for _, row in self.order_items_df.iterrows():
            self.order_items[row["order_id"]].append(row.to_dict())
        self.products = {row["sku"]: row.to_dict() for _, row in self.products_df.iterrows()}
        self.pincodes = {row["pincode"]: row.to_dict() for _, row in self.pincodes_df.iterrows()}
        self.holidays = set(self.holidays_df["date"].tolist())

    # =========================================================================
    # Relational Join Methods
    # =========================================================================

    def get_customer(self, customer_id: str) -> Optional[Dict[str, str]]:
        """Retrieve customer details by customer_id."""
        return self.customers.get(str(customer_id).strip())

    def get_order(self, order_id: str) -> Optional[Dict[str, str]]:
        """Retrieve raw order row by order_id."""
        oid = "".join(ch for ch in str(order_id) if ch.isdigit())
        return self.orders.get(oid)

    def get_order_items(self, order_id: str) -> List[Dict[str, str]]:
        """Retrieve order items for an order_id."""
        oid = "".join(ch for ch in str(order_id) if ch.isdigit())
        return self.order_items.get(oid, [])

    def get_product(self, sku: str) -> Optional[Dict[str, str]]:
        """Retrieve product details by sku."""
        return self.products.get(str(sku).strip().upper())

    def get_product_by_title_or_query(self, query: str) -> Optional[Dict[str, str]]:
        """Fuzzy match a product by title or SKU in query."""
        q_lower = query.lower()
        # Direct SKU match
        for sku, prod in self.products.items():
            if sku.lower() in q_lower:
                return prod
        # Title matching (exact substring)
        for _, prod in self.products.items():
            title_lower = prod["title"].lower()
            if title_lower in q_lower or q_lower in title_lower:
                return prod
            # Match distinctive multi-word tokens
            parts = [p for p in title_lower.split() if len(p) > 3]
            if len(parts) >= 2 and all(part in q_lower for part in parts[:2]):
                return prod
        return None

    def get_pincode_info(self, pincode: str) -> Optional[Dict[str, str]]:
        """Retrieve delivery / serviceability info for a pincode."""
        pin = "".join(ch for ch in str(pincode) if ch.isdigit())
        return self.pincodes.get(pin)

    def get_order_full_details(self, order_id: str) -> Optional[Dict[str, Any]]:
        """
        Execute full relational join for an order:
        orders -> customers -> order_items -> products -> pincodes
        """
        oid = "".join(ch for ch in str(order_id) if ch.isdigit())
        order = self.orders.get(oid)
        if not order:
            return None

        joined = dict(order)

        # Customer join
        customer_id = order.get("customer_id", "")
        customer = self.customers.get(customer_id, {})
        joined["customer"] = customer
        joined["customer_name"] = customer.get("name", "")
        joined["customer_phone"] = customer.get("phone", "")
        joined["is_plus"] = customer.get("is_plus", "False") == "True"

        # Pincode join
        del_pin = order.get("delivery_pincode", "")
        pincode_info = self.pincodes.get(del_pin, {})
        joined["delivery_pincode_info"] = pincode_info

        # Items + Products join
        raw_items = self.order_items.get(oid, [])
        joined_items = []
        for item in raw_items:
            item_dict = dict(item)
            sku = item.get("sku", "")
            prod = self.products.get(sku, {})
            item_dict["product_details"] = prod
            item_dict["brand"] = prod.get("brand", "")
            item_dict["assured"] = prod.get("assured", "False") == "True"
            item_dict["warranty_months"] = prod.get("warranty_months", "0")
            joined_items.append(item_dict)

        joined["items"] = joined_items
        return joined

    def get_joined_orders_dataframe(self) -> pd.DataFrame:
        """
        Generate a fully denormalized Pandas DataFrame joining:
        customers -> orders -> order_items -> products -> pincodes.
        """
        # Join order_items with products
        items_prod = self.order_items_df.merge(
            self.products_df,
            on="sku",
            how="left",
            suffixes=("", "_product")
        )
        # Join orders with customers
        orders_cust = self.orders_df.merge(
            self.customers_df,
            on="customer_id",
            how="left",
            suffixes=("", "_customer")
        )
        # Join with items
        joined = orders_cust.merge(
            items_prod,
            on="order_id",
            how="left",
            suffixes=("", "_item")
        )
        # Join delivery pincode
        joined = joined.merge(
            self.pincodes_df,
            left_on="delivery_pincode",
            right_on="pincode",
            how="left",
            suffixes=("", "_pincode")
        )
        return joined

    # =========================================================================
    # Calendar & Business Days Logic
    # =========================================================================

    def parse_date(self, d: str | date | datetime) -> date:
        """Safely parse input string or datetime into a date object."""
        if isinstance(d, datetime):
            return d.date()
        if isinstance(d, date):
            return d
        clean_d = str(d).strip().split()[0]  # strip time component if present
        return datetime.strptime(clean_d, "%Y-%m-%d").date()

    def is_business_day(self, d: str | date | datetime) -> bool:
        """
        Check if a given date is a business day:
        - Monday to Saturday (d.weekday() in 0..5, where Sunday is 6)
        - Not in data/holidays.csv
        """
        dt = self.parse_date(d)
        if dt.weekday() == 6:  # Sunday
            return False
        date_str = dt.strftime("%Y-%m-%d")
        if date_str in self.holidays:
            return False
        return True

    def add_business_days(self, start_date: str | date | datetime, num_days: int) -> date:
        """
        Add `num_days` business days to `start_date`.
        Business days are Mon-Sat excluding holidays.
        """
        curr = self.parse_date(start_date)
        added = 0
        while added < num_days:
            curr += timedelta(days=1)
            if self.is_business_day(curr):
                added += 1
        return curr

    def count_business_days_between(
        self, start_date: str | date | datetime, end_date: str | date | datetime
    ) -> int:
        """
        Count business days strictly between (start_date, end_date] (end_date inclusive).
        """
        s = self.parse_date(start_date)
        e = self.parse_date(end_date)
        if s >= e:
            return 0
        curr = s
        count = 0
        while curr < e:
            curr += timedelta(days=1)
            if self.is_business_day(curr):
                count += 1
        return count

    def add_calendar_days(self, start_date: str | date | datetime, num_days: int) -> date:
        """Add calendar days (used for return eligibility windows)."""
        curr = self.parse_date(start_date)
        return curr + timedelta(days=num_days)

    def calendar_days_between(
        self, start_date: str | date | datetime, end_date: str | date | datetime
    ) -> int:
        """Count calendar days between start_date and end_date."""
        s = self.parse_date(start_date)
        e = self.parse_date(end_date)
        return (e - s).days

    def format_date_readable(self, d: str | date | datetime, lang: str = "en") -> str:
        """Format date into standard readable representation (e.g. '24 October 2026')."""
        dt = self.parse_date(d)
        month_names_en = [
            "", "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"
        ]
        month_names_hi = [
            "", "जनवरी", "फ़रवरी", "मार्च", "अप्रैल", "मई", "जून",
            "जुलाई", "अगस्त", "सितंबर", "अक्टूबर", "नवंबर", "दिसंबर"
        ]
        if lang == "hi":
            return f"{dt.day} {month_names_hi[dt.month]} {dt.year}"
        return f"{dt.day} {month_names_en[dt.month]} {dt.year}"
