from mcp.server.fastmcp import FastMCP

mcp = FastMCP("sample-order-server")

ORDERS = {
    "A100": {
        "customer": "alice",
        "status": "shipped",
        "estimated_delivery": "2026-10-10",
        "items": [{"sku": "KB-01", "quantity": 1}, {"sku": "MS-02", "quantity": 2}],
        "total": 129.97,
    },
    "A200": {
        "customer": "bob",
        "status": "processing",
        "estimated_delivery": "2026-10-14",
        "items": [{"sku": "MN-27", "quantity": 1}],
        "total": 299.0,
    },
    "A300": {
        "customer": "alice",
        "status": "delivered",
        "estimated_delivery": "2026-10-01",
        "items": [{"sku": "HD-09", "quantity": 1}],
        "total": 59.5,
    },
}

INVENTORY = {
    "KB-01": {"name": "Mechanical Keyboard", "price": 79.99, "in_stock": 25},
    "MS-02": {"name": "Wireless Mouse", "price": 24.99, "in_stock": 0},
    "MN-27": {"name": "27-inch Monitor", "price": 299.0, "in_stock": 7},
    "HD-09": {"name": "USB-C Hub", "price": 59.5, "in_stock": 112},
}

CANCELLABLE_STATUSES = {"processing"}


@mcp.tool()
def get_order_status(order_id: str) -> dict:
    """Look up the status and estimated delivery date for an order."""
    order = ORDERS.get(order_id.upper())
    if order is None:
        return {"order_id": order_id, "status": "not found"}
    return {
        "order_id": order_id.upper(),
        "status": order["status"],
        "estimated_delivery": order["estimated_delivery"],
    }


@mcp.tool()
def get_order_details(order_id: str) -> dict:
    """Get the items, quantities, and total price of an order."""
    order = ORDERS.get(order_id.upper())
    if order is None:
        return {"order_id": order_id, "error": "order not found"}
    items = [
        {
            "sku": item["sku"],
            "name": INVENTORY[item["sku"]]["name"],
            "quantity": item["quantity"],
        }
        for item in order["items"]
    ]
    return {"order_id": order_id.upper(), "items": items, "total": order["total"]}


@mcp.tool()
def list_customer_orders(customer: str) -> dict:
    """List all order IDs and their statuses for a customer name."""
    orders = [
        {"order_id": order_id, "status": order["status"]}
        for order_id, order in ORDERS.items()
        if order["customer"] == customer.lower()
    ]
    return {"customer": customer, "orders": orders}


@mcp.tool()
def check_inventory(sku: str) -> dict:
    """Check the product name, price, and stock level for a product SKU."""
    product = INVENTORY.get(sku.upper())
    if product is None:
        return {"sku": sku, "error": "product not found"}
    return {"sku": sku.upper(), **product, "available": product["in_stock"] > 0}


@mcp.tool()
def cancel_order(order_id: str) -> dict:
    """Cancel an order if it has not shipped yet."""
    order = ORDERS.get(order_id.upper())
    if order is None:
        return {"order_id": order_id, "error": "order not found"}
    if order["status"] not in CANCELLABLE_STATUSES:
        return {
            "order_id": order_id.upper(),
            "cancelled": False,
            "reason": f"order is already {order['status']}",
        }
    order["status"] = "cancelled"
    return {"order_id": order_id.upper(), "cancelled": True}


if __name__ == "__main__":
    mcp.run(transport="stdio")
