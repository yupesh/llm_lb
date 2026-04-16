"""OpenAI-function-calling JSON schemas for the 15 retail tools.

These schemas are passed to the LLM via the `tools` param of the chat
completions endpoint. We hand-write them rather than auto-generate from
pydantic signatures because the faithfulness of argument descriptions
(and `Args:` prose from the original tau2 docstrings) materially changes
agent behaviour — we want to preserve the original phrasing verbatim.
"""
from __future__ import annotations


def _t(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


RETAIL_TOOL_SCHEMAS: list[dict] = [
    _t(
        "calculate",
        "Calculate the result of a mathematical expression.",
        {
            "expression": {
                "type": "string",
                "description": "The mathematical expression to calculate, such as '2 + 2'. The expression can contain numbers, operators (+, -, *, /), parentheses, and spaces.",
            },
        },
        ["expression"],
    ),
    _t(
        "cancel_pending_order",
        "Cancel a pending order. If the order is already processed or delivered, it cannot be cancelled. The agent needs to explain the cancellation detail and ask for explicit user confirmation (yes/no) to proceed. If the user confirms, the order status will be changed to 'cancelled' and the payment will be refunded. The refund will be added to the user's gift card balance immediately if the payment was made using a gift card, otherwise the refund would take 5-7 business days to process. The function returns the order details after the cancellation.",
        {
            "order_id": {
                "type": "string",
                "description": "The order id, such as '#W0000000'. Be careful there is a '#' symbol at the beginning of the order id.",
            },
            "reason": {
                "type": "string",
                "enum": ["no longer needed", "ordered by mistake"],
                "description": "The reason for cancellation.",
            },
        },
        ["order_id", "reason"],
    ),
    _t(
        "exchange_delivered_order_items",
        "Exchange items in a delivered order to new items of the same product type. For a delivered order, return or exchange can be only done once by the agent. The agent needs to explain the exchange detail and ask for explicit user confirmation (yes/no) to proceed.",
        {
            "order_id": {"type": "string", "description": "The order id, such as '#W0000000'."},
            "item_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "The item ids to be exchanged, each such as '1008292230'. There could be duplicate items in the list.",
            },
            "new_item_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "The item ids to be exchanged for. Each new item id should match the item id in the same position and be of the same product.",
            },
            "payment_method_id": {
                "type": "string",
                "description": "The payment method id to pay or receive refund for the item price difference, such as 'gift_card_0000000' or 'credit_card_0000000'.",
            },
        },
        ["order_id", "item_ids", "new_item_ids", "payment_method_id"],
    ),
    _t(
        "find_user_id_by_name_zip",
        "Find user id by first name, last name, and zip code. If the user is not found, the function will return an error message. By default, find user id by email, and only call this function if the user is not found by email or cannot remember email.",
        {
            "first_name": {"type": "string", "description": "The first name of the customer, such as 'John'."},
            "last_name": {"type": "string", "description": "The last name of the customer, such as 'Doe'."},
            "zip": {"type": "string", "description": "The zip code of the customer, such as '12345'."},
        },
        ["first_name", "last_name", "zip"],
    ),
    _t(
        "find_user_id_by_email",
        "Find user id by email. If the user is not found, the function will return an error message.",
        {"email": {"type": "string", "description": "The email of the user, such as 'something@example.com'."}},
        ["email"],
    ),
    _t(
        "get_order_details",
        "Get the status and details of an order.",
        {"order_id": {"type": "string", "description": "The order id, such as '#W0000000'."}},
        ["order_id"],
    ),
    _t(
        "get_product_details",
        "Get the inventory details of a product.",
        {"product_id": {"type": "string", "description": "The product id, such as '6086499569'. Be careful the product id is different from the item id."}},
        ["product_id"],
    ),
    _t(
        "get_user_details",
        "Get the details of a user, including their orders.",
        {"user_id": {"type": "string", "description": "The user id, such as 'sara_doe_496'."}},
        ["user_id"],
    ),
    _t(
        "list_all_product_types",
        "List the name and product id of all product types. Each product type has a variety of different items with unique item ids and options.",
        {},
        [],
    ),
    _t(
        "modify_pending_order_address",
        "Modify the shipping address of a pending order. The agent needs to explain the modification detail and ask for explicit user confirmation (yes/no) to proceed.",
        {
            "order_id": {"type": "string", "description": "The order id, such as '#W0000000'."},
            "address1": {"type": "string", "description": "Primary address line."},
            "address2": {"type": "string", "description": "Secondary address line."},
            "city": {"type": "string"},
            "state": {"type": "string"},
            "country": {"type": "string"},
            "zip": {"type": "string"},
        },
        ["order_id", "address1", "address2", "city", "state", "country", "zip"],
    ),
    _t(
        "modify_pending_order_items",
        "Modify items in a pending order to new items of the same product type. For a pending order, this function can only be called once. The agent needs to explain the exchange detail and ask for explicit user confirmation (yes/no) to proceed.",
        {
            "order_id": {"type": "string", "description": "The order id, such as '#W0000000'."},
            "item_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "The item ids to be modified, each such as '1008292230'. There could be duplicate items in the list.",
            },
            "new_item_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "The item ids to be modified for. Each new item id should match the item id in the same position and be of the same product.",
            },
            "payment_method_id": {
                "type": "string",
                "description": "The payment method id to pay or receive refund for the item price difference, such as 'gift_card_0000000' or 'credit_card_0000000'.",
            },
        },
        ["order_id", "item_ids", "new_item_ids", "payment_method_id"],
    ),
    _t(
        "modify_pending_order_payment",
        "Modify the payment method of a pending order. The agent needs to explain the modification detail and ask for explicit user confirmation (yes/no) to proceed.",
        {
            "order_id": {"type": "string", "description": "The order id, such as '#W0000000'."},
            "payment_method_id": {
                "type": "string",
                "description": "The payment method id to pay or receive refund for the item price difference, such as 'gift_card_0000000' or 'credit_card_0000000'.",
            },
        },
        ["order_id", "payment_method_id"],
    ),
    _t(
        "modify_user_address",
        "Modify the default address of a user. The agent needs to explain the modification detail and ask for explicit user confirmation (yes/no) to proceed.",
        {
            "user_id": {"type": "string", "description": "The user id, such as 'sara_doe_496'."},
            "address1": {"type": "string", "description": "Primary address line."},
            "address2": {"type": "string", "description": "Secondary address line."},
            "city": {"type": "string"},
            "state": {"type": "string"},
            "country": {"type": "string"},
            "zip": {"type": "string"},
        },
        ["user_id", "address1", "address2", "city", "state", "country", "zip"],
    ),
    _t(
        "return_delivered_order_items",
        "Return some items of a delivered order. The order status will be changed to 'return requested'. The agent needs to explain the return detail and ask for explicit user confirmation (yes/no) to proceed. The refund will be added to the original payment method after 5-7 business days.",
        {
            "order_id": {"type": "string", "description": "The order id, such as '#W0000000'."},
            "item_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "The item ids to be returned, each such as '1008292230'. There could be duplicate items in the list.",
            },
            "payment_method_id": {
                "type": "string",
                "description": "The payment method id to pay or receive refund for the item price difference, such as 'gift_card_0000000' or 'credit_card_0000000'. This should be the original payment method.",
            },
        },
        ["order_id", "item_ids", "payment_method_id"],
    ),
    _t(
        "transfer_to_human_agents",
        "Transfer the user to a human agent, with a summary of the user's issue. Only transfer if the user explicitly asks for a human agent, or if the user's issue cannot be resolved by the agent with the available tools.",
        {
            "summary": {
                "type": "string",
                "description": "A summary of the user's issue.",
            },
        },
        ["summary"],
    ),
]
