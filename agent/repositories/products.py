"""Read-only access to the store's live product catalog.

This queries the real Supabase `products`/`categories` tables using the
public anon key - the same read-only access the storefront website itself
uses for every visitor. No order or customer data is reachable with this
key (Row Level Security blocks it), only the public catalog.

See docs/STORE_CONTEXT.md for how this was discovered and the full schema.
"""
import logging
import time

import requests

from agent import config

logger = logging.getLogger(__name__)

# How long list_all_product_names() reuses its cached result before
# re-fetching - the catalog doesn't change minute-to-minute, and without
# this every single product-related WhatsApp message would re-fetch the
# whole catalog just to build LLM context.
_CATALOG_CACHE_SECONDS = 300
_catalog_cache: list[dict] | None = None
_catalog_cache_at: float = 0.0

_HEADERS = {
    "apikey": config.SUPABASE_ANON_KEY or "",
    "Authorization": f"Bearer {config.SUPABASE_ANON_KEY or ''}",
}


def _get(table: str, params: dict) -> list[dict]:
    url = f"{config.SUPABASE_URL}/rest/v1/{table}"
    try:
        response = requests.get(url, headers=_HEADERS, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        logger.error("Supabase request to %s failed: %s", table, e)
        return []


def _base_params(limit: int) -> dict:
    return {
        "select": "id,name,description,category,price,unit_type,hechsher,in_stock,parent_product_id",
        "is_deleted": "eq.false",
        "is_internal": "eq.false",
        "limit": str(limit),
    }


def search_products(query: str | None = None, category: str | None = None, limit: int = 8) -> list[dict]:
    """Searches active, customer-facing products by name and/or category.

    Product names are short Hebrew phrases (e.g. "טחון בקר 500 גרם") where
    word order doesn't match how a customer phrases a request (e.g. "בשר
    טחון" = "ground meat"). A strict "name ILIKE the whole phrase" filter
    misses these, so if the exact phrase finds nothing, we fall back to
    broader matches before giving up - logged at each step so it's clear
    which tier actually found the result.

    IMPORTANT ordering: when a specific query is given, word-by-word name
    search is tried BEFORE falling back to category-only. The category
    here is just the LLM's guess at which category a product is in, and
    that guess is sometimes wrong (e.g. "carp roe" was guessed as "gefilte
    fish" instead of "fresh fish") - if category-only ran first, it would
    confidently return 8 unrelated products from the WRONG category
    instead of ever trying to actually find the named product by its real
    name. Category-only is only a sensible fallback when there's no
    specific product name to search for at all.
    """
    logger.info("search_products(query=%r, category=%r)", query, category)

    if query and category:
        params = _base_params(limit)
        params["name"] = f"ilike.*{query}*"
        params["category"] = f"ilike.*{category}*"
        results = _get("products", params)
        if results:
            logger.info("search_products -> %d result(s) (name+category match)", len(results))
            return results
        logger.info("search_products: no match for name+category together, widening to word search")

    if query:
        # Try each significant word on its own (OR'd together), ignoring
        # category entirely - catches both word-order mismatches and a
        # wrongly-guessed category, since the real product name is what
        # actually matters here.
        # Strip punctuation so "הודו:" matches "הודו" in product names
        words = [w.strip('":,./()') for w in query.split()]
        words = [w for w in words if len(w) > 1]
        if words:
            or_filter = ",".join(f"name.ilike.*{w}*" for w in words)
            params = _base_params(limit)
            params["or"] = f"({or_filter})"
            results = _get("products", params)
            if results:
                logger.info("search_products -> %d result(s) (word-by-word match on %r)", len(results), words)
                return results
        logger.info("search_products: no word match either, falling back to category only")

    if category:
        params = _base_params(limit)
        params["category"] = f"ilike.*{category}*"
        results = _get("products", params)
        if results:
            logger.info("search_products -> %d result(s) (category-only match)", len(results))
            return results

    logger.info("search_products -> 0 results for query=%r category=%r (all fallbacks exhausted)", query, category)
    return []


def list_all_product_names() -> list[dict]:
    """All active, customer-facing products' name+category.

    Used to give the LLM the REAL catalog to match a customer's words
    against (see extract_query_terms in nodes.py), instead of asking it to
    blindly translate/guess a Hebrew search term that might not actually
    exist in the catalog - that guessing is what caused real product/
    category mismatches (e.g. "fresh fish" -> a generic translated term
    that coincidentally matched unrelated products instead of any real
    fresh-fish item). Cached briefly since this rarely changes.
    """
    global _catalog_cache, _catalog_cache_at
    if _catalog_cache is not None and (time.time() - _catalog_cache_at) < _CATALOG_CACHE_SECONDS:
        return _catalog_cache

    params = _base_params(500)
    params["select"] = "name,category"
    results = _get("products", params)
    logger.info("list_all_product_names -> %d products (refreshed cache)", len(results))
    _catalog_cache = results
    _catalog_cache_at = time.time()
    return results


def get_product_by_id(product_id: str) -> dict | None:
    results = _get("products", {"select": "*", "id": f"eq.{product_id}", "limit": "1"})
    return results[0] if results else None


def get_parent_products(category: str) -> list[dict]:
    """Fetches top-level (parent) products for a category — those with no parent_product_id.

    These are the second-level selection options shown to the customer after
    they name a category: kashrut brands in meat/poultry, fish species in
    fresh fish, etc. Each parent groups the orderable child products under it.
    """
    params = _base_params(20)
    params["category"] = f"ilike.*{category}*"
    params["parent_product_id"] = "is.null"
    results = _get("products", params)
    logger.info("get_parent_products(category=%r) -> %d results", category, len(results))
    return results


def get_child_products(parent_product_id: str) -> list[dict]:
    """Fetches orderable child products under a given parent product."""
    params = _base_params(20)
    params["parent_product_id"] = f"eq.{parent_product_id}"
    results = _get("products", params)
    logger.info("get_child_products(parent_product_id=%r) -> %d results", parent_product_id, len(results))
    return results


def check_has_children(product_ids: list[str]) -> set[str]:
    """Returns the subset of product_ids that have at least one child product.

    Used to distinguish navigation nodes (groups with sub-products) from
    orderable leaf products. One Supabase query regardless of list size.
    """
    if not product_ids:
        return set()
    id_list = ",".join(product_ids)
    params = {
        "select": "parent_product_id",
        "is_deleted": "eq.false",
        "is_internal": "eq.false",
        "parent_product_id": f"in.({id_list})",
        "limit": "200",
    }
    results = _get("products", params)
    found = {r["parent_product_id"] for r in results if r.get("parent_product_id")}
    logger.info("check_has_children(%d ids) -> %d groups found", len(product_ids), len(found))
    return found


def list_categories() -> list[dict]:
    results = _get(
        "categories",
        {"select": "category_name,category_description", "is_visible": "eq.true", "order": "sort_order"},
    )
    logger.info("list_categories -> %d categories", len(results))
    return results
