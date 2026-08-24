# Gold Fish (Beitar Illit) — Store Context

This file documents what we know about the live website (`https://goldfishbeitar.co.il`),
gathered by reading the public site and its public, read-only Supabase API. It's the
context the WhatsApp agent uses until we get direct credentials from the store owner.

## Business info

- **Name:** Gold Fish / גולד פיש — boutique fishmonger & butcher
- **Address:** Rabbi Akiva 23, Beitar Illit
- **Phone:** 02-581-9955
- **WhatsApp:** 972-2-581-9955
- **Email:** z5819955@gmail.com
- **Serves:** Beitar Illit, Beit Shemesh and surrounding areas
- **Site language:** Hebrew (RTL) — customers will mostly write in Hebrew

### Opening hours
| Day | Hours |
|---|---|
| Sunday | Closed |
| Monday–Tuesday | 10:00–17:00 |
| Wednesday–Thursday | 10:00–20:00 |
| Friday | 10:00–12:00 |
| Holidays | Closed |

**Ordering rule:** orders accepted daily until 15:00 to guarantee availability;
customers are encouraged to order 2 days ahead.

### Kosher certifications used across products
עד"ח (Eidah Chareidit), חב"ד שייקביץ, קרסיק חב"ד, רובין, ביסטריצקי — varies per product
(see `hechsher` field per product).

## Product catalog

326 active, customer-facing products across 10 categories (counts as of the scrape date):

| Category (Hebrew) | Products |
|---|---|
| עופות (Poultry) | 122 |
| בשר (Meat) | 67 |
| דגים טריים (Fresh fish) | 56 |
| הודו (Turkey) | 18 |
| גפילטע פיש (Gefilte fish) | 15 |
| שונות (Misc) | 13 |
| פסטרמות (Pastrami) | 12 |
| דגים קפואים (Frozen fish) | 11 |
| דגים טחונים (Ground fish) | 6 |
| הערינגים מעושנים וסלטים (Smoked herring & salads) | 6 |

Full structured data: [`data/products.json`](../data/products.json) and
[`data/categories.json`](../data/categories.json) — pulled live from the real
Supabase project (see below), filtered to only products that are visible,
not deleted, and not internal-only.

## Real backend (discovered, not guessed)

The site is built on **Supabase** (project ref `rawdfxukxncpkqmtmvbg`), found via
the public anon key shipped in the site's own JS bundle — the same access any
visitor's browser has. Using this **read-only**, we confirmed the real schema:

### `products` table
```
id                      uuid
name                    text
description             text
image                   text (Supabase storage URL)
category                text   (denormalized category name)
category_id             uuid   (FK -> categories.id)
hechsher                text   (kosher certification)
price                   numeric (ILS)
unit_type               text   (e.g. יחידה=unit, קילו=kg)
estimated_weight_kg     numeric, nullable
in_stock                boolean
stock_qty               integer, nullable
is_deleted              boolean (soft delete)
is_internal             boolean (hidden from storefront)
is_kosher_for_passover  boolean
parent_product_id       uuid, nullable (links size/variant products together)
recipe_url              text
created_at / updated_at timestamps
```

### `categories` table
```
id                  uuid
category_name       text
category_description text, nullable
sort_order          integer
is_visible          boolean
created_at / updated_at timestamps
```

### Tables that exist but are protected (empty/inaccessible with the public anon key — expected, this is correct RLS security)
- `orders` — exists, RLS hides all rows from anon/unauthenticated access
- `order_items` — same
- `profiles` — same (likely the customer account table)

**This is good news and bad news:** good, because it means the storefront's
order/customer data is properly locked down. Bad, because it means **we cannot
read or write real orders without the store owner giving us either:**
1. A service-role key (full access — would need to be handled very carefully), or
2. A scoped Postgres role/RPC function dedicated to the WhatsApp agent (preferred —
   e.g. a `create_whatsapp_order(...)` Postgres function the owner creates and grants
   to a new role, so the agent never touches raw tables for writes).

## What this means for the current build

- **Product browsing / Q&A / pricing:** can run against the **real, live** Supabase
  `products`/`categories` tables right now, read-only, via the public anon key —
  no extra credentials needed for this part.
- **Placing orders / checking order status:** cannot touch the real `orders` table
  yet. The agent will record orders to a **local pending-orders store** (clearly
  logged as "not yet synced to Supabase") until the store owner provides write
  access. Swapping this for the real table/RPC later is a one-file change — see
  `agent/repositories/orders.py`.
- **Delivery info / business hours / FAQ:** no dedicated table found for this —
  using the hours/area info documented above (from the public site) as static
  context for now. If the store owner has a structured source for this, it should
  replace the static values.

## Open items for the store owner

1. Confirm the `orders` / `order_items` schema (column names) so the agent's order
   payload matches exactly.
2. Provide either a service-role key (with the warning above) or create a
   dedicated `create_whatsapp_order` Postgres RPC function we can call.
3. Confirm how customers are identified today (phone number? email? guest
   checkout?) so WhatsApp orders can be linked to the right customer record.
4. Confirm if there's a real delivery-zones/pricing table, or if delivery info
   should stay as static text.
